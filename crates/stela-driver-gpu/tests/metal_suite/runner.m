/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * The Metal suite's runner: executes every case of a plan that
 * `cargo xtask gpu metal-suite plan` wrote and stores each output's bytes. The same
 * arm64 binary runs on a Mac, giving the reference, and in a macOS guest on
 * Stela, so both read the same libraries and inputs.
 *
 *   runner PLAN_DIR RESULTS_DIR [--only SUBSTRING]... [--skip-done]
 *
 * Each case writes RESULTS_DIR/<case>/status.json before it runs ("running")
 * and after ("ok", "error" or "timeout", with the GPU time its work took as
 * gpu_ms), and one <output>.bin per output. A
 * case that does not complete within its timeout stops the runner with exit
 * status 2, since the device may not recover; --skip-done resumes after it.
 * Nothing here interprets results: the plan says what to bind and dispatch.
 *
 * The binary runs from Runner.app, whose Share.xpc service is the same
 * binary: launched by launchd for `host` cases that share a texture and an
 * event with another process.
 */
#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <CoreText/CoreText.h>
#import <IOSurface/IOSurface.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <MetalKit/MetalKit.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <QuartzCore/QuartzCore.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <servers/bootstrap.h>
#include <objc/message.h>
#include <stdio.h>
#include <sys/mman.h>

static NSString *plan_dir;
/* GPU time of the case's own command buffers, not its output readback. */
static double gpu_seconds;
static BOOL timing;

@interface Case : NSObject
@property(retain) NSDictionary *spec;
@property(retain) NSString *name;
@property(retain) NSString *dir;
@property(retain) id<MTLDevice> device;
@property(retain) NSMutableDictionary<NSString *, id<MTLBuffer>> *buffers;
@property(retain) NSMutableDictionary<NSString *, id<MTLTexture>> *textures;
@property(retain) NSMutableDictionary<NSString *, id<MTLSamplerState>> *samplers;
@property(retain) NSMutableArray *resident;
@property(retain) NSMutableDictionary<NSString *, id<MTLFence>> *fences;
@property(retain) NSMutableDictionary<NSString *, id<MTLEvent>> *events;
@property(retain) NSMutableDictionary<NSString *, id<MTLIndirectCommandBuffer>> *icbs;
@property(retain) NSMutableDictionary<NSString *, id<MTLHeap>> *heaps;
@property(retain) NSMutableDictionary *libraries;
@property(retain) NSDictionary *plan;
@property(retain) id<MTLCommandQueue> queue;
@end
@implementation Case
@end

static NSException *failure(NSString *format, ...) NS_FORMAT_FUNCTION(1, 2);
static NSException *failure(NSString *format, ...)
{
	va_list args;
	va_start(args, format);
	NSString *reason = [[NSString alloc] initWithFormat:format arguments:args];
	va_end(args);
	return [NSException exceptionWithName:@"CaseFailure" reason:reason userInfo:nil];
}

static NSUInteger number(NSDictionary *d, NSString *key, NSUInteger fallback)
{
	id v = d[key];
	return v ? [v unsignedIntegerValue] : fallback;
}

/* Every named value a case looked up, by table, and every table's names:
 * `check` holds a complete run to using each one. */
static NSMutableDictionary<NSString *, NSMutableSet<NSString *> *> *case_lookups, *lookup_tables;

/* How the current case creates its functions and its libraries from
 * source: "name" or "sync" calls by default, or the case's `creation`
 * ("descriptor", "async"), which also selects its pipelines' calls. */
static NSString *creation;

/* Waits for an asynchronous creation's handler to run. */
static void await_handler(void (^start)(dispatch_semaphore_t done))
{
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	start(done);
	if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC)))
		@throw [NSException exceptionWithName:@"CaseFailure" reason:@"creation handler never ran" userInfo:nil];
}

/* Records that the current case used these names of a table. */
static void note_lookups(NSDictionary *table, NSString *kind, NSArray<NSString *> *names)
{
	@synchronized(lookup_tables) {
		if (!lookup_tables[kind])
			lookup_tables[kind] = [NSMutableSet new];
		[lookup_tables[kind] addObjectsFromArray:table.allKeys];
		if (!case_lookups[kind])
			case_lookups[kind] = [NSMutableSet new];
		[case_lookups[kind] addObjectsFromArray:names];
	}
}

static NSUInteger lookup(NSDictionary<NSString *, NSNumber *> *table, NSString *kind, id name)
{
	if ([name isKindOfClass:[NSNumber class]])
		return [name unsignedIntegerValue];
	note_lookups(table, kind, @[ name ]);
	NSNumber *v = table[name];
	if (!v)
		@throw failure(@"unknown %@ '%@'", kind, name);
	return v.unsignedIntegerValue;
}

static NSDictionary<NSString *, NSNumber *> *primitive_types(void)
{
	return @{@"point" : @(MTLPrimitiveTypePoint), @"line" : @(MTLPrimitiveTypeLine), @"line_strip" : @(MTLPrimitiveTypeLineStrip),
		 @"triangle" : @(MTLPrimitiveTypeTriangle), @"triangle_strip" : @(MTLPrimitiveTypeTriangleStrip)};
}

static NSDictionary<NSString *, NSNumber *> *pixel_formats(void)
{
	static NSDictionary *t;
	if (!t)
		t = @{
			@"a8unorm" : @(MTLPixelFormatA8Unorm), @"r8unorm" : @(MTLPixelFormatR8Unorm),
			@"r8unorm_srgb" : @(MTLPixelFormatR8Unorm_sRGB), @"r8snorm" : @(MTLPixelFormatR8Snorm),
			@"r8uint" : @(MTLPixelFormatR8Uint), @"r8sint" : @(MTLPixelFormatR8Sint),
			@"r16unorm" : @(MTLPixelFormatR16Unorm), @"r16snorm" : @(MTLPixelFormatR16Snorm),
			@"r16uint" : @(MTLPixelFormatR16Uint), @"r16sint" : @(MTLPixelFormatR16Sint),
			@"r16float" : @(MTLPixelFormatR16Float), @"rg8unorm" : @(MTLPixelFormatRG8Unorm),
			@"rg8snorm" : @(MTLPixelFormatRG8Snorm), @"rg8uint" : @(MTLPixelFormatRG8Uint),
			@"rg8sint" : @(MTLPixelFormatRG8Sint), @"rg8unorm_srgb" : @(MTLPixelFormatRG8Unorm_sRGB),
			@"b5g6r5unorm" : @(MTLPixelFormatB5G6R5Unorm), @"a1bgr5unorm" : @(MTLPixelFormatA1BGR5Unorm),
			@"abgr4unorm" : @(MTLPixelFormatABGR4Unorm), @"bgr5a1unorm" : @(MTLPixelFormatBGR5A1Unorm),
			@"r32uint" : @(MTLPixelFormatR32Uint), @"r32sint" : @(MTLPixelFormatR32Sint),
			@"r32float" : @(MTLPixelFormatR32Float), @"rg16unorm" : @(MTLPixelFormatRG16Unorm),
			@"rg16snorm" : @(MTLPixelFormatRG16Snorm), @"rg16uint" : @(MTLPixelFormatRG16Uint),
			@"rg16sint" : @(MTLPixelFormatRG16Sint), @"rg16float" : @(MTLPixelFormatRG16Float),
			@"rgba8unorm" : @(MTLPixelFormatRGBA8Unorm), @"rgba8unorm_srgb" : @(MTLPixelFormatRGBA8Unorm_sRGB),
			@"rgba8snorm" : @(MTLPixelFormatRGBA8Snorm), @"rgba8uint" : @(MTLPixelFormatRGBA8Uint),
			@"rgba8sint" : @(MTLPixelFormatRGBA8Sint), @"bgra8unorm" : @(MTLPixelFormatBGRA8Unorm),
			@"bgra8unorm_srgb" : @(MTLPixelFormatBGRA8Unorm_sRGB),
			@"rgb10a2unorm" : @(MTLPixelFormatRGB10A2Unorm), @"rgb10a2uint" : @(MTLPixelFormatRGB10A2Uint),
			@"bgr10a2unorm" : @(MTLPixelFormatBGR10A2Unorm), @"rg11b10float" : @(MTLPixelFormatRG11B10Float),
			@"rgb9e5float" : @(MTLPixelFormatRGB9E5Float), @"rg32uint" : @(MTLPixelFormatRG32Uint),
			@"rg32sint" : @(MTLPixelFormatRG32Sint), @"rg32float" : @(MTLPixelFormatRG32Float),
			@"rgba16unorm" : @(MTLPixelFormatRGBA16Unorm), @"rgba16snorm" : @(MTLPixelFormatRGBA16Snorm),
			@"rgba16uint" : @(MTLPixelFormatRGBA16Uint), @"rgba16sint" : @(MTLPixelFormatRGBA16Sint),
			@"rgba16float" : @(MTLPixelFormatRGBA16Float), @"rgba32uint" : @(MTLPixelFormatRGBA32Uint),
			@"rgba32sint" : @(MTLPixelFormatRGBA32Sint), @"rgba32float" : @(MTLPixelFormatRGBA32Float),
			@"bc1_rgba" : @(MTLPixelFormatBC1_RGBA), @"bc3_rgba" : @(MTLPixelFormatBC3_RGBA),
			@"bc4_runorm" : @(MTLPixelFormatBC4_RUnorm), @"bc5_rgunorm" : @(MTLPixelFormatBC5_RGUnorm),
			@"bc7_rgbaunorm" : @(MTLPixelFormatBC7_RGBAUnorm), @"bc7_rgbaunorm_srgb" : @(MTLPixelFormatBC7_RGBAUnorm_sRGB),
			@"bc1_rgba_srgb" : @(MTLPixelFormatBC1_RGBA_sRGB), @"bc2_rgba" : @(MTLPixelFormatBC2_RGBA),
			@"bc2_rgba_srgb" : @(MTLPixelFormatBC2_RGBA_sRGB), @"bc3_rgba_srgb" : @(MTLPixelFormatBC3_RGBA_sRGB),
			@"bc4_rsnorm" : @(MTLPixelFormatBC4_RSnorm), @"bc5_rgsnorm" : @(MTLPixelFormatBC5_RGSnorm),
			@"bc6h_rgbfloat" : @(MTLPixelFormatBC6H_RGBFloat), @"bc6h_rgbufloat" : @(MTLPixelFormatBC6H_RGBUfloat),
			@"etc2_rgb8" : @(MTLPixelFormatETC2_RGB8), @"etc2_rgb8_srgb" : @(MTLPixelFormatETC2_RGB8_sRGB),
			@"etc2_rgb8a1" : @(MTLPixelFormatETC2_RGB8A1), @"etc2_rgb8a1_srgb" : @(MTLPixelFormatETC2_RGB8A1_sRGB),
			@"eac_rgba8" : @(MTLPixelFormatEAC_RGBA8), @"eac_rgba8_srgb" : @(MTLPixelFormatEAC_RGBA8_sRGB),
			@"eac_r11unorm" : @(MTLPixelFormatEAC_R11Unorm), @"eac_r11snorm" : @(MTLPixelFormatEAC_R11Snorm),
			@"eac_rg11unorm" : @(MTLPixelFormatEAC_RG11Unorm), @"eac_rg11snorm" : @(MTLPixelFormatEAC_RG11Snorm),
			@"astc_4x4_ldr" : @(MTLPixelFormatASTC_4x4_LDR), @"astc_4x4_srgb" : @(MTLPixelFormatASTC_4x4_sRGB),
			@"astc_4x4_hdr" : @(MTLPixelFormatASTC_4x4_HDR), @"astc_6x6_hdr" : @(MTLPixelFormatASTC_6x6_HDR),
			@"astc_8x8_hdr" : @(MTLPixelFormatASTC_8x8_HDR), @"astc_12x12_hdr" : @(MTLPixelFormatASTC_12x12_HDR),
			@"astc_5x4_ldr" : @(MTLPixelFormatASTC_5x4_LDR), @"astc_5x4_srgb" : @(MTLPixelFormatASTC_5x4_sRGB),
			@"astc_5x5_ldr" : @(MTLPixelFormatASTC_5x5_LDR), @"astc_5x5_srgb" : @(MTLPixelFormatASTC_5x5_sRGB),
			@"astc_6x5_ldr" : @(MTLPixelFormatASTC_6x5_LDR), @"astc_6x5_srgb" : @(MTLPixelFormatASTC_6x5_sRGB),
			@"astc_6x6_ldr" : @(MTLPixelFormatASTC_6x6_LDR), @"astc_6x6_srgb" : @(MTLPixelFormatASTC_6x6_sRGB),
			@"astc_8x5_ldr" : @(MTLPixelFormatASTC_8x5_LDR), @"astc_8x5_srgb" : @(MTLPixelFormatASTC_8x5_sRGB),
			@"astc_8x6_ldr" : @(MTLPixelFormatASTC_8x6_LDR), @"astc_8x6_srgb" : @(MTLPixelFormatASTC_8x6_sRGB),
			@"astc_8x8_ldr" : @(MTLPixelFormatASTC_8x8_LDR), @"astc_8x8_srgb" : @(MTLPixelFormatASTC_8x8_sRGB),
			@"astc_10x5_ldr" : @(MTLPixelFormatASTC_10x5_LDR), @"astc_10x5_srgb" : @(MTLPixelFormatASTC_10x5_sRGB),
			@"astc_10x6_ldr" : @(MTLPixelFormatASTC_10x6_LDR), @"astc_10x6_srgb" : @(MTLPixelFormatASTC_10x6_sRGB),
			@"astc_10x8_ldr" : @(MTLPixelFormatASTC_10x8_LDR), @"astc_10x8_srgb" : @(MTLPixelFormatASTC_10x8_sRGB),
			@"astc_10x10_ldr" : @(MTLPixelFormatASTC_10x10_LDR), @"astc_10x10_srgb" : @(MTLPixelFormatASTC_10x10_sRGB),
			@"astc_12x10_ldr" : @(MTLPixelFormatASTC_12x10_LDR), @"astc_12x10_srgb" : @(MTLPixelFormatASTC_12x10_sRGB),
			@"astc_12x12_ldr" : @(MTLPixelFormatASTC_12x12_LDR), @"astc_12x12_srgb" : @(MTLPixelFormatASTC_12x12_sRGB),
			@"pvrtc_rgb_2bpp" : @(MTLPixelFormatPVRTC_RGB_2BPP), @"pvrtc_rgb_2bpp_srgb" : @(MTLPixelFormatPVRTC_RGB_2BPP_sRGB),
			@"pvrtc_rgb_4bpp" : @(MTLPixelFormatPVRTC_RGB_4BPP), @"pvrtc_rgb_4bpp_srgb" : @(MTLPixelFormatPVRTC_RGB_4BPP_sRGB),
			@"pvrtc_rgba_2bpp" : @(MTLPixelFormatPVRTC_RGBA_2BPP), @"pvrtc_rgba_2bpp_srgb" : @(MTLPixelFormatPVRTC_RGBA_2BPP_sRGB),
			@"pvrtc_rgba_4bpp" : @(MTLPixelFormatPVRTC_RGBA_4BPP), @"pvrtc_rgba_4bpp_srgb" : @(MTLPixelFormatPVRTC_RGBA_4BPP_sRGB),
			@"depth24unorm_stencil8" : @(MTLPixelFormatDepth24Unorm_Stencil8),
			@"x32_stencil8" : @(MTLPixelFormatX32_Stencil8), @"x24_stencil8" : @(MTLPixelFormatX24_Stencil8),
			@"depth16unorm" : @(MTLPixelFormatDepth16Unorm), @"depth32float" : @(MTLPixelFormatDepth32Float),
			@"stencil8" : @(MTLPixelFormatStencil8),
			@"depth32float_stencil8" : @(MTLPixelFormatDepth32Float_Stencil8),
			@"bgr10_xr" : @(MTLPixelFormatBGR10_XR), @"bgr10_xr_srgb" : @(MTLPixelFormatBGR10_XR_sRGB),
			@"bgra10_xr" : @(MTLPixelFormatBGRA10_XR),
			@"bgra10_xr_srgb" : @(MTLPixelFormatBGRA10_XR_sRGB),
			/* Private formats without SDK names: 10-bit extended-range
			 * color in one plane, 8-bit alpha in another. */
			@"rgb10a8_2p_xr10" : @550, @"rgb10a8_2p_xr10_srgb" : @551,
		};
	return t;
}

static NSDictionary<NSString *, NSNumber *> *vertex_formats(void)
{
	static NSDictionary *t;
	if (!t)
		t = @{
			@"uchar2" : @(MTLVertexFormatUChar2), @"uchar3" : @(MTLVertexFormatUChar3),
			@"uchar4" : @(MTLVertexFormatUChar4), @"char2" : @(MTLVertexFormatChar2),
			@"char3" : @(MTLVertexFormatChar3), @"char4" : @(MTLVertexFormatChar4),
			@"uchar2normalized" : @(MTLVertexFormatUChar2Normalized),
			@"uchar3normalized" : @(MTLVertexFormatUChar3Normalized),
			@"uchar4normalized" : @(MTLVertexFormatUChar4Normalized),
			@"char2normalized" : @(MTLVertexFormatChar2Normalized),
			@"char3normalized" : @(MTLVertexFormatChar3Normalized),
			@"char4normalized" : @(MTLVertexFormatChar4Normalized),
			@"ushort2" : @(MTLVertexFormatUShort2), @"ushort3" : @(MTLVertexFormatUShort3),
			@"ushort4" : @(MTLVertexFormatUShort4), @"short2" : @(MTLVertexFormatShort2),
			@"short3" : @(MTLVertexFormatShort3), @"short4" : @(MTLVertexFormatShort4),
			@"ushort2normalized" : @(MTLVertexFormatUShort2Normalized),
			@"ushort3normalized" : @(MTLVertexFormatUShort3Normalized),
			@"ushort4normalized" : @(MTLVertexFormatUShort4Normalized),
			@"short2normalized" : @(MTLVertexFormatShort2Normalized),
			@"short3normalized" : @(MTLVertexFormatShort3Normalized),
			@"short4normalized" : @(MTLVertexFormatShort4Normalized),
			@"half2" : @(MTLVertexFormatHalf2), @"half3" : @(MTLVertexFormatHalf3),
			@"half4" : @(MTLVertexFormatHalf4), @"float" : @(MTLVertexFormatFloat),
			@"float2" : @(MTLVertexFormatFloat2), @"float3" : @(MTLVertexFormatFloat3),
			@"float4" : @(MTLVertexFormatFloat4), @"int" : @(MTLVertexFormatInt),
			@"int2" : @(MTLVertexFormatInt2), @"int3" : @(MTLVertexFormatInt3),
			@"int4" : @(MTLVertexFormatInt4), @"uint" : @(MTLVertexFormatUInt),
			@"uint2" : @(MTLVertexFormatUInt2), @"uint3" : @(MTLVertexFormatUInt3),
			@"uint4" : @(MTLVertexFormatUInt4),
			@"int1010102normalized" : @(MTLVertexFormatInt1010102Normalized),
			@"uint1010102normalized" : @(MTLVertexFormatUInt1010102Normalized),
			@"uchar4normalized_bgra" : @(MTLVertexFormatUChar4Normalized_BGRA),
			@"uchar" : @(MTLVertexFormatUChar), @"char" : @(MTLVertexFormatChar),
			@"ucharnormalized" : @(MTLVertexFormatUCharNormalized),
			@"charnormalized" : @(MTLVertexFormatCharNormalized), @"ushort" : @(MTLVertexFormatUShort),
			@"short" : @(MTLVertexFormatShort), @"ushortnormalized" : @(MTLVertexFormatUShortNormalized),
			@"shortnormalized" : @(MTLVertexFormatShortNormalized), @"half" : @(MTLVertexFormatHalf),
			@"floatrg11b10" : @(MTLVertexFormatFloatRG11B10), @"floatrgb9e5" : @(MTLVertexFormatFloatRGB9E5),
		};
	return t;
}

static NSDictionary<NSString *, NSNumber *> *blend_factors(void)
{
	static NSDictionary *t;
	if (!t)
		t = @{
			@"zero" : @(MTLBlendFactorZero), @"one" : @(MTLBlendFactorOne),
			@"source_color" : @(MTLBlendFactorSourceColor),
			@"one_minus_source_color" : @(MTLBlendFactorOneMinusSourceColor),
			@"source_alpha" : @(MTLBlendFactorSourceAlpha),
			@"one_minus_source_alpha" : @(MTLBlendFactorOneMinusSourceAlpha),
			@"destination_color" : @(MTLBlendFactorDestinationColor),
			@"one_minus_destination_color" : @(MTLBlendFactorOneMinusDestinationColor),
			@"destination_alpha" : @(MTLBlendFactorDestinationAlpha),
			@"one_minus_destination_alpha" : @(MTLBlendFactorOneMinusDestinationAlpha),
			@"source_alpha_saturated" : @(MTLBlendFactorSourceAlphaSaturated),
			@"blend_color" : @(MTLBlendFactorBlendColor),
			@"one_minus_blend_color" : @(MTLBlendFactorOneMinusBlendColor),
			@"blend_alpha" : @(MTLBlendFactorBlendAlpha),
			@"one_minus_blend_alpha" : @(MTLBlendFactorOneMinusBlendAlpha),
			@"source1_color" : @(MTLBlendFactorSource1Color),
			@"one_minus_source1_color" : @(MTLBlendFactorOneMinusSource1Color),
			@"source1_alpha" : @(MTLBlendFactorSource1Alpha),
			@"one_minus_source1_alpha" : @(MTLBlendFactorOneMinusSource1Alpha),
		};
	return t;
}

static NSDictionary<NSString *, NSNumber *> *blend_operations(void)
{
	return @{
		@"add" : @(MTLBlendOperationAdd), @"subtract" : @(MTLBlendOperationSubtract),
		@"reverse_subtract" : @(MTLBlendOperationReverseSubtract), @"min" : @(MTLBlendOperationMin),
		@"max" : @(MTLBlendOperationMax),
	};
}

static NSDictionary<NSString *, NSNumber *> *compare_functions(void)
{
	return @{
		@"never" : @(MTLCompareFunctionNever), @"less" : @(MTLCompareFunctionLess),
		@"equal" : @(MTLCompareFunctionEqual), @"less_equal" : @(MTLCompareFunctionLessEqual),
		@"greater" : @(MTLCompareFunctionGreater), @"not_equal" : @(MTLCompareFunctionNotEqual),
		@"greater_equal" : @(MTLCompareFunctionGreaterEqual), @"always" : @(MTLCompareFunctionAlways),
	};
}

static NSDictionary<NSString *, NSNumber *> *stencil_operations(void)
{
	return @{
		@"keep" : @(MTLStencilOperationKeep), @"zero" : @(MTLStencilOperationZero),
		@"replace" : @(MTLStencilOperationReplace),
		@"increment_clamp" : @(MTLStencilOperationIncrementClamp),
		@"decrement_clamp" : @(MTLStencilOperationDecrementClamp), @"invert" : @(MTLStencilOperationInvert),
		@"increment_wrap" : @(MTLStencilOperationIncrementWrap),
		@"decrement_wrap" : @(MTLStencilOperationDecrementWrap),
	};
}

static NSDictionary<NSString *, NSNumber *> *texture_types(void)
{
	return @{
		@"1d" : @(MTLTextureType1D), @"1d_array" : @(MTLTextureType1DArray), @"2d" : @(MTLTextureType2D),
		@"2d_array" : @(MTLTextureType2DArray), @"2d_ms" : @(MTLTextureType2DMultisample),
		@"2d_ms_array" : @(MTLTextureType2DMultisampleArray), @"cube" : @(MTLTextureTypeCube),
		@"cube_array" : @(MTLTextureTypeCubeArray), @"3d" : @(MTLTextureType3D),
	};
}

static NSDictionary<NSString *, NSNumber *> *data_types(void)
{
	return @{
		@"bool" : @(MTLDataTypeBool), @"char" : @(MTLDataTypeChar), @"uchar" : @(MTLDataTypeUChar),
		@"short" : @(MTLDataTypeShort), @"ushort" : @(MTLDataTypeUShort), @"int" : @(MTLDataTypeInt),
		@"uint" : @(MTLDataTypeUInt), @"long" : @(MTLDataTypeLong), @"ulong" : @(MTLDataTypeULong),
		@"half" : @(MTLDataTypeHalf), @"float" : @(MTLDataTypeFloat), @"float2" : @(MTLDataTypeFloat2),
		@"float4" : @(MTLDataTypeFloat4), @"int2" : @(MTLDataTypeInt2), @"int4" : @(MTLDataTypeInt4),
		@"uint4" : @(MTLDataTypeUInt4), @"half4" : @(MTLDataTypeHalf4),
	};
}

static NSData *read_file(NSString *relative)
{
	NSString *path = [plan_dir stringByAppendingPathComponent:relative];
	NSData *data = [NSData dataWithContentsOfFile:path];
	if (!data)
		@throw failure(@"cannot read %@", path);
	return data;
}

static void write_json(NSString *path, NSDictionary *value)
{
	NSData *data = [NSJSONSerialization dataWithJSONObject:value
						       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
							 error:nil];
	[data writeToFile:path atomically:YES];
}

/* The plan's library: a corpus metallib beside the plan, or the first of a
 * list of system paths that exists. */
static id<MTLLibrary> library_named(id<MTLDevice> device, NSDictionary *plan, NSString *name, NSMutableDictionary *cache)
{
	if (cache[name])
		return cache[name];
	id where = plan[@"libraries"][name];
	if (!where)
		@throw failure(@"plan has no library %@", name);
	if ([where isKindOfClass:[NSDictionary class]]) {
		/* Compiled from source by the device at run time, with the
		 * options an application passes. */
		NSDictionary *w = where;
		NSString *source = [[NSString alloc] initWithData:read_file(w[@"source"]) encoding:NSUTF8StringEncoding];
		MTLCompileOptions *o = [MTLCompileOptions new];
		/* MTLLanguageVersion is the version's major << 16 | minor. */
		if (w[@"language"]) {
			NSArray<NSString *> *parts = [w[@"language"] componentsSeparatedByString:@"."];
			o.languageVersion = (MTLLanguageVersion)((NSUInteger)parts[0].integerValue << 16 | (NSUInteger)parts[1].integerValue);
		}
		if (w[@"fast_math"])
			o.fastMathEnabled = [w[@"fast_math"] boolValue];
		if (w[@"math_mode"] || w[@"math_functions"]) {
			if (@available(macOS 15.0, *)) {
				NSDictionary *modes = @{@"safe" : @(MTLMathModeSafe), @"relaxed" : @(MTLMathModeRelaxed), @"fast" : @(MTLMathModeFast)};
				NSDictionary *functions = @{@"precise" : @(MTLMathFloatingPointFunctionsPrecise),
							    @"fast" : @(MTLMathFloatingPointFunctionsFast)};
				o.mathMode = lookup(modes, @"math mode", w[@"math_mode"] ?: @"fast");
				o.mathFloatingPointFunctions = lookup(functions, @"math functions", w[@"math_functions"] ?: @"fast");
			} else {
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"math modes need macOS 15" userInfo:nil];
			}
		}
		if (w[@"macros"])
			o.preprocessorMacros = w[@"macros"];
		o.preserveInvariance = [w[@"preserve_invariance"] boolValue];
		/* A dynamic library, installed under the name it gives. */
		if (w[@"install_name"]) {
			o.libraryType = MTLLibraryTypeDynamic;
			o.installName = w[@"install_name"];
		}
		/* Dynamic libraries the library links against as it compiles,
		 * which its pipelines load by their install names: one under
		 * the executable's directory is serialized there first. */
		if (w[@"libraries"]) {
			NSMutableArray *dylibs = [NSMutableArray new];
			NSString *executable = @"@executable_path/";
			for (NSString *n in w[@"libraries"]) {
				NSError *error = nil;
				id<MTLDynamicLibrary> d = [device newDynamicLibrary:library_named(device, plan, n, cache) error:&error];
				if (!d)
					@throw failure(@"dynamic library %@: %@", n, error.localizedDescription);
				if ([d.installName hasPrefix:executable]) {
					NSString *at = [[[NSBundle mainBundle].executablePath stringByDeletingLastPathComponent]
						stringByAppendingPathComponent:[d.installName substringFromIndex:executable.length]];
					if (![d serializeToURL:[NSURL fileURLWithPath:at] error:&error])
						@throw failure(@"installing dynamic library %@: %@", n, error.localizedDescription);
				}
				[dylibs addObject:d];
			}
			o.libraries = dylibs;
		}
		/* External functions left for a pipeline's linked or preloaded
		 * libraries to supply, and symbols kept from other libraries. */
		o.allowReferencingUndefinedSymbols = [w[@"allow_undefined"] boolValue];
		o.compileSymbolVisibility = lookup(@{@"default" : @(MTLCompileSymbolVisibilityDefault), @"hidden" : @(MTLCompileSymbolVisibilityHidden)},
						   @"symbol visibility", w[@"symbol_visibility"] ?: @"default");
		if (w[@"logging"]) {
			if (@available(macOS 15.0, *))
				o.enableLogging = [w[@"logging"] boolValue];
			else
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"shader logging needs macOS 15" userInfo:nil];
		}
		if (w[@"required_threads"]) {
			if (@available(macOS 26.0, *)) {
				NSArray *t = w[@"required_threads"];
				o.requiredThreadsPerThreadgroup = MTLSizeMake([t[0] unsignedIntegerValue], [t[1] unsignedIntegerValue], [t[2] unsignedIntegerValue]);
			} else {
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"required threadgroup sizes need macOS 26" userInfo:nil];
			}
		}
		if (w[@"conversion_rounding"]) {
			if (@available(macOS 27.0, *))
				o.floatingPointConversionRoundingMode =
					lookup(@{@"nearest_even" : @(MTLFloatingPointConversionRoundingModeToNearestEven),
						 @"toward_zero" : @(MTLFloatingPointConversionRoundingModeTowardZero)},
					       @"conversion rounding", w[@"conversion_rounding"]);
			else
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"conversion rounding modes need macOS 27" userInfo:nil];
		}
		if ([w[@"optimization"] isEqualToString:@"size"])
			o.optimizationLevel = MTLLibraryOptimizationLevelSize;
		__block NSError *error = nil;
		__block id<MTLLibrary> library = nil;
		if ([w[@"async"] boolValue])
			await_handler(^(dispatch_semaphore_t done) {
				[device newLibraryWithSource:source
						     options:o
					   completionHandler:^(id<MTLLibrary> l, NSError *e) {
						   library = l;
						   error = e;
						   dispatch_semaphore_signal(done);
					   }];
			});
		else
			library = [device newLibraryWithSource:source options:o error:&error];
		if (!library)
			@throw failure(@"library %@ from source: %@", name, error.localizedDescription);
		cache[name] = library;
		return library;
	}
	NSArray *candidates = [where isKindOfClass:[NSArray class]] ? where : @[ [plan_dir stringByAppendingPathComponent:where] ];
	for (NSString *path in candidates) {
		if (![[NSFileManager defaultManager] fileExistsAtPath:path])
			continue;
		NSError *error = nil;
		id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
		if (!library)
			@throw failure(@"library %@: %@", path, error.localizedDescription);
		cache[name] = library;
		return library;
	}
	@throw failure(@"library %@ is not present", name);
}

/* The dynamic libraries a pipeline preloads, by plan library name. */
static NSArray<id<MTLDynamicLibrary>> *preloaded(Case *c, NSArray *names)
{
	NSMutableArray *out = [NSMutableArray new];
	for (NSString *name in names) {
		NSError *error = nil;
		id<MTLDynamicLibrary> dylib = [c.device newDynamicLibrary:library_named(c.device, c.plan, name, c.libraries) error:&error];
		if (!dylib)
			@throw failure(@"dynamic library %@: %@", name, error.localizedDescription);
		[out addObject:dylib];
	}
	return out;
}

static MTLFunctionConstantValues *constants_of(NSArray *constants)
{
	MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
	for (NSDictionary *c in constants) {
		NSUInteger type = lookup(data_types(), @"data type", c[@"type"]);
		NSData *bytes;
		if (c[@"bytes"]) {
			bytes = [[NSData alloc] initWithBase64EncodedString:c[@"bytes"] options:0];
		} else {
			union {
				uint8_t u8;
				uint16_t u16;
				uint32_t u32;
				uint64_t u64;
				float f;
				__fp16 h;
			} v = {0};
			size_t size = 4;
			double value = [c[@"value"] doubleValue];
			switch (type) {
			case MTLDataTypeBool: case MTLDataTypeChar: case MTLDataTypeUChar:
				v.u8 = (uint8_t)(int64_t)value; size = 1; break;
			case MTLDataTypeShort: case MTLDataTypeUShort:
				v.u16 = (uint16_t)(int64_t)value; size = 2; break;
			case MTLDataTypeHalf: v.h = (__fp16)value; size = 2; break;
			case MTLDataTypeFloat: v.f = (float)value; break;
			case MTLDataTypeLong: case MTLDataTypeULong:
				v.u64 = (uint64_t)[c[@"value"] longLongValue]; size = 8; break;
			default: v.u32 = (uint32_t)[c[@"value"] longLongValue]; break;
			}
			bytes = [NSData dataWithBytes:&v length:size];
		}
		/* Consecutive constants of one type from one array of bytes. */
		if (c[@"range"])
			[values setConstantValues:bytes.bytes type:type
					withRange:NSMakeRange([c[@"range"][0] unsignedIntegerValue], [c[@"range"][1] unsignedIntegerValue])];
		else if (c[@"name"])
			[values setConstantValue:bytes.bytes type:type withName:c[@"name"]];
		else
			[values setConstantValue:bytes.bytes type:type atIndex:number(c, @"index", 0)];
	}
	return values;
}

static id<MTLFunction> function_named(id<MTLLibrary> library, NSString *name, NSArray *constants)
{
	__block NSError *error = nil;
	__block id<MTLFunction> function;
	BOOL async = [creation isEqualToString:@"async"];
	if ([creation isEqualToString:@"descriptor"] || (async && constants.count == 0)) {
		/* A descriptor naming the function, its constants and the name
		 * the specialized function takes. */
		MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
		fd.name = name;
		if (constants.count) {
			fd.constantValues = constants_of(constants);
			fd.specializedName = [name stringByAppendingString:@"_specialized"];
		}
		if (async)
			await_handler(^(dispatch_semaphore_t done) {
				[library newFunctionWithDescriptor:fd
					       completionHandler:^(id<MTLFunction> f, NSError *e) {
						       function = f;
						       error = e;
						       dispatch_semaphore_signal(done);
					       }];
			});
		else
			function = [library newFunctionWithDescriptor:fd error:&error];
	} else if (async) {
		await_handler(^(dispatch_semaphore_t done) {
			[library newFunctionWithName:name
					constantValues:constants_of(constants)
				     completionHandler:^(id<MTLFunction> f, NSError *e) {
					     function = f;
					     error = e;
					     dispatch_semaphore_signal(done);
				     }];
		});
	} else if (constants.count) {
		function = [library newFunctionWithName:name constantValues:constants_of(constants) error:&error];
	} else {
		function = [library newFunctionWithName:name];
	}
	if (!function)
		@throw failure(@"function %@: %@", name, error ? error.localizedDescription : @"not in library");
	return function;
}

/* Shared storage, without hazard tracking when the case orders its uses
 * itself with fences or events. */
/* A heap: automatic (the heap places resources) or placement (each names
 * its offset), shared or private, tracked or untracked, of a given size or
 * of the size the device answers for the textures it names. */
static NSUInteger answered_heap_size(Case *c, NSDictionary *h, NSArray *textures);

static id<MTLHeap> make_heap(Case *c, NSDictionary *h, NSArray *textures)
{
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = h[@"size_of"] ? answered_heap_size(c, h, textures) - number(h, @"size_less", 0) : number(h, @"size", 65536);
	d.type = [h[@"type"] isEqualToString:@"placement"] ? MTLHeapTypePlacement : MTLHeapTypeAutomatic;
	d.storageMode = [h[@"storage"] isEqualToString:@"private"] ? MTLStorageModePrivate : MTLStorageModeShared;
	d.hazardTrackingMode = [h[@"untracked"] boolValue] ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked;
	id<MTLHeap> heap = [c.device newHeapWithDescriptor:d];
	if (!heap)
		@throw failure(@"heap %@ creation failed", h[@"name"]);
	heap.label = h[@"name"];
	return heap;
}

static MTLResourceOptions storage_option(NSString *storage)
{
	if ([storage isEqualToString:@"private"])
		return MTLResourceStorageModePrivate;
	if ([storage isEqualToString:@"managed"])
		return MTLResourceStorageModeManaged;
	if (storage && ![storage isEqualToString:@"shared"])
		@throw failure(@"unknown storage mode %@", storage);
	return MTLResourceStorageModeShared;
}

static MTLResourceOptions buffer_options(NSDictionary *b)
{
	MTLResourceOptions options = storage_option(b[@"storage"]);
	if ([b[@"untracked"] boolValue])
		options |= MTLResourceHazardTrackingModeUntracked;
	if ([b[@"write_combined"] boolValue])
		options |= MTLResourceCPUCacheModeWriteCombined;
	return options;
}

/* A buffer of `length` bytes: from the device, or from the named heap at
 * an offset when that heap places its resources. */
static id<MTLBuffer> allocate_buffer(Case *c, NSDictionary *b, NSUInteger length)
{
	if ([b[@"no_copy"] boolValue]) {
		/* Memory the application owns, wrapped without a copy. */
		NSUInteger page = (NSUInteger)getpagesize(), bytes = (length + page - 1) / page * page;
		void *memory = NULL;
		if (posix_memalign(&memory, page, bytes))
			return nil;
		return [c.device newBufferWithBytesNoCopy:memory length:bytes options:buffer_options(b)
					      deallocator:^(void *pointer, NSUInteger _) { free(pointer); }];
	}
	if (!b[@"heap"])
		return [c.device newBufferWithLength:length options:buffer_options(b)];
	id<MTLHeap> heap = c.heaps[b[@"heap"]];
	if (!heap)
		@throw failure(@"buffer %@ names no heap", b[@"name"]);
	MTLResourceOptions options = buffer_options(b) & ~MTLResourceHazardTrackingModeMask;
	options |= heap.resourceOptions & MTLResourceHazardTrackingModeMask;
	if (heap.type == MTLHeapTypePlacement)
		return [heap newBufferWithLength:length options:options offset:number(b, @"heap_offset", 0)];
	return [heap newBufferWithLength:length options:options];
}

/* Runs one blit on the case's queue and waits for it. */
static void blit_now(Case *c, void (^encode)(id<MTLBlitCommandEncoder>))
{
	id<MTLCommandBuffer> cb = [c.queue commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	encode(e);
	[e endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.error)
		@throw failure(@"blit: %@", cb.error.localizedDescription);
}

static id<MTLBuffer> make_buffer(Case *c, NSDictionary *b)
{
	NSData *data = b[@"file"] ? read_file(b[@"file"]) : nil;
	NSUInteger length = MAX(MAX(number(b, @"length", data ? data.length : 16), data.length), (NSUInteger)16);
	id<MTLBuffer> buffer = allocate_buffer(c, b, length);
	if (!buffer)
		@throw failure(@"buffer %@ allocation failed", b[@"name"]);
	buffer.label = b[@"name"];
	/* An alias keeps the bytes of the resource it overlaps. */
	if (b[@"alias_of"])
		return buffer;
	/* The initial bytes go in through the CPU, or through a staging
	 * buffer and a blit when the CPU cannot reach the storage. */
	id<MTLBuffer> target = buffer;
	if (buffer.storageMode == MTLStorageModePrivate)
		target = [c.device newBufferWithLength:buffer.length options:MTLResourceStorageModeShared];
	memset(target.contents, (int)number(b, @"fill", 0), target.length);
	if (data)
		memcpy(target.contents, data.bytes, data.length);
	if (buffer.storageMode == MTLStorageModeManaged)
		[buffer didModifyRange:NSMakeRange(0, buffer.length)];
	if (target != buffer)
		blit_now(c, ^(id<MTLBlitCommandEncoder> e) {
			[e copyFromBuffer:target sourceOffset:0 toBuffer:buffer destinationOffset:0 size:buffer.length];
		});
	return buffer;
}

/* Bytes of one block of a format the runner uploads or reads back, and its
 * width and height: 1x1 except for the compressed formats. */
static NSUInteger pixel_bytes(MTLPixelFormat f, NSUInteger *block, NSUInteger *block_height)
{
	*block = *block_height = 1;
	switch (f) {
	case MTLPixelFormatA8Unorm: case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Unorm_sRGB:
	case MTLPixelFormatR8Snorm: case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint:
	case MTLPixelFormatStencil8:
		return 1;
	case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Uint:
	case MTLPixelFormatR16Sint: case MTLPixelFormatR16Float: case MTLPixelFormatRG8Unorm:
	case MTLPixelFormatRG8Snorm: case MTLPixelFormatRG8Uint: case MTLPixelFormatB5G6R5Unorm:
	case MTLPixelFormatRG8Sint: case MTLPixelFormatRG8Unorm_sRGB:
	case MTLPixelFormatA1BGR5Unorm: case MTLPixelFormatABGR4Unorm: case MTLPixelFormatBGR5A1Unorm:
	case MTLPixelFormatDepth16Unorm:
		return 2;
	case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRG32Float:
	case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRGBA16Uint:
	case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatBGRA10_XR: case MTLPixelFormatBGRA10_XR_sRGB:
		return 8;
	case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Sint: case MTLPixelFormatRGBA32Float:
		return 16;
	case MTLPixelFormatBC1_RGBA: case MTLPixelFormatBC1_RGBA_sRGB: case MTLPixelFormatBC4_RUnorm: case MTLPixelFormatBC4_RSnorm:
		*block = *block_height = 4;
		return 8;
	case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC2_RGBA_sRGB: case MTLPixelFormatBC3_RGBA: case MTLPixelFormatBC3_RGBA_sRGB:
	case MTLPixelFormatBC5_RGUnorm: case MTLPixelFormatBC5_RGSnorm: case MTLPixelFormatBC6H_RGBFloat:
	case MTLPixelFormatBC6H_RGBUfloat: case MTLPixelFormatBC7_RGBAUnorm: case MTLPixelFormatBC7_RGBAUnorm_sRGB:
	case MTLPixelFormatEAC_RGBA8: case MTLPixelFormatEAC_RGBA8_sRGB: case MTLPixelFormatEAC_RG11Unorm:
	case MTLPixelFormatEAC_RG11Snorm:
		*block = *block_height = 4;
		return 16;
	case MTLPixelFormatETC2_RGB8: case MTLPixelFormatETC2_RGB8_sRGB: case MTLPixelFormatETC2_RGB8A1:
	case MTLPixelFormatETC2_RGB8A1_sRGB: case MTLPixelFormatEAC_R11Unorm: case MTLPixelFormatEAC_R11Snorm:
	case MTLPixelFormatPVRTC_RGB_4BPP: case MTLPixelFormatPVRTC_RGB_4BPP_sRGB:
	case MTLPixelFormatPVRTC_RGBA_4BPP: case MTLPixelFormatPVRTC_RGBA_4BPP_sRGB:
		*block = *block_height = 4;
		return 8;
	case MTLPixelFormatPVRTC_RGB_2BPP: case MTLPixelFormatPVRTC_RGB_2BPP_sRGB:
	case MTLPixelFormatPVRTC_RGBA_2BPP: case MTLPixelFormatPVRTC_RGBA_2BPP_sRGB:
		*block = 8;
		*block_height = 4;
		return 8;
	case MTLPixelFormatASTC_4x4_LDR: case MTLPixelFormatASTC_4x4_sRGB: case MTLPixelFormatASTC_4x4_HDR:
		*block = 4;
		*block_height = 4;
		return 16;
	case MTLPixelFormatASTC_5x4_LDR: case MTLPixelFormatASTC_5x4_sRGB:
		*block = 5;
		*block_height = 4;
		return 16;
	case MTLPixelFormatASTC_5x5_LDR: case MTLPixelFormatASTC_5x5_sRGB:
		*block = 5;
		*block_height = 5;
		return 16;
	case MTLPixelFormatASTC_6x5_LDR: case MTLPixelFormatASTC_6x5_sRGB:
		*block = 6;
		*block_height = 5;
		return 16;
	case MTLPixelFormatASTC_6x6_LDR: case MTLPixelFormatASTC_6x6_sRGB: case MTLPixelFormatASTC_6x6_HDR:
		*block = 6;
		*block_height = 6;
		return 16;
	case MTLPixelFormatASTC_8x5_LDR: case MTLPixelFormatASTC_8x5_sRGB:
		*block = 8;
		*block_height = 5;
		return 16;
	case MTLPixelFormatASTC_8x6_LDR: case MTLPixelFormatASTC_8x6_sRGB:
		*block = 8;
		*block_height = 6;
		return 16;
	case MTLPixelFormatASTC_8x8_LDR: case MTLPixelFormatASTC_8x8_sRGB: case MTLPixelFormatASTC_8x8_HDR:
		*block = 8;
		*block_height = 8;
		return 16;
	case MTLPixelFormatASTC_10x5_LDR: case MTLPixelFormatASTC_10x5_sRGB:
		*block = 10;
		*block_height = 5;
		return 16;
	case MTLPixelFormatASTC_10x6_LDR: case MTLPixelFormatASTC_10x6_sRGB:
		*block = 10;
		*block_height = 6;
		return 16;
	case MTLPixelFormatASTC_10x8_LDR: case MTLPixelFormatASTC_10x8_sRGB:
		*block = 10;
		*block_height = 8;
		return 16;
	case MTLPixelFormatASTC_10x10_LDR: case MTLPixelFormatASTC_10x10_sRGB:
		*block = 10;
		*block_height = 10;
		return 16;
	case MTLPixelFormatASTC_12x10_LDR: case MTLPixelFormatASTC_12x10_sRGB:
		*block = 12;
		*block_height = 10;
		return 16;
	case MTLPixelFormatASTC_12x12_LDR: case MTLPixelFormatASTC_12x12_sRGB: case MTLPixelFormatASTC_12x12_HDR:
		*block = 12;
		*block_height = 12;
		return 16;
	default:
		return 4;
	}
}

static NSUInteger level_extent(NSUInteger size, NSUInteger level)
{
	return MAX((NSUInteger)1, size >> level);
}

/* The bytes of each (slice, level) of a texture, tightly packed, slices
 * outermost and levels inner: how inputs are uploaded and outputs read. */
struct region {
	NSUInteger slice, level, offset, bytes_per_row, bytes_per_image;
	MTLSize size;
};

static NSArray<NSValue *> *regions_of(id<MTLTexture> t, NSString *aspect, NSUInteger *total)
{
	NSMutableArray *out = [NSMutableArray new];
	NSUInteger block, block_height, bytes = pixel_bytes(t.pixelFormat, &block, &block_height), offset = 0;
	if (t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 || t.pixelFormat == MTLPixelFormatDepth24Unorm_Stencil8)
		bytes = [aspect isEqualToString:@"stencil"] ? 1 : 4;
	NSUInteger slices = t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
	for (NSUInteger slice = 0; slice < slices; slice++) {
		for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
			struct region r = {slice, level, offset, 0, 0, MTLSizeMake(0, 0, 0)};
			r.size = MTLSizeMake(level_extent(t.width, level), level_extent(t.height, level), level_extent(t.depth, level));
			r.bytes_per_row = (r.size.width + block - 1) / block * bytes;
			r.bytes_per_image = r.bytes_per_row * ((r.size.height + block_height - 1) / block_height);
			offset += r.bytes_per_image * r.size.depth;
			[out addObject:[NSValue value:&r withObjCType:@encode(struct region)]];
		}
	}
	*total = offset;
	return out;
}

static MTLBlitOption blit_option(MTLPixelFormat f, NSString *aspect)
{
	if (f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatDepth24Unorm_Stencil8)
		return [aspect isEqualToString:@"stencil"] ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;
	return MTLBlitOptionNone;
}

static MTLTextureUsage texture_usage(NSDictionary *t)
{
	MTLTextureUsage usage = 0;
	for (NSString *u in t[@"usage"] ?: @[ @"read" ]) {
		if ([u isEqualToString:@"read"]) usage |= MTLTextureUsageShaderRead;
		else if ([u isEqualToString:@"write"]) usage |= MTLTextureUsageShaderWrite;
		else if ([u isEqualToString:@"render"]) usage |= MTLTextureUsageRenderTarget;
		else if ([u isEqualToString:@"atomic"]) usage |= MTLTextureUsageShaderAtomic;
		else if ([u isEqualToString:@"view"]) usage |= MTLTextureUsagePixelFormatView;
		else @throw failure(@"unknown texture usage %@", u);
	}
	return usage;
}

/* The descriptor of a texture a case makes on the device or a heap. */
static MTLTextureDescriptor *texture_descriptor(NSDictionary *t)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor new];
	d.textureType = lookup(texture_types(), @"texture type", t[@"type"] ?: @"2d");
	d.pixelFormat = lookup(pixel_formats(), @"pixel format", t[@"format"]);
	d.width = number(t, @"width", 1);
	d.height = number(t, @"height", 1);
	d.depth = number(t, @"depth", 1);
	d.arrayLength = number(t, @"array", 1);
	d.mipmapLevelCount = number(t, @"mipmaps", 1);
	d.sampleCount = number(t, @"samples", 1);
	d.storageMode = t[@"storage"] ? (MTLStorageMode)(storage_option(t[@"storage"]) >> MTLResourceStorageModeShift)
				      : MTLStorageModePrivate;
	d.usage = texture_usage(t);
	if ([t[@"write_combined"] boolValue])
		d.cpuCacheMode = MTLCPUCacheModeWriteCombined;
	return d;
}

/* Writes the device's `heapTextureSizeAndAlignWithDescriptor:` answer for
 * each texture a shared buffer names in `size_and_align_of`, as pairs of
 * 64-bit size and alignment. */
static void write_heap_answers(Case *c, NSDictionary *b, NSArray *textures)
{
	id<MTLBuffer> buffer = c.buffers[b[@"name"]];
	NSArray *names = b[@"size_and_align_of"];
	if (buffer.storageMode != MTLStorageModeShared || buffer.length < names.count * 16)
		@throw failure(@"buffer %@ cannot hold %lu answers", b[@"name"], (unsigned long)names.count);
	uint64_t *out = buffer.contents;
	for (NSUInteger i = 0; i < names.count; i++) {
		NSDictionary *t = nil;
		for (NSDictionary *each in textures)
			if ([each[@"name"] isEqualToString:names[i]])
				t = each;
		if (!t)
			@throw failure(@"buffer %@ names no texture %@", b[@"name"], names[i]);
		MTLSizeAndAlign answer = [c.device heapTextureSizeAndAlignWithDescriptor:texture_descriptor(t)];
		out[2 * i] = answer.size;
		out[2 * i + 1] = answer.align;
	}
}

/* A heap's size as its caller computes it from the device's answers: the
 * sum of `heapTextureSizeAndAlignWithDescriptor:` sizes, each rounded up to
 * its alignment, of the case's textures the heap names in `size_of`, less
 * `size_less` bytes. */
static NSUInteger answered_heap_size(Case *c, NSDictionary *h, NSArray *textures)
{
	NSUInteger size = 0;
	for (NSString *name in h[@"size_of"]) {
		NSDictionary *t = nil;
		for (NSDictionary *each in textures)
			if ([each[@"name"] isEqualToString:name])
				t = each;
		if (!t)
			@throw failure(@"heap %@ is sized for no texture %@", h[@"name"], name);
		MTLTextureDescriptor *d = texture_descriptor(t);
		d.storageMode = [h[@"storage"] isEqualToString:@"private"] ? MTLStorageModePrivate : MTLStorageModeShared;
		MTLSizeAndAlign answer = [c.device heapTextureSizeAndAlignWithDescriptor:d];
		if (!answer.size || !answer.align)
			@throw failure(@"no heap size for texture %@", name);
		size += (answer.size + answer.align - 1) / answer.align * answer.align;
	}
	return size;
}

/* A texture buffer: a linear view of one of the case's buffers. */
static id<MTLTexture> make_texture_buffer(Case *c, NSDictionary *t)
{
	id<MTLBuffer> storage = c.buffers[t[@"buffer"]];
	if (!storage)
		@throw failure(@"texture buffer %@ names no buffer", t[@"name"]);
	MTLPixelFormat format = lookup(pixel_formats(), @"pixel format", t[@"format"]);
	NSUInteger block, block_height, width = number(t, @"width", 1);
	NSUInteger row = width * pixel_bytes(format, &block, &block_height);
	NSUInteger align = [c.device minimumTextureBufferAlignmentForPixelFormat:format];
	MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format width:width
									    resourceOptions:storage.resourceOptions
										      usage:texture_usage(t)];
	id<MTLTexture> texture = [storage newTextureWithDescriptor:d offset:number(t, @"offset", 0)
						      bytesPerRow:(row + align - 1) / align * align];
	if (!texture)
		@throw failure(@"texture buffer %@ creation failed", t[@"name"]);
	texture.label = t[@"name"];
	return texture;
}

static id<MTLTexture> make_texture(Case *c, NSDictionary *t, id<MTLCommandQueue> queue)
{
	if ([t[@"type"] isEqual:@"buffer"])
		return make_texture_buffer(c, t);
	if (t[@"view_of"]) {
		/* A view of an earlier texture: another format, type, mipmap and
		 * slice range or component swizzle over the same memory. */
		id<MTLTexture> source = c.textures[t[@"view_of"]];
		if (!source)
			@throw failure(@"view %@ names no texture", t[@"name"]);
		NSArray *levels = t[@"levels"] ?: @[ @0, @(source.mipmapLevelCount) ];
		NSArray *slices = t[@"slices"] ?: @[ @0, @(source.textureType == MTLTextureTypeCube ? 6 : source.arrayLength) ];
		NSDictionary *components = @{@"r" : @(MTLTextureSwizzleRed), @"g" : @(MTLTextureSwizzleGreen), @"b" : @(MTLTextureSwizzleBlue),
					     @"a" : @(MTLTextureSwizzleAlpha), @"0" : @(MTLTextureSwizzleZero), @"1" : @(MTLTextureSwizzleOne)};
		NSString *sw = t[@"swizzle"] ?: @"rgba";
		MTLTextureSwizzle channel[4];
		for (NSUInteger k = 0; k < 4; k++)
			channel[k] = lookup(components, @"swizzle", [sw substringWithRange:NSMakeRange(k, 1)]);
		MTLPixelFormat format = t[@"format"] ? lookup(pixel_formats(), @"pixel format", t[@"format"]) : source.pixelFormat;
		MTLTextureType type = t[@"type"] ? lookup(texture_types(), @"texture type", t[@"type"]) : source.textureType;
		NSRange level_range = NSMakeRange([levels[0] unsignedIntegerValue], [levels[1] unsignedIntegerValue]);
		NSRange slice_range = NSMakeRange([slices[0] unsignedIntegerValue], [slices[1] unsignedIntegerValue]);
		MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(channel[0], channel[1], channel[2], channel[3]);
		BOOL whole = !t[@"levels"] && !t[@"slices"] && !t[@"type"], identity = [sw isEqualToString:@"rgba"];
		id<MTLTexture> view = nil;
		if (t[@"min_lod"] || [t[@"view_descriptor"] boolValue]) {
			/* A descriptor, which alone carries a minimum LOD. */
			if (@available(macOS 26.0, *)) {
				MTLTextureViewDescriptor *vd = [MTLTextureViewDescriptor new];
				vd.pixelFormat = format;
				vd.textureType = type;
				vd.levelRange = level_range;
				vd.sliceRange = slice_range;
				vd.swizzle = swizzle;
				if (t[@"min_lod"]) {
					if (@available(macOS 27.0, *))
						vd.minLOD = [t[@"min_lod"] floatValue];
					else
						@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"view minimum LODs need macOS 27" userInfo:nil];
				}
				view = [source newTextureViewWithDescriptor:vd];
			} else {
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"view descriptors need macOS 26" userInfo:nil];
			}
		} else if (whole && identity) {
			/* The shortest call that states the view. */
			view = [source newTextureViewWithPixelFormat:format];
		} else if (identity) {
			view = [source newTextureViewWithPixelFormat:format textureType:type levels:level_range slices:slice_range];
		} else {
			view = [source newTextureViewWithPixelFormat:format textureType:type levels:level_range slices:slice_range swizzle:swizzle];
		}
		if (!view)
			@throw failure(@"view %@ creation failed", t[@"name"]);
		view.label = t[@"name"];
		return view;
	}
	MTLTextureDescriptor *d = texture_descriptor(t);
	id<MTLTexture> texture;
	if (t[@"over_buffer"]) {
		/* A linear 2D texture over a buffer's bytes, rows `bytes_per_row`
		 * apart, the buffer's contents its texels. */
		id<MTLBuffer> storage = c.buffers[t[@"over_buffer"]];
		if (!storage)
			@throw failure(@"texture %@ names no buffer", t[@"name"]);
		d.storageMode = storage.storageMode;
		d.resourceOptions = storage.resourceOptions;
		texture = [storage newTextureWithDescriptor:d offset:number(t, @"offset", 0) bytesPerRow:number(t, @"bytes_per_row", 256)];
	} else if (t[@"heap"]) {
		id<MTLHeap> heap = c.heaps[t[@"heap"]];
		if (!heap)
			@throw failure(@"texture %@ names no heap", t[@"name"]);
		d.storageMode = heap.storageMode;
		d.hazardTrackingMode = heap.hazardTrackingMode;
		texture = heap.type == MTLHeapTypePlacement ? [heap newTextureWithDescriptor:d offset:number(t, @"heap_offset", 0)]
							    : [heap newTextureWithDescriptor:d];
	} else {
		texture = [c.device newTextureWithDescriptor:d];
	}
	/* A heap without room answers nil, which a case can expect. */
	if ([t[@"expect_nil"] boolValue]) {
		if (texture)
			@throw failure(@"texture %@ allocated though its heap has no room", t[@"name"]);
		return nil;
	}
	if (!texture)
		@throw failure(@"texture %@ allocation failed", t[@"name"]);
	texture.label = t[@"name"];
	for (NSString *key in @[ @"file", @"stencil_file" ]) {
		if (!t[key])
			continue;
		NSData *data = read_file(t[key]);
		NSString *aspect = [key isEqualToString:@"stencil_file"] ? @"stencil" : t[@"aspect"];
		NSUInteger total;
		NSArray *regions = regions_of(texture, aspect, &total);
		if (data.length < total)
			@throw failure(@"texture %@ input has %lu bytes, needs %lu", t[@"name"], (unsigned long)data.length, (unsigned long)total);
		if (texture.storageMode != MTLStorageModePrivate) {
			/* A texture the CPU reaches takes its texels from the CPU,
			 * rows and images `row_padding` bytes apart beyond their
			 * texels. */
			NSUInteger pad = number(t, @"row_padding", 0);
			for (NSValue *v in regions) {
				struct region r;
				[v getValue:&r];
				NSUInteger rows = r.bytes_per_image / r.bytes_per_row, row = r.bytes_per_row + pad;
				NSMutableData *padded = [NSMutableData dataWithLength:row * rows * r.size.depth];
				for (NSUInteger k = 0; k < rows * r.size.depth; k++)
					memcpy((uint8_t *)padded.mutableBytes + k * row, (const uint8_t *)data.bytes + r.offset + k * r.bytes_per_row,
					       r.bytes_per_row);
				[texture replaceRegion:MTLRegionMake3D(0, 0, 0, r.size.width, r.size.height, r.size.depth)
					   mipmapLevel:r.level slice:r.slice withBytes:padded.bytes bytesPerRow:row bytesPerImage:row * rows];
			}
			continue;
		}
		id<MTLBuffer> staging = [c.device newBufferWithBytes:data.bytes length:total options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		for (NSValue *v in regions) {
			struct region r;
			[v getValue:&r];
			[blit copyFromBuffer:staging sourceOffset:r.offset sourceBytesPerRow:r.bytes_per_row
			     sourceBytesPerImage:r.bytes_per_image sourceSize:r.size toTexture:texture
				destinationSlice:r.slice destinationLevel:r.level destinationOrigin:MTLOriginMake(0, 0, 0)
					 options:blit_option(texture.pixelFormat, aspect)];
		}
		[blit endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		if (cb.error)
			@throw failure(@"texture %@ upload: %@", t[@"name"], cb.error.localizedDescription);
	}
	return texture;
}

static NSDictionary<NSString *, NSNumber *> *address_modes(void)
{
	return @{
		@"clamp_to_edge" : @(MTLSamplerAddressModeClampToEdge),
		@"mirror_clamp_to_edge" : @(MTLSamplerAddressModeMirrorClampToEdge),
		@"repeat" : @(MTLSamplerAddressModeRepeat), @"mirror_repeat" : @(MTLSamplerAddressModeMirrorRepeat),
		@"clamp_to_zero" : @(MTLSamplerAddressModeClampToZero),
		@"clamp_to_border" : @(MTLSamplerAddressModeClampToBorderColor),
	};
}

static id<MTLSamplerState> make_sampler(Case *c, NSDictionary *s)
{
	MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
	// Bicubic, 2, is a filter no header names.
	NSDictionary *filters = @{@"nearest" : @(MTLSamplerMinMagFilterNearest), @"linear" : @(MTLSamplerMinMagFilterLinear), @"bicubic" : @2};
	NSDictionary *mips = @{@"none" : @(MTLSamplerMipFilterNotMipmapped), @"nearest" : @(MTLSamplerMipFilterNearest),
			       @"linear" : @(MTLSamplerMipFilterLinear)};
	d.minFilter = lookup(filters, @"filter", s[@"min"] ?: @"nearest");
	d.magFilter = lookup(filters, @"filter", s[@"mag"] ?: @"nearest");
	d.mipFilter = lookup(mips, @"mip filter", s[@"mip"] ?: @"none");
	NSArray *address = s[@"address"] ?: @[ @"clamp_to_edge" ];
	d.sAddressMode = lookup(address_modes(), @"address mode", address[0]);
	d.tAddressMode = lookup(address_modes(), @"address mode", address[MIN(1u, address.count - 1)]);
	d.rAddressMode = lookup(address_modes(), @"address mode", address[MIN(2u, address.count - 1)]);
	NSDictionary *borders = @{@"transparent_black" : @(MTLSamplerBorderColorTransparentBlack),
				  @"opaque_black" : @(MTLSamplerBorderColorOpaqueBlack),
				  @"opaque_white" : @(MTLSamplerBorderColorOpaqueWhite)};
	d.borderColor = lookup(borders, @"border color", s[@"border"] ?: @"transparent_black");
	d.normalizedCoordinates = s[@"normalized"] ? [s[@"normalized"] boolValue] : YES;
	d.lodMinClamp = s[@"lod_min"] ? [s[@"lod_min"] floatValue] : 0.0f;
	d.lodMaxClamp = s[@"lod_max"] ? [s[@"lod_max"] floatValue] : FLT_MAX;
	d.maxAnisotropy = number(s, @"anisotropy", 1);
	/* A hint an implementation may ignore: averaging the LOD over a quad. */
	d.lodAverage = [s[@"lod_average"] boolValue];
	if (s[@"compare"])
		d.compareFunction = lookup(compare_functions(), @"compare function", s[@"compare"]);
	d.supportArgumentBuffers = [s[@"argument_buffers"] boolValue];
	if (s[@"reduction"] || s[@"lod_bias"]) {
		if (@available(macOS 26.0, *)) {
			NSDictionary *reductions = @{@"weighted_average" : @(MTLSamplerReductionModeWeightedAverage),
						     @"minimum" : @(MTLSamplerReductionModeMinimum),
						     @"maximum" : @(MTLSamplerReductionModeMaximum)};
			d.reductionMode = lookup(reductions, @"reduction mode", s[@"reduction"] ?: @"weighted_average");
			d.lodBias = [s[@"lod_bias"] floatValue];
		} else {
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"sampler reduction and bias need macOS 26" userInfo:nil];
		}
	}
	id<MTLSamplerState> sampler = [c.device newSamplerStateWithDescriptor:d];
	if (!sampler)
		@throw failure(@"sampler %@ creation failed", s[@"name"]);
	return sampler;
}

static void write_words(Case *c, NSString *name, NSArray<NSNumber *> *words);

static BOOL power_of_two(NSUInteger v)
{
	return v && !(v & (v - 1));
}

/* A pipeline or state an argument buffer entry names, made resident. */
static id<MTLComputePipelineState> argument_compute_pipeline(Case *c, id<MTLLibrary> library, NSDictionary *e)
{
	MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
	pd.computeFunction = function_named(library, e[@"function"], nil);
	pd.supportIndirectCommandBuffers = YES;
	NSError *error = nil;
	id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone
										   reflection:nil error:&error];
	if (!pipeline)
		@throw failure(@"argument compute pipeline: %@", error.localizedDescription);
	[c.resident addObject:pipeline];
	return pipeline;
}

static id<MTLRenderPipelineState> argument_render_pipeline(Case *c, id<MTLLibrary> library, NSDictionary *e)
{
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	pd.vertexFunction = function_named(library, e[@"vertex"], nil);
	pd.fragmentFunction = function_named(library, e[@"fragment"], nil);
	pd.colorAttachments[0].pixelFormat = lookup(pixel_formats(), @"pixel format", e[@"format"]);
	pd.supportIndirectCommandBuffers = YES;
	NSError *error = nil;
	id<MTLRenderPipelineState> pipeline = [c.device newRenderPipelineStateWithDescriptor:pd error:&error];
	if (!pipeline)
		@throw failure(@"argument render pipeline: %@", error.localizedDescription);
	[c.resident addObject:pipeline];
	return pipeline;
}

static id<MTLDepthStencilState> argument_depth_stencil(Case *c, NSDictionary *e)
{
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	dd.depthCompareFunction = lookup(compare_functions(), @"compare function", e[@"compare"] ?: @"always");
	dd.depthWriteEnabled = [e[@"write"] boolValue];
	id<MTLDepthStencilState> state = [c.device newDepthStencilStateWithDescriptor:dd];
	[c.resident addObject:state];
	return state;
}

/* The argument encoder an entry list's `encoder` names: from a function's
 * buffer argument, through the older call that also reflects it, or from
 * the binding a pipeline's reflection gives. */
static id<MTLArgumentEncoder> argument_encoder(Case *c, id<MTLLibrary> library, NSDictionary *source)
{
	id<MTLFunction> f = function_named(library, source[@"function"], nil);
	NSUInteger index = number(source, @"index", 0);
	NSString *from = source[@"from"] ?: @"function";
	if ([from isEqualToString:@"function"])
		return [f newArgumentEncoderWithBufferIndex:index];
	if ([from isEqualToString:@"reflected"]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		MTLArgument *reflection = nil;
		id<MTLArgumentEncoder> ae = [f newArgumentEncoderWithBufferIndex:index reflection:&reflection];
		if (reflection.index != index || reflection.type != MTLArgumentTypeBuffer || !reflection.bufferPointerType.elementIsArgumentBuffer)
			@throw failure(@"argument encoder reflects %@ at %lu", reflection.name, (unsigned long)reflection.index);
#pragma clang diagnostic pop
		return ae;
	}
	if ([from isEqualToString:@"binding"]) {
		MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
		pd.computeFunction = f;
		MTLComputePipelineReflection *reflection = nil;
		NSError *error = nil;
		if (![c.device newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
							  reflection:&reflection error:&error])
			@throw failure(@"argument encoder pipeline: %@", error.localizedDescription);
		for (id<MTLBinding> b in reflection.bindings)
			if (b.type == MTLBindingTypeBuffer && b.index == index)
				return [c.device newArgumentEncoderWithBufferBinding:(id<MTLBufferBinding>)b];
		@throw failure(@"no buffer binding at %lu", (unsigned long)index);
	}
	@throw failure(@"argument encoder from %@", from);
}

/* Encodes entries at their [[id]]s: one resource each, or `resources`
 * of one kind over consecutive ids; `nested` entries encode a struct of
 * their own into another buffer through the encoder for that id. */
static void encode_entries(Case *c, id<MTLLibrary> library, id<MTLArgumentEncoder> ae, NSArray *entries)
{
	for (NSDictionary *e in entries) {
		NSString *kind = e[@"kind"], *name = e[@"resource"];
		NSUInteger index = number(e, @"id", 0);
		NSArray *many = e[@"resources"];
		NSRange range = NSMakeRange(index, many.count);
		if ([kind isEqualToString:@"buffer"] && many) {
			__unsafe_unretained id<MTLBuffer> list[many.count];
			NSUInteger offsets[many.count];
			for (NSUInteger k = 0; k < many.count; k++) {
				list[k] = c.buffers[many[k]];
				offsets[k] = 0;
				[c.resident addObject:list[k]];
			}
			[ae setBuffers:list offsets:offsets withRange:range];
		} else if ([kind isEqualToString:@"buffer"]) {
			[ae setBuffer:c.buffers[name] offset:number(e, @"resource_offset", 0) atIndex:index];
			[c.resident addObject:c.buffers[name]];
		} else if ([kind isEqualToString:@"texture"] && many) {
			__unsafe_unretained id<MTLTexture> list[many.count];
			for (NSUInteger k = 0; k < many.count; k++) {
				list[k] = c.textures[many[k]];
				[c.resident addObject:list[k]];
			}
			[ae setTextures:list withRange:range];
		} else if ([kind isEqualToString:@"texture"]) {
			[ae setTexture:c.textures[name] atIndex:index];
			[c.resident addObject:c.textures[name]];
		} else if ([kind isEqualToString:@"sampler"] && many) {
			__unsafe_unretained id<MTLSamplerState> list[many.count];
			for (NSUInteger k = 0; k < many.count; k++)
				list[k] = c.samplers[many[k]];
			[ae setSamplerStates:list withRange:range];
		} else if ([kind isEqualToString:@"sampler"]) {
			[ae setSamplerState:c.samplers[name] atIndex:index];
		} else if ([kind isEqualToString:@"constant"]) {
			uint32_t value = (uint32_t)number(e, @"value", 0);
			memcpy([ae constantDataAtIndex:index], &value, 4);
		} else if ([kind isEqualToString:@"icb"] && many) {
			__unsafe_unretained id<MTLIndirectCommandBuffer> list[many.count];
			for (NSUInteger k = 0; k < many.count; k++)
				list[k] = c.icbs[many[k]];
			[ae setIndirectCommandBuffers:list withRange:range];
		} else if ([kind isEqualToString:@"icb"]) {
			[ae setIndirectCommandBuffer:c.icbs[name] atIndex:index];
		} else if ([kind isEqualToString:@"compute_pipeline"] && e[@"functions"]) {
			NSArray *functions = e[@"functions"];
			__unsafe_unretained id<MTLComputePipelineState> list[functions.count];
			NSMutableArray *keep = [NSMutableArray new];
			for (NSUInteger k = 0; k < functions.count; k++) {
				[keep addObject:argument_compute_pipeline(c, library, @{@"function" : functions[k]})];
				list[k] = keep[k];
			}
			[ae setComputePipelineStates:list withRange:NSMakeRange(index, functions.count)];
		} else if ([kind isEqualToString:@"compute_pipeline"]) {
			[ae setComputePipelineState:argument_compute_pipeline(c, library, e) atIndex:index];
		} else if ([kind isEqualToString:@"render_pipeline"] && e[@"pipelines"]) {
			NSArray *pipelines = e[@"pipelines"];
			__unsafe_unretained id<MTLRenderPipelineState> list[pipelines.count];
			NSMutableArray *keep = [NSMutableArray new];
			for (NSUInteger k = 0; k < pipelines.count; k++) {
				[keep addObject:argument_render_pipeline(c, library, pipelines[k])];
				list[k] = keep[k];
			}
			[ae setRenderPipelineStates:list withRange:NSMakeRange(index, pipelines.count)];
		} else if ([kind isEqualToString:@"render_pipeline"]) {
			[ae setRenderPipelineState:argument_render_pipeline(c, library, e) atIndex:index];
		} else if ([kind isEqualToString:@"depth_stencil"] && e[@"states"]) {
			NSArray *states = e[@"states"];
			__unsafe_unretained id<MTLDepthStencilState> list[states.count];
			NSMutableArray *keep = [NSMutableArray new];
			for (NSUInteger k = 0; k < states.count; k++) {
				[keep addObject:argument_depth_stencil(c, states[k])];
				list[k] = keep[k];
			}
			if (@available(macOS 26.0, *))
				[ae setDepthStencilStates:list withRange:NSMakeRange(index, states.count)];
			else
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"encoded depth-stencil states need macOS 26" userInfo:nil];
		} else if ([kind isEqualToString:@"depth_stencil"]) {
			if (@available(macOS 26.0, *))
				[ae setDepthStencilState:argument_depth_stencil(c, e) atIndex:index];
			else
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"encoded depth-stencil states need macOS 26" userInfo:nil];
		} else if ([kind isEqualToString:@"nested"]) {
			id<MTLArgumentEncoder> inner = [ae newArgumentEncoderForBufferAtIndex:index];
			id<MTLBuffer> target = c.buffers[e[@"buffer"]];
			[ae setBuffer:target offset:number(e, @"offset", 0) atIndex:index];
			[c.resident addObject:target];
			[inner setArgumentBuffer:target offset:number(e, @"offset", 0)];
			encode_entries(c, library, inner, e[@"entries"]);
		} else {
			@throw failure(@"unknown encoded argument kind %@", kind);
		}
	}
}

/* Fills an argument buffer's entries with GPU addresses and resource IDs,
 * the Tier 2 layout of a struct of pointers, textures and samplers. */
static void encode_arguments(Case *c, NSDictionary *a)
{
	id<MTLBuffer> buffer = c.buffers[a[@"buffer"]];
	if (!buffer)
		@throw failure(@"argument buffer %@ is not a buffer", a[@"buffer"]);
	uint8_t *bytes = buffer.contents;
	/* The case's library, loaded only by entries that name functions. */
	id<MTLLibrary> (^library)(void) = ^{
		return library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	};
	if (a[@"encoder"]) {
		/* The argument encoder a function's buffer argument describes,
		 * each entry at its [[id]]: the path before Tier 2 layouts. An
		 * array of structs encodes each element from its start. */
		id<MTLArgumentEncoder> ae = argument_encoder(c, library(), a[@"encoder"]);
		if (a[@"element"])
			[ae setArgumentBuffer:buffer startOffset:number(a, @"offset", 0) arrayElement:number(a, @"element", 0)];
		else
			[ae setArgumentBuffer:buffer offset:number(a, @"offset", 0)];
		encode_entries(c, library(), ae, a[@"entries"]);
		/* What the specification asks of the encoder's layout, which
		 * the device chooses: some length, which fits the buffer from
		 * its offset, at a power-of-two alignment. */
		if (a[@"layout"])
			write_words(c, a[@"layout"], @[ @(ae.encodedLength > 0), @(number(a, @"offset", 0) + ae.encodedLength <= buffer.length),
							@(power_of_two(ae.alignment)) ]);
		return;
	}
	for (NSDictionary *e in a[@"entries"]) {
		NSUInteger offset = number(e, @"offset", 0);
		NSString *kind = e[@"kind"], *name = e[@"resource"];
		if ([kind isEqualToString:@"buffer"]) {
			id<MTLBuffer> target = c.buffers[name];
			if (!target) @throw failure(@"argument %@ names no buffer", name);
			uint64_t address = target.gpuAddress + number(e, @"resource_offset", 0);
			memcpy(bytes + offset, &address, 8);
			/* A heap's resources are made resident by useHeap, and a
			 * residency set's by the set. */
			if (!target.heap && ![e[@"residency_set"] boolValue])
				[c.resident addObject:target];
		} else if ([kind isEqualToString:@"texture"]) {
			id<MTLTexture> target = c.textures[name];
			if (!target) @throw failure(@"argument %@ names no texture", name);
			MTLResourceID id_ = target.gpuResourceID;
			memcpy(bytes + offset, &id_, 8);
			if (!target.heap)
				[c.resident addObject:target];
		} else if ([kind isEqualToString:@"icb"]) {
			id<MTLIndirectCommandBuffer> target = c.icbs[name];
			if (!target) @throw failure(@"argument %@ names no indirect command buffer", name);
			MTLResourceID id_ = target.gpuResourceID;
			memcpy(bytes + offset, &id_, 8);
			[c.resident addObject:target];
		} else if ([kind isEqualToString:@"sampler"]) {
			id<MTLSamplerState> target = c.samplers[name];
			if (!target) @throw failure(@"argument %@ names no sampler", name);
			MTLResourceID id_ = target.gpuResourceID;
			memcpy(bytes + offset, &id_, 8);
		} else if ([kind isEqualToString:@"compute_pipeline"]) {
			/* A pipeline an indirect command sets, by its resource ID. */
			MTLResourceID id_ = argument_compute_pipeline(c, library(), e).gpuResourceID;
			memcpy(bytes + offset, &id_, 8);
		} else if ([kind isEqualToString:@"depth_stencil"]) {
			/* A depth-stencil state an indirect command sets. */
			if (@available(macOS 26.0, *)) {
				MTLResourceID id_ = argument_depth_stencil(c, e).gpuResourceID;
				memcpy(bytes + offset, &id_, 8);
			} else {
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"depth-stencil resource IDs need macOS 26" userInfo:nil];
			}
		} else if ([kind isEqualToString:@"render_pipeline"]) {
			MTLResourceID id_ = argument_render_pipeline(c, library(), e).gpuResourceID;
			memcpy(bytes + offset, &id_, 8);
		} else {
			@throw failure(@"unknown argument kind %@", kind);
		}
	}
}

/* The bytes an inline buffer records: its declared length, or all of it. */
static NSUInteger inline_length(NSDictionary *b, id<MTLBuffer> buffer)
{
	NSUInteger length = number(b, @"length", 0);
	return length ? MIN(length, buffer.length) : buffer.length;
}

/* The stages a binding names, as bits of a mask: a compute encoder has
 * one, a render encoder four, and "both" is the vertex and fragment. */
typedef NS_ENUM(NSUInteger, Stage) { StageCompute, StageVertex, StageFragment, StageObject, StageMesh, StageTile, StageCount };

enum { buffer_slots = 31, texture_slots = 128, sampler_slots = 16 };

/* What one stage binds, by index. A buffer without an attribute stride
 * holds MTLAttributeStrideStatic; a sampler's LOD clamps are its own
 * unless the binding overrides them. */
typedef struct {
	__unsafe_unretained id<MTLBuffer> buffers[buffer_slots];
	NSUInteger offsets[buffer_slots], strides[buffer_slots];
	__unsafe_unretained id<MTLTexture> textures[texture_slots];
	__unsafe_unretained id<MTLSamplerState> samplers[sampler_slots];
	float lod_min[sampler_slots], lod_max[sampler_slots];
	BOOL clamped[sampler_slots];
} Slots;

static NSUInteger stages_of(NSDictionary *binding, BOOL compute)
{
	if (compute)
		return 1u << StageCompute;
	NSDictionary *stages = @{@"vertex" : @(1u << StageVertex), @"fragment" : @(1u << StageFragment),
				 @"both" : @(1u << StageVertex | 1u << StageFragment), @"object" : @(1u << StageObject),
				 @"mesh" : @(1u << StageMesh), @"tile" : @(1u << StageTile)};
	return lookup(stages, @"binding stage", binding[@"stage"] ?: @"fragment");
}

static void need_static(Stage s, NSUInteger stride)
{
	if (s != StageCompute && s != StageVertex && stride != MTLAttributeStrideStatic)
		@throw failure(@"attribute strides bind only compute and vertex buffers");
}

static void set_buffer(id e, Stage s, id<MTLBuffer> b, NSUInteger offset, NSUInteger stride, NSUInteger i)
{
	need_static(s, stride);
	BOOL dynamic = stride != MTLAttributeStrideStatic;
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	switch (s) {
	case StageCompute:
		if (dynamic)
			[ce setBuffer:b offset:offset attributeStride:stride atIndex:i];
		else
			[ce setBuffer:b offset:offset atIndex:i];
		break;
	case StageVertex:
		if (dynamic)
			[re setVertexBuffer:b offset:offset attributeStride:stride atIndex:i];
		else
			[re setVertexBuffer:b offset:offset atIndex:i];
		break;
	case StageFragment: [re setFragmentBuffer:b offset:offset atIndex:i]; break;
	case StageObject: [re setObjectBuffer:b offset:offset atIndex:i]; break;
	case StageMesh: [re setMeshBuffer:b offset:offset atIndex:i]; break;
	case StageTile: [re setTileBuffer:b offset:offset atIndex:i]; break;
	default: break;
	}
}

static void set_offset(id e, Stage s, NSUInteger offset, NSUInteger stride, NSUInteger i)
{
	need_static(s, stride);
	BOOL dynamic = stride != MTLAttributeStrideStatic;
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	switch (s) {
	case StageCompute:
		if (dynamic)
			[ce setBufferOffset:offset attributeStride:stride atIndex:i];
		else
			[ce setBufferOffset:offset atIndex:i];
		break;
	case StageVertex:
		if (dynamic)
			[re setVertexBufferOffset:offset attributeStride:stride atIndex:i];
		else
			[re setVertexBufferOffset:offset atIndex:i];
		break;
	case StageFragment: [re setFragmentBufferOffset:offset atIndex:i]; break;
	case StageObject: [re setObjectBufferOffset:offset atIndex:i]; break;
	case StageMesh: [re setMeshBufferOffset:offset atIndex:i]; break;
	case StageTile: [re setTileBufferOffset:offset atIndex:i]; break;
	default: break;
	}
}

static void set_buffers(id e, Stage s, Slots *t, NSRange r)
{
	BOOL dynamic = NO;
	for (NSUInteger i = r.location; i < NSMaxRange(r); i++) {
		need_static(s, t->strides[i]);
		dynamic |= t->strides[i] != MTLAttributeStrideStatic;
	}
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	id<MTLBuffer> const __unsafe_unretained *b = t->buffers + r.location;
	const NSUInteger *o = t->offsets + r.location, *st = t->strides + r.location;
	switch (s) {
	case StageCompute:
		if (dynamic)
			[ce setBuffers:b offsets:o attributeStrides:st withRange:r];
		else
			[ce setBuffers:b offsets:o withRange:r];
		break;
	case StageVertex:
		if (dynamic)
			[re setVertexBuffers:b offsets:o attributeStrides:st withRange:r];
		else
			[re setVertexBuffers:b offsets:o withRange:r];
		break;
	case StageFragment: [re setFragmentBuffers:b offsets:o withRange:r]; break;
	case StageObject: [re setObjectBuffers:b offsets:o withRange:r]; break;
	case StageMesh: [re setMeshBuffers:b offsets:o withRange:r]; break;
	case StageTile: [re setTileBuffers:b offsets:o withRange:r]; break;
	default: break;
	}
}

static void set_bytes(id e, Stage s, const void *bytes, NSUInteger length, NSUInteger stride, NSUInteger i)
{
	need_static(s, stride);
	BOOL dynamic = stride != MTLAttributeStrideStatic;
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	switch (s) {
	case StageCompute:
		if (dynamic)
			[ce setBytes:bytes length:length attributeStride:stride atIndex:i];
		else
			[ce setBytes:bytes length:length atIndex:i];
		break;
	case StageVertex:
		if (dynamic)
			[re setVertexBytes:bytes length:length attributeStride:stride atIndex:i];
		else
			[re setVertexBytes:bytes length:length atIndex:i];
		break;
	case StageFragment: [re setFragmentBytes:bytes length:length atIndex:i]; break;
	case StageObject: [re setObjectBytes:bytes length:length atIndex:i]; break;
	case StageMesh: [re setMeshBytes:bytes length:length atIndex:i]; break;
	case StageTile: [re setTileBytes:bytes length:length atIndex:i]; break;
	default: break;
	}
}

static void set_textures(id e, Stage s, Slots *t, NSRange r)
{
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	id<MTLTexture> const __unsafe_unretained *x = t->textures + r.location;
	if (r.length == 1) {
		NSUInteger i = r.location;
		switch (s) {
		case StageCompute: [ce setTexture:x[0] atIndex:i]; break;
		case StageVertex: [re setVertexTexture:x[0] atIndex:i]; break;
		case StageFragment: [re setFragmentTexture:x[0] atIndex:i]; break;
		case StageObject: [re setObjectTexture:x[0] atIndex:i]; break;
		case StageMesh: [re setMeshTexture:x[0] atIndex:i]; break;
		case StageTile: [re setTileTexture:x[0] atIndex:i]; break;
		default: break;
		}
		return;
	}
	switch (s) {
	case StageCompute: [ce setTextures:x withRange:r]; break;
	case StageVertex: [re setVertexTextures:x withRange:r]; break;
	case StageFragment: [re setFragmentTextures:x withRange:r]; break;
	case StageObject: [re setObjectTextures:x withRange:r]; break;
	case StageMesh: [re setMeshTextures:x withRange:r]; break;
	case StageTile: [re setTileTextures:x withRange:r]; break;
	default: break;
	}
}

static void set_samplers(id e, Stage s, Slots *t, NSRange r)
{
	BOOL clamped = NO;
	for (NSUInteger i = r.location; i < NSMaxRange(r); i++)
		clamped |= t->clamped[i];
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	id<MTLSamplerState> const __unsafe_unretained *x = t->samplers + r.location;
	const float *lo = t->lod_min + r.location, *hi = t->lod_max + r.location;
	if (r.length == 1) {
		NSUInteger i = r.location;
		switch (s) {
		case StageCompute:
			if (clamped)
				[ce setSamplerState:x[0] lodMinClamp:lo[0] lodMaxClamp:hi[0] atIndex:i];
			else
				[ce setSamplerState:x[0] atIndex:i];
			break;
		case StageVertex:
			if (clamped)
				[re setVertexSamplerState:x[0] lodMinClamp:lo[0] lodMaxClamp:hi[0] atIndex:i];
			else
				[re setVertexSamplerState:x[0] atIndex:i];
			break;
		case StageFragment:
			if (clamped)
				[re setFragmentSamplerState:x[0] lodMinClamp:lo[0] lodMaxClamp:hi[0] atIndex:i];
			else
				[re setFragmentSamplerState:x[0] atIndex:i];
			break;
		case StageObject:
			if (clamped)
				[re setObjectSamplerState:x[0] lodMinClamp:lo[0] lodMaxClamp:hi[0] atIndex:i];
			else
				[re setObjectSamplerState:x[0] atIndex:i];
			break;
		case StageMesh:
			if (clamped)
				[re setMeshSamplerState:x[0] lodMinClamp:lo[0] lodMaxClamp:hi[0] atIndex:i];
			else
				[re setMeshSamplerState:x[0] atIndex:i];
			break;
		case StageTile:
			if (clamped)
				[re setTileSamplerState:x[0] lodMinClamp:lo[0] lodMaxClamp:hi[0] atIndex:i];
			else
				[re setTileSamplerState:x[0] atIndex:i];
			break;
		default: break;
		}
		return;
	}
	switch (s) {
	case StageCompute:
		if (clamped)
			[ce setSamplerStates:x lodMinClamps:lo lodMaxClamps:hi withRange:r];
		else
			[ce setSamplerStates:x withRange:r];
		break;
	case StageVertex:
		if (clamped)
			[re setVertexSamplerStates:x lodMinClamps:lo lodMaxClamps:hi withRange:r];
		else
			[re setVertexSamplerStates:x withRange:r];
		break;
	case StageFragment:
		if (clamped)
			[re setFragmentSamplerStates:x lodMinClamps:lo lodMaxClamps:hi withRange:r];
		else
			[re setFragmentSamplerStates:x withRange:r];
		break;
	case StageObject:
		if (clamped)
			[re setObjectSamplerStates:x lodMinClamps:lo lodMaxClamps:hi withRange:r];
		else
			[re setObjectSamplerStates:x withRange:r];
		break;
	case StageMesh:
		if (clamped)
			[re setMeshSamplerStates:x lodMinClamps:lo lodMaxClamps:hi withRange:r];
		else
			[re setMeshSamplerStates:x withRange:r];
		break;
	case StageTile:
		if (clamped)
			[re setTileSamplerStates:x lodMinClamps:lo lodMaxClamps:hi withRange:r];
		else
			[re setTileSamplerStates:x withRange:r];
		break;
	default: break;
	}
}

/* Each run of consecutive bound indices in `bound` below `count`. */
static void each_run(NSUInteger count, BOOL (^bound)(NSUInteger), BOOL ranges, void (^run)(NSRange))
{
	for (NSUInteger i = 0; i < count;) {
		if (!bound(i)) {
			i++;
			continue;
		}
		NSUInteger end = i + 1;
		while (ranges && end < count && bound(end))
			end++;
		run(NSMakeRange(i, end - i));
		i = end;
	}
}

/* Binds a case's buffers, textures and samplers to the encoder's stages
 * the way the case's `bind` names: "single" calls per resource, "ranges"
 * calls over each run of consecutive indices, "offsets", each buffer at
 * offset 0 and then moved to its own, or "legacy", single calls with the
 * deprecated residency calls that name no stages; then makes resident
 * what its argument buffers reach. */
static void bind_resources(Case *c, id e, NSDictionary *spec, BOOL compute, MTLRenderStages render_stages)
{
	NSUInteger mode = lookup(@{@"single" : @0, @"ranges" : @1, @"offsets" : @2, @"legacy" : @3}, @"bind mode", spec[@"bind"] ?: @"single");
	BOOL ranges = mode == 1, offsets = mode == 2, legacy = mode == 3;
	NSMutableData *storage = [NSMutableData dataWithLength:StageCount * sizeof(Slots)];
	Slots *slots = storage.mutableBytes;
	for (NSUInteger s = 0; s < StageCount; s++) {
		for (NSUInteger i = 0; i < buffer_slots; i++)
			slots[s].strides[i] = MTLAttributeStrideStatic;
	}
	NSUInteger used = 0;
	for (NSDictionary *b in spec[@"buffers"]) {
		if (!b[@"index"])
			continue;
		NSUInteger index = number(b, @"index", 0), mask = stages_of(b, compute);
		NSUInteger stride = b[@"attribute_stride"] ? number(b, @"attribute_stride", 0) : MTLAttributeStrideStatic;
		id<MTLBuffer> buffer = c.buffers[b[@"name"]];
		if (index >= buffer_slots)
			@throw failure(@"buffer index %lu", (unsigned long)index);
		for (NSUInteger s = 0; s < StageCount; s++) {
			if (!(mask & 1u << s))
				continue;
			/* An inline buffer's bytes are recorded into the command. */
			if ([b[@"inline"] boolValue]) {
				set_bytes(e, s, buffer.contents, inline_length(b, buffer), stride, index);
				continue;
			}
			slots[s].buffers[index] = buffer;
			slots[s].offsets[index] = number(b, @"offset", 0);
			slots[s].strides[index] = stride;
			used |= 1u << s;
		}
	}
	for (NSDictionary *t in spec[@"textures"]) {
		if (!t[@"index"])
			continue;
		NSUInteger index = number(t, @"index", 0), mask = stages_of(t, compute);
		if (index >= texture_slots)
			@throw failure(@"texture index %lu", (unsigned long)index);
		for (NSUInteger s = 0; s < StageCount; s++)
			if (mask & 1u << s) {
				slots[s].textures[index] = c.textures[t[@"name"]];
				used |= 1u << s;
			}
	}
	for (NSDictionary *d in spec[@"samplers"]) {
		if (!d[@"index"])
			continue;
		NSUInteger index = number(d, @"index", 0), mask = stages_of(d, compute);
		if (index >= sampler_slots)
			@throw failure(@"sampler index %lu", (unsigned long)index);
		/* `bind_lod` replaces the state's LOD clamps for this binding. */
		NSArray *clamp = d[@"bind_lod"];
		for (NSUInteger s = 0; s < StageCount; s++)
			if (mask & 1u << s) {
				slots[s].samplers[index] = c.samplers[d[@"name"]];
				slots[s].clamped[index] = clamp != nil;
				slots[s].lod_min[index] = clamp ? [clamp[0] floatValue] : d[@"lod_min"] ? [d[@"lod_min"] floatValue] : 0.0f;
				slots[s].lod_max[index] = clamp ? [clamp[1] floatValue] : d[@"lod_max"] ? [d[@"lod_max"] floatValue] : FLT_MAX;
				used |= 1u << s;
			}
	}
	for (NSUInteger s = 0; s < StageCount; s++) {
		if (!(used & 1u << s))
			continue;
		Slots *t = &slots[s];
		each_run(buffer_slots, ^BOOL(NSUInteger i) { return t->buffers[i] != nil; }, ranges, ^(NSRange r) {
			if (ranges) {
				set_buffers(e, s, t, r);
				return;
			}
			NSUInteger i = r.location;
			set_buffer(e, s, t->buffers[i], offsets ? 0 : t->offsets[i], t->strides[i], i);
			if (offsets)
				set_offset(e, s, t->offsets[i], t->strides[i], i);
		});
		each_run(texture_slots, ^BOOL(NSUInteger i) { return t->textures[i] != nil; }, ranges, ^(NSRange r) {
			set_textures(e, s, t, r);
		});
		each_run(sampler_slots, ^BOOL(NSUInteger i) { return t->samplers[i] != nil; }, ranges, ^(NSRange r) {
			set_samplers(e, s, t, r);
		});
	}
	NSMutableArray<id<MTLResource>> *resident = [NSMutableArray new];
	for (id r in c.resident)
		if ([r conformsToProtocol:@protocol(MTLResource)])
			[resident addObject:r];
	NSMutableArray<id<MTLHeap>> *heaps = [NSMutableArray new];
	for (NSString *name in spec[@"use_heaps"])
		[heaps addObject:c.heaps[name]];
	/* Arrays of resources and heaps made resident in one call each. */
	NSUInteger count = MAX(resident.count, heaps.count);
	__unsafe_unretained id<MTLResource> resources[MAX(count, 1u)];
	__unsafe_unretained id<MTLHeap> heap_list[MAX(count, 1u)];
	[resident getObjects:resources range:NSMakeRange(0, resident.count)];
	[heaps getObjects:heap_list range:NSMakeRange(0, heaps.count)];
	MTLResourceUsage usage = MTLResourceUsageRead | MTLResourceUsageWrite;
	if (compute) {
		id<MTLComputeCommandEncoder> ce = e;
		if (ranges && resident.count)
			[ce useResources:resources count:resident.count usage:usage];
		else
			for (id<MTLResource> r in resident)
				[ce useResource:r usage:usage];
		if (ranges && heaps.count)
			[ce useHeaps:heap_list count:heaps.count];
		else
			for (id<MTLHeap> h in heaps)
				[ce useHeap:h];
		for (NSDictionary *m in spec[@"threadgroup_memory"])
			[ce setThreadgroupMemoryLength:number(m, @"length", 16) atIndex:number(m, @"index", 0)];
		/* A kernel's explicit imageblock's width and height. */
		NSArray *imageblock = spec[@"imageblock"];
		if (imageblock)
			[ce setImageblockWidth:[imageblock[0] unsignedIntegerValue] height:[imageblock[1] unsignedIntegerValue]];
		return;
	}
	id<MTLRenderCommandEncoder> re = e;
	if (legacy) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		if (resident.count)
			[re useResources:resources count:resident.count usage:usage];
		if (heaps.count)
			[re useHeaps:heap_list count:heaps.count];
		for (id<MTLResource> r in resident)
			[re useResource:r usage:usage];
		for (id<MTLHeap> h in heaps)
			[re useHeap:h];
#pragma clang diagnostic pop
		return;
	}
	if (ranges && resident.count)
		[re useResources:resources count:resident.count usage:usage stages:render_stages];
	else
		for (id<MTLResource> r in resident)
			[re useResource:r usage:usage stages:render_stages];
	if (ranges && heaps.count)
		[re useHeaps:heap_list count:heaps.count stages:render_stages];
	else
		for (id<MTLHeap> h in heaps)
			[re useHeap:h stages:render_stages];
}

/* A function a pipeline links: a name in the case's library, or an entry
 * naming another of the plan's libraries. */
static id<MTLFunction> linked_function(Case *c, id<MTLLibrary> library, id entry)
{
	if ([entry isKindOfClass:[NSString class]])
		return function_named(library, entry, nil);
	id<MTLLibrary> from = library_named(c.device, c.plan, entry[@"library"], c.libraries);
	if (!entry[@"specialized_name"])
		return function_named(from, entry[@"name"], nil);
	/* A specialization, which pipelines link by the name it takes. */
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = entry[@"name"];
	fd.specializedName = entry[@"specialized_name"];
	fd.constantValues = constants_of(entry[@"constants"] ?: @[]);
	NSError *error = nil;
	id<MTLFunction> f = [from newFunctionWithDescriptor:fd error:&error];
	if (!f)
		@throw failure(@"specialized function %@: %@", entry[@"name"], error.localizedDescription);
	return f;
}

/* Adds a stage's `<prefix>private_functions`, linked for direct calls
 * alone, and `<prefix>function_groups`, call-site groups of its linked
 * functions by name. */
static void link_extras(Case *c, id<MTLLibrary> library, NSDictionary *spec, NSString *prefix, MTLLinkedFunctions *l)
{
	NSMutableArray *private = [NSMutableArray new];
	for (id entry in spec[[prefix stringByAppendingString:@"private_functions"]])
		[private addObject:linked_function(c, library, entry)];
	if (private.count)
		l.privateFunctions = private;
	NSDictionary *groups = spec[[prefix stringByAppendingString:@"function_groups"]];
	if (!groups)
		return;
	NSArray<id<MTLFunction>> *all = [(l.functions ?: @[]) arrayByAddingObjectsFromArray:l.binaryFunctions ?: @[]];
	NSMutableDictionary *g = [NSMutableDictionary new];
	for (NSString *name in groups) {
		NSMutableArray *members = [NSMutableArray new];
		for (NSString *f in groups[name])
			for (id<MTLFunction> v in all)
				if ([v.name isEqualToString:f])
					[members addObject:v];
		g[name] = members;
	}
	l.groups = g;
}

/* A stage's linked functions: `visible`, callable through its tables,
 * then the extras `link_extras` adds; nil when it links none. */
static MTLLinkedFunctions *stage_links(Case *c, id<MTLLibrary> library, NSDictionary *spec, NSString *prefix, NSArray *visible)
{
	MTLLinkedFunctions *l = [MTLLinkedFunctions new];
	l.functions = visible;
	link_extras(c, library, spec, prefix, l);
	return visible.count || l.privateFunctions.count || l.groups ? l : nil;
}

/* A visible function table's handles, set one entry at a time or, when
 * the case binds by ranges, all in one call. */
static void fill_table(id<MTLVisibleFunctionTable> vft, NSArray<id<MTLFunctionHandle>> *handles, BOOL ranges)
{
	if (!ranges) {
		for (NSUInteger i = 0; i < handles.count; i++)
			[vft setFunction:handles[i] atIndex:i];
		return;
	}
	__unsafe_unretained id<MTLFunctionHandle> list[MAX(handles.count, 1u)];
	[handles getObjects:list range:NSMakeRange(0, handles.count)];
	[vft setFunctions:list withRange:NSMakeRange(0, handles.count)];
}

/* Binds a stage's visible function tables by buffer index, singly or
 * over each run of consecutive indices. */
static void bind_tables(id e, Stage s, NSDictionary<NSNumber *, id<MTLVisibleFunctionTable>> *tables, BOOL ranges)
{
	__unsafe_unretained id<MTLVisibleFunctionTable> storage[buffer_slots] = {nil};
	id<MTLVisibleFunctionTable> __unsafe_unretained *slots = storage;
	for (NSNumber *k in tables) {
		if (k.unsignedIntegerValue >= buffer_slots)
			@throw failure(@"function table index %@", k);
		slots[k.unsignedIntegerValue] = tables[k];
	}
	id<MTLComputeCommandEncoder> ce = e;
	id<MTLRenderCommandEncoder> re = e;
	each_run(buffer_slots, ^BOOL(NSUInteger i) { return slots[i] != nil; }, ranges, ^(NSRange r) {
		id<MTLVisibleFunctionTable> const __unsafe_unretained *t = slots + r.location;
		if (r.length == 1) {
			if (s == StageCompute)
				[ce setVisibleFunctionTable:t[0] atBufferIndex:r.location];
			else if (s == StageVertex)
				[re setVertexVisibleFunctionTable:t[0] atBufferIndex:r.location];
			else if (s == StageTile)
				[re setTileVisibleFunctionTable:t[0] atBufferIndex:r.location];
			else
				[re setFragmentVisibleFunctionTable:t[0] atBufferIndex:r.location];
		} else if (s == StageCompute) {
			[ce setVisibleFunctionTables:t withBufferRange:r];
		} else if (s == StageVertex) {
			[re setVertexVisibleFunctionTables:t withBufferRange:r];
		} else if (s == StageTile) {
			[re setTileVisibleFunctionTables:t withBufferRange:r];
		} else {
			[re setFragmentVisibleFunctionTables:t withBufferRange:r];
		}
	});
}

/* Visible function tables a compute function reaches through an argument
 * buffer, each at its [[id]]: set by the argument encoder of the
 * function's buffer, singly or over each run of consecutive ids, or
 * written as resource IDs in the Tier 2 layout. The tables are made
 * resident either way. */
static void encode_tables(Case *c, id<MTLComputeCommandEncoder> e, id<MTLFunction> function, NSDictionary *a,
			  NSDictionary<NSNumber *, id<MTLVisibleFunctionTable>> *tables, BOOL ranges)
{
	if (!tables.count)
		return;
	id<MTLBuffer> buffer = c.buffers[a[@"buffer"]];
	if (!buffer)
		@throw failure(@"table argument buffer %@ is not a buffer", a[@"buffer"]);
	NSUInteger count = 0;
	for (NSNumber *k in tables)
		count = MAX(count, k.unsignedIntegerValue + 1);
	if (count > buffer_slots)
		@throw failure(@"function table id %lu", (unsigned long)count - 1);
	__unsafe_unretained id<MTLVisibleFunctionTable> storage[buffer_slots] = {nil};
	id<MTLVisibleFunctionTable> __unsafe_unretained *slots = storage;
	for (NSNumber *k in tables) {
		slots[k.unsignedIntegerValue] = tables[k];
		[e useResource:tables[k] usage:MTLResourceUsageRead];
	}
	if (![a[@"encoder"] boolValue]) {
		for (NSUInteger i = 0; i < count; i++) {
			if (!slots[i])
				continue;
			MTLResourceID id_ = slots[i].gpuResourceID;
			memcpy((uint8_t *)buffer.contents + 8 * i, &id_, 8);
		}
		return;
	}
	id<MTLArgumentEncoder> ae = [function newArgumentEncoderWithBufferIndex:number(a, @"index", 0)];
	[ae setArgumentBuffer:buffer offset:0];
	each_run(count, ^BOOL(NSUInteger i) { return slots[i] != nil; }, ranges, ^(NSRange r) {
		if (r.length == 1)
			[ae setVisibleFunctionTable:slots[r.location] atIndex:r.location];
		else
			[ae setVisibleFunctionTables:slots + r.location withRange:r];
	});
}

/* A stitched function from the plan's graph: each node calls a stitchable
 * library function on the graph's inputs or earlier nodes. */
static id<MTLFunction> stitched_function(Case *c, id<MTLLibrary> library, NSDictionary *g)
{
	NSMutableArray *functions = [NSMutableArray new], *nodes = [NSMutableArray new];
	NSMutableDictionary *named = [NSMutableDictionary new];
	for (NSUInteger i = 0; i < number(g, @"inputs", 0); i++)
		named[[NSString stringWithFormat:@"in%lu", (unsigned long)i]] = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:i];
	for (NSDictionary *n in g[@"nodes"]) {
		id<MTLFunction> f = function_named(library, n[@"function"], nil);
		if (![functions containsObject:f])
			[functions addObject:f];
		NSMutableArray *arguments = [NSMutableArray new];
		for (NSString *a in n[@"args"]) {
			if (!named[a]) @throw failure(@"stitching argument %@ is undefined", a);
			[arguments addObject:named[a]];
		}
		MTLFunctionStitchingFunctionNode *node =
			[[MTLFunctionStitchingFunctionNode alloc] initWithName:n[@"function"] arguments:arguments controlDependencies:@[]];
		named[n[@"id"]] = node;
		[nodes addObject:node];
	}
	MTLFunctionStitchingGraph *graph = [[MTLFunctionStitchingGraph alloc] initWithFunctionName:g[@"name"]
										    nodes:[nodes subarrayWithRange:NSMakeRange(0, nodes.count - 1)]
									       outputNode:nodes.lastObject
									       attributes:@[]];
	/* The graph holds what it was built from. */
	if (![graph.functionName isEqualToString:g[@"name"]] || graph.outputNode != nodes.lastObject || graph.nodes.count + 1 != nodes.count)
		@throw failure(@"stitching graph %@ does not hold its nodes", g[@"name"]);
	for (MTLFunctionStitchingFunctionNode *node in nodes)
		for (id<MTLFunctionStitchingNode> a in node.arguments)
			if ([(NSObject *)a isKindOfClass:[MTLFunctionStitchingInputNode class]] &&
			    ((MTLFunctionStitchingInputNode *)a).argumentIndex >= number(g, @"inputs", 0))
				@throw failure(@"stitching input past the graph's inputs");
	if (((MTLFunctionStitchingFunctionNode *)nodes.lastObject).controlDependencies.count)
		@throw failure(@"stitching node has control dependencies it was not given");
	MTLStitchedLibraryDescriptor *d = [MTLStitchedLibraryDescriptor new];
	d.functions = functions;
	d.functionGraphs = @[ graph ];
	__block NSError *error = nil;
	__block id<MTLLibrary> stitched = nil;
	if ([creation isEqualToString:@"async"])
		await_handler(^(dispatch_semaphore_t done) {
			[c.device newLibraryWithStitchedDescriptor:d
						 completionHandler:^(id<MTLLibrary> l, NSError *e) {
							 stitched = l;
							 error = e;
							 dispatch_semaphore_signal(done);
						 }];
		});
	else
		stitched = [c.device newLibraryWithStitchedDescriptor:d error:&error];
	if (!stitched)
		@throw failure(@"stitched library: %@", error.localizedDescription);
	return function_named(stitched, g[@"name"], nil);
}

/* The device's counter set of a common name: timestamp or statistic. */
static id<MTLCounterSet> counter_set(id<MTLDevice> d, NSString *name)
{
	NSString *common = [name isEqualToString:@"statistic"] ? MTLCommonCounterSetStatistic : MTLCommonCounterSetTimestamp;
	/* The set must hold the counter its results are laid out by. */
	NSString *counter = [name isEqualToString:@"statistic"] ? MTLCommonCounterComputeKernelInvocations : MTLCommonCounterTimestamp;
	for (id<MTLCounterSet> set in d.counterSets) {
		if (![set.name isEqualToString:common])
			continue;
		for (id<MTLCounter> k in set.counters)
			if ([k.name isEqualToString:counter])
				return set;
		@throw failure(@"%@ counter set lacks %@", name, counter);
	}
	@throw [NSException exceptionWithName:@"CaseUnsupported" reason:[NSString stringWithFormat:@"no %@ counter set", name] userInfo:nil];
}

/* Once the command buffer completes, the samples resolved into the named
 * buffer: timestamps as sampled, statistics as the counts between each
 * pair of samples. */
static void resolve_counters_after(Case *c, id<MTLCommandBuffer> cb, id<MTLCounterSampleBuffer> samples, NSUInteger count,
				   NSDictionary *spec)
{
	id<MTLBuffer> out = c.buffers[spec[@"buffer"]];
	if (!out)
		@throw failure(@"counters name no buffer %@", spec[@"buffer"]);
	BOOL statistic = [spec[@"counters"] isEqualToString:@"statistic"];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> _) {
		NSData *data = [samples resolveCounterRange:NSMakeRange(0, count)];
		memset(out.contents, 0, out.length);
		if (!statistic) {
			memcpy(out.contents, data.bytes, MIN(data.length, out.length));
			return;
		}
		const MTLCounterResultStatistic *s = data.bytes;
		uint64_t *o = out.contents;
		const NSUInteger fields = sizeof(MTLCounterResultStatistic) / sizeof(uint64_t);
		for (NSUInteger pair = 0; pair + 1 < count && (pair / 2 + 1) * fields * 8 <= out.length; pair += 2) {
			const uint64_t *from = (const uint64_t *)&s[pair], *to = (const uint64_t *)&s[pair + 1];
			for (NSUInteger f = 0; f < fields; f++)
				o[pair / 2 * fields + f] = to[f] - from[f];
		}
	}];
}

/* The named buffer's bytes replaced by a text and zeros after it. */
static void write_text(Case *c, NSString *name, NSString *text)
{
	id<MTLBuffer> out = c.buffers[name];
	if (!out)
		@throw failure(@"text output names no buffer %@", name);
	NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding];
	memset(out.contents, 0, out.length);
	memcpy(out.contents, bytes.bytes, MIN(bytes.length, out.length));
}

/* Words, each a uint32, into the named buffer; nothing when unnamed. */
static void write_words(Case *c, NSString *name, NSArray<NSNumber *> *words)
{
	if (!name)
		return;
	id<MTLBuffer> out = c.buffers[name];
	if (!out)
		@throw failure(@"words output names no buffer %@", name);
	uint32_t *w = out.contents;
	for (NSUInteger i = 0; i < words.count && (i + 1) * 4 <= out.length; i++)
		w[i] = words[i].unsignedIntValue;
}

/* A command buffer, with a log state that collects what its shaders log
 * when the case asks for `log`. */
static id<MTLCommandBuffer> command_buffer(Case *c, id<MTLCommandQueue> queue, NSDictionary *spec, NSMutableArray *lines)
{
	if (!spec[@"log"])
		return [queue commandBuffer];
	if (@available(macOS 15.0, *)) {
		MTLLogStateDescriptor *ld = [MTLLogStateDescriptor new];
		ld.level = MTLLogLevelDebug;
		ld.bufferSize = 65536;
		NSError *error = nil;
		id<MTLLogState> state = [c.device newLogStateWithDescriptor:ld error:&error];
		if (!state)
			@throw failure(@"log state: %@", error.localizedDescription);
		[state addLogHandler:^(NSString *subsystem, NSString *category, MTLLogLevel level, NSString *message) {
			@synchronized(lines) {
				[lines addObject:[NSString stringWithFormat:@"%@ %@ %ld %@", subsystem ?: @"-", category ?: @"-", (long)level, message]];
			}
		}];
		MTLCommandBufferDescriptor *cd = [MTLCommandBufferDescriptor new];
		cd.logState = state;
		return [queue commandBufferWithDescriptor:cd];
	}
	@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"shader logging needs macOS 15" userInfo:nil];
}

/* The logged lines, once `count` have arrived or two seconds passed, in
 * the order the log handler received them. */
static void store_log(Case *c, NSDictionary *spec, NSMutableArray *lines)
{
	NSUInteger want = number(spec[@"log"], @"lines", 1);
	for (int k = 0; k < 200; k++) {
		@synchronized(lines) {
			if (lines.count >= want)
				break;
		}
		usleep(10000);
	}
	@synchronized(lines) {
		write_text(c, spec[@"log"][@"buffer"], [[lines componentsJoinedByString:@"\n"] stringByAppendingString:@"\n"]);
	}
}

static NSString *type_text(MTLType *t);

/* A struct's members, each as name@offset#argument index:data type and
 * its own type where it has one, after checking each is found by name. */
static NSString *struct_text(MTLStructType *st)
{
	NSMutableArray *members = [NSMutableArray new];
	for (MTLStructMember *m in st.members) {
		if (![[st memberByName:m.name].name isEqualToString:m.name])
			@throw failure(@"struct member %@ is not found by its name", m.name);
		NSMutableString *line = [NSMutableString stringWithFormat:@"%@@%lu#%lu:%ld", m.name, (unsigned long)m.offset,
									  (unsigned long)m.argumentIndex, (long)m.dataType];
		MTLType *inner = m.structType ?: m.arrayType ?: m.pointerType ?: (MTLType *)m.textureReferenceType;
		if (inner)
			[line appendFormat:@"=%@", type_text(inner)];
		[members addObject:line];
	}
	return [NSString stringWithFormat:@"{%@}", [members componentsJoinedByString:@" "]];
}

/* A reflected type in full: structs' members, arrays' length, stride,
 * argument index stride and element, pointers' access, alignment, size
 * and pointee, and textures' type, data type, access and depth. */
static NSString *type_text(MTLType *t)
{
	if ([t isKindOfClass:[MTLStructType class]])
		return struct_text((MTLStructType *)t);
	if ([t isKindOfClass:[MTLArrayType class]]) {
		MTLArrayType *a = (MTLArrayType *)t;
		MTLType *e = a.elementStructType ?: a.elementArrayType ?: a.elementPointerType ?: (MTLType *)a.elementTextureReferenceType;
		return [NSString stringWithFormat:@"[%lu stride=%lu ids=%lu %ld%@]", (unsigned long)a.arrayLength, (unsigned long)a.stride,
						  (unsigned long)a.argumentIndexStride, (long)a.elementType,
						  e ? [@" " stringByAppendingString:type_text(e)] : @""];
	}
	if ([t isKindOfClass:[MTLPointerType class]]) {
		MTLPointerType *p = (MTLPointerType *)t;
		MTLType *e = p.elementStructType ?: (MTLType *)p.elementArrayType;
		return [NSString stringWithFormat:@"*(%ld access=%ld align=%lu size=%lu argument_buffer=%d%@)", (long)p.elementType,
						  (long)p.access, (unsigned long)p.alignment, (unsigned long)p.dataSize, p.elementIsArgumentBuffer,
						  e ? [@" " stringByAppendingString:type_text(e)] : @""];
	}
	if ([t isKindOfClass:[MTLTextureReferenceType class]]) {
		MTLTextureReferenceType *x = (MTLTextureReferenceType *)t;
		return [NSString stringWithFormat:@"texture(%ld data=%ld access=%ld depth=%d)", (long)x.textureType, (long)x.textureDataType,
						  (long)x.access, x.isDepthTexture];
	}
	return [NSString stringWithFormat:@"%ld", (long)t.dataType];
}

/* A pipeline's bindings, one line each in index order within each kind:
 * what an application learns from reflection; with `types`, each buffer's
 * pointee in full. */
static NSString *bindings_text(NSArray<id<MTLBinding>> *bindings, BOOL types)
{
	NSMutableArray *lines = [NSMutableArray new];
	NSArray *sorted = [bindings sortedArrayUsingComparator:^NSComparisonResult(id<MTLBinding> a, id<MTLBinding> b) {
		if (a.type != b.type)
			return a.type < b.type ? NSOrderedAscending : NSOrderedDescending;
		return a.index < b.index ? NSOrderedAscending : a.index > b.index ? NSOrderedDescending : NSOrderedSame;
	}];
	for (id<MTLBinding> b in sorted) {
		NSMutableString *line = [NSMutableString stringWithFormat:@"%@ type=%ld index=%lu access=%ld used=%d argument=%d", b.name,
						(long)b.type, (unsigned long)b.index, (long)b.access, b.used, b.argument];
		if ([b conformsToProtocol:@protocol(MTLBufferBinding)]) {
			id<MTLBufferBinding> bb = (id<MTLBufferBinding>)b;
			[line appendFormat:@" size=%lu align=%lu data=%ld", (unsigned long)bb.bufferDataSize, (unsigned long)bb.bufferAlignment,
					   (long)bb.bufferDataType];
			if (bb.bufferStructType)
				for (MTLStructMember *m in bb.bufferStructType.members)
					[line appendFormat:@" %@@%lu:%ld", m.name, (unsigned long)m.offset, (long)m.dataType];
			if (types && bb.bufferPointerType)
				[line appendFormat:@" pointer=%@", type_text(bb.bufferPointerType)];
		} else if ([b conformsToProtocol:@protocol(MTLTextureBinding)]) {
			id<MTLTextureBinding> tb = (id<MTLTextureBinding>)b;
			[line appendFormat:@" texture=%ld data=%ld depth=%d array=%lu", (long)tb.textureType, (long)tb.textureDataType,
					   tb.depthTexture, (unsigned long)tb.arrayLength];
		} else if ([b conformsToProtocol:@protocol(MTLThreadgroupBinding)]) {
			id<MTLThreadgroupBinding> gb = (id<MTLThreadgroupBinding>)b;
			[line appendFormat:@" size=%lu align=%lu", (unsigned long)gb.threadgroupMemoryDataSize,
					   (unsigned long)gb.threadgroupMemoryAlignment];
		} else if ([b conformsToProtocol:@protocol(MTLObjectPayloadBinding)]) {
			id<MTLObjectPayloadBinding> pb = (id<MTLObjectPayloadBinding>)b;
			[line appendFormat:@" size=%lu align=%lu", (unsigned long)pb.objectPayloadDataSize, (unsigned long)pb.objectPayloadAlignment];
		}
		[lines addObject:line];
	}
	return [lines componentsJoinedByString:@"\n"];
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
/* The older reflection's arguments, one line each like `bindings_text`,
 * with whether each is active and its pointee in full. */
static NSString *arguments_text(NSArray<MTLArgument *> *arguments)
{
	NSMutableArray *lines = [NSMutableArray new];
	NSArray *sorted = [arguments sortedArrayUsingComparator:^NSComparisonResult(MTLArgument *a, MTLArgument *b) {
		if (a.type != b.type)
			return a.type < b.type ? NSOrderedAscending : NSOrderedDescending;
		return a.index < b.index ? NSOrderedAscending : a.index > b.index ? NSOrderedDescending : NSOrderedSame;
	}];
	for (MTLArgument *a in sorted) {
		NSMutableString *line = [NSMutableString stringWithFormat:@"%@ type=%ld index=%lu access=%ld active=%d", a.name, (long)a.type,
						(unsigned long)a.index, (long)a.access, a.active];
		if (a.type == MTLArgumentTypeBuffer) {
			[line appendFormat:@" size=%lu align=%lu data=%ld", (unsigned long)a.bufferDataSize, (unsigned long)a.bufferAlignment,
					   (long)a.bufferDataType];
			if (a.bufferPointerType)
				[line appendFormat:@" pointer=%@", type_text(a.bufferPointerType)];
		} else if (a.type == MTLArgumentTypeTexture) {
			[line appendFormat:@" texture=%ld data=%ld depth=%d array=%lu", (long)a.textureType, (long)a.textureDataType,
					   a.isDepthTexture, (unsigned long)a.arrayLength];
		} else if (a.type == MTLArgumentTypeThreadgroupMemory) {
			[line appendFormat:@" size=%lu align=%lu", (unsigned long)a.threadgroupMemoryDataSize,
					   (unsigned long)a.threadgroupMemoryAlignment];
		}
		[lines addObject:line];
	}
	return [lines componentsJoinedByString:@"\n"];
}

/* A render pipeline's reflection, stage by stage. */
static NSString *render_reflection_text(MTLRenderPipelineReflection *r, BOOL mesh, BOOL legacy, BOOL types)
{
	if (legacy)
		return [NSString stringWithFormat:@"vertex\n%@\nfragment\n%@\n", arguments_text(r.vertexArguments),
						  arguments_text(r.fragmentArguments)];
	if (mesh)
		return [NSString stringWithFormat:@"object\n%@\nmesh\n%@\nfragment\n%@\n", bindings_text(r.objectBindings, types),
						  bindings_text(r.meshBindings, types), bindings_text(r.fragmentBindings, types)];
	return [NSString stringWithFormat:@"vertex\n%@\nfragment\n%@\n", bindings_text(r.vertexBindings, types),
					  bindings_text(r.fragmentBindings, types)];
}
#pragma clang diagnostic pop

/* Words a render pipeline reports that its descriptor fixes, and its
 * limits, which the device chooses. */
static void store_render_properties(Case *c, NSDictionary *spec, id<MTLRenderPipelineState> p, BOOL mesh)
{
	NSDictionary *names = spec[@"pipeline_properties"];
	if (!names)
		return;
	NSMutableArray *fixed = [NSMutableArray arrayWithObject:@(p.supportIndirectCommandBuffers)];
	NSMutableArray *limits = [NSMutableArray new];
	if (mesh) {
		[fixed addObject:@(p.maxTotalThreadgroupsPerMeshGrid)];
		if (@available(macOS 26.0, *))
			[fixed addObjectsFromArray:@[ @(p.requiredThreadsPerObjectThreadgroup.width), @(p.requiredThreadsPerMeshThreadgroup.width) ]];
		[limits addObjectsFromArray:@[ @(p.objectThreadExecutionWidth), @(p.meshThreadExecutionWidth),
					       @(p.maxTotalThreadsPerObjectThreadgroup), @(p.maxTotalThreadsPerMeshThreadgroup) ]];
	}
	write_words(c, names[@"fixed"], fixed);
	write_words(c, names[@"limits"], limits);
}

static void wait_for(id<MTLCommandBuffer> cb, NSDictionary *spec)
{
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[cb addCompletedHandler:^(id<MTLCommandBuffer> _) {
		dispatch_semaphore_signal(done);
	}];
	[cb commit];
	double seconds = spec[@"timeout"] ? [spec[@"timeout"] doubleValue] : 20.0;
	if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))))
		@throw [NSException exceptionWithName:@"CaseTimeout" reason:@"command buffer did not complete" userInfo:nil];
	if (cb.error)
		@throw failure(@"command buffer: %@", cb.error.localizedDescription);
	if (timing && cb.GPUEndTime > cb.GPUStartTime)
		gpu_seconds += cb.GPUEndTime - cb.GPUStartTime;
}

static id<MTLFence> fence_named(Case *c, NSString *name)
{
	if (!c.fences[name])
		c.fences[name] = [c.device newFence];
	return c.fences[name];
}

/* Waits for a step's fences before its work and updates its fences after. */
static void wait_fences(Case *c, id<MTLCommandEncoder> e, NSDictionary *spec)
{
	for (NSString *name in spec[@"wait_fences"]) {
		if ([e conformsToProtocol:@protocol(MTLComputeCommandEncoder)])
			[(id<MTLComputeCommandEncoder>)e waitForFence:fence_named(c, name)];
		else if ([e conformsToProtocol:@protocol(MTLBlitCommandEncoder)])
			[(id<MTLBlitCommandEncoder>)e waitForFence:fence_named(c, name)];
		else
			[(id<MTLRenderCommandEncoder>)e waitForFence:fence_named(c, name) beforeStages:MTLRenderStageVertex];
	}
}

static void update_fences(Case *c, id<MTLCommandEncoder> e, NSDictionary *spec)
{
	for (NSString *name in spec[@"update_fences"]) {
		if ([e conformsToProtocol:@protocol(MTLComputeCommandEncoder)])
			[(id<MTLComputeCommandEncoder>)e updateFence:fence_named(c, name)];
		else if ([e conformsToProtocol:@protocol(MTLBlitCommandEncoder)])
			[(id<MTLBlitCommandEncoder>)e updateFence:fence_named(c, name)];
		else
			[(id<MTLRenderCommandEncoder>)e updateFence:fence_named(c, name) afterStages:MTLRenderStageFragment];
	}
}

/* Encodes a compute step into cb, or into `open` when the step continues
 * the previous compute encoder; returns the encoder, still open. */
/* Compute stage-in: attributes fetched by grid position. */
static MTLStageInputOutputDescriptor *stage_input_descriptor(NSDictionary *stage_input)
{
	MTLStageInputOutputDescriptor *si = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
	for (NSDictionary *a in stage_input[@"attributes"]) {
		MTLAttributeDescriptor *ad = si.attributes[number(a, @"index", 0)];
		ad.format = (MTLAttributeFormat)lookup(vertex_formats(), @"attribute format", a[@"format"]);
		ad.offset = number(a, @"offset", 0);
		ad.bufferIndex = number(a, @"buffer", 0);
	}
	NSDictionary *steps = @{@"grid_x" : @(MTLStepFunctionThreadPositionInGridX), @"grid_y" : @(MTLStepFunctionThreadPositionInGridY),
				@"grid_x_indexed" : @(MTLStepFunctionThreadPositionInGridXIndexed),
				@"grid_y_indexed" : @(MTLStepFunctionThreadPositionInGridYIndexed), @"constant" : @(MTLStepFunctionConstant)};
	/* The indexed steps fetch through one index buffer, by the thread's x
	 * or y position. */
	if (stage_input[@"index_buffer"]) {
		si.indexBufferIndex = number(stage_input, @"index_buffer", 0);
		si.indexType = [stage_input[@"index_type"] isEqualToString:@"uint16"] ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
	}
	for (NSDictionary *l in stage_input[@"layouts"]) {
		MTLBufferLayoutDescriptor *ld = si.layouts[number(l, @"buffer", 0)];
		ld.stride = [l[@"stride"] isEqual:@"dynamic"] ? MTLBufferLayoutStrideDynamic : number(l, @"stride", 16);
		ld.stepFunction = lookup(steps, @"step function", l[@"step"] ?: @"grid_x");
		/* A constant attribute has no rate. */
		ld.stepRate = [l[@"step"] isEqualToString:@"constant"] ? 0 : number(l, @"rate", 1);
	}
	return si;
}

/* Runs an indirect command buffer's commands: a range the plan names, or
 * one an indirect buffer holds. */
static void execute_icb(Case *c, id encoder, NSDictionary *x)
{
	id<MTLIndirectCommandBuffer> icb = c.icbs[x[@"icb"]];
	if (x[@"range_buffer"])
		[encoder executeCommandsInBuffer:icb indirectBuffer:c.buffers[x[@"range_buffer"]] indirectBufferOffset:0];
	else
		[encoder executeCommandsInBuffer:icb withRange:NSMakeRange(number(x, @"start", 0), number(x, @"count", 1))];
}

/* A visible function compiled to a binary on its own. */
static id<MTLFunction> binary_function(id<MTLLibrary> library, NSString *name)
{
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = name;
	fd.options = MTLFunctionOptionCompileToBinary;
	NSError *error = nil;
	id<MTLFunction> f = [library newFunctionWithDescriptor:fd error:&error];
	if (!f)
		@throw failure(@"binary function %@: %@", name, error.localizedDescription);
	return f;
}

/* The mutability a case declares for the buffers it binds to a stage:
 * an immutable buffer is a promise the stage never writes it. */
static void declare_mutability(NSDictionary *spec, BOOL compute, NSUInteger stage, MTLPipelineBufferDescriptorArray *buffers)
{
	for (NSDictionary *b in spec[@"buffers"])
		if (b[@"index"] && b[@"mutability"] && (stages_of(b, compute) & 1u << stage))
			buffers[number(b, @"index", 0)].mutability =
				lookup(@{@"default" : @(MTLMutabilityDefault), @"mutable" : @(MTLMutabilityMutable),
					 @"immutable" : @(MTLMutabilityImmutable)},
				       @"mutability", b[@"mutability"]);
}

static MTLSize size3(NSArray *a)
{
	return MTLSizeMake([a[0] unsignedIntegerValue], a.count > 1 ? [a[1] unsignedIntegerValue] : 1, a.count > 2 ? [a[2] unsignedIntegerValue] : 1);
}

/* Words a compute pipeline reports that its descriptor fixes, and its
 * limits, which the device chooses: written to the case's named outputs. */
static void store_compute_properties(Case *c, NSDictionary *spec, id<MTLComputePipelineState> p)
{
	NSDictionary *names = spec[@"pipeline_properties"];
	if (!names)
		return;
	NSMutableArray *fixed = [NSMutableArray arrayWithObjects:@(p.staticThreadgroupMemoryLength), @(p.supportIndirectCommandBuffers), nil];
	if (@available(macOS 26.0, *))
		[fixed addObjectsFromArray:@[ @(p.requiredThreadsPerThreadgroup.width), @(p.requiredThreadsPerThreadgroup.height),
					      @(p.requiredThreadsPerThreadgroup.depth) ]];
	/* The imageblock memory of each width and height. */
	for (NSArray *size in spec[@"imageblock_lengths"])
		[fixed addObject:@([p imageblockMemoryLengthForDimensions:MTLSizeMake([size[0] unsignedIntegerValue],
										      [size[1] unsignedIntegerValue], 1)])];
	NSMutableArray *limits = [NSMutableArray arrayWithObjects:@(p.threadExecutionWidth), @(p.maxTotalThreadsPerThreadgroup), nil];
	if (@available(macOS 27.0, *)) {
		/* Automatic usage leaves the pipeline's to the device. */
		if (spec[@"forward_progress"] && ![spec[@"forward_progress"] isEqual:@"automatic"])
			[fixed addObject:@(p.forwardProgressUsage)];
		/* How many threadgroups a persistent kernel should launch. */
		if ([spec[@"persistent"] boolValue])
			[limits addObject:@([p recommendedPersistentThreadgroupsPerGridForThreadsPerThreadgroup:MTLSizeMake(p.threadExecutionWidth, 1, 1)])];
	}
	write_words(c, names[@"fixed"], fixed);
	write_words(c, names[@"limits"], limits);
}

static id<MTLComputeCommandEncoder> encode_compute(Case *c, id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> open,
						    id<MTLLibrary> library, NSDictionary *spec)
{
	id<MTLFunction> function = function_named(library, spec[@"function"], spec[@"constants"]);
	MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
	d.computeFunction = function;
	d.supportIndirectCommandBuffers = [spec[@"supports_icb"] boolValue];
	if (spec[@"preload"])
		d.preloadedLibraries = preloaded(c, spec[@"preload"]);
	/* The older name for preloaded libraries. */
	if (spec[@"insert"]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		d.insertLibraries = preloaded(c, spec[@"insert"]);
#pragma clang diagnostic pop
	}
	/* Promises and hints that leave what the kernel computes alone. */
	d.threadGroupSizeIsMultipleOfThreadExecutionWidth = [spec[@"group_multiple_of_width"] boolValue];
	if (spec[@"call_stack_depth"])
		d.maxCallStackDepth = number(spec, @"call_stack_depth", 1);
	d.supportAddingBinaryFunctions = spec[@"added_binary_functions"] != nil;
	declare_mutability(spec, YES, StageCompute, d.buffers);
	if (spec[@"required_threads"]) {
		if (@available(macOS 26.0, *))
			d.requiredThreadsPerThreadgroup = size3(spec[@"required_threads"]);
		else
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"required threadgroup sizes need macOS 26" userInfo:nil];
	}
	if (spec[@"forward_progress"] || spec[@"contention_relief"] || spec[@"persistent"]) {
		if (@available(macOS 27.0, *)) {
			d.forwardProgressUsage = lookup(@{@"automatic" : @(MTLForwardProgressUsageAutomatic), @"weak" : @(MTLForwardProgressUsageWeak),
							  @"simdgroup_parallel" : @(MTLForwardProgressUsageSIMDGroupParallel)},
							@"forward progress", spec[@"forward_progress"] ?: @"automatic");
			d.contentionRelief = lookup(@{@"automatic" : @(MTLContentionReliefAutomatic), @"none" : @(MTLContentionReliefNone)},
						    @"contention relief", spec[@"contention_relief"] ?: @"automatic");
			d.optimizeForPersistentKernel = [spec[@"persistent"] boolValue];
		} else {
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"forward progress hints need macOS 27" userInfo:nil];
		}
	}
	if (spec[@"stage_input"])
		d.stageInputDescriptor = stage_input_descriptor(spec[@"stage_input"]);
	NSMutableArray *linked = [NSMutableArray new];
	for (id entry in spec[@"visible_functions"])
		[linked addObject:linked_function(c, library, entry)];
	/* Visible functions compiled to binaries on their own, then linked. */
	NSMutableArray *binaries = [NSMutableArray new];
	for (NSString *name in spec[@"binary_functions"]) {
		id<MTLFunction> f = binary_function(library, name);
		[binaries addObject:f];
		[linked addObject:f];
	}
	NSMutableDictionary *stitched = [NSMutableDictionary new];
	for (NSDictionary *g in spec[@"stitched"]) {
		id<MTLFunction> f = stitched_function(c, library, g);
		stitched[g[@"name"]] = f;
		[linked addObject:f];
	}
	MTLLinkedFunctions *l = [MTLLinkedFunctions new];
	NSMutableArray *source = [linked mutableCopy];
	[source removeObjectsInArray:binaries];
	l.functions = source;
	l.binaryFunctions = binaries;
	link_extras(c, library, spec, @"", l);
	if (linked.count || l.privateFunctions.count || l.groups)
		d.linkedFunctions = l;
	__block NSError *error = nil;
	__block MTLComputePipelineReflection *reflection = nil;
	BOOL legacy_reflection = [spec[@"reflection_arguments"] boolValue];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	MTLPipelineOption options = !spec[@"reflection"] ? MTLPipelineOptionNone
			  : legacy_reflection ? MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo
					      : MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
#pragma clang diagnostic pop
	/* From the descriptor, or from the function alone when that is all
	 * the case sets; at once, or through a completion handler. */
	__block id<MTLComputePipelineState> pipeline = nil;
	NSString *made = spec[@"pipeline_creation"] ?: @"descriptor";
	void (^take)(id<MTLComputePipelineState>, MTLComputePipelineReflection *, NSError *, dispatch_semaphore_t) =
		^(id<MTLComputePipelineState> p, MTLComputePipelineReflection *r, NSError *e, dispatch_semaphore_t done) {
			pipeline = p;
			reflection = r;
			error = e;
			dispatch_semaphore_signal(done);
		};
	if ([made isEqualToString:@"descriptor"]) {
		pipeline = [c.device newComputePipelineStateWithDescriptor:d options:options reflection:spec[@"reflection"] ? &reflection : nil
								     error:&error];
	} else if ([made isEqualToString:@"descriptor_async"]) {
		await_handler(^(dispatch_semaphore_t done) {
			[c.device newComputePipelineStateWithDescriptor:d options:options
						     completionHandler:^(id<MTLComputePipelineState> p, MTLComputePipelineReflection *r, NSError *e) {
							     take(p, r, e, done);
						     }];
		});
	} else if ([made isEqualToString:@"function"]) {
		pipeline = [c.device newComputePipelineStateWithFunction:function error:&error];
	} else if ([made isEqualToString:@"function_options"]) {
		pipeline = [c.device newComputePipelineStateWithFunction:function options:options
							      reflection:spec[@"reflection"] ? &reflection : nil error:&error];
	} else if ([made isEqualToString:@"function_async"]) {
		await_handler(^(dispatch_semaphore_t done) {
			[c.device newComputePipelineStateWithFunction:function
						    completionHandler:^(id<MTLComputePipelineState> p, NSError *e) {
							    take(p, nil, e, done);
						    }];
		});
	} else if ([made isEqualToString:@"function_options_async"]) {
		await_handler(^(dispatch_semaphore_t done) {
			[c.device newComputePipelineStateWithFunction:function options:options
						    completionHandler:^(id<MTLComputePipelineState> p, MTLComputePipelineReflection *r, NSError *e) {
							    take(p, r, e, done);
						    }];
		});
	} else {
		@throw failure(@"pipeline creation %@", made);
	}
	if (!pipeline)
		@throw failure(@"compute pipeline: %@", error.localizedDescription);
	/* Binary functions added to the finished pipeline, whose new state
	 * the dispatch then uses. */
	if (spec[@"added_binary_functions"]) {
		NSMutableArray *added = [NSMutableArray new];
		for (NSString *name in spec[@"added_binary_functions"]) {
			id<MTLFunction> f = binary_function(library, name);
			[added addObject:f];
			[linked addObject:f];
		}
		pipeline = [pipeline newComputePipelineStateWithAdditionalBinaryFunctions:added error:&error];
		if (!pipeline)
			@throw failure(@"pipeline with added functions: %@", error.localizedDescription);
	}
	store_compute_properties(c, spec, pipeline);
	if (spec[@"reflection"]) {
		BOOL types = [spec[@"reflection_types"] boolValue];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		NSString *listed = legacy_reflection ? arguments_text(reflection.arguments) : bindings_text(reflection.bindings, types);
#pragma clang diagnostic pop
		NSMutableString *text = [NSMutableString stringWithFormat:@"%@\nstatic_threadgroup_memory=%lu\n", listed,
									  (unsigned long)pipeline.staticThreadgroupMemoryLength];
		/* The reflection a pipeline keeps, read back from the state. */
		if (types) {
			if (@available(macOS 26.0, *))
				[text appendFormat:@"state=%@\n", pipeline.reflection ? bindings_text(pipeline.reflection.bindings, YES) : @"none"];
		}
		write_text(c, spec[@"reflection"], text);
		/* Reflection alone: the function's resources stay unbound, which
		 * no dispatch may leave them. */
		if ([spec[@"reflection_only"] boolValue])
			return open ?: [cb computeCommandEncoder];
	}
	id<MTLComputeCommandEncoder> e = open;
	/* A barrier between dispatches of one encoder, scoped as named. */
	if (e && spec[@"barrier"]) {
		NSString *b = spec[@"barrier"];
		MTLBarrierScope scope = ([b containsString:@"buffers"] ? MTLBarrierScopeBuffers : 0) |
					([b containsString:@"textures"] ? MTLBarrierScopeTextures : 0);
		[e memoryBarrierWithScope:scope];
	}
	/* Concurrent dispatches, asked of the command buffer or of a pass
	 * descriptor; the encoder reports the type it was given. */
	MTLDispatchType dispatch_type = spec[@"concurrent"] ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
	if (!e && [spec[@"concurrent"] isEqual:@"pass"]) {
		MTLComputePassDescriptor *cp = [MTLComputePassDescriptor computePassDescriptor];
		cp.dispatchType = MTLDispatchTypeConcurrent;
		e = [cb computeCommandEncoderWithDescriptor:cp];
	} else if (!e && [spec[@"concurrent"] boolValue]) {
		e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	}
	if (!e && spec[@"timestamps"]) {
		/* Samples at the start and end of the encoder's work. */
		MTLCounterSampleBufferDescriptor *sd = [MTLCounterSampleBufferDescriptor new];
		sd.counterSet = counter_set(c.device, spec[@"timestamps"][@"counters"] ?: @"timestamp");
		sd.sampleCount = 2;
		sd.storageMode = MTLStorageModeShared;
		id<MTLCounterSampleBuffer> samples = [c.device newCounterSampleBufferWithDescriptor:sd error:&error];
		if (!samples)
			@throw failure(@"counter sample buffer: %@", error.localizedDescription);
		MTLComputePassDescriptor *cp = [MTLComputePassDescriptor computePassDescriptor];
		cp.sampleBufferAttachments[0].sampleBuffer = samples;
		cp.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
		cp.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
		resolve_counters_after(c, cb, samples, 2, spec[@"timestamps"]);
		e = [cb computeCommandEncoderWithDescriptor:cp];
	}
	e = e ?: [cb computeCommandEncoder];
	if (!open && e.dispatchType != dispatch_type)
		@throw failure(@"encoder dispatch type %ld, asked for %ld", (long)e.dispatchType, (long)dispatch_type);
	if (!open)
		wait_fences(c, e, spec);
	[e setComputePipelineState:pipeline];
	bind_resources(c, e, spec, YES, 0);
	BOOL ranges = [spec[@"bind"] isEqual:@"ranges"];
	NSMutableDictionary *tables = [NSMutableDictionary new], *encoded = [NSMutableDictionary new];
	for (NSDictionary *table in spec[@"function_tables"]) {
		MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
		NSArray *names = table[@"functions"];
		td.functionCount = names.count;
		id<MTLVisibleFunctionTable> vft = [pipeline newVisibleFunctionTableWithDescriptor:td];
		if (!vft)
			@throw failure(@"visible function table creation failed");
		NSMutableArray *handles = [NSMutableArray new];
		for (NSUInteger i = 0; i < names.count; i++) {
			id<MTLFunction> f = stitched[names[i]] ?: nil;
			if (!f) {
				for (id<MTLFunction> l in linked)
					if ([l.name isEqualToString:names[i]]) f = l;
			}
			id<MTLFunctionHandle> handle = nil;
			/* By name, the lookup that needs no function object. */
			if ([spec[@"handles_by_name"] boolValue]) {
				if (@available(macOS 26.0, *))
					handle = [pipeline functionHandleWithName:names[i]];
				else
					@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"handles by name need macOS 26" userInfo:nil];
			} else {
				handle = [pipeline functionHandleWithFunction:f];
			}
			if (!handle)
				@throw failure(@"no function handle for %@", names[i]);
			[handles addObject:handle];
		}
		fill_table(vft, handles, ranges);
		if (table[@"id"])
			encoded[@(number(table, @"id", 0))] = vft;
		else
			tables[@(number(table, @"index", 0))] = vft;
	}
	bind_tables(e, StageCompute, tables, ranges);
	encode_tables(c, e, function, spec[@"table_arguments"], encoded, ranges);
	NSArray *grid = spec[@"grid"] ?: @[ @1, @1, @1 ], *group = spec[@"threadgroup"] ?: @[ @64, @1, @1 ];
	MTLSize g = MTLSizeMake([grid[0] unsignedIntegerValue], [grid[1] unsignedIntegerValue], [grid[2] unsignedIntegerValue]);
	MTLSize t = MTLSizeMake([group[0] unsignedIntegerValue], [group[1] unsignedIntegerValue], [group[2] unsignedIntegerValue]);
	if (spec[@"execute_icb"]) {
		NSDictionary *x = spec[@"execute_icb"];
		for (id r in c.resident)
			if ([r conformsToProtocol:@protocol(MTLResource)])
				[e useResource:r usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		for (NSString *name in spec[@"use"])
			[e useResource:c.buffers[name] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		execute_icb(c, e, x);
	} else if (spec[@"indirect"]) {
		NSDictionary *i = spec[@"indirect"];
		if (spec[@"stage_input"])
			[e setStageInRegion:MTLRegionMake3D(0, 0, 0, g.width, g.height, g.depth)];
		[e dispatchThreadgroupsWithIndirectBuffer:c.buffers[i[@"buffer"]] indirectBufferOffset:number(i, @"offset", 0)
				    threadsPerThreadgroup:t];
	} else if ([spec[@"dispatch"] isEqualToString:@"threads"]) {
		if (spec[@"stage_in_region_buffer"])
			[e setStageInRegionWithIndirectBuffer:c.buffers[spec[@"stage_in_region_buffer"]] indirectBufferOffset:0];
		else if (spec[@"stage_input"])
			[e setStageInRegion:MTLRegionMake3D(0, 0, 0, g.width, g.height, g.depth)];
		[e dispatchThreads:g threadsPerThreadgroup:t];
	} else {
		[e dispatchThreadgroups:MTLSizeMake((g.width + t.width - 1) / t.width, (g.height + t.height - 1) / t.height,
						    (g.depth + t.depth - 1) / t.depth)
		  threadsPerThreadgroup:t];
	}
	return e;
}

static void run_compute(Case *c, id<MTLCommandQueue> queue, id<MTLLibrary> library, NSDictionary *spec)
{
	NSMutableArray *lines = [NSMutableArray new];
	id<MTLCommandBuffer> cb = command_buffer(c, queue, spec, lines);
	if (spec[@"residency_set"]) {
		/* The named buffers resident through a residency set alone. */
		if (@available(macOS 15.0, *)) {
			NSError *error = nil;
			id<MTLResidencySet> set = [c.device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
			if (!set)
				@throw failure(@"residency set: %@", error.localizedDescription);
			for (NSString *name in spec[@"residency_set"])
				[set addAllocation:c.buffers[name]];
			[set commit];
			[set requestResidency];
			[cb useResidencySet:set];
		} else {
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"residency sets need macOS 15" userInfo:nil];
		}
	}
	id<MTLComputeCommandEncoder> e = encode_compute(c, cb, nil, library, spec);
	update_fences(c, e, spec);
	[e endEncoding];
	wait_for(cb, spec);
	if (spec[@"log"])
		store_log(c, spec, lines);
}

static void encode_render(Case *c, id<MTLCommandBuffer> cb, id<MTLLibrary> library, NSDictionary *spec)
{
	NSDictionary *r = spec[@"render"];
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	NSDictionary *mesh = r[@"mesh"];
	if (!mesh)
		d.vertexFunction = function_named(library, spec[@"vertex"], spec[@"constants"]);
	d.maxVertexAmplificationCount = number(r[@"amplification"], @"count", 1);
	if (r[@"preload"]) {
		d.vertexPreloadedLibraries = preloaded(c, r[@"preload"]);
		d.fragmentPreloadedLibraries = d.vertexPreloadedLibraries;
	}
	NSMutableArray *fragment_linked = [NSMutableArray new], *vertex_linked = [NSMutableArray new];
	for (id entry in r[@"fragment_visible_functions"])
		[fragment_linked addObject:linked_function(c, library, entry)];
	for (id entry in r[@"vertex_visible_functions"])
		[vertex_linked addObject:linked_function(c, library, entry)];
	d.fragmentLinkedFunctions = stage_links(c, library, r, @"fragment_", fragment_linked);
	d.vertexLinkedFunctions = stage_links(c, library, r, @"vertex_", vertex_linked);
	if (spec[@"fragment"])
		d.fragmentFunction = function_named(library, spec[@"fragment"], spec[@"constants"]);
	d.rasterSampleCount = number(r, @"samples", 1);
	d.alphaToCoverageEnabled = [r[@"alpha_to_coverage"] boolValue];
	d.alphaToOneEnabled = [r[@"alpha_to_one"] boolValue];
	d.supportIndirectCommandBuffers = [r[@"supports_icb"] boolValue];
	NSDictionary *tess = r[@"tessellation"];
	if (tess) {
		NSDictionary *modes = @{@"pow2" : @(MTLTessellationPartitionModePow2), @"integer" : @(MTLTessellationPartitionModeInteger),
					@"fractional_odd" : @(MTLTessellationPartitionModeFractionalOdd),
					@"fractional_even" : @(MTLTessellationPartitionModeFractionalEven)};
		d.tessellationPartitionMode = lookup(modes, @"partition mode", tess[@"partition"] ?: @"integer");
		d.maxTessellationFactor = number(tess, @"max_factor", 16);
		/* Instances read their own factors when the draw gives them a
		 * stride, unless the case names the step. */
		NSDictionary *factor_steps = @{@"constant" : @(MTLTessellationFactorStepFunctionConstant),
					       @"per_patch" : @(MTLTessellationFactorStepFunctionPerPatch),
					       @"per_instance" : @(MTLTessellationFactorStepFunctionPerInstance),
					       @"per_patch_and_per_instance" : @(MTLTessellationFactorStepFunctionPerPatchAndPerInstance)};
		d.tessellationFactorStepFunction =
			lookup(factor_steps, @"factor step function",
			       tess[@"factor_step"] ?: (tess[@"instance_stride"] ? @"per_patch_and_per_instance" : @"per_patch"));
		d.tessellationOutputWindingOrder = [tess[@"winding"] isEqualToString:@"cw"] ? MTLWindingClockwise : MTLWindingCounterClockwise;
		d.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
		/* The encoder's factor scale applies only where the pipeline allows it. */
		d.tessellationFactorScaleEnabled = tess[@"factor_scale"] != nil;
		/* Control points fetched through an index buffer of this type. */
		if (tess[@"control_point_index"])
			d.tessellationControlPointIndexType = [tess[@"control_point_index_type"] isEqualToString:@"uint16"]
								      ? MTLTessellationControlPointIndexTypeUInt16
								      : MTLTessellationControlPointIndexTypeUInt32;
	}
	if (r[@"rasterize"] && ![r[@"rasterize"] boolValue])
		d.rasterizationEnabled = NO;
	if (r[@"input_primitive"]) {
		NSDictionary *topologies = @{@"point" : @(MTLPrimitiveTopologyClassPoint), @"line" : @(MTLPrimitiveTopologyClassLine),
					     @"triangle" : @(MTLPrimitiveTopologyClassTriangle)};
		d.inputPrimitiveTopology = lookup(topologies, @"primitive class", r[@"input_primitive"]);
	}
	if (r[@"vertex_descriptor"]) {
		MTLVertexDescriptor *v = [MTLVertexDescriptor vertexDescriptor];
		for (NSDictionary *a in r[@"vertex_descriptor"][@"attributes"]) {
			MTLVertexAttributeDescriptor *ad = v.attributes[number(a, @"index", 0)];
			ad.format = lookup(vertex_formats(), @"vertex format", a[@"format"]);
			ad.offset = number(a, @"offset", 0);
			ad.bufferIndex = number(a, @"buffer", 0);
		}
		for (NSDictionary *l in r[@"vertex_descriptor"][@"layouts"]) {
			MTLVertexBufferLayoutDescriptor *ld = v.layouts[number(l, @"buffer", 0)];
			ld.stride = [l[@"stride"] isEqual:@"dynamic"] ? MTLBufferLayoutStrideDynamic : number(l, @"stride", 16);
			NSString *step = l[@"step"] ?: @"vertex";
			ld.stepFunction = [step isEqualToString:@"instance"] ? MTLVertexStepFunctionPerInstance :
					  [step isEqualToString:@"constant"] ? MTLVertexStepFunctionConstant :
					  [step isEqualToString:@"patch"] ? MTLVertexStepFunctionPerPatch :
					  [step isEqualToString:@"patch_control_point"] ? MTLVertexStepFunctionPerPatchControlPoint :
									       MTLVertexStepFunctionPerVertex;
			ld.stepRate = [step isEqualToString:@"constant"] ? 0 : number(l, @"rate", 1);
		}
		d.vertexDescriptor = v;
	}
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	if (r[@"sample_positions"]) {
		NSArray *list = r[@"sample_positions"];
		MTLSamplePosition positions[16];
		NSUInteger count = MIN(list.count, (NSUInteger)16);
		for (NSUInteger k = 0; k < count; k++)
			positions[k] = MTLSamplePositionMake([list[k][0] floatValue], [list[k][1] floatValue]);
		[pass setSamplePositions:positions count:count];
		/* The pass hands back how many positions it holds, and then
		 * the positions. */
		MTLSamplePosition held[16];
		if ([pass getSamplePositions:NULL count:0] != count || [pass getSamplePositions:held count:count] != count ||
		    memcmp(held, positions, count * sizeof *held))
			@throw failure(@"the pass holds other sample positions than it was given");
	}
	NSArray *colors = r[@"colors"] ?: @[];
	if (r[@"visibility"]) {
		pass.visibilityResultBuffer = c.buffers[r[@"visibility"][@"buffer"]];
		/* Whether the pass starts from zeroed results or adds to what
		 * the buffer holds. */
		if (r[@"visibility"][@"type"]) {
			if (@available(macOS 26.0, *))
				pass.visibilityResultType = lookup(@{@"reset" : @(MTLVisibilityResultTypeReset), @"accumulate" : @(MTLVisibilityResultTypeAccumulate)},
								   @"visibility result type", r[@"visibility"][@"type"]);
			else
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"visibility result types need macOS 26" userInfo:nil];
		}
	}
	if (r[@"timestamps"]) {
		/* Timestamps at the start and end of the vertex and fragment
		 * stages, resolved after the pass into the named buffer. */
		MTLCounterSampleBufferDescriptor *sd = [MTLCounterSampleBufferDescriptor new];
		sd.counterSet = counter_set(c.device, r[@"timestamps"][@"counters"] ?: @"timestamp");
		sd.sampleCount = 4;
		sd.storageMode = MTLStorageModeShared;
		NSError *error = nil;
		id<MTLCounterSampleBuffer> samples = [c.device newCounterSampleBufferWithDescriptor:sd error:&error];
		if (!samples)
			@throw failure(@"counter sample buffer: %@", error.localizedDescription);
		MTLRenderPassSampleBufferAttachmentDescriptor *sa = pass.sampleBufferAttachments[0];
		sa.sampleBuffer = samples;
		sa.startOfVertexSampleIndex = 0;
		sa.endOfVertexSampleIndex = 1;
		sa.startOfFragmentSampleIndex = 2;
		sa.endOfFragmentSampleIndex = 3;
		resolve_counters_after(c, cb, samples, 4, r[@"timestamps"]);
	}
	if (r[@"tile_size"]) {
		pass.tileWidth = [r[@"tile_size"][0] unsignedIntegerValue];
		pass.tileHeight = [r[@"tile_size"][1] unsignedIntegerValue];
	}
	pass.threadgroupMemoryLength = number(r, @"tile_memory", 0);
	pass.imageblockSampleLength = number(r, @"imageblock_sample_length", 0);
	if (r[@"target_size"]) {
		pass.renderTargetWidth = [r[@"target_size"][0] unsignedIntegerValue];
		pass.renderTargetHeight = [r[@"target_size"][1] unsignedIntegerValue];
		pass.defaultRasterSampleCount = number(r, @"samples", 1);
	}
	for (NSUInteger i = 0; i < colors.count; i++) {
		NSDictionary *a = colors[i];
		id<MTLTexture> t = c.textures[a[@"texture"]];
		if (!t) @throw failure(@"color attachment %lu names no texture", (unsigned long)i);
		MTLRenderPipelineColorAttachmentDescriptor *pa = d.colorAttachments[i];
		pa.pixelFormat = t.pixelFormat;
		NSDictionary *blend = a[@"blend"];
		if (blend) {
			pa.blendingEnabled = YES;
			pa.rgbBlendOperation = lookup(blend_operations(), @"blend operation", blend[@"rgb"] ?: @"add");
			pa.alphaBlendOperation = lookup(blend_operations(), @"blend operation", blend[@"alpha"] ?: @"add");
			pa.sourceRGBBlendFactor = lookup(blend_factors(), @"blend factor", blend[@"src_rgb"] ?: @"one");
			pa.destinationRGBBlendFactor = lookup(blend_factors(), @"blend factor", blend[@"dst_rgb"] ?: @"zero");
			pa.sourceAlphaBlendFactor = lookup(blend_factors(), @"blend factor", blend[@"src_alpha"] ?: @"one");
			pa.destinationAlphaBlendFactor = lookup(blend_factors(), @"blend factor", blend[@"dst_alpha"] ?: @"zero");
		}
		if (a[@"write_mask"]) {
			MTLColorWriteMask mask = 0;
			NSString *m = a[@"write_mask"];
			if ([m containsString:@"r"]) mask |= MTLColorWriteMaskRed;
			if ([m containsString:@"g"]) mask |= MTLColorWriteMaskGreen;
			if ([m containsString:@"b"]) mask |= MTLColorWriteMaskBlue;
			if ([m containsString:@"a"]) mask |= MTLColorWriteMaskAlpha;
			pa.writeMask = mask;
		}
		MTLRenderPassColorAttachmentDescriptor *ca = pass.colorAttachments[i];
		ca.texture = t;
		/* One level, slice or 3D plane of the texture. */
		ca.level = number(a, @"level", 0);
		ca.slice = number(a, @"slice", 0);
		ca.depthPlane = number(a, @"depth_plane", 0);
		ca.resolveLevel = number(a, @"resolve_level", 0);
		ca.resolveSlice = number(a, @"resolve_slice", 0);
		/* The plane of a 3D resolve texture the samples resolve into. */
		ca.resolveDepthPlane = number(a, @"resolve_depth_plane", 0);
		/* An array target is layered, the vertex function picking the
		 * slice, unless the attachment names one. */
		if ((t.textureType == MTLTextureType2DArray || t.textureType == MTLTextureType2DMultisampleArray) && !a[@"slice"])
			pass.renderTargetArrayLength = t.arrayLength;
		NSArray *clear = a[@"clear"];
		ca.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
		if (clear)
			ca.clearColor = MTLClearColorMake([clear[0] doubleValue], [clear[1] doubleValue], [clear[2] doubleValue], [clear[3] doubleValue]);
		ca.storeAction = MTLStoreActionStore;
		if (a[@"resolve"]) {
			ca.resolveTexture = c.textures[a[@"resolve"]];
			ca.storeAction = [a[@"store"] isEqualToString:@"resolve"] ? MTLStoreActionMultisampleResolve
										   : MTLStoreActionStoreAndMultisampleResolve;
		}
	}
	MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
	if (r[@"depth"]) {
		NSDictionary *a = r[@"depth"];
		id<MTLTexture> t = c.textures[a[@"texture"]];
		d.depthAttachmentPixelFormat = t.pixelFormat;
		pass.depthAttachment.texture = t;
		pass.depthAttachment.loadAction = a[@"clear"] ? MTLLoadActionClear : MTLLoadActionLoad;
		pass.depthAttachment.clearDepth = [a[@"clear"] doubleValue];
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		/* Depth kept for later passes that use the same custom sample
		 * positions. */
		if ([a[@"custom_sample_positions"] boolValue])
			pass.depthAttachment.storeActionOptions = MTLStoreActionOptionCustomSamplePositions;
		if (a[@"resolve"]) {
			NSDictionary *filters = @{@"sample0" : @(MTLMultisampleDepthResolveFilterSample0),
						  @"min" : @(MTLMultisampleDepthResolveFilterMin), @"max" : @(MTLMultisampleDepthResolveFilterMax)};
			pass.depthAttachment.resolveTexture = c.textures[a[@"resolve"]];
			pass.depthAttachment.depthResolveFilter = lookup(filters, @"depth resolve filter", a[@"resolve_filter"] ?: @"sample0");
			pass.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
		}
		ds.depthCompareFunction = lookup(compare_functions(), @"compare function", a[@"compare"] ?: @"always");
		ds.depthWriteEnabled = a[@"write"] ? [a[@"write"] boolValue] : YES;
	}
	if (r[@"stencil"]) {
		NSDictionary *a = r[@"stencil"];
		id<MTLTexture> t = c.textures[a[@"texture"]];
		d.stencilAttachmentPixelFormat = t.pixelFormat;
		pass.stencilAttachment.texture = t;
		pass.stencilAttachment.loadAction = a[@"clear"] ? MTLLoadActionClear : MTLLoadActionLoad;
		pass.stencilAttachment.clearStencil = (uint32_t)number(a, @"clear", 0);
		pass.stencilAttachment.storeAction = MTLStoreActionStore;
		if (a[@"resolve"]) {
			NSDictionary *filters = @{@"sample0" : @(MTLMultisampleStencilResolveFilterSample0),
						  @"depth_resolved" : @(MTLMultisampleStencilResolveFilterDepthResolvedSample)};
			pass.stencilAttachment.resolveTexture = c.textures[a[@"resolve"]];
			pass.stencilAttachment.stencilResolveFilter = lookup(filters, @"stencil resolve filter", a[@"resolve_filter"] ?: @"sample0");
			pass.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
		}
		for (NSString *face in @[ @"front", @"back" ]) {
			MTLStencilDescriptor *s = [MTLStencilDescriptor new];
			NSDictionary *f = a[face] ?: a;
			s.stencilCompareFunction = lookup(compare_functions(), @"compare function", f[@"compare"] ?: @"always");
			s.stencilFailureOperation = lookup(stencil_operations(), @"stencil operation", f[@"fail"] ?: @"keep");
			s.depthFailureOperation = lookup(stencil_operations(), @"stencil operation", f[@"depth_fail"] ?: @"keep");
			s.depthStencilPassOperation = lookup(stencil_operations(), @"stencil operation", f[@"pass"] ?: @"keep");
			s.readMask = (uint32_t)number(f, @"read_mask", 0xff);
			s.writeMask = (uint32_t)number(f, @"write_mask", 0xff);
			if ([face isEqualToString:@"front"]) ds.frontFaceStencil = s;
			else ds.backFaceStencil = s;
		}
	}
	/* Promises and limits that leave what the stages compute alone. */
	if (r[@"call_stack_depth"]) {
		d.maxVertexCallStackDepth = number(r, @"call_stack_depth", 1);
		d.maxFragmentCallStackDepth = number(r, @"call_stack_depth", 1);
	}
	d.supportAddingVertexBinaryFunctions = r[@"added_vertex_binary_functions"] != nil;
	d.supportAddingFragmentBinaryFunctions = r[@"added_fragment_binary_functions"] != nil;
	declare_mutability(spec, NO, StageVertex, d.vertexBuffers);
	declare_mutability(spec, NO, StageFragment, d.fragmentBuffers);
	__block NSError *error = nil;
	__block id<MTLRenderPipelineState> pipeline = nil;
	/* Reflection, when the case records it: bindings, and the older
	 * arguments when it asks for those. */
	BOOL legacy_reflection = [spec[@"reflection_arguments"] boolValue];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	MTLPipelineOption options = !spec[@"reflection"] ? MTLPipelineOptionNone
			  : legacy_reflection ? MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo
					      : MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
#pragma clang diagnostic pop
	__block MTLRenderPipelineReflection *reflection = nil;
	/* At once, or through a completion handler with or without options. */
	NSString *made = spec[@"pipeline_creation"] ?: @"descriptor";
	void (^take)(id<MTLRenderPipelineState>, MTLRenderPipelineReflection *, NSError *, dispatch_semaphore_t) =
		^(id<MTLRenderPipelineState> p, MTLRenderPipelineReflection *rr, NSError *e, dispatch_semaphore_t done) {
			pipeline = p;
			reflection = rr;
			error = e;
			dispatch_semaphore_signal(done);
		};
	if (mesh) {
		/* Object (optional) and mesh functions in place of a vertex
		 * function, with the same attachments and fragment function. */
		MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
		if (mesh[@"object"])
			m.objectFunction = function_named(library, mesh[@"object"], spec[@"constants"]);
		m.meshFunction = function_named(library, mesh[@"mesh"], spec[@"constants"]);
		m.fragmentFunction = d.fragmentFunction;
		m.rasterSampleCount = d.rasterSampleCount;
		m.alphaToCoverageEnabled = d.alphaToCoverageEnabled;
		for (NSUInteger i = 0; i < colors.count; i++)
			m.colorAttachments[i] = d.colorAttachments[i];
		m.depthAttachmentPixelFormat = d.depthAttachmentPixelFormat;
		m.stencilAttachmentPixelFormat = d.stencilAttachmentPixelFormat;
		m.maxTotalThreadsPerObjectThreadgroup = number(mesh, @"max_object_threads", 0);
		m.maxTotalThreadsPerMeshThreadgroup = number(mesh, @"max_mesh_threads", 0);
		if (@available(macOS 14.0, *))
			m.supportIndirectCommandBuffers = d.supportIndirectCommandBuffers;
		m.objectLinkedFunctions = stage_links(c, library, mesh, @"object_", @[]);
		m.meshLinkedFunctions = stage_links(c, library, mesh, @"mesh_", @[]);
		m.fragmentLinkedFunctions = d.fragmentLinkedFunctions;
		declare_mutability(spec, NO, StageObject, m.objectBuffers);
		declare_mutability(spec, NO, StageMesh, m.meshBuffers);
		declare_mutability(spec, NO, StageFragment, m.fragmentBuffers);
		/* The payload's size and the grid's largest threadgroup count
		 * when the case states them, and threadgroup sizes promised
		 * to be multiples of the execution width. */
		if (mesh[@"payload_length"])
			m.payloadMemoryLength = number(mesh, @"payload_length", 0);
		if (mesh[@"max_threadgroups"])
			m.maxTotalThreadgroupsPerMeshGrid = number(mesh, @"max_threadgroups", 0);
		m.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth = [mesh[@"group_multiple_of_width"] boolValue];
		m.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth = [mesh[@"group_multiple_of_width"] boolValue];
		if (mesh[@"required_object_threads"] || mesh[@"required_mesh_threads"]) {
			if (@available(macOS 26.0, *)) {
				if (mesh[@"required_object_threads"])
					m.requiredThreadsPerObjectThreadgroup = size3(mesh[@"required_object_threads"]);
				if (mesh[@"required_mesh_threads"])
					m.requiredThreadsPerMeshThreadgroup = size3(mesh[@"required_mesh_threads"]);
			} else {
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"required threadgroup sizes need macOS 26" userInfo:nil];
			}
		}
		if ([made isEqualToString:@"async"])
			await_handler(^(dispatch_semaphore_t done) {
				[c.device newRenderPipelineStateWithMeshDescriptor:m options:options
							      completionHandler:^(id<MTLRenderPipelineState> p, MTLRenderPipelineReflection *rr, NSError *e) {
								      take(p, rr, e, done);
							      }];
			});
		else
			pipeline = [c.device newRenderPipelineStateWithMeshDescriptor:m options:options
									   reflection:spec[@"reflection"] ? &reflection : nil error:&error];
	} else if ([made isEqualToString:@"async"] && !spec[@"reflection"]) {
		await_handler(^(dispatch_semaphore_t done) {
			[c.device newRenderPipelineStateWithDescriptor:d
						    completionHandler:^(id<MTLRenderPipelineState> p, NSError *e) {
							    take(p, nil, e, done);
						    }];
		});
	} else if ([made isEqualToString:@"async"]) {
		await_handler(^(dispatch_semaphore_t done) {
			[c.device newRenderPipelineStateWithDescriptor:d options:options
						    completionHandler:^(id<MTLRenderPipelineState> p, MTLRenderPipelineReflection *rr, NSError *e) {
							    take(p, rr, e, done);
						    }];
		});
	} else if (spec[@"reflection"]) {
		pipeline = [c.device newRenderPipelineStateWithDescriptor:d options:options reflection:&reflection error:&error];
	} else {
		pipeline = [c.device newRenderPipelineStateWithDescriptor:d error:&error];
	}
	if (pipeline && (r[@"added_vertex_binary_functions"] || r[@"added_fragment_binary_functions"])) {
		/* Binary functions added to the finished pipeline, which its
		 * tables then call. */
		MTLRenderPipelineFunctionsDescriptor *fd = [MTLRenderPipelineFunctionsDescriptor new];
		NSMutableArray *vertex_added = [NSMutableArray new], *fragment_added = [NSMutableArray new];
		for (NSString *name in r[@"added_vertex_binary_functions"])
			[vertex_added addObject:binary_function(library, name)];
		for (NSString *name in r[@"added_fragment_binary_functions"])
			[fragment_added addObject:binary_function(library, name)];
		fd.vertexAdditionalBinaryFunctions = vertex_added;
		fd.fragmentAdditionalBinaryFunctions = fragment_added;
		[vertex_linked addObjectsFromArray:vertex_added];
		[fragment_linked addObjectsFromArray:fragment_added];
		pipeline = [pipeline newRenderPipelineStateWithAdditionalBinaryFunctions:fd error:&error];
	}
	if (pipeline && spec[@"reflection"])
		write_text(c, spec[@"reflection"], render_reflection_text(reflection, mesh != nil, legacy_reflection, [spec[@"reflection_types"] boolValue]));
	if (pipeline)
		store_render_properties(c, spec, pipeline, mesh != nil);
	if (!pipeline)
		@throw failure(@"render pipeline: %@", error.localizedDescription);
	id<MTLDepthStencilState> depth_stencil = [c.device newDepthStencilStateWithDescriptor:ds];
	/* Tile pipelines of the pass's color attachments, which dispatch after
	 * the draws in order. */
	NSMutableArray<id<MTLRenderPipelineState>> *tile_pipelines = [NSMutableArray new];
	NSMutableArray<NSDictionary *> *tile_tables = [NSMutableArray new];
	for (NSDictionary *tile in r[@"tiles"]) {
		MTLTileRenderPipelineDescriptor *td = [MTLTileRenderPipelineDescriptor new];
		td.tileFunction = function_named(library, tile[@"function"], spec[@"constants"]);
		td.rasterSampleCount = number(r, @"samples", 1);
		td.threadgroupSizeMatchesTileSize = [tile[@"matches_tile_size"] boolValue];
		td.maxTotalThreadsPerThreadgroup = number(tile, @"max_total_threads", 0);
		for (NSUInteger i = 0; i < colors.count; i++)
			td.colorAttachments[i].pixelFormat = c.textures[colors[i][@"texture"]].pixelFormat;
		for (NSNumber *i in tile[@"immutable_buffers"])
			td.tileBuffers[i.unsignedIntegerValue].mutability = MTLMutabilityImmutable;
		if (tile[@"required_threads"]) {
			NSArray *t = tile[@"required_threads"];
			if (@available(macOS 26.0, *))
				td.requiredThreadsPerThreadgroup = MTLSizeMake([t[0] unsignedIntegerValue], [t[1] unsignedIntegerValue], [t[2] unsignedIntegerValue]);
			else
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"required tile threads need macOS 26" userInfo:nil];
		}
		NSMutableArray *linked = [NSMutableArray new];
		for (id entry in tile[@"visible_functions"])
			[linked addObject:linked_function(c, library, entry)];
		td.linkedFunctions = stage_links(c, library, tile, @"", linked);
		td.supportAddingBinaryFunctions = tile[@"added_binary_functions"] != nil;
		/* An archive given the descriptor's functions, which then makes
		 * the pipeline or fails. */
		MTLPipelineOption options = tile[@"reflection"] ? MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo : 0;
		if ([tile[@"archive"] boolValue]) {
			id<MTLBinaryArchive> archive = [c.device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
			if (!archive || ![archive addTileRenderPipelineFunctionsWithDescriptor:td error:&error])
				@throw failure(@"tile archive: %@", error.localizedDescription);
			td.binaryArchives = @[ archive ];
			options |= MTLPipelineOptionFailOnBinaryArchiveMiss;
		}
		__block id<MTLRenderPipelineState> tp = nil;
		__block MTLRenderPipelineReflection *reflection = nil;
		__block NSError *made_error = nil;
		if ([tile[@"made"] isEqual:@"async"]) {
			await_handler(^(dispatch_semaphore_t done) {
				[c.device newRenderPipelineStateWithTileDescriptor:td
									   options:options
								 completionHandler:^(id<MTLRenderPipelineState> p, MTLRenderPipelineReflection *rr, NSError *e) {
									 tp = p;
									 reflection = rr;
									 made_error = e;
									 dispatch_semaphore_signal(done);
								 }];
			});
		} else {
			tp = [c.device newRenderPipelineStateWithTileDescriptor:td options:options reflection:&reflection error:&made_error];
		}
		if (!tp)
			@throw failure(@"tile pipeline: %@", made_error.localizedDescription);
		/* A limit asked for is the pipeline's (an Apple GPU reports it). */
		if (tile[@"max_total_threads"] && tp.maxTotalThreadsPerThreadgroup != number(tile, @"max_total_threads", 0))
			@throw failure(@"tile pipeline maxTotalThreadsPerThreadgroup %lu, asked %@",
				       (unsigned long)tp.maxTotalThreadsPerThreadgroup, tile[@"max_total_threads"]);
		if (tile[@"added_binary_functions"]) {
			/* Binary functions added to the finished pipeline, which its
			 * table then calls. */
			MTLRenderPipelineFunctionsDescriptor *fd = [MTLRenderPipelineFunctionsDescriptor new];
			NSMutableArray *added = [NSMutableArray new];
			for (NSString *name in tile[@"added_binary_functions"])
				[added addObject:binary_function(library, name)];
			fd.tileAdditionalBinaryFunctions = added;
			[linked addObjectsFromArray:added];
			tp = [tp newRenderPipelineStateWithAdditionalBinaryFunctions:fd error:&made_error];
			if (!tp)
				@throw failure(@"tile binary functions: %@", made_error.localizedDescription);
		}
		if (tile[@"reflection"]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			NSUInteger arguments = [reflection.tileArguments count];
#pragma clang diagnostic pop
			MTLSize t = MTLSizeMake(0, 0, 0);
			if (@available(macOS 26.0, *))
				t = tp.requiredThreadsPerTileThreadgroup;
			write_text(c, tile[@"reflection"],
				   [NSString stringWithFormat:@"tile\n%@\narguments=%lu\nrequired=%lu,%lu,%lu\n",
							      bindings_text(reflection.tileBindings, NO),
							      (unsigned long)arguments, (unsigned long)t.width,
							      (unsigned long)t.height, (unsigned long)t.depth]);
		}
		/* The tile's tables of its linked functions, by buffer index. */
		NSMutableDictionary *tables = [NSMutableDictionary new];
		for (NSDictionary *table in tile[@"tables"]) {
			MTLVisibleFunctionTableDescriptor *vd = [MTLVisibleFunctionTableDescriptor visibleFunctionTableDescriptor];
			NSArray *names = table[@"functions"];
			vd.functionCount = names.count;
			id<MTLVisibleFunctionTable> vft = [tp newVisibleFunctionTableWithDescriptor:vd stage:MTLRenderStageTile];
			if (!vft)
				@throw failure(@"tile visible function table creation failed");
			NSMutableArray *handles = [NSMutableArray new];
			for (NSString *name in names) {
				id<MTLFunction> f = nil;
				for (id<MTLFunction> l in linked)
					if ([l.name isEqualToString:name]) f = l;
				id<MTLFunctionHandle> handle = f ? [tp functionHandleWithFunction:f stage:MTLRenderStageTile] : nil;
				if (!handle)
					@throw failure(@"no tile function handle for %@", name);
				[handles addObject:handle];
			}
			fill_table(vft, handles, [spec[@"bind"] isEqual:@"ranges"]);
			tables[@(number(table, @"index", 0))] = vft;
		}
		[tile_pipelines addObject:tp];
		[tile_tables addObject:tables];
	}
	/* Everything an encoder of this pass records, from `spec` and its
	 * bindings: once, or once per sub-encoder of a parallel encoder. */
	void (^encode)(id<MTLRenderCommandEncoder>, NSDictionary *) = ^(id<MTLRenderCommandEncoder> e, NSDictionary *spec) {
	wait_fences(c, e, spec);
	[e setRenderPipelineState:pipeline];
	[e setDepthStencilState:depth_stencil];
	if (r[@"stencil"]) {
		uint32_t front = (uint32_t)number(r[@"stencil"], @"reference", 0);
		[e setStencilFrontReferenceValue:front backReferenceValue:(uint32_t)number(r[@"stencil"], @"reference_back", front)];
	}
	if (r[@"depth_bias"]) {
		NSArray *b = r[@"depth_bias"];
		[e setDepthBias:[b[0] floatValue] slopeScale:[b[1] floatValue] clamp:[b[2] floatValue]];
	}
	if ([r[@"depth_clip"] isEqualToString:@"clamp"])
		[e setDepthClipMode:MTLDepthClipModeClamp];
	if ([r[@"fill"] isEqualToString:@"lines"])
		[e setTriangleFillMode:MTLTriangleFillModeLines];
	if (r[@"blend_color"]) {
		NSArray *b = r[@"blend_color"];
		[e setBlendColorRed:[b[0] floatValue] green:[b[1] floatValue] blue:[b[2] floatValue] alpha:[b[3] floatValue]];
	}
	NSString *cull = r[@"cull"] ?: @"none";
	[e setCullMode:[cull isEqualToString:@"front"] ? MTLCullModeFront : [cull isEqualToString:@"back"] ? MTLCullModeBack : MTLCullModeNone];
	[e setFrontFacingWinding:[r[@"winding"] isEqualToString:@"cw"] ? MTLWindingClockwise : MTLWindingCounterClockwise];
	if (r[@"viewport"]) {
		NSArray *v = r[@"viewport"];
		[e setViewport:(MTLViewport){[v[0] doubleValue], [v[1] doubleValue], [v[2] doubleValue], [v[3] doubleValue], 0.0, 1.0}];
	}
	if (r[@"viewports"]) {
		NSArray *list = r[@"viewports"];
		MTLViewport viewports[16];
		NSUInteger count = MIN(list.count, (NSUInteger)16);
		for (NSUInteger k = 0; k < count; k++) {
			NSArray *v = list[k];
			viewports[k] = (MTLViewport){[v[0] doubleValue], [v[1] doubleValue], [v[2] doubleValue], [v[3] doubleValue], 0.0, 1.0};
		}
		[e setViewports:viewports count:count];
	}
	if (r[@"scissor"]) {
		NSArray *s = r[@"scissor"];
		[e setScissorRect:(MTLScissorRect){[s[0] unsignedIntegerValue], [s[1] unsignedIntegerValue], [s[2] unsignedIntegerValue], [s[3] unsignedIntegerValue]}];
	}
	bind_resources(c, e, spec, NO,
		       MTLRenderStageVertex | MTLRenderStageFragment | (mesh ? MTLRenderStageObject | MTLRenderStageMesh : 0)
			       | (r[@"tiles"] ? MTLRenderStageTile : 0));
	/* The tile dispatches before the draws or after them, in order. */
	NSArray *tiles = r[@"tiles"];
	void (^dispatch_tiles)(BOOL) = ^(BOOL before) {
		for (NSUInteger k = 0; k < tiles.count; k++) {
			if ([tiles[k][@"before"] boolValue] != before)
				continue;
			[e setRenderPipelineState:tile_pipelines[k]];
			bind_tables(e, StageTile, tile_tables[k], [spec[@"bind"] isEqual:@"ranges"]);
			for (NSDictionary *m in tiles[k][@"threadgroup_memory"])
				[e setThreadgroupMemoryLength:number(m, @"length", 16)
						       offset:number(m, @"offset", 0)
						      atIndex:number(m, @"index", 0)];
			NSArray *threads = tiles[k][@"threads"];
			[e dispatchThreadsPerTile:MTLSizeMake([threads[0] unsignedIntegerValue],
							      [threads[1] unsignedIntegerValue], 1)];
			if (before)
				[e setRenderPipelineState:pipeline];
		}
	};
	dispatch_tiles(YES);
	/* Each stage's tables of the functions linked to it. */
	BOOL ranges = [spec[@"bind"] isEqual:@"ranges"];
	for (NSString *stage in @[ @"vertex", @"fragment" ]) {
		BOOL vertex = [stage isEqualToString:@"vertex"];
		MTLRenderStages render_stage = vertex ? MTLRenderStageVertex : MTLRenderStageFragment;
		NSMutableDictionary *tables = [NSMutableDictionary new];
		for (NSDictionary *table in r[[stage stringByAppendingString:@"_tables"]]) {
			MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor visibleFunctionTableDescriptor];
			NSArray *names = table[@"functions"];
			td.functionCount = names.count;
			id<MTLVisibleFunctionTable> vft = [pipeline newVisibleFunctionTableWithDescriptor:td stage:render_stage];
			if (!vft)
				@throw failure(@"%@ visible function table creation failed", stage);
			NSMutableArray *handles = [NSMutableArray new];
			for (NSUInteger i = 0; i < names.count; i++) {
				id<MTLFunction> f = nil;
				for (id<MTLFunction> l in vertex ? vertex_linked : fragment_linked)
					if ([l.name isEqualToString:names[i]]) f = l;
				id<MTLFunctionHandle> handle = nil;
				if ([r[@"handles_by_name"] boolValue]) {
					if (@available(macOS 26.0, *))
						handle = [pipeline functionHandleWithName:names[i] stage:render_stage];
					else
						@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"handles by name need macOS 26" userInfo:nil];
				} else {
					handle = [pipeline functionHandleWithFunction:f stage:render_stage];
				}
				if (!handle)
					@throw failure(@"no %@ function handle for %@", stage, names[i]);
				[handles addObject:handle];
			}
			fill_table(vft, handles, ranges);
			tables[@(number(table, @"index", 0))] = vft;
		}
		bind_tables(e, vertex ? StageVertex : StageFragment, tables, ranges);
	}
	MTLPrimitiveType primitive = lookup(primitive_types(), @"primitive", r[@"primitive"] ?: @"triangle");
	NSUInteger base_instance = number(r, @"base_instance", 0);
	if (r[@"amplification"]) {
		/* Each view's offsets into the viewports and target layers. */
		NSArray *list = r[@"amplification"][@"mappings"] ?: @[];
		MTLVertexAmplificationViewMapping mappings[8];
		NSUInteger count = MIN(list.count, (NSUInteger)8);
		for (NSUInteger k = 0; k < count; k++)
			mappings[k] = (MTLVertexAmplificationViewMapping){[list[k][0] unsignedIntValue], [list[k][1] unsignedIntValue]};
		[e setVertexAmplificationCount:number(r[@"amplification"], @"count", 1) viewMappings:count ? mappings : NULL];
	}
	if (mesh && !r[@"execute_icb"]) {
		NSArray *g = mesh[@"threadgroups"] ?: @[ @1 ], *o = mesh[@"object_threads"] ?: @[ @1 ], *t = mesh[@"mesh_threads"] ?: @[ @1 ];
		MTLSize (^size)(NSArray *) = ^(NSArray *a) {
			return MTLSizeMake([a[0] unsignedIntegerValue], a.count > 1 ? [a[1] unsignedIntegerValue] : 1,
					   a.count > 2 ? [a[2] unsignedIntegerValue] : 1);
		};
		if (mesh[@"object_threadgroup_memory"])
			[e setObjectThreadgroupMemoryLength:number(mesh, @"object_threadgroup_memory", 16) atIndex:0];
		if (mesh[@"indirect"])
			[e drawMeshThreadgroupsWithIndirectBuffer:c.buffers[mesh[@"indirect"]] indirectBufferOffset:0
				      threadsPerObjectThreadgroup:size(o) threadsPerMeshThreadgroup:size(t)];
		else if (mesh[@"threads"])
			[e drawMeshThreads:size(mesh[@"threads"]) threadsPerObjectThreadgroup:size(o) threadsPerMeshThreadgroup:size(t)];
		else
			[e drawMeshThreadgroups:size(g) threadsPerObjectThreadgroup:size(o) threadsPerMeshThreadgroup:size(t)];
	} else if (tess && !r[@"execute_icb"]) {
		/* Factors per patch, and per instance at a stride; patches by
		 * index buffer and draws from an indirect buffer when named. */
		[e setTessellationFactorBuffer:c.buffers[tess[@"factors"]] offset:0
				instanceStride:number(tess, @"instance_stride", 0)];
		if (tess[@"factor_scale"])
			[e setTessellationFactorScale:[tess[@"factor_scale"] floatValue]];
		id<MTLBuffer> index = tess[@"patch_index"] ? c.buffers[tess[@"patch_index"]] : nil;
		id<MTLBuffer> point_index = tess[@"control_point_index"] ? c.buffers[tess[@"control_point_index"]] : nil;
		NSUInteger points = number(tess, @"control_points", 4);
		if (tess[@"indirect"] && point_index)
			[e drawIndexedPatches:points patchIndexBuffer:index patchIndexBufferOffset:0 controlPointIndexBuffer:point_index
			    controlPointIndexBufferOffset:0 indirectBuffer:c.buffers[tess[@"indirect"]] indirectBufferOffset:0];
		else if (tess[@"indirect"])
			[e drawPatches:points patchIndexBuffer:index patchIndexBufferOffset:0
				indirectBuffer:c.buffers[tess[@"indirect"]] indirectBufferOffset:0];
		else if (point_index)
			[e drawIndexedPatches:points patchStart:number(tess, @"patch_start", 0) patchCount:number(tess, @"patch_count", 1)
				  patchIndexBuffer:index patchIndexBufferOffset:0 controlPointIndexBuffer:point_index
		     controlPointIndexBufferOffset:0 instanceCount:number(tess, @"instance_count", 1) baseInstance:0];
		else
			[e drawPatches:points patchStart:number(tess, @"patch_start", 0)
				patchCount:number(tess, @"patch_count", 1) patchIndexBuffer:index patchIndexBufferOffset:0
			     instanceCount:number(tess, @"instance_count", 1) baseInstance:0];
	} else if (r[@"execute_icb"]) {
		NSDictionary *x = r[@"execute_icb"];
		MTLRenderStages stages = MTLRenderStageVertex | MTLRenderStageFragment | (mesh ? MTLRenderStageObject | MTLRenderStageMesh : 0);
		for (id res in c.resident)
			if ([res conformsToProtocol:@protocol(MTLResource)])
				[e useResource:res usage:MTLResourceUsageRead stages:stages];
		for (NSString *name in spec[@"use"])
			[e useResource:c.buffers[name] usage:MTLResourceUsageRead stages:stages];
		execute_icb(c, e, x);
	} else if (r[@"indirect"]) {
		NSDictionary *i = r[@"indirect"];
		if (r[@"index_buffer"])
			[e drawIndexedPrimitives:primitive indexType:[r[@"index_type"] isEqualToString:@"uint32"] ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
				     indexBuffer:c.buffers[r[@"index_buffer"]] indexBufferOffset:number(r, @"index_offset", 0)
				  indirectBuffer:c.buffers[i[@"buffer"]] indirectBufferOffset:number(i, @"offset", 0)];
		else
			[e drawPrimitives:primitive indirectBuffer:c.buffers[i[@"buffer"]] indirectBufferOffset:number(i, @"offset", 0)];
	} else if (r[@"draws"]) {
		/* Several draws, each counted into its own visibility result. */
		NSDictionary *modes = @{@"boolean" : @(MTLVisibilityResultModeBoolean), @"counting" : @(MTLVisibilityResultModeCounting),
					@"disabled" : @(MTLVisibilityResultModeDisabled)};
		for (NSDictionary *draw in r[@"draws"]) {
			if (r[@"visibility"])
				[e setVisibilityResultMode:lookup(modes, @"visibility mode",
								  draw[@"visibility_mode"] ?: r[@"visibility"][@"mode"] ?: @"counting")
						    offset:number(draw, @"visibility_offset", 0)];
			[e drawPrimitives:primitive vertexStart:number(draw, @"vertex_start", 0) vertexCount:number(draw, @"vertex_count", 3)
			    instanceCount:number(draw, @"instance_count", 1) baseInstance:base_instance];
		}
	} else if (r[@"index_buffer"]) {
		/* The shortest call that states the draw. */
		id<MTLBuffer> indices = c.buffers[r[@"index_buffer"]];
		MTLIndexType type = [r[@"index_type"] isEqualToString:@"uint32"] ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
		NSUInteger count = number(r, @"vertex_count", 3), offset = number(r, @"index_offset", 0);
		NSInteger base_vertex = [r[@"base_vertex"] integerValue];
		if (base_vertex || base_instance)
			[e drawIndexedPrimitives:primitive indexCount:count indexType:type indexBuffer:indices indexBufferOffset:offset
				   instanceCount:number(r, @"instance_count", 1) baseVertex:base_vertex baseInstance:base_instance];
		else if (r[@"instance_count"])
			[e drawIndexedPrimitives:primitive indexCount:count indexType:type indexBuffer:indices indexBufferOffset:offset
				   instanceCount:number(r, @"instance_count", 1)];
		else
			[e drawIndexedPrimitives:primitive indexCount:count indexType:type indexBuffer:indices indexBufferOffset:offset];
	} else {
		NSUInteger start = number(r, @"vertex_start", 0), count = number(r, @"vertex_count", 3);
		if (base_instance)
			[e drawPrimitives:primitive vertexStart:start vertexCount:count instanceCount:number(r, @"instance_count", 1)
			     baseInstance:base_instance];
		else if (r[@"instance_count"])
			[e drawPrimitives:primitive vertexStart:start vertexCount:count instanceCount:number(r, @"instance_count", 1)];
		else
			[e drawPrimitives:primitive vertexStart:start vertexCount:count];
	}
	dispatch_tiles(NO);
	update_fences(c, e, spec);
	[e endEncoding];
	};
	NSArray *parts = r[@"parallel"];
	if (parts) {
		/* Each part's bindings replace the step's; the parts' encoders
		 * execute in the order they were created. */
		id<MTLParallelRenderCommandEncoder> pe = [cb parallelRenderCommandEncoderWithDescriptor:pass];
		NSMutableArray *encoders = [NSMutableArray new];
		for (NSUInteger k = 0; k < parts.count; k++)
			[encoders addObject:[pe renderCommandEncoder]];
		/* Recorded last part first: order comes from creation alone. */
		for (NSUInteger k = parts.count; k-- > 0;) {
			NSMutableDictionary *part = [spec mutableCopy];
			[part addEntriesFromDictionary:parts[k]];
			encode(encoders[k], part);
		}
		[pe endEncoding];
	} else {
		encode([cb renderCommandEncoderWithDescriptor:pass], spec);
	}
}

static void run_render(Case *c, id<MTLCommandQueue> queue, id<MTLLibrary> library, NSDictionary *spec)
{
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	encode_render(c, cb, library, spec);
	wait_for(cb, spec);
}

static MTLSize size_of(NSArray *a)
{
	return MTLSizeMake([a[0] unsignedIntegerValue], a.count > 1 ? [a[1] unsignedIntegerValue] : 1,
			   a.count > 2 ? [a[2] unsignedIntegerValue] : 1);
}

static MTLOrigin origin_of(NSArray *a)
{
	if (!a) return MTLOriginMake(0, 0, 0);
	return MTLOriginMake([a[0] unsignedIntegerValue], a.count > 1 ? [a[1] unsignedIntegerValue] : 0,
			     a.count > 2 ? [a[2] unsignedIntegerValue] : 0);
}

static void encode_blits(Case *c, id<MTLCommandBuffer> cb, NSDictionary *spec)
{
	id<MTLBlitCommandEncoder> e = nil;
	if (spec[@"timestamps"]) {
		/* Samples at the start and end of the pass's work. */
		MTLCounterSampleBufferDescriptor *sd = [MTLCounterSampleBufferDescriptor new];
		sd.counterSet = counter_set(c.device, @"timestamp");
		sd.sampleCount = 2;
		sd.storageMode = MTLStorageModeShared;
		NSError *error = nil;
		id<MTLCounterSampleBuffer> samples = [c.device newCounterSampleBufferWithDescriptor:sd error:&error];
		if (!samples)
			@throw failure(@"counter sample buffer: %@", error.localizedDescription);
		MTLBlitPassDescriptor *bp = [MTLBlitPassDescriptor blitPassDescriptor];
		bp.sampleBufferAttachments[0].sampleBuffer = samples;
		bp.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
		bp.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
		resolve_counters_after(c, cb, samples, 2, spec[@"timestamps"]);
		e = [cb blitCommandEncoderWithDescriptor:bp];
	}
	e = e ?: [cb blitCommandEncoder];
	wait_fences(c, e, spec);
	for (NSDictionary *op in spec[@"blits"]) {
		NSString *kind = op[@"op"];
		if ([kind isEqualToString:@"copy_buffer"]) {
			[e copyFromBuffer:c.buffers[op[@"src"]] sourceOffset:number(op, @"src_offset", 0)
				 toBuffer:c.buffers[op[@"dst"]] destinationOffset:number(op, @"dst_offset", 0)
				     size:number(op, @"size", 0)];
		} else if ([kind isEqualToString:@"fill_buffer"]) {
			[e fillBuffer:c.buffers[op[@"dst"]] range:NSMakeRange(number(op, @"offset", 0), number(op, @"size", 0))
				value:(uint8_t)number(op, @"value", 0)];
		} else if ([kind isEqualToString:@"buffer_to_texture"]) {
			[e copyFromBuffer:c.buffers[op[@"src"]] sourceOffset:number(op, @"src_offset", 0)
			     sourceBytesPerRow:number(op, @"bytes_per_row", 0) sourceBytesPerImage:number(op, @"bytes_per_image", 0)
				    sourceSize:size_of(op[@"size"]) toTexture:c.textures[op[@"dst"]]
			      destinationSlice:number(op, @"slice", 0) destinationLevel:number(op, @"level", 0)
			     destinationOrigin:origin_of(op[@"origin"])];
		} else if ([kind isEqualToString:@"texture_to_buffer"]) {
			[e copyFromTexture:c.textures[op[@"src"]] sourceSlice:number(op, @"slice", 0) sourceLevel:number(op, @"level", 0)
				 sourceOrigin:origin_of(op[@"origin"]) sourceSize:size_of(op[@"size"])
				     toBuffer:c.buffers[op[@"dst"]] destinationOffset:number(op, @"dst_offset", 0)
			    destinationBytesPerRow:number(op, @"bytes_per_row", 0)
			  destinationBytesPerImage:number(op, @"bytes_per_image", 0)];
		} else if ([kind isEqualToString:@"copy_texture"]) {
			[e copyFromTexture:c.textures[op[@"src"]] sourceSlice:number(op, @"src_slice", 0)
				   sourceLevel:number(op, @"src_level", 0) sourceOrigin:origin_of(op[@"src_origin"])
				    sourceSize:size_of(op[@"size"]) toTexture:c.textures[op[@"dst"]]
			      destinationSlice:number(op, @"dst_slice", 0) destinationLevel:number(op, @"dst_level", 0)
			     destinationOrigin:origin_of(op[@"dst_origin"])];
		} else if ([kind isEqualToString:@"buffer_to_texture_plane"]) {
			/* One plane of a depth-stencil texture, chosen by aspect. */
			id<MTLTexture> t = c.textures[op[@"dst"]];
			[e copyFromBuffer:c.buffers[op[@"src"]] sourceOffset:number(op, @"src_offset", 0)
			     sourceBytesPerRow:number(op, @"bytes_per_row", 0) sourceBytesPerImage:number(op, @"bytes_per_image", 0)
				    sourceSize:size_of(op[@"size"]) toTexture:t destinationSlice:number(op, @"slice", 0)
			      destinationLevel:number(op, @"level", 0) destinationOrigin:origin_of(op[@"origin"])
				       options:blit_option(t.pixelFormat, op[@"aspect"])];
		} else if ([kind isEqualToString:@"texture_to_buffer_plane"]) {
			id<MTLTexture> t = c.textures[op[@"src"]];
			[e copyFromTexture:t sourceSlice:number(op, @"slice", 0) sourceLevel:number(op, @"level", 0)
				 sourceOrigin:origin_of(op[@"origin"]) sourceSize:size_of(op[@"size"])
				     toBuffer:c.buffers[op[@"dst"]] destinationOffset:number(op, @"dst_offset", 0)
			    destinationBytesPerRow:number(op, @"bytes_per_row", 0)
			  destinationBytesPerImage:number(op, @"bytes_per_image", 0)
					   options:blit_option(t.pixelFormat, op[@"aspect"])];
		} else if ([kind isEqualToString:@"copy_whole_texture"]) {
			[e copyFromTexture:c.textures[op[@"src"]] toTexture:c.textures[op[@"dst"]]];
		} else if ([kind isEqualToString:@"copy_slices"]) {
			[e copyFromTexture:c.textures[op[@"src"]] sourceSlice:number(op, @"src_slice", 0)
				   sourceLevel:number(op, @"src_level", 0) toTexture:c.textures[op[@"dst"]]
			      destinationSlice:number(op, @"dst_slice", 0) destinationLevel:number(op, @"dst_level", 0)
				    sliceCount:number(op, @"slices", 1) levelCount:number(op, @"levels", 1)];
		} else if ([kind isEqualToString:@"optimize_gpu_access"]) {
			/* The whole texture, or one slice's level. */
			if (op[@"level"])
				[e optimizeContentsForGPUAccess:c.textures[op[@"texture"]] slice:number(op, @"slice", 0) level:number(op, @"level", 0)];
			else
				[e optimizeContentsForGPUAccess:c.textures[op[@"texture"]]];
		} else if ([kind isEqualToString:@"optimize_cpu_access"]) {
			if (op[@"level"])
				[e optimizeContentsForCPUAccess:c.textures[op[@"texture"]] slice:number(op, @"slice", 0) level:number(op, @"level", 0)];
			else
				[e optimizeContentsForCPUAccess:c.textures[op[@"texture"]]];
		} else if ([kind isEqualToString:@"generate_mipmaps"]) {
			[e generateMipmapsForTexture:c.textures[op[@"texture"]]];
		} else if ([kind isEqualToString:@"reset_icb"]) {
			[e resetCommandsInBuffer:c.icbs[op[@"icb"]] withRange:NSMakeRange(number(op, @"start", 0), number(op, @"count", 1))];
		} else if ([kind isEqualToString:@"optimize_icb"]) {
			[e optimizeIndirectCommandBuffer:c.icbs[op[@"icb"]] withRange:NSMakeRange(number(op, @"start", 0), number(op, @"count", 1))];
		} else if ([kind isEqualToString:@"copy_icb"]) {
			[e copyIndirectCommandBuffer:c.icbs[op[@"src"]] sourceRange:NSMakeRange(number(op, @"start", 0), number(op, @"count", 1))
					 destination:c.icbs[op[@"dst"]] destinationIndex:number(op, @"dst_start", 0)];
		} else {
			@throw failure(@"unknown blit %@", kind);
		}
	}
	update_fences(c, e, spec);
	[e endEncoding];
}

static void run_blits(Case *c, id<MTLCommandQueue> queue, NSDictionary *spec)
{
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	encode_blits(c, cb, spec);
	wait_for(cb, spec);
}

/* A case's event: shared, unless the case names it among its
 * `device_events`, which only the GPU signals and waits for. */
static id<MTLEvent> event_named(Case *c, NSString *name)
{
	if (!c.events[name])
		c.events[name] = [c.spec[@"device_events"] containsObject:name] ? [c.device newEvent] : [c.device newSharedEvent];
	return c.events[name];
}

/* Records, per command buffer in commit order, when it was scheduled and
 * completed, for checking the order a queue keeps. */
@interface Handlers : NSObject
@property(retain) NSMutableArray<NSNumber *> *completed;
@property NSUInteger scheduled;
@property BOOL scheduled_first;
@end
@implementation Handlers
@end

/* Steps of compute, render and blit work over the case's resources, in one
 * command buffer unless a step starts a new one. A compute step may
 * continue the previous compute encoder; a step may wait for and update
 * fences, and wait for or signal an event's value between encoders. Every
 * command buffer is committed in order and the last one waited for. */
static void run_sequence(Case *c, id<MTLCommandQueue> queue, NSDictionary *plan, NSDictionary *spec,
			 NSMutableDictionary *libraries)
{
	/* Steps run on the case's queue or on others it creates by index;
	 * command buffers retain their resources unless `unretained`. */
	NSMutableDictionary<NSNumber *, id<MTLCommandQueue>> *queues = [NSMutableDictionary dictionaryWithObject:queue forKey:@0];
	BOOL unretained = [spec[@"unretained"] boolValue];
	Handlers *handlers = [Handlers new];
	handlers.completed = [NSMutableArray new];
	handlers.scheduled_first = YES;
	__block NSUInteger committed = 0;
	id<MTLCommandBuffer> (^make)(id<MTLCommandQueue>) = ^(id<MTLCommandQueue> q) {
		id<MTLCommandBuffer> b = unretained ? [q commandBufferWithUnretainedReferences] : [q commandBuffer];
		NSUInteger k = committed++;
		if (spec[@"handlers"]) {
			[b addScheduledHandler:^(id<MTLCommandBuffer> _) {
				@synchronized(handlers) {
					handlers.scheduled++;
				}
			}];
			[b addCompletedHandler:^(id<MTLCommandBuffer> _) {
				@synchronized(handlers) {
					if (handlers.scheduled < handlers.completed.count + 1)
						handlers.scheduled_first = NO;
					[handlers.completed addObject:@(k)];
				}
			}];
		}
		return b;
	};
	NSNumber *on = @0;
	id<MTLCommandBuffer> cb = make(queue);
	id<MTLComputeCommandEncoder> open = nil;
	NSDictionary *open_step = nil;
	for (NSDictionary *step in spec[@"steps"]) {
		NSString *kind = step[@"kind"];
		BOOL continues = [kind isEqualToString:@"compute"] && [step[@"continue"] boolValue] && open;
		if (open && !continues) {
			update_fences(c, open, open_step);
			[open endEncoding];
			open = nil;
		}
		NSNumber *wanted = step[@"queue"] ?: @0;
		if (!queues[wanted])
			queues[wanted] = [c.device newCommandQueue];
		if ([step[@"new_command_buffer"] boolValue] || ![wanted isEqual:on]) {
			[cb commit];
			on = wanted;
			cb = make(queues[on]);
		}
		for (NSArray *w in step[@"wait_events"])
			[cb encodeWaitForEvent:event_named(c, w[0]) value:[w[1] unsignedLongLongValue]];
		id<MTLLibrary> library = nil;
		if (![kind isEqualToString:@"blit"])
			library = library_named(c.device, plan, step[@"library"] ?: spec[@"library"], libraries);
		if ([kind isEqualToString:@"compute"]) {
			open = encode_compute(c, cb, continues ? open : nil, library, step);
			open_step = step;
		} else if ([kind isEqualToString:@"render"]) {
			encode_render(c, cb, library, step);
		} else if ([kind isEqualToString:@"blit"]) {
			encode_blits(c, cb, step);
		} else {
			@throw failure(@"unknown step kind %@", kind);
		}
		if (step[@"signal_events"] && open) {
			update_fences(c, open, open_step);
			[open endEncoding];
			open = nil;
		}
		for (NSArray *s in step[@"signal_events"])
			[cb encodeSignalEvent:event_named(c, s[0]) value:[s[1] unsignedLongLongValue]];
	}
	if (open) {
		update_fences(c, open, open_step);
		[open endEncoding];
	}
	/* Values the host signals once everything is committed, after a
	 * delay, so the GPU's waits for them are real; a signal may first wait
	 * for each [event, value] it lists, as a listener of the GPU's would. */
	NSArray *host = spec[@"host_signals"];
	for (NSArray *h in host) {
		event_named(c, h[0]);
		for (NSArray *w in h.count > 2 ? h[2] : @[])
			event_named(c, w[0]);
	}
	if (host.count)
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC), dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
			for (NSArray *h in host) {
				for (NSArray *w in h.count > 2 ? h[2] : @[])
					[(id<MTLSharedEvent>)c.events[w[0]] waitUntilSignaledValue:[w[1] unsignedLongLongValue] timeoutMS:10000];
				((id<MTLSharedEvent>)c.events[h[0]]).signaledValue = [h[1] unsignedLongLongValue];
			}
		});
	/* Listeners the host registers for values the GPU signals. */
	MTLSharedEventListener *listener = nil;
	NSMutableSet *heard = [NSMutableSet new];
	for (NSArray *l in spec[@"listen"]) {
		/* A listener on a queue of the case's, on the queue of its own
		 * it makes, or the one the framework shares. */
		if (!listener) {
			NSString *from = spec[@"listener"] ?: @"queue";
			if ([from isEqualToString:@"queue"]) {
				dispatch_queue_t on = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
				listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:on];
				if (listener.dispatchQueue != on)
					@throw failure(@"event listener runs on another queue");
			} else if ([from isEqualToString:@"own"]) {
				listener = [[MTLSharedEventListener alloc] init];
			} else if (![from isEqualToString:@"shared"]) {
				@throw failure(@"unknown listener %@", from);
			} else if (@available(macOS 26.0, *)) {
				listener = [MTLSharedEventListener sharedListener];
			} else {
				@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"the shared listener needs macOS 26" userInfo:nil];
			}
			if (!listener.dispatchQueue)
				@throw failure(@"event listener has no queue");
		}
		id<MTLSharedEvent> event = (id<MTLSharedEvent>)event_named(c, l[0]);
		NSString *name = l[2];
		[event notifyListener:listener atValue:[l[1] unsignedLongLongValue] block:^(id<MTLSharedEvent> e, uint64_t value) {
			@synchronized(heard) {
				[heard addObject:name];
			}
		}];
	}
	wait_for(cb, spec);
	for (NSArray *l in spec[@"listen"]) {
		BOOL got = NO;
		for (int k = 0; k < 200 && !got; k++) {
			@synchronized(heard) {
				got = [heard containsObject:l[2]];
			}
			if (!got)
				usleep(10000);
		}
		uint32_t *out = c.buffers[l[2]].contents;
		out[0] = got;
		out[1] = (uint32_t)((id<MTLSharedEvent>)c.events[l[0]]).signaledValue;
	}
	if (spec[@"handlers"]) {
		/* Completion in commit order, each after its scheduling, and the
		 * last buffer's final status. */
		for (int k = 0; k < 200; k++) {
			@synchronized(handlers) {
				if (handlers.completed.count == committed)
					break;
			}
			usleep(10000);
		}
		uint32_t *out = c.buffers[spec[@"handlers"]].contents;
		@synchronized(handlers) {
			for (NSUInteger k = 0; k < handlers.completed.count && k < 8; k++)
				out[k] = handlers.completed[k].unsignedIntValue;
			out[8] = handlers.scheduled_first;
			out[9] = (uint32_t)cb.status;
		}
	}
}

static void store_outputs(Case *c, id<MTLCommandQueue> queue, NSDictionary *spec)
{
	timing = NO;
	for (NSDictionary *o in spec[@"outputs"]) {
		NSString *name = o[@"name"];
		NSString *path = [c.dir stringByAppendingPathComponent:[(o[@"as"] ?: name) stringByAppendingString:@".bin"]];
		if (c.buffers[name]) {
			id<MTLBuffer> b = c.buffers[name], shared = b;
			if (b.storageMode == MTLStorageModePrivate)
				shared = [c.device newBufferWithLength:b.length options:MTLResourceStorageModeShared];
			if (b.storageMode != MTLStorageModeShared)
				blit_now(c, ^(id<MTLBlitCommandEncoder> e) {
					if (shared != b)
						[e copyFromBuffer:b sourceOffset:0 toBuffer:shared destinationOffset:0 size:b.length];
					else
						[e synchronizeResource:b];
				});
			NSUInteger length = number(o, @"length", b.length);
			[[NSData dataWithBytes:shared.contents length:MIN(length, b.length)] writeToFile:path atomically:YES];
			continue;
		}
		id<MTLTexture> t = c.textures[name];
		if (!t)
			@throw failure(@"output %@ is neither a buffer nor a texture", name);
		NSUInteger total;
		NSArray *regions = regions_of(t, o[@"aspect"], &total);
		if (t.storageMode != MTLStorageModePrivate && !t.buffer) {
			/* Read back by the CPU, after a synchronization when managed. */
			if (t.storageMode == MTLStorageModeManaged)
				blit_now(c, ^(id<MTLBlitCommandEncoder> e) {
					[e synchronizeResource:t];
				});
			NSMutableData *out = [NSMutableData dataWithLength:total];
			for (NSValue *v in regions) {
				struct region r;
				[v getValue:&r];
				/* The shorter call where it states the same image. */
				if (r.slice == 0 && r.size.depth == 1 && t.textureType != MTLTextureType3D)
					[t getBytes:(uint8_t *)out.mutableBytes + r.offset bytesPerRow:r.bytes_per_row
					 fromRegion:MTLRegionMake2D(0, 0, r.size.width, r.size.height) mipmapLevel:r.level];
				else
					[t getBytes:(uint8_t *)out.mutableBytes + r.offset bytesPerRow:r.bytes_per_row bytesPerImage:r.bytes_per_image
					 fromRegion:MTLRegionMake3D(0, 0, 0, r.size.width, r.size.height, r.size.depth) mipmapLevel:r.level slice:r.slice];
			}
			[out writeToFile:path atomically:YES];
			continue;
		}
		id<MTLBuffer> staging = [c.device newBufferWithLength:MAX(total, 16) options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		for (NSValue *v in regions) {
			struct region r;
			[v getValue:&r];
			[e copyFromTexture:t sourceSlice:r.slice sourceLevel:r.level sourceOrigin:MTLOriginMake(0, 0, 0)
				 sourceSize:r.size toBuffer:staging destinationOffset:r.offset
			    destinationBytesPerRow:r.bytes_per_row destinationBytesPerImage:r.bytes_per_image
					   options:blit_option(t.pixelFormat, o[@"aspect"])];
		}
		[e endEncoding];
		wait_for(cb, spec);
		[[NSData dataWithBytes:staging.contents length:total] writeToFile:path atomically:YES];
	}
}


static NSDictionary<NSString *, NSNumber *> *gpu_families(void)
{
	NSMutableDictionary *families = [@{
		@"apple1" : @(MTLGPUFamilyApple1), @"apple2" : @(MTLGPUFamilyApple2), @"apple3" : @(MTLGPUFamilyApple3),
		@"apple4" : @(MTLGPUFamilyApple4), @"apple5" : @(MTLGPUFamilyApple5), @"apple6" : @(MTLGPUFamilyApple6),
		@"apple7" : @(MTLGPUFamilyApple7), @"apple8" : @(MTLGPUFamilyApple8), @"apple9" : @(MTLGPUFamilyApple9),
		@"apple10" : @(MTLGPUFamilyApple10), @"apple11" : @(MTLGPUFamilyApple11), @"mac2" : @(MTLGPUFamilyMac2), @"common1" : @(MTLGPUFamilyCommon1),
		@"common2" : @(MTLGPUFamilyCommon2), @"common3" : @(MTLGPUFamilyCommon3), @"metal3" : @(MTLGPUFamilyMetal3),
	} mutableCopy];
	if (@available(macOS 26.0, *))
		families[@"metal4"] = @(MTLGPUFamilyMetal4);
	return families;
}

/* Whether the device has a feature a case names; nil for a name the runner
 * does not know, which is an error in the plan. */
static BOOL has_counter_set(id<MTLDevice> d, NSString *name)
{
	for (id<MTLCounterSet> set in d.counterSets)
		if ([set.name isEqualToString:name])
			return YES;
	return NO;
}

static BOOL logging_available(void)
{
	if (@available(macOS 15.0, *))
		return YES;
	return NO;
}

/* A capability MTLDevice answers outside its public header, as the
 * frameworks ask it; no for a device without the query. */
static BOOL device_answers(id<MTLDevice> d, SEL query)
{
	return [d respondsToSelector:query] && ((BOOL (*)(id, SEL))objc_msgSend)(d, query);
}

static NSNumber *feature(id<MTLDevice> d, NSString *name)
{
	if ([name hasPrefix:@"samples"])
		return @([d supportsTextureSampleCount:(NSUInteger)[[name substringFromIndex:7] integerValue]]);
	NSDictionary *t = @{
		@"function_pointers" : @(d.supportsFunctionPointers),
		@"function_pointers_render" : @(d.supportsFunctionPointersFromRender),
		@"raytracing" : @(d.supportsRaytracing),
		@"dynamic_libraries" : @(d.supportsDynamicLibraries),
		@"render_dynamic_libraries" : @(d.supportsRenderDynamicLibraries),
		@"float32_filtering" : @(d.supports32BitFloatFiltering),
		@"float32_msaa" : @(d.supports32BitMSAA),
		@"bc_compression" : @(d.supportsBCTextureCompression),
		@"query_texture_lod" : @(d.supportsQueryTextureLOD),
		@"barycentrics" : @(d.supportsShaderBarycentricCoordinates),
		@"pull_model_interpolation" : @(d.supportsPullModelInterpolation),
		@"raster_order_groups" : @(d.rasterOrderGroupsSupported),
		@"programmable_sample_positions" : @(d.programmableSamplePositionsSupported),
		@"depth24_stencil8" : @(d.depth24Stencil8PixelFormatSupported),
		@"argument_buffers_tier2" : @(d.argumentBuffersSupport >= MTLArgumentBuffersTier2),
		@"read_write_textures_tier2" : @(d.readWriteTextureSupport >= MTLReadWriteTextureTier2),
		@"primitive_motion_blur" : @(d.supportsPrimitiveMotionBlur),
		@"memoryless_render_targets" : @(device_answers(d, @selector(supportsMemorylessRenderTargets))),
		@"framebuffer_reads" : @(device_answers(d, @selector(isFramebufferReadSupported))),
		@"vertex_amplification2" : @([d supportsVertexAmplificationCount:2]),
		@"simd_reduction" : @([d supportsFamily:MTLGPUFamilyApple7] || [d supportsFamily:MTLGPUFamilyMac2]),
		@"counters_stage_boundary" : @([d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),
		@"counters_dispatch_boundary" : @([d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
		@"counter_set_timestamp" : @(has_counter_set(d, MTLCommonCounterSetTimestamp)),
		@"counter_set_statistic" : @(has_counter_set(d, MTLCommonCounterSetStatistic)),
		@"logging" : @(logging_available()),
	};
	return t[name];
}

/* The case's unmet requirements, or nil when the device has them all. */
static NSString *unmet(id<MTLDevice> d, NSDictionary *requires)
{
	NSMutableArray *missing = [NSMutableArray new];
	for (NSString *family in requires[@"families"]) {
		if (![d supportsFamily:lookup(gpu_families(), @"GPU family", family)])
			[missing addObject:family];
	}
	for (NSString *name in requires[@"features"]) {
		NSNumber *has = feature(d, name);
		if (!has)
			@throw failure(@"unknown feature %@", name);
		if (!has.boolValue)
			[missing addObject:name];
	}
	return missing.count ? [missing componentsJoinedByString:@", "] : nil;
}

/* Every capability query MTLDevice answers, for comparing what two
 * devices claim. */
static NSDictionary *device_properties(id<MTLDevice> d)
{
	NSMutableDictionary *p = [NSMutableDictionary new];
	NSMutableDictionary *families = [NSMutableDictionary new];
	NSDictionary *known = gpu_families();
	note_lookups(known, @"GPU family", known.allKeys);
	[known enumerateKeysAndObjectsUsingBlock:^(NSString *k, NSNumber *v, BOOL *stop) {
		families[k] = @([d supportsFamily:v.integerValue]);
	}];
	p[@"families"] = families;
	NSMutableDictionary *features = [NSMutableDictionary new];
	for (NSString *name in @[ @"function_pointers", @"function_pointers_render", @"raytracing", @"dynamic_libraries",
				   @"render_dynamic_libraries", @"float32_filtering", @"float32_msaa", @"bc_compression",
				   @"query_texture_lod", @"barycentrics", @"pull_model_interpolation", @"raster_order_groups",
				   @"programmable_sample_positions", @"depth24_stencil8", @"argument_buffers_tier2",
				   @"read_write_textures_tier2", @"primitive_motion_blur", @"memoryless_render_targets",
				   @"framebuffer_reads", @"vertex_amplification2",
				   @"samples1", @"samples2", @"samples4", @"samples8", @"samples16" ])
		features[name] = feature(d, name);
	/* The older name for barycentric support, and the compiler's
	 * concurrency once asked to maximize it. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	features[@"barycentric_coords"] = @(d.barycentricCoordsSupported);
	features[@"feature_set_macos_family2_v1"] = @([d supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1]);
#pragma clang diagnostic pop
	if (@available(macOS 13.3, *)) {
		d.shouldMaximizeConcurrentCompilation = YES;
		features[@"maximized_compilation"] = @(d.shouldMaximizeConcurrentCompilation);
		features[@"concurrent_compilation"] = @(d.maximumConcurrentCompilationTaskCount > 0);
	}
	p[@"features"] = features;
	MTLSize tg = d.maxThreadsPerThreadgroup;
	p[@"limits"] = @{
		@"max_threads_per_threadgroup" : @[ @(tg.width), @(tg.height), @(tg.depth) ],
		@"max_threadgroup_memory" : @(d.maxThreadgroupMemoryLength),
		@"max_buffer_length" : @(d.maxBufferLength),
		@"max_argument_buffer_samplers" : @(d.maxArgumentBufferSamplerCount),
		@"recommended_max_working_set" : @(d.recommendedMaxWorkingSetSize),
		@"sparse_tile_bytes" : @(d.sparseTileSizeInBytes),
		@"argument_buffers_tier" : @(d.argumentBuffersSupport),
		@"read_write_texture_tier" : @(d.readWriteTextureSupport),
		@"max_transfer_rate" : @(d.maxTransferRate),
		@"linear_texture_alignment_rgba8" : @([d minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm]),
		@"linear_texture_alignment_rgba32f" : @([d minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA32Float]),
		@"texture_buffer_alignment_r8" : @([d minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatR8Unorm]),
	};
	p[@"identity"] = @{
		@"name" : d.name, @"unified_memory" : @(d.hasUnifiedMemory), @"low_power" : @(d.isLowPower),
		@"headless" : @(d.isHeadless), @"removable" : @(d.isRemovable), @"location" : @(d.location),
		@"architecture" : d.architecture.name ?: @"",
	};
	NSMutableDictionary *positions = [NSMutableDictionary new];
	for (NSUInteger count = 2; count <= 8; count *= 2) {
		if (![d supportsTextureSampleCount:count])
			continue;
		MTLSamplePosition sp[16];
		[d getDefaultSamplePositions:sp count:count];
		NSMutableArray *list = [NSMutableArray new];
		for (NSUInteger i = 0; i < count; i++)
			[list addObject:@[ @(sp[i].x), @(sp[i].y) ]];
		positions[[NSString stringWithFormat:@"%lu", (unsigned long)count]] = list;
	}
	p[@"sample_positions"] = positions;
	return p;
}

/* An indirect command buffer of dispatches or draws, and the commands the
 * CPU encodes into it. A draw command inherits its pipeline from the
 * executing encoder; a dispatch names its function or inherits too. */
static id<MTLIndirectCommandBuffer> make_icb(Case *c, NSDictionary *b)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	BOOL mesh = [b[@"type"] isEqualToString:@"mesh"], patches = [b[@"type"] isEqualToString:@"patches"];
	BOOL draw = mesh || patches || [b[@"type"] isEqualToString:@"draw"];
	d.commandTypes = mesh	   ? MTLIndirectCommandTypeDrawMeshThreadgroups | MTLIndirectCommandTypeDrawMeshThreads
			 : patches ? MTLIndirectCommandTypeDrawPatches | MTLIndirectCommandTypeDrawIndexedPatches
			 : draw	   ? MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed
				   : MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
	if ([b[@"dynamic_stride"] boolValue]) {
		if (@available(macOS 14.0, *))
			d.supportDynamicAttributeStride = YES;
		else
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"dynamic strides need macOS 14" userInfo:nil];
	}
	/* Draws that set their own fixed-function state rather than inherit
	 * the executing encoder's. */
	if (b[@"inherit_state"] && ![b[@"inherit_state"] boolValue]) {
		if (@available(macOS 26.0, *)) {
			d.inheritCullMode = NO;
			d.inheritDepthStencilState = NO;
			d.inheritDepthBias = NO;
			d.inheritDepthClipMode = NO;
			d.inheritFrontFacingWinding = NO;
			d.inheritTriangleFillMode = NO;
		} else {
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"indirect draw state needs macOS 26" userInfo:nil];
		}
	}
	if (mesh) {
		if (@available(macOS 14.0, *)) {
			d.maxObjectBufferBindCount = number(b, @"max_buffers", 8);
			d.maxMeshBufferBindCount = number(b, @"max_buffers", 8);
			d.maxObjectThreadgroupMemoryBindCount = 1;
		} else {
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"indirect mesh draws need macOS 14" userInfo:nil];
		}
	}
	/* Draws inherit the executing encoder's pipeline unless told not to;
	 * their commands then set their own. */
	d.inheritPipelineState = b[@"inherit_pipeline"] ? [b[@"inherit_pipeline"] boolValue] : draw;
	d.inheritBuffers = [b[@"inherit_buffers"] boolValue];
	NSUInteger bindings = number(b, @"max_buffers", 8);
	if (draw) {
		d.maxVertexBufferBindCount = bindings;
		d.maxFragmentBufferBindCount = bindings;
	} else {
		d.maxKernelBufferBindCount = bindings;
		/* Threadgroup memory a command sets, at the indices it names. */
		if (@available(macOS 14.0, *))
			d.maxKernelThreadgroupMemoryBindCount = number(b, @"max_threadgroup_memory", 0);
	}
	id<MTLIndirectCommandBuffer> icb = [c.device newIndirectCommandBufferWithDescriptor:d
									 maxCommandCount:number(b, @"count", 1)
										 options:0];
	if (!icb)
		@throw failure(@"indirect command buffer %@ creation failed", b[@"name"]);
	icb.label = b[@"name"];
	[c.resident addObject:icb];
	NSArray *commands = b[@"commands"] ?: @[];
	for (NSUInteger k = 0; k < commands.count; k++) {
		NSDictionary *command = commands[k];
		if (draw) {
			id<MTLIndirectRenderCommand> r = [icb indirectRenderCommandAtIndex:k];
			for (NSDictionary *u in command[@"buffers"]) {
				id<MTLBuffer> buffer = c.buffers[u[@"name"]];
				[c.resident addObject:buffer];
				NSUInteger offset = number(u, @"offset", 0), index = number(u, @"index", 0);
				if (u[@"attribute_stride"]) {
					if (@available(macOS 14.0, *))
						[r setVertexBuffer:buffer offset:offset attributeStride:number(u, @"attribute_stride", 0) atIndex:index];
				} else if ([u[@"stage"] isEqualToString:@"vertex"])
					[r setVertexBuffer:buffer offset:offset atIndex:index];
				else if (@available(macOS 14.0, *)) {
					if ([u[@"stage"] isEqualToString:@"object"])
						[r setObjectBuffer:buffer offset:offset atIndex:index];
					else if ([u[@"stage"] isEqualToString:@"mesh"])
						[r setMeshBuffer:buffer offset:offset atIndex:index];
					else
						[r setFragmentBuffer:buffer offset:offset atIndex:index];
				} else {
					[r setFragmentBuffer:buffer offset:offset atIndex:index];
				}
			}
			NSDictionary *st = command[@"state"];
			if (@available(macOS 26.0, *)) {
				if (st) {
					NSDictionary *culls = @{@"none" : @(MTLCullModeNone), @"front" : @(MTLCullModeFront), @"back" : @(MTLCullModeBack)};
					NSDictionary *windings = @{@"clockwise" : @(MTLWindingClockwise),
								   @"counter_clockwise" : @(MTLWindingCounterClockwise)};
					NSDictionary *fills = @{@"fill" : @(MTLTriangleFillModeFill), @"lines" : @(MTLTriangleFillModeLines)};
					NSDictionary *clips = @{@"clip" : @(MTLDepthClipModeClip), @"clamp" : @(MTLDepthClipModeClamp)};
					[r setCullMode:lookup(culls, @"cull mode", st[@"cull"] ?: @"none")];
					[r setFrontFacingWinding:lookup(windings, @"winding", st[@"winding"] ?: @"clockwise")];
					[r setTriangleFillMode:lookup(fills, @"fill mode", st[@"fill"] ?: @"fill")];
					[r setDepthClipMode:lookup(clips, @"depth clip mode", st[@"depth_clip"] ?: @"clip")];
					NSArray *bias = st[@"depth_bias"] ?: @[ @0, @0, @0 ];
					[r setDepthBias:[bias[0] floatValue] slopeScale:[bias[1] floatValue] clamp:[bias[2] floatValue]];
					MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
					dd.depthCompareFunction = lookup(compare_functions(), @"compare function", st[@"depth_compare"] ?: @"always");
					dd.depthWriteEnabled = [st[@"depth_write"] boolValue];
					id<MTLDepthStencilState> ds = [c.device newDepthStencilStateWithDescriptor:dd];
					[c.resident addObject:ds];
					[r setDepthStencilState:ds];
				}
			}
			if (@available(macOS 14.0, *)) {
				if ([command[@"barrier"] boolValue])
					[r setBarrier];
				if ([command[@"clear_barrier"] boolValue])
					[r clearBarrier];
			}
			MTLPrimitiveType primitive = lookup(primitive_types(), @"indirect primitive", command[@"primitive"] ?: @"triangle");
			if (mesh) {
				if (@available(macOS 14.0, *)) {
					NSDictionary *m = command[@"mesh"];
					MTLSize (^size)(NSArray *) = ^(NSArray *a) {
						return MTLSizeMake([a[0] unsignedIntegerValue], a.count > 1 ? [a[1] unsignedIntegerValue] : 1,
								   a.count > 2 ? [a[2] unsignedIntegerValue] : 1);
					};
					if (m[@"object_threadgroup_memory"])
						[r setObjectThreadgroupMemoryLength:number(m, @"object_threadgroup_memory", 16) atIndex:0];
					if (m[@"threads"])
						[r drawMeshThreads:size(m[@"threads"]) threadsPerObjectThreadgroup:size(m[@"object_threads"] ?: @[ @1 ])
							threadsPerMeshThreadgroup:size(m[@"mesh_threads"] ?: @[ @1 ])];
					else
						[r drawMeshThreadgroups:size(m[@"threadgroups"] ?: @[ @1 ])
						    threadsPerObjectThreadgroup:size(m[@"object_threads"] ?: @[ @1 ])
						      threadsPerMeshThreadgroup:size(m[@"mesh_threads"] ?: @[ @1 ])];
				}
			} else if (patches) {
				NSDictionary *t = command[@"patches"];
				id<MTLBuffer> factors = c.buffers[t[@"factors"]];
				[c.resident addObject:factors];
				id<MTLBuffer> point_index = t[@"control_point_index"] ? c.buffers[t[@"control_point_index"]] : nil;
				if (point_index) {
					[c.resident addObject:point_index];
					[r drawIndexedPatches:number(t, @"control_points", 4) patchStart:number(t, @"patch_start", 0)
							   patchCount:number(t, @"patch_count", 1) patchIndexBuffer:nil patchIndexBufferOffset:0
						  controlPointIndexBuffer:point_index controlPointIndexBufferOffset:0
							instanceCount:number(t, @"instance_count", 1) baseInstance:0
						 tessellationFactorBuffer:factors tessellationFactorBufferOffset:0
					   tessellationFactorBufferInstanceStride:0];
				} else {
					[r drawPatches:number(t, @"control_points", 4) patchStart:number(t, @"patch_start", 0)
							patchCount:number(t, @"patch_count", 1) patchIndexBuffer:nil patchIndexBufferOffset:0
						     instanceCount:number(t, @"instance_count", 1) baseInstance:0 tessellationFactorBuffer:factors
					 tessellationFactorBufferOffset:0 tessellationFactorBufferInstanceStride:0];
				}
			} else if (command[@"indices"]) {
				id<MTLBuffer> indices = c.buffers[command[@"indices"]];
				[c.resident addObject:indices];
				[r drawIndexedPrimitives:primitive indexCount:number(command, @"index_count", 3)
					       indexType:[command[@"index_type"] isEqualToString:@"uint32"] ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
					     indexBuffer:indices indexBufferOffset:number(command, @"index_offset", 0)
					   instanceCount:number(command, @"instance_count", 1)
					      baseVertex:[command[@"base_vertex"] integerValue]
					    baseInstance:number(command, @"base_instance", 0)];
			} else {
				[r drawPrimitives:primitive vertexStart:number(command, @"vertex_start", 0)
				      vertexCount:number(command, @"vertex_count", 3)
				    instanceCount:number(command, @"instance_count", 1)
				     baseInstance:number(command, @"base_instance", 0)];
			}
			/* A command encoded, then reset: it does nothing. */
			if ([command[@"reset"] boolValue])
				[r reset];
		} else {
			id<MTLIndirectComputeCommand> k_ = [icb indirectComputeCommandAtIndex:k];
			if (command[@"function"]) {
				id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
				MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
				pd.computeFunction = function_named(library, command[@"function"], nil);
				pd.supportIndirectCommandBuffers = YES;
				if (command[@"stage_input"])
					pd.stageInputDescriptor = stage_input_descriptor(command[@"stage_input"]);
				NSError *error = nil;
				id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithDescriptor:pd
												      options:MTLPipelineOptionNone
												   reflection:nil
													error:&error];
				if (!pipeline)
					@throw failure(@"indirect compute pipeline: %@", error.localizedDescription);
				/* The command buffer does not retain its pipelines. */
				[c.resident addObject:pipeline];
				[k_ setComputePipelineState:pipeline];
			}
			for (NSDictionary *u in command[@"buffers"]) {
				id<MTLBuffer> buffer = c.buffers[u[@"name"]];
				[c.resident addObject:buffer];
				if (u[@"attribute_stride"]) {
					if (@available(macOS 14.0, *))
						[k_ setKernelBuffer:buffer offset:number(u, @"offset", 0)
						     attributeStride:number(u, @"attribute_stride", 0) atIndex:number(u, @"index", 0)];
				} else {
					[k_ setKernelBuffer:buffer offset:number(u, @"offset", 0) atIndex:number(u, @"index", 0)];
				}
			}
			NSArray *g = command[@"threadgroups"] ?: @[ @1 ], *t = command[@"threadgroup"] ?: @[ @64 ];
			MTLSize groups = MTLSizeMake([g[0] unsignedIntegerValue], g.count > 1 ? [g[1] unsignedIntegerValue] : 1,
						     g.count > 2 ? [g[2] unsignedIntegerValue] : 1);
			MTLSize threads = MTLSizeMake([t[0] unsignedIntegerValue], t.count > 1 ? [t[1] unsignedIntegerValue] : 1,
						      t.count > 2 ? [t[2] unsignedIntegerValue] : 1);
			for (NSDictionary *m in command[@"threadgroup_memory"])
				[k_ setThreadgroupMemoryLength:number(m, @"length", 16) atIndex:number(m, @"index", 0)];
			NSArray *imageblock = command[@"imageblock"];
			if (imageblock)
				[k_ setImageblockWidth:[imageblock[0] unsignedIntegerValue] height:[imageblock[1] unsignedIntegerValue]];
			NSArray *region = command[@"stage_in_region"];
			if (region)
				[k_ setStageInRegion:MTLRegionMake3D(0, 0, 0, [region[0] unsignedIntegerValue],
								     region.count > 1 ? [region[1] unsignedIntegerValue] : 1,
								     region.count > 2 ? [region[2] unsignedIntegerValue] : 1)];
			/* A grid of threads, its edge groups partial, or of groups. */
			if (command[@"threads_per_grid"])
				[k_ concurrentDispatchThreads:size3(command[@"threads_per_grid"]) threadsPerThreadgroup:threads];
			else
				[k_ concurrentDispatchThreadgroups:groups threadsPerThreadgroup:threads];
			/* The command waits for every earlier command of the buffer,
			 * or no longer does. */
			if ([command[@"barrier"] boolValue])
				[k_ setBarrier];
			if ([command[@"clear_barrier"] boolValue])
				[k_ clearBarrier];
			if ([command[@"reset"] boolValue])
				[k_ reset];
		}
	}
	/* Commands reset from the CPU after encoding. */
	NSArray *reset = b[@"reset_range"];
	if (reset)
		[icb resetWithRange:NSMakeRange([reset[0] unsignedIntegerValue], [reset[1] unsignedIntegerValue])];
	return icb;
}

/* A Core Animation filter of `type`, by its private class. */
static id ca_filter(NSString *type)
{
	return ((id (*)(id, SEL, id))objc_msgSend)(NSClassFromString(@"CAFilter"), sel_registerName("filterWithType:"),
						    type);
}

/* A horizontal red and vertical green ramp over blue 0.6, `w` by `h`. */
static CGImageRef ca_gradient(size_t w, size_t h)
{
	CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, w, h, 8, w * 4, rgb,
						     (CGBitmapInfo)kCGImageAlphaPremultipliedFirst);
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++) {
			CGContextSetRGBFillColor(context, x / (double)w, y / (double)h, 0.6, 1);
			CGContextFillRect(context, CGRectMake(x, y, 1, 1));
		}
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	CGColorSpaceRelease(rgb);
	return image;
}

/* Two overlapping opaque discs on clear, `w` by `h`: a shape only its alpha
 * describes, as a glyph's is. */
static CGImageRef ca_discs(size_t w, size_t h)
{
	CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, w, h, 8, w * 4, rgb,
						     (CGBitmapInfo)kCGImageAlphaPremultipliedFirst);
	CGContextSetRGBFillColor(context, 1, 1, 1, 1);
	CGContextFillEllipseInRect(context, CGRectMake(2, 2, h - 4, h - 4));
	CGContextFillEllipseInRect(context, CGRectMake(w - h + 2, 2, h - 4, h - 4));
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	CGColorSpaceRelease(rgb);
	return image;
}

/* A backdrop layer at `frame` filtering what lies below it by `f`'s
 * "backdrop" filters: shaped by an SDF sublayer when `f` has "sdf", by an
 * image sublayer's alpha when it has "mask", or else by its rounded bounds. */
static CALayer *ca_backdrop(NSDictionary *f, CGRect frame)
{
	NSArray *backdrop = f[@"backdrop"];
	CALayer *layer = [NSClassFromString(@"CABackdropLayer") layer];
	layer.frame = frame;
	NSDictionary *sdf = f[@"sdf"];
	BOOL mask = f[@"mask"] != nil;
	if (mask) {
		/* Glass shaped by an image, as text is: the filters name a
		 * sublayer whose contents' alpha is the shape. */
		[layer setValue:@YES forKey:@"enabled"];
		[layer setValue:@YES forKey:@"sourceSublayerDefinesBounds"];
		[layer setValue:@0.5 forKey:@"marginWidth"];
		[layer setValue:@0.5 forKey:@"scale"];
		CALayer *shape = [CALayer layer];
		shape.frame = layer.bounds;
		shape.name = @"@0";
		CGImageRef discs = ca_discs((size_t)frame.size.width, (size_t)frame.size.height);
		shape.contents = (__bridge id)discs;
		CGImageRelease(discs);
		[layer addSublayer:shape];
	} else if (sdf) {
		/* Glass as SwiftUI builds it: the shape is an SDF sublayer the
		 * filters name, one rounded element over the whole backdrop. */
		[layer setValue:@YES forKey:@"enabled"];
		[layer setValue:@YES forKey:@"sourceSublayerDefinesBounds"];
		[layer setValue:@0.5 forKey:@"marginWidth"];
		[layer setValue:@0.5 forKey:@"scale"];
		CALayer *shape = [NSClassFromString(@"CASDFLayer") layer];
		shape.frame = layer.bounds;
		shape.name = @"@0";
		[shape setValue:[NSClassFromString(@"CASDFOutputEffect") new] forKey:@"effect"];
		[shape setValue:@8 forKey:@"smoothness"];
		CALayer *element = [NSClassFromString(@"CASDFElementLayer") layer];
		element.frame = shape.bounds;
		if (sdf[@"contents"]) {
			/* An element shaped by an image's coverage, as a glyph's
			 * glass is: Core Animation finds its distances itself. */
			CGImageRef discs = ca_discs((size_t)frame.size.width, (size_t)frame.size.height);
			element.contents = (__bridge id)discs;
			CGImageRelease(discs);
			[element setValue:@"contents" forKey:@"mode"];
		} else {
			element.cornerRadius = [sdf[@"corner_radius"] doubleValue];
			[element setValue:@"bounds" forKey:@"mode"];
		}
		[shape addSublayer:element];
		[layer addSublayer:shape];
	} else {
		layer.cornerRadius = 8;
		layer.masksToBounds = YES;
	}
	NSMutableArray *filters = [NSMutableArray new];
	for (NSDictionary *spec in backdrop) {
		id filter = ca_filter(spec[@"type"]);
		if (sdf || mask)
			[filter setValue:@"@0" forKey:@"inputSourceSublayerName"];
		for (NSString *key in spec[@"values"])
			[filter setValue:spec[@"values"][key] forKey:key];
		/* Colors as sRGB components with alpha, sizes as width and height. */
		for (NSString *key in spec[@"colors"]) {
			NSArray *v = spec[@"colors"][key];
			CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
			CGFloat c[4] = {[v[0] doubleValue], [v[1] doubleValue], [v[2] doubleValue], [v[3] doubleValue]};
			CGColorRef color = CGColorCreate(srgb, c);
			[filter setValue:(__bridge id)color forKey:key];
			CGColorRelease(color);
			CGColorSpaceRelease(srgb);
		}
		for (NSString *key in spec[@"sizes"]) {
			NSArray *v = spec[@"sizes"][key];
			CGSize size = CGSizeMake([v[0] doubleValue], [v[1] doubleValue]);
			[filter setValue:[NSValue valueWithBytes:&size objCType:@encode(CGSize)] forKey:key];
		}
		if (spec[@"matrix"]) {
			float m[20];
			for (int i = 0; i < 20; i++)
				m[i] = [spec[@"matrix"][i] floatValue];
			[filter setValue:[NSValue valueWithBytes:m objCType:"{CAColorMatrix=ffffffffffffffffffff}"]
				  forKey:@"inputColorMatrix"];
		}
		[filters addObject:filter];
	}
	layer.filters = filters;
	return layer;
}

/* A push button's label, "Get Started", in the system font of `size` points,
 * black, its baseline at `origin`. */
static void draw_label(CGContextRef context, CGFloat size, CGPoint origin)
{
	CTFontRef font = CTFontCreateUIFontForLanguage(kCTFontUIFontSystem, size, NULL);
	CGColorRef black = CGColorCreateGenericGray(0, 1);
	NSDictionary *attributes = @{(id)kCTFontAttributeName : (__bridge id)font,
				     (id)kCTForegroundColorAttributeName : (__bridge id)black};
	NSAttributedString *label = [[NSAttributedString alloc] initWithString:@"Get Started" attributes:attributes];
	CTLineRef line = CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)label);
	CGContextSetTextPosition(context, origin.x, origin.y);
	CTLineDraw(line, context);
	CFRelease(line);
	CGColorRelease(black);
	CFRelease(font);
}

/* Draws the label into a layer's backing store, as views and SwiftUI draw
 * their text, and animates nothing. */
@interface LabelDelegate : NSObject <CALayerDelegate>
@end

@implementation LabelDelegate
- (void)drawLayer:(CALayer *)layer inContext:(CGContextRef)context
{
	draw_label(context, 17, CGPointMake(6, 6));
}

- (id<CAAction>)actionForLayer:(CALayer *)layer forKey:(NSString *)key
{
	return (id<CAAction>)[NSNull null];
}
@end

/* The label drawn into a layer of `frame`, as `text` names: a text layer
 * ("layer"), an alpha-only image Core Animation tints ("tinted", as it tints
 * template images), or an A8 backing store the layer's delegate draws
 * ("backing", as SwiftUI draws text). With "vibrant", the label takes the
 * color matrix filter SwiftUI gives text over glass. */
static CALayer *ca_label(NSDictionary *text, CGRect frame)
{
	static LabelDelegate *delegate;
	NSString *kind = text[@"kind"];
	CALayer *layer;
	if ([kind isEqualToString:@"layer"]) {
		CATextLayer *label = [CATextLayer layer];
		label.string = @"Get Started";
		CTFontRef font = CTFontCreateUIFontForLanguage(kCTFontUIFontSystem, 17, NULL);
		label.font = font;
		CFRelease(font);
		label.fontSize = 17;
		CGColorRef white = CGColorCreateGenericGray(1, 1);
		label.foregroundColor = white;
		CGColorRelease(white);
		layer = label;
	} else if ([kind isEqualToString:@"tinted"]) {
		size_t w = (size_t)frame.size.width, h = (size_t)frame.size.height;
		CGContextRef context = CGBitmapContextCreate(NULL, w, h, 8, w, NULL, (CGBitmapInfo)kCGImageAlphaOnly);
		draw_label(context, 17, CGPointMake(6, 6));
		CGImageRef mask = CGBitmapContextCreateImage(context);
		CGContextRelease(context);
		CGColorRef tint = CGColorCreateGenericRGB(0.1, 0.1, 0.1, 1);
		layer = [CALayer layer];
		layer.contents = ((id (*)(id, SEL, CGImageRef, CGColorRef))objc_msgSend)(
			NSClassFromString(@"CATintedImage"), sel_registerName("tintedImageWithCGImage:tint:"), mask, tint);
		CGColorRelease(tint);
		CGImageRelease(mask);
	} else if ([kind isEqualToString:@"backing"]) {
		if (!delegate)
			delegate = [LabelDelegate new];
		layer = [CALayer layer];
		layer.delegate = delegate;
		layer.contentsFormat = @"A8";
		layer.frame = frame;
		[layer setNeedsDisplay];
	} else {
		@throw failure(@"unknown label %@", kind);
	}
	layer.frame = frame;
	if ([text[@"vibrant"] boolValue]) {
		/* SwiftUI lifts the label over dark glass and lowers it over
		 * light glass, which it marks as aware of the backdrop. */
		id filter = ca_filter(@"vibrantColorMatrix");
		float o = text[@"offset"] ? [text[@"offset"] floatValue] : 0.2f;
		float m[20] = {1, 0, 0, 0, o, 0, 1, 0, 0, o, 0, 0, 1, 0, o, 0, 0, 0, 1, 0};
		if ([text[@"backdrop_aware"] boolValue])
			[filter setValue:@YES forKey:@"inputBackdropAware"];
		[filter setValue:[NSValue valueWithBytes:m objCType:"{CAColorMatrix=ffffffffffffffffffff}"]
			  forKey:@"inputColorMatrix"];
		layer.filters = @[ filter ];
	}
	return layer;
}

/* A button's label over the gradient, by `text` (see ca_label). With
 * "portal", the label shows only through a portal layer that hides its
 * source, and with "glass" too, above a glass backdrop of the button's shape
 * (`scene`'s "backdrop" and "sdf"), as SwiftUI composes a glass button: the
 * label's container comes first, and a portal above the glass shows it. */
static CALayer *ca_text(NSDictionary *text, NSDictionary *scene)
{
	CGRect button = CGRectMake(16, 6, 128, 36);
	CALayer *host = [CALayer layer];
	host.frame = button;
	/* SwiftUI's containers have no size: they only place their sublayers. */
	CALayer *content = [CALayer layer];
	[content addSublayer:ca_label(text, CGRectMake(8, 8, 112, 24))];
	[host addSublayer:content];
	if (![text[@"portal"] boolValue])
		return host;
	CALayer *glass = [CALayer layer];
	if (scene[@"backdrop"])
		[glass addSublayer:ca_backdrop(scene, host.bounds)];
	CALayer *portal = [NSClassFromString(@"CAPortalLayer") layer];
	[portal setValue:content forKey:@"sourceLayer"];
	[portal setValue:@YES forKey:@"hidesSourceLayer"];
	[portal setValue:@YES forKey:@"allowsBackdropGroups"];
	[glass addSublayer:portal];
	[host addSublayer:glass];
	return host;
}

/* Two discs joined by a bar as glyph-like coverage, `w` by `h` pixels, in a
 * 10-bit RGB image whose two high bits mark covered pixels, as SwiftUI hands
 * glass text its glyphs: Core Animation takes such contents as data, not
 * color. */
static CGImageRef ca_glyphs10(size_t w, size_t h)
{
	CGContextRef context = CGBitmapContextCreate(NULL, w, h, 8, w, NULL, (CGBitmapInfo)kCGImageAlphaOnly);
	CGContextSetRGBFillColor(context, 0, 0, 0, 1);
	CGContextFillEllipseInRect(context, CGRectMake(4, 4, h - 8, h - 8));
	CGContextFillEllipseInRect(context, CGRectMake(w - h + 4, 4, h - 8, h - 8));
	CGContextFillRect(context, CGRectMake(h / 2, h / 2 - 5, w - h, 10));
	const uint8_t *coverage = CGBitmapContextGetData(context);
	size_t stride = CGBitmapContextGetBytesPerRow(context);
	NSMutableData *pixels = [NSMutableData dataWithLength:w * h * 4];
	uint32_t *p = pixels.mutableBytes;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++) {
			uint32_t v = (coverage[y * stride + x] * 1023u + 127) / 255;
			p[y * w + x] = (v ? 3u << 30 : 0) | v << 20 | v << 10 | v;
		}
	CGContextRelease(context);
	CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	CGDataProviderRef data = CGDataProviderCreateWithCFData((__bridge CFDataRef)pixels);
	CGImageRef image = CGImageCreate(w, h, 10, 32, w * 4, srgb,
					 kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little | kCGImagePixelFormatRGB101010,
					 data, NULL, true, kCGRenderingIntentDefault);
	CGDataProviderRelease(data);
	CGColorSpaceRelease(srgb);
	if (!image)
		@throw failure(@"no 10-bit image");
	return image;
}

/* A vibrant color matrix filter of row-major `m`, aware of the backdrop or
 * not, clamping at `clamp`. */
static id ca_vibrant(const float m[20], BOOL aware, float clamp)
{
	id filter = ca_filter(@"vibrantColorMatrix");
	[filter setValue:[NSValue valueWithBytes:m objCType:"{CAColorMatrix=ffffffffffffffffffff}"] forKey:@"inputColorMatrix"];
	if (aware)
		[filter setValue:@YES forKey:@"inputBackdropAware"];
	[filter setValue:@(clamp) forKey:@"inputClamp"];
	return filter;
}

/* Glass text as SwiftUI builds it for large glass lettering such as the
 * login window's clock: a backdrop layer whose glass filter refracts by the
 * SDF of an element's 10-bit glyph contents, then, as `f` lists in "parts",
 * a gradient "fill" of that SDF and a key and fill "highlight" of it, each
 * seen through a portal of the element and lifted by a vibrant color
 * matrix. The fill writes the glyphs' distances to a memoryless attachment
 * and the highlight reads them back in the same pass. */
static CALayer *ca_glass_text(NSDictionary *f, CGRect frame)
{
	NSArray *parts = f[@"parts"];
	CALayer *group = [CALayer layer];
	group.frame = frame;
	CGRect b = group.bounds;
	CALayer *backdrop = [NSClassFromString(@"CABackdropLayer") layer];
	backdrop.frame = b;
	backdrop.name = @"@0";
	[backdrop setValue:@YES forKey:@"enabled"];
	[backdrop setValue:@YES forKey:@"sourceSublayerDefinesBounds"];
	[backdrop setValue:@0.5 forKey:@"marginWidth"];
	[backdrop setValue:@0.25 forKey:@"scale"];
	CALayer *shape = [NSClassFromString(@"CASDFLayer") layer];
	shape.frame = b;
	shape.name = @"@0";
	[shape setValue:[NSClassFromString(@"CASDFOutputEffect") new] forKey:@"effect"];
	/* The glyphs' container, which the portals show. */
	CALayer *glyphs = [CALayer layer];
	CALayer *element = [NSClassFromString(@"CASDFElementLayer") layer];
	element.frame = CGRectInset(b, -2, -2);
	element.contentsScale = 2;
	CGImageRef contents = ca_glyphs10((size_t)(element.frame.size.width * 2), (size_t)(element.frame.size.height * 2));
	element.contents = (__bridge id)contents;
	CGImageRelease(contents);
	[element setValue:@"contents" forKey:@"mode"];
	[element setValue:@1.5 forKey:@"contentsZeroValueDistance"];
	[element setValue:@-24 forKey:@"contentsOneValueDistance"];
	[glyphs addSublayer:element];
	[shape addSublayer:glyphs];
	[backdrop addSublayer:shape];
	id glass = ca_filter(@"glassBackground");
	NSDictionary *values = @{
		@"inputSourceSublayerName" : @"@0", @"inputInnerRefractionAmount" : @-57.7, @"inputInnerRefractionHeight" : @6,
		@"inputRefractionDistance0" : @-1, @"inputRefractionDistance1" : @10, @"inputBlurRadius" : @3,
		@"inputBlurOpacity0" : @0.5, @"inputBlurOpacity1" : @1, @"inputBlurOpacity2" : @0.5, @"inputBlurOpacity3" : @1,
		@"inputBlurDistance0" : @-24, @"inputBlurDistance1" : @-1, @"inputFaceOpacity" : @0,
		@"inputFaceColorMatrixBlack" : @0.5, @"inputMaxHeadroom" : @1.2, @"inputClamp" : @1,
		@"inputKeyFillHighlightHeight" : @1, @"inputKeyFillHighlightSpread" : @(M_PI / 2),
		@"inputKeyFillHighlightColorBias" : @-0.2,
	};
	for (NSString *key in values)
		[glass setValue:values[key] forKey:key];
	backdrop.filters = @[ glass ];
	[group addSublayer:backdrop];
	CGColorSpaceRef extended = CGColorSpaceCreateWithName(kCGColorSpaceExtendedSRGB);
	CGFloat one[4] = {1, 1, 1, 1};
	CGColorRef white = CGColorCreate(extended, one);
	if ([parts containsObject:@"fill"]) {
		CALayer *fill = [CALayer layer];
		fill.frame = b;
		fill.name = @"@1";
		CALayer *gradient = [NSClassFromString(@"CASDFLayer") layer];
		gradient.frame = b;
		gradient.name = @"@0";
		id effect = [NSClassFromString(@"CASDFGradientEffect") new];
		[effect setValue:@[ (__bridge id)white, (__bridge id)white ] forKey:@"colors"];
		[effect setValue:@[ @-1, @0, @0 ] forKey:@"distances"];
		[gradient setValue:effect forKey:@"effect"];
		const float lift[20] = {1.389f, -0.372f, -0.037f, 0, 0, -0.11f, 1.2f, -0.09f, 0, 0,
					-0.11f, -0.37f, 1.48f, 0, 0, 0, 0, 0, 1, 0};
		gradient.filters = @[ ca_vibrant(lift, YES, 1.2f) ];
		CALayer *portal = [NSClassFromString(@"CAPortalLayer") layer];
		[portal setValue:glyphs forKey:@"sourceLayer"];
		[gradient addSublayer:portal];
		[fill addSublayer:gradient];
		[group addSublayer:fill];
	}
	if ([parts containsObject:@"highlight"]) {
		CALayer *highlight = [NSClassFromString(@"CASDFLayer") layer];
		highlight.frame = b;
		highlight.name = @"@2";
		id effect = [NSClassFromString(@"CASDFKeyFillHighlightEffect") new];
		[effect setValue:(__bridge id)white forKey:@"keyColor"];
		[effect setValue:(__bridge id)white forKey:@"fillColor"];
		NSDictionary *light = @{
			@"keyHeight" : @0.75, @"keyAngle" : @0, @"keySpread" : @0.8727, @"keyAmount" : @0.5,
			@"fillHeight" : @0.75, @"fillAngle" : @(M_PI), @"fillSpread" : @0.8727, @"fillAmount" : @0.5,
			@"curvature" : @0.7,
		};
		for (NSString *key in light)
			[effect setValue:light[key] forKey:key];
		[highlight setValue:effect forKey:@"effect"];
		const float lift[20] = {1.325f, -0.477f, -0.048f, 0, 0, -0.14f, 1.26f, -0.12f, 0, 0,
					-0.14f, -0.48f, 1.62f, 0, 0, 0, 0, 0, 1, 0};
		highlight.filters = @[ ca_vibrant(lift, NO, 1.2f) ];
		CALayer *portal = [NSClassFromString(@"CAPortalLayer") layer];
		[portal setValue:glyphs forKey:@"sourceLayer"];
		[highlight addSublayer:portal];
		[group addSublayer:highlight];
	}
	CGColorRelease(white);
	CGColorSpaceRelease(extended);
	return group;
}

/* The layer `scene` names, over the gradient: a translucent orange rounded
 * panel, which no blend mode leaves the gradient under, alone or compositing
 * by `compositing`, a backdrop layer of the same shape (ca_backdrop), a
 * button's label when the scene has `text` (ca_text), or glass lettering
 * when it has `glass_text` (ca_glass_text). */
static CALayer *ca_scene(NSDictionary *f)
{
	CGRect frame = CGRectMake(12, 20, 40, 24);
	if (f[@"text"])
		return ca_text(f[@"text"], f);
	if (f[@"glass_text"])
		return ca_glass_text(f[@"glass_text"], CGRectMake(8, 6, 144, 36));
	if (f[@"backdrop"])
		return ca_backdrop(f, frame);
	CALayer *layer = [CALayer layer];
	layer.frame = frame;
	layer.cornerRadius = 8;
	CGColorRef orange = CGColorCreateGenericRGB(0.9, 0.3, 0.1, 0.7);
	layer.backgroundColor = orange;
	CGColorRelease(orange);
	NSDictionary *compositing = f[@"compositing"];
	if (compositing) {
		id filter = ca_filter(compositing[@"type"]);
		if (compositing[@"matrix"]) {
			float m[20];
			for (int i = 0; i < 20; i++)
				m[i] = [compositing[@"matrix"][i] floatValue];
			[filter setValue:[NSValue valueWithBytes:m objCType:"{CAColorMatrix=ffffffffffffffffffff}"]
				  forKey:@"inputColorMatrix"];
		}
		layer.compositingFilter = filter;
	}
	if (f[@"filter"])
		layer.filters = @[ ca_filter(f[@"filter"]) ];
	/* A drop shadow, of the layer's own shape or of a path, as windows and
	 * panels cast them. */
	NSDictionary *shadow = f[@"shadow"];
	if (shadow) {
		CGColorRef black = CGColorCreateGenericRGB(0, 0, 0, 1);
		layer.shadowColor = black;
		CGColorRelease(black);
		layer.shadowOpacity = [shadow[@"opacity"] floatValue];
		layer.shadowRadius = [shadow[@"radius"] doubleValue];
		layer.shadowOffset = CGSizeMake([shadow[@"offset"][0] doubleValue], [shadow[@"offset"][1] doubleValue]);
		if ([shadow[@"path"] boolValue]) {
			CGPathRef path = CGPathCreateWithRoundedRect(layer.bounds, 8, 8, NULL);
			layer.shadowPath = path;
			CGPathRelease(path);
		}
	}
	return layer;
}

/* An AppKit push button of the glass bezel, "Get Started", in a borderless
 * window that is never shown, over a plain blue: AppKit hosts it in SwiftUI,
 * which draws its label into an A8 backing store and shows it through a
 * portal above the glass. The window's content layer, which the caller
 * renders. The runner links AppKit, whose fonts must bridge Core Text's
 * from launch, and starts its application only here. A process outside a
 * window server session that can still reach the server, such as one
 * launchctl bsexec places in the login window's bootstrap, makes its window;
 * one that reaches none makes no window, and the case is unsupported.
 * Glass and its label take their colors from the appearance, which would
 * otherwise follow the user's light or dark setting, so the case names it:
 * `dark` for the dark one, else the light one a Mac shows before any user
 * has chosen. */
static CALayer *appkit_button(CGSize size, BOOL dark)
{
	CFDictionaryRef session = CGSessionCopyCurrentDictionary();
	mach_port_t server = MACH_PORT_NULL;
	if (session)
		CFRelease(session);
	else if (bootstrap_look_up(bootstrap_port, "com.apple.windowserver.active", &server) != KERN_SUCCESS)
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"no window server session" userInfo:nil];
	else
		mach_port_deallocate(mach_task_self(), server);
	id (*send)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
	id app = send(NSClassFromString(@"NSApplication"), sel_registerName("sharedApplication"));
	id appearance = ((id (*)(id, SEL, id))objc_msgSend)(
		NSClassFromString(@"NSAppearance"), sel_registerName("appearanceNamed:"),
		dark ? @"NSAppearanceNameDarkAqua" : @"NSAppearanceNameAqua");
	((void (*)(id, SEL, id))objc_msgSend)(app, sel_registerName("setAppearance:"), appearance);
	id button = ((id (*)(id, SEL, id, id, SEL))objc_msgSend)(NSClassFromString(@"NSButton"),
								  sel_registerName("buttonWithTitle:target:action:"),
								  @"Get Started", nil, NULL);
	/* NSBezelStyleGlass. */
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(button, sel_registerName("setBezelStyle:"), 16);
	CGRect bounds = CGRectMake(0, 0, size.width, size.height);
	id view = ((id (*)(id, SEL, CGRect))objc_msgSend)(send(NSClassFromString(@"NSView"), @selector(alloc)),
							  sel_registerName("initWithFrame:"), bounds);
	((void (*)(id, SEL, BOOL))objc_msgSend)(view, sel_registerName("setWantsLayer:"), YES);
	((void (*)(id, SEL, id))objc_msgSend)(view, sel_registerName("addSubview:"), button);
	((void (*)(id, SEL, CGRect))objc_msgSend)(button, sel_registerName("setFrame:"),
						  CGRectMake((size.width - 106) / 2, (size.height - 36) / 2, 106, 36));
	/* NSWindowStyleMaskBorderless, NSBackingStoreBuffered, deferred. */
	id window = ((id (*)(id, SEL, CGRect, NSUInteger, NSUInteger, BOOL))objc_msgSend)(
		send(NSClassFromString(@"NSWindow"), @selector(alloc)),
		sel_registerName("initWithContentRect:styleMask:backing:defer:"), bounds, 0, 2, YES);
	((void (*)(id, SEL, id))objc_msgSend)(window, sel_registerName("setAppearance:"), appearance);
	((void (*)(id, SEL, id))objc_msgSend)(window, sel_registerName("setContentView:"), view);
	CALayer *layer = send(view, sel_registerName("layer"));
	CGColorRef blue = CGColorCreateGenericRGB(0.3, 0.5, 0.7, 1);
	layer.backgroundColor = blue;
	CGColorRelease(blue);
	/* AppKit lays the button out and SwiftUI draws it over a few turns of
	 * the run loop. */
	for (int turn = 0; turn < 10; turn++) {
		send(view, sel_registerName("layoutSubtreeIfNeeded"));
		send(view, sel_registerName("displayIfNeeded"));
		[CATransaction flush];
		CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.02, false);
	}
	return layer;
}

static IOSurfaceRef make_surface(NSDictionary *properties);
static id<MTLTexture> surface_texture(Case *c, IOSurfaceRef s, size_t plane, MTLPixelFormat format, MTLTextureUsage usage);

/* Core Animation's renderer drawing a scene into the case's `out`, on the
 * case's queue. Core Animation builds pipelines its archive lacks while it
 * draws and leaves their layers out until they are ready, and glass eases
 * toward the luma beneath it, so frames repeat until two in a row are
 * identical. Between frames the run loop turns, which AppKit's and
 * SwiftUI's updates wait on. */
static void run_core_animation(Case *c, id<MTLCommandQueue> queue, NSDictionary *f)
{
	id<MTLTexture> out = c.textures[@"out"];
	if (!out)
		@throw failure(@"core animation draws into an `out` texture");
	/* With "surface", Core Animation draws into a texture over an
	 * IOSurface, as a window server draws into its displays' framebuffers,
	 * whose texels are linear rows in shared memory; `out` takes a copy. */
	id<MTLTexture> target = out;
	if ([f[@"surface"] boolValue]) {
		IOSurfaceRef surface = make_surface(@{
			(__bridge id)kIOSurfaceWidth : @(out.width), (__bridge id)kIOSurfaceHeight : @(out.height),
			(__bridge id)kIOSurfaceBytesPerElement : @4, (__bridge id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
			(__bridge id)kIOSurfaceBytesPerRow : @(IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, out.width * 4)),
		});
		target = surface_texture(c, surface, 0, out.pixelFormat, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
		CFRelease(surface);
	}
	CARenderer *renderer = [CARenderer rendererWithMTLTexture:target options:@{kCARendererMetalCommandQueue : queue}];
	CALayer *root;
	if ([f[@"appkit_button"] boolValue]) {
		root = appkit_button(CGSizeMake(out.width, out.height), [f[@"appearance"] isEqual:@"dark"]);
	} else {
		root = [CALayer layer];
		root.frame = CGRectMake(0, 0, out.width, out.height);
		CALayer *ground = [CALayer layer];
		ground.frame = root.bounds;
		CGImageRef gradient = ca_gradient(out.width, out.height);
		ground.contents = (__bridge id)gradient;
		CGImageRelease(gradient);
		[root addSublayer:ground];
		[root addSublayer:ca_scene(f)];
	}
	renderer.layer = root;
	renderer.bounds = root.bounds;
	[CATransaction flush];
	NSUInteger bytes = out.width * out.height * 4;
	id<MTLBuffer> frames[2] = {[c.device newBufferWithLength:bytes options:MTLResourceStorageModeShared],
				   [c.device newBufferWithLength:bytes options:MTLResourceStorageModeShared]};
	for (unsigned frame = 0;; frame++) {
		[renderer beginFrameAtTime:CACurrentMediaTime() timeStamp:NULL];
		[renderer addUpdateRect:renderer.bounds];
		[renderer render];
		[renderer endFrame];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			   sourceSize:MTLSizeMake(out.width, out.height, 1) toBuffer:frames[frame % 2]
		    destinationOffset:0 destinationBytesPerRow:out.width * 4 destinationBytesPerImage:bytes];
		if (target != out)
			[blit copyFromTexture:target toTexture:out];
		[blit endEncoding];
		wait_for(cb, c.spec);
		if (frame > 0 && !memcmp(frames[0].contents, frames[1].contents, bytes))
			return;
		/* Glass adapts to the luma beneath it over about 25 frames. */
		if (frame == 47)
			@throw failure(@"core animation's frames never settled");
		if (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.25, false) == kCFRunLoopRunFinished)
			usleep(250000);
		[CATransaction flush];
	}
}

/* A push button's label set by Core Text on the CPU into a bitmap context
 * of `out`'s size, as text reaches every layer's backing store: in BGRA over
 * clear, over opaque white with font smoothing, or alpha only (`f`'s
 * "format"). The bitmap's rows are copied into `out` as they are. */
static void run_core_text(Case *c, id<MTLCommandQueue> queue, NSDictionary *f)
{
	id<MTLTexture> out = c.textures[@"out"];
	if (!out)
		@throw failure(@"core text draws into an `out` texture");
	size_t w = out.width, h = out.height;
	NSString *format = f[@"format"];
	BOOL alpha = [format isEqualToString:@"alpha"], smoothed = [format isEqualToString:@"smoothed"];
	size_t row = alpha ? w : 4 * w;
	CGColorSpaceRef space = alpha ? NULL : CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	CGContextRef context = CGBitmapContextCreate(
		NULL, w, h, 8, row, space,
		alpha ? (CGBitmapInfo)kCGImageAlphaOnly
		      : (CGBitmapInfo)kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
	CGColorSpaceRelease(space);
	if (!context)
		@throw failure(@"no bitmap context of format %@", format);
	if (smoothed) {
		CGContextSetRGBFillColor(context, 1, 1, 1, 1);
		CGContextFillRect(context, CGRectMake(0, 0, w, h));
	}
	CGContextSetAllowsFontSmoothing(context, smoothed);
	CGContextSetShouldSmoothFonts(context, smoothed);
	/* Two pixels a point, as on a Retina display. */
	CGContextScaleCTM(context, 2, 2);
	draw_label(context, 13, CGPointMake(4, 6));
	size_t stride = CGBitmapContextGetBytesPerRow(context);
	id<MTLBuffer> bitmap = [c.device newBufferWithBytes:CGBitmapContextGetData(context)
						     length:stride * h
						    options:MTLResourceStorageModeShared];
	CGContextRelease(context);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit copyFromBuffer:bitmap sourceOffset:0 sourceBytesPerRow:stride sourceBytesPerImage:stride * h
		  sourceSize:MTLSizeMake(w, h, 1) toTexture:out destinationSlice:0 destinationLevel:0
	   destinationOrigin:MTLOriginMake(0, 0, 0)];
	[blit endEncoding];
	wait_for(cb, c.spec);
}

/* A desktop picture as an app hands it on: a gradient under a checker,
 * opaque, of an odd size, with its alpha last and not premultiplied, as
 * AppKit's images give it: 8-bit sRGB, or 16-bit linear sRGB (`wide`). */
static CGImageRef desktop_picture(size_t w, size_t h, BOOL wide)
{
	size_t bytes = wide ? 8 : 4;
	NSMutableData *pixels = [NSMutableData dataWithLength:w * h * bytes];
	uint8_t *p = pixels.mutableBytes;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++, p += bytes) {
			uint8_t v[4] = {(uint8_t)(x * 255 / (w - 1)), (uint8_t)(y * 255 / (h - 1)), ((x / 4 + y / 4) & 1) ? 224 : 32, 255};
			for (int k = 0; k < 4; k++) {
				if (wide)
					((uint16_t *)p)[k] = (uint16_t)(v[k] * 257);
				else
					p[k] = v[k];
			}
		}
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)pixels);
	CGColorSpaceRef space = CGColorSpaceCreateWithName(wide ? kCGColorSpaceLinearSRGB : kCGColorSpaceSRGB);
	CGBitmapInfo info = (CGBitmapInfo)kCGImageAlphaLast | (wide ? kCGBitmapByteOrder16Little : 0);
	CGImageRef image = CGImageCreate(w, h, wide ? 16 : 8, bytes * 8, w * bytes, space, info, provider, NULL, false,
					 kCGRenderingIntentDefault);
	CGColorSpaceRelease(space);
	CGDataProviderRelease(provider);
	return image;
}

/* The image file of `type` ImageIO writes for `image`, which it releases. */
static NSData *image_file(CGImageRef image, CFStringRef type)
{
	NSMutableData *file = [NSMutableData data];
	CGImageDestinationRef destination = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)file, type, 1, NULL);
	CGImageDestinationAddImage(destination, image, NULL);
	BOOL encoded = CGImageDestinationFinalize(destination);
	CFRelease(destination);
	CGImageRelease(image);
	if (!encoded)
		@throw failure(@"ImageIO wrote no %@", (__bridge NSString *)type);
	return file;
}

/* A desktop picture loaded by MTKTextureLoader with sRGB and generated
 * mipmaps: the 8-bit one as TIFF, as AppKit's TIFFRepresentation gives it,
 * or the 16-bit linear one as PNG (`f`'s "bits"), which MetalKit loads as
 * RGBA16Unorm. The out buffer holds the texture's pixel format, level
 * count, width and height (32 bits each), then every level's texels packed
 * at 4 or 8 bytes. */
static void run_texture_loader(Case *c, id<MTLCommandQueue> queue, NSDictionary *f)
{
	id<MTLBuffer> out = c.buffers[@"out"];
	size_t w = number(f, @"width", 1), h = number(f, @"height", 1);
	BOOL wide = number(f, @"bits", 8) == 16;
	NSData *file = image_file(desktop_picture(w, h, wide), wide ? CFSTR("public.png") : CFSTR("public.tiff"));
	NSError *error = nil;
	MTKTextureLoader *loader = [[MTKTextureLoader alloc] initWithDevice:c.device];
	id<MTLTexture> t = [loader newTextureWithData:file
					      options:@{MTKTextureLoaderOptionGenerateMipmaps : @YES, MTKTextureLoaderOptionSRGB : @YES}
						error:&error];
	if (!t)
		@throw failure(@"MetalKit loaded no texture: %@", error);
	uint32_t header[4] = {(uint32_t)t.pixelFormat, (uint32_t)t.mipmapLevelCount, (uint32_t)t.width, (uint32_t)t.height};
	memcpy(out.contents, header, sizeof header);
	NSUInteger at = sizeof header, bytes = wide ? 8 : 4;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
		NSUInteger lw = MAX(t.width >> level, 1u), lh = MAX(t.height >> level, 1u);
		if (at + lw * lh * bytes > out.length)
			@throw failure(@"MetalKit's texture (format %lu, %lu levels) outgrows the out buffer", (unsigned long)t.pixelFormat,
				       (unsigned long)t.mipmapLevelCount);
		[blit copyFromTexture:t sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0)
			   sourceSize:MTLSizeMake(lw, lh, 1) toBuffer:out destinationOffset:at
		    destinationBytesPerRow:lw * bytes destinationBytesPerImage:lw * lh * bytes];
		at += lw * lh * bytes;
	}
	[blit endEncoding];
	wait_for(cb, c.spec);
}

/* The 8-bit desktop picture's TIFF as Core Image renders it into memory
 * for a texture: a bitmap destination of 16-bit RGBA in linear sRGB, alpha
 * not premultiplied. Core Image converts on the device, into a surface it
 * then copies to the bitmap, the case's out buffer. */
static void run_core_image_bitmap(Case *c, NSDictionary *f)
{
	id<MTLBuffer> out = c.buffers[@"out"];
	size_t w = number(f, @"width", 1), h = number(f, @"height", 1);
	if (out.length < w * h * 8)
		@throw failure(@"the out buffer holds no %zux%zu RGBA16 bitmap", w, h);
	CIImage *image = [CIImage imageWithData:image_file(desktop_picture(w, h, NO), CFSTR("public.tiff"))];
	if (!image)
		@throw failure(@"Core Image read no TIFF");
	CIContext *context = [CIContext contextWithMTLDevice:c.device];
	CIRenderDestination *destination = [[CIRenderDestination alloc] initWithBitmapData:out.contents
										    width:w
										   height:h
									      bytesPerRow:w * 8
										   format:kCIFormatRGBA16];
	CGColorSpaceRef linear = CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB);
	destination.colorSpace = linear;
	CGColorSpaceRelease(linear);
	destination.alphaMode = CIRenderDestinationAlphaUnpremultiplied;
	NSError *error = nil;
	CIRenderTask *task = [context startTaskToRender:image toDestination:destination error:&error];
	if (!task || ![task waitUntilCompletedAndReturnError:&error])
		@throw failure(@"Core Image rendered no bitmap: %@", error);
}

/* Work Apple's frameworks encode with their own shaders: an MPS matrix
 * product and box filter, an MPSGraph fused kernel, MetalFX's spatial and
 * temporal upscalers, a Core Image filter and a Core Animation scene, each
 * over the case's resources; and text Core Text sets on the CPU. */
static void run_framework(Case *c, id<MTLCommandQueue> queue, NSDictionary *spec)
{
	NSDictionary *f = spec[@"framework"];
	NSString *kind = f[@"kind"];
	if ([kind isEqualToString:@"core_animation"]) {
		run_core_animation(c, queue, f);
		return;
	}
	if ([kind isEqualToString:@"core_text"]) {
		run_core_text(c, queue, f);
		return;
	}
	if ([kind isEqualToString:@"texture_loader"]) {
		run_texture_loader(c, queue, f);
		return;
	}
	if ([kind isEqualToString:@"core_image_bitmap"]) {
		run_core_image_bitmap(c, f);
		return;
	}
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	if ([kind hasPrefix:@"mps"] && !MPSSupportsMTLDevice(c.device))
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"MPS does not support the device" userInfo:nil];
	if ([kind isEqualToString:@"mps_matrix_multiply"]) {
		NSUInteger m = number(f, @"rows", 1), n = number(f, @"columns", 1), k = number(f, @"interior", 1);
		MPSMatrix *(^matrix)(NSString *, NSUInteger, NSUInteger) = ^(NSString *name, NSUInteger rows, NSUInteger columns) {
			MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
										  rowBytes:columns * sizeof(float)
										  dataType:MPSDataTypeFloat32];
			return [[MPSMatrix alloc] initWithBuffer:c.buffers[name] descriptor:d];
		};
		MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:c.device transposeLeft:NO transposeRight:NO
										   resultRows:m resultColumns:n interiorColumns:k
											alpha:1.0 beta:0.0];
		[mm encodeToCommandBuffer:cb leftMatrix:matrix(@"a", m, k) rightMatrix:matrix(@"b", k, n) resultMatrix:matrix(@"c", m, n)];
	} else if ([kind isEqualToString:@"mps_gaussian_blur"]) {
		/* Edges read zero, as MPS's default edge mode. */
		MPSImageGaussianBlur *blur = [[MPSImageGaussianBlur alloc] initWithDevice:c.device sigma:[f[@"sigma"] floatValue]];
		blur.edgeMode = MPSImageEdgeModeZero;
		[blur encodeToCommandBuffer:cb sourceTexture:c.textures[@"in"] destinationTexture:c.textures[@"out"]];
	} else if ([kind isEqualToString:@"mps_image_box"]) {
		MPSImageBox *box = [[MPSImageBox alloc] initWithDevice:c.device kernelWidth:3 kernelHeight:3];
		box.edgeMode = MPSImageEdgeModeClamp;
		[box encodeToCommandBuffer:cb sourceTexture:c.textures[@"in"] destinationTexture:c.textures[@"out"]];
	} else if ([kind isEqualToString:@"mps_graph_fused"]) {
		/* MPSGraph's elementwise operations, relu(sin(a * b + offset)),
		 * which it fuses into one kernel stitched from its own visible
		 * functions. */
		MPSShape *shape = @[ @(number(f, @"count", 1)) ];
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor *a = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"a"];
		MPSGraphTensor *b = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"b"];
		MPSGraphTensor *offset = [graph constantWithScalar:[f[@"offset"] doubleValue] dataType:MPSDataTypeFloat32];
		MPSGraphTensor *sum = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]
						       secondaryTensor:offset
								  name:nil];
		MPSGraphTensor *result = [graph reLUWithTensor:[graph sinWithTensor:sum name:nil] name:nil];
		MPSGraphTensorData *(^data)(NSString *) = ^(NSString *name) {
			return [[MPSGraphTensorData alloc] initWithMTLBuffer:c.buffers[name] shape:shape dataType:MPSDataTypeFloat32];
		};
		[graph runWithMTLCommandQueue:queue
				       feeds:@{a : data(@"a"), b : data(@"b")}
			    targetOperations:nil
			   resultsDictionary:@{result : data(@"out")}];
	} else if ([kind isEqualToString:@"metalfx_spatial"]) {
		/* MetalFX's spatial upscaler, from the input's size to the output's. */
		if (![MTLFXSpatialScalerDescriptor supportsDevice:c.device])
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"MetalFX has no spatial scaler for the device" userInfo:nil];
		id<MTLTexture> in = c.textures[@"in"], out = c.textures[@"out"];
		MTLFXSpatialScalerDescriptor *d = [MTLFXSpatialScalerDescriptor new];
		d.inputWidth = in.width;
		d.inputHeight = in.height;
		d.outputWidth = out.width;
		d.outputHeight = out.height;
		d.colorTextureFormat = in.pixelFormat;
		d.outputTextureFormat = out.pixelFormat;
		d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
		id<MTLFXSpatialScaler> scaler = [d newSpatialScalerWithDevice:c.device];
		if (!scaler)
			@throw failure(@"MetalFX made no spatial scaler");
		scaler.colorTexture = in;
		scaler.outputTexture = out;
		scaler.inputContentWidth = in.width;
		scaler.inputContentHeight = in.height;
		[scaler encodeToCommandBuffer:cb];
	} else if ([kind isEqualToString:@"metalfx_temporal"]) {
		/* MetalFX's temporal upscaler on its first frame: no jitter, no
		 * motion and no history, so the output is the color upscaled.
		 * MetalFX makes one only for a device in Apple7. */
		if (![MTLFXTemporalScalerDescriptor supportsDevice:c.device])
			@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"MetalFX has no temporal scaler for the device" userInfo:nil];
		id<MTLTexture> color = c.textures[@"color"], depth = c.textures[@"depth"];
		id<MTLTexture> motion = c.textures[@"motion"], out = c.textures[@"out"];
		MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
		d.inputWidth = color.width;
		d.inputHeight = color.height;
		d.outputWidth = out.width;
		d.outputHeight = out.height;
		d.colorTextureFormat = color.pixelFormat;
		d.depthTextureFormat = depth.pixelFormat;
		d.motionTextureFormat = motion.pixelFormat;
		d.outputTextureFormat = out.pixelFormat;
		id<MTLFXTemporalScaler> scaler = [d newTemporalScalerWithDevice:c.device];
		if (!scaler)
			@throw failure(@"MetalFX made no temporal scaler");
		scaler.colorTexture = color;
		scaler.depthTexture = depth;
		scaler.motionTexture = motion;
		scaler.outputTexture = out;
		scaler.inputContentWidth = color.width;
		scaler.inputContentHeight = color.height;
		scaler.jitterOffsetX = 0;
		scaler.jitterOffsetY = 0;
		scaler.motionVectorScaleX = 1;
		scaler.motionVectorScaleY = 1;
		scaler.reset = YES;
		[scaler encodeToCommandBuffer:cb];
	} else if ([kind isEqualToString:@"core_image_invert"]) {
		/* No colour management: the filter sees the texels as stored. */
		CIContext *context = [CIContext contextWithMTLDevice:c.device
							     options:@{kCIContextWorkingColorSpace : [NSNull null],
								       kCIContextOutputColorSpace : [NSNull null]}];
		CIImage *image = [CIImage imageWithMTLTexture:c.textures[@"in"] options:@{kCIImageColorSpace : [NSNull null]}];
		CIFilter *invert = [CIFilter filterWithName:@"CIColorInvert"];
		[invert setValue:image forKey:kCIInputImageKey];
		id<MTLTexture> out = c.textures[@"out"];
		/* A null output space skips colour matching, though the header marks it nonnull. */
		CGColorSpaceRef unmatched = NULL;
		[context render:invert.outputImage toMTLTexture:out commandBuffer:cb
			 bounds:CGRectMake(0, 0, out.width, out.height) colorSpace:unmatched];
	} else if ([kind isEqualToString:@"core_image_lanczos"]) {
		/* Core Image's Lanczos scaler, of the input or of it mirrored, which
		 * Core Image samples through an affine transform: its stitched
		 * kernels pass a sampler one library builds to another's filters
		 * (ci_lanczosDownH_affine). No colour management. */
		CIContext *context = [CIContext contextWithMTLDevice:c.device
							     options:@{kCIContextWorkingColorSpace : [NSNull null],
								       kCIContextOutputColorSpace : [NSNull null]}];
		CIImage *image = [CIImage imageWithMTLTexture:c.textures[@"in"] options:@{kCIImageColorSpace : [NSNull null]}];
		if ([f[@"mirrored"] boolValue])
			image = [image imageByApplyingTransform:CGAffineTransformMakeScale(-1, 1)];
		CIFilter *lanczos = [CIFilter filterWithName:@"CILanczosScaleTransform"];
		[lanczos setValue:image forKey:kCIInputImageKey];
		[lanczos setValue:f[@"scale"] forKey:kCIInputScaleKey];
		[lanczos setValue:@1.0 forKey:kCIInputAspectRatioKey];
		CIImage *scaled = lanczos.outputImage;
		id<MTLTexture> out = c.textures[@"out"];
		CGColorSpaceRef unmatched = NULL;
		[context render:scaled toMTLTexture:out commandBuffer:cb
			 bounds:CGRectMake(floor(scaled.extent.origin.x), floor(scaled.extent.origin.y), out.width, out.height)
		     colorSpace:unmatched];
	} else {
		@throw failure(@"unknown framework work %@", kind);
	}
	wait_for(cb, spec);
}

/*
 * Host work around the device: drawables of a layer, IOSurface-backed
 * textures, textures and events another process opens, binary archives,
 * the errors creation calls report, command buffer status and order,
 * purgeable state, and work at scale. Each writes what it observed into
 * the case's buffers as plain numbers; the plan holds the expectations.
 */
static void put(Case *c, NSString *buffer, NSUInteger index, uint32_t value)
{
	id<MTLBuffer> b = c.buffers[buffer];
	if (!b || (index + 1) * sizeof(uint32_t) > b.length)
		@throw failure(@"no room for observation %lu in %@", (unsigned long)index, buffer);
	((uint32_t *)b.contents)[index] = value;
}

static id<MTLComputePipelineState> host_pipeline(Case *c, NSString *function)
{
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	NSError *error = nil;
	id<MTLComputePipelineState> p = [c.device newComputePipelineStateWithFunction:function_named(library, function, nil)
										error:&error];
	if (!p)
		@throw failure(@"pipeline %@: %@", function, error.localizedDescription);
	return p;
}

static void dispatch_one(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, NSArray<id<MTLBuffer>> *buffers,
			 uint32_t constant, MTLSize threads)
{
	[e setComputePipelineState:p];
	for (NSUInteger i = 0; i < buffers.count; i++)
		[e setBuffer:buffers[i] offset:0 atIndex:i];
	[e setBytes:&constant length:sizeof constant atIndex:buffers.count];
	[e dispatchThreads:threads threadsPerThreadgroup:MTLSizeMake(MIN(threads.width, 64), 1, 1)];
}

static id<MTLRenderPipelineState> host_render_pipeline(Case *c, MTLRenderPipelineDescriptor *d, NSString *vertex,
						       NSString *fragment, MTLPixelFormat format, NSError **error)
{
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	d.vertexFunction = function_named(library, vertex, nil);
	d.fragmentFunction = function_named(library, fragment, nil);
	d.colorAttachments[0].pixelFormat = format;
	return [c.device newRenderPipelineStateWithDescriptor:d error:error];
}

/* Clears a target, then draws its left half in `color`. */
static void clear_and_draw_left(Case *c, id<MTLCommandBuffer> cb, id<MTLRenderPipelineState> p, id<MTLTexture> t,
				NSArray *clear, NSArray *color)
{
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = t;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.colorAttachments[0].clearColor =
		MTLClearColorMake([clear[0] doubleValue], [clear[1] doubleValue], [clear[2] doubleValue], [clear[3] doubleValue]);
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
	float rgba[4] = {[color[0] floatValue], [color[1] floatValue], [color[2] floatValue], [color[3] floatValue]};
	[e setRenderPipelineState:p];
	[e setFragmentBytes:rgba length:sizeof rgba atIndex:0];
	[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[e endEncoding];
}

/* Drawables of a layer in no window: each rendered, read back unless the
 * layer is framebuffer only, and presented. */
static void host_drawables(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	MTLPixelFormat format = lookup(pixel_formats(), @"pixel format", h[@"format"]);
	NSUInteger w = number(h, @"width", 1), height = number(h, @"height", 1), count = number(h, @"count", 1), block = 1, block_height = 1;
	BOOL framebuffer_only = [h[@"framebuffer_only"] boolValue];
	CAMetalLayer *layer = [CAMetalLayer layer];
	layer.device = c.device;
	layer.pixelFormat = format;
	layer.framebufferOnly = framebuffer_only;
	layer.drawableSize = CGSizeMake(w, height);
	layer.maximumDrawableCount = 3;
	NSError *error = nil;
	id<MTLRenderPipelineState> p =
		host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_left_half", @"frag_constant", format, &error);
	if (!p)
		@throw failure(@"drawable pipeline: %@", error.localizedDescription);
	NSUInteger texel = pixel_bytes(format, &block, &block_height), image = w * height * texel;
	for (NSUInteger i = 0; i < count; i++) {
		id<CAMetalDrawable> drawable = [layer nextDrawable];
		if (!drawable)
			@throw failure(@"no drawable %lu", (unsigned long)i);
		id<MTLTexture> t = drawable.texture;
		NSUInteger at = i * 5;
		put(c, @"props", at, (uint32_t)t.width);
		put(c, @"props", at + 1, (uint32_t)t.height);
		put(c, @"props", at + 2, (uint32_t)t.pixelFormat);
		put(c, @"props", at + 3, (t.usage & MTLTextureUsageRenderTarget) != 0);
		put(c, @"props", at + 4, (t.usage & MTLTextureUsageShaderRead) != 0);
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		clear_and_draw_left(c, cb, p, t, h[@"clears"][i], h[@"color"]);
		if (!framebuffer_only) {
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
				sourceSize:MTLSizeMake(w, height, 1) toBuffer:c.buffers[@"out"] destinationOffset:i * image
			   destinationBytesPerRow:w * texel destinationBytesPerImage:image];
			[b endEncoding];
		}
		/* Presented by the command buffer or by the drawable, at once,
		 * at a time or after a minimum duration; each presentation's
		 * handler, which reads its time, noted. */
		NSArray *modes = h[@"present"];
		NSString *mode = modes.count ? modes[i % modes.count] : @"command_buffer";
		if (modes.count) {
			NSUInteger slot = i;
			[drawable addPresentedHandler:^(id<MTLDrawable> d) {
				(void)d.presentedTime;
				(void)d.drawableID;
				put(c, @"presented", slot, 1);
			}];
		}
		CFTimeInterval soon = CACurrentMediaTime() + 0.001;
		if ([mode isEqualToString:@"command_buffer"])
			[cb presentDrawable:drawable];
		else if ([mode isEqualToString:@"command_buffer_at"])
			[cb presentDrawable:drawable atTime:soon];
		else if ([mode isEqualToString:@"command_buffer_after"])
			[cb presentDrawable:drawable afterMinimumDuration:0.001];
		wait_for(cb, c.spec);
		if ([mode isEqualToString:@"drawable"])
			[drawable present];
		else if ([mode isEqualToString:@"drawable_at"])
			[drawable presentAtTime:soon];
		else if ([mode isEqualToString:@"drawable_after"])
			[drawable presentAfterMinimumDuration:0.001];
	}
	/* Presentation handlers run once the display is done with each. */
	if ([h[@"present"] count])
		usleep(500000);
}

static IOSurfaceRef make_surface(NSDictionary *properties)
{
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
	if (!s)
		@throw failure(@"IOSurfaceCreate refused %@", properties);
	return s;
}

/* Copies tight rows into a surface plane, or out of it, under its lock. */
static void surface_rows(IOSurfaceRef s, size_t plane, NSData *in, NSMutableData *out, size_t row, size_t rows)
{
	IOSurfaceLock(s, in ? 0 : kIOSurfaceLockReadOnly, NULL);
	uint8_t *base = IOSurfaceGetBaseAddressOfPlane(s, plane);
	size_t stride = IOSurfaceGetBytesPerRowOfPlane(s, plane);
	for (size_t y = 0; y < rows; y++) {
		if (in)
			memcpy(base + y * stride, (const uint8_t *)in.bytes + y * row, row);
		else
			memcpy((uint8_t *)out.mutableBytes + y * row, base + y * stride, row);
	}
	IOSurfaceUnlock(s, in ? 0 : kIOSurfaceLockReadOnly, NULL);
}

static id<MTLTexture> surface_texture(Case *c, IOSurfaceRef s, size_t plane, MTLPixelFormat format, MTLTextureUsage usage)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
										     width:IOSurfaceGetWidthOfPlane(s, plane)
										    height:IOSurfaceGetHeightOfPlane(s, plane)
										 mipmapped:NO];
	d.usage = usage;
	d.storageMode = c.device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
	id<MTLTexture> t = [c.device newTextureWithDescriptor:d iosurface:s plane:plane];
	if (!t)
		@throw failure(@"no texture over plane %zu", plane);
	return t;
}

static NSData *contents(id<MTLBuffer> b)
{
	return [NSData dataWithBytes:b.contents length:b.length];
}

/* A BGRA surface the CPU fills and a kernel inverts into a second one the
 * CPU reads; then a two-plane 4:2:0 surface a kernel reads both planes of. */
static void host_iosurface(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	size_t w = number(h, @"width", 1), height = number(h, @"height", 1);
	NSDictionary *bgra = @{
		(__bridge id)kIOSurfaceWidth : @(w), (__bridge id)kIOSurfaceHeight : @(height), (__bridge id)kIOSurfaceBytesPerElement : @4,
		(__bridge id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
		(__bridge id)kIOSurfaceBytesPerRow : @(IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, w * 4)),
	};
	IOSurfaceRef src = make_surface(bgra), dst = make_surface(bgra);
	surface_rows(src, 0, contents(c.buffers[@"in"]), nil, w * 4, height);
	id<MTLTexture> ts = surface_texture(c, src, 0, MTLPixelFormatBGRA8Unorm, MTLTextureUsageShaderRead);
	id<MTLTexture> td = surface_texture(c, dst, 0, MTLPixelFormatBGRA8Unorm, MTLTextureUsageShaderWrite);
	put(c, @"props", 0, ts.iosurface == src);
	put(c, @"props", 1, (uint32_t)ts.iosurfacePlane);
	put(c, @"props", 2, (uint32_t)td.width);
	put(c, @"props", 3, (uint32_t)td.height);

	size_t uw = w / 2, uh = height / 2;
	size_t y_row = IOSurfaceAlignProperty(kIOSurfacePlaneBytesPerRow, w);
	size_t uv_row = IOSurfaceAlignProperty(kIOSurfacePlaneBytesPerRow, uw * 2);
	size_t uv_offset = IOSurfaceAlignProperty(kIOSurfacePlaneOffset, y_row * height);
	NSDictionary *planar = @{
		(__bridge id)kIOSurfaceWidth : @(w), (__bridge id)kIOSurfaceHeight : @(height),
		(__bridge id)kIOSurfacePixelFormat : @((uint32_t)'420v'),
		(__bridge id)kIOSurfaceAllocSize : @(uv_offset + uv_row * uh),
		(__bridge id)kIOSurfacePlaneInfo : @[
			@{(__bridge id)kIOSurfacePlaneWidth : @(w), (__bridge id)kIOSurfacePlaneHeight : @(height),
			  (__bridge id)kIOSurfacePlaneBytesPerElement : @1, (__bridge id)kIOSurfacePlaneBytesPerRow : @(y_row),
			  (__bridge id)kIOSurfacePlaneOffset : @0, (__bridge id)kIOSurfacePlaneSize : @(y_row * height)},
			@{(__bridge id)kIOSurfacePlaneWidth : @(uw), (__bridge id)kIOSurfacePlaneHeight : @(uh),
			  (__bridge id)kIOSurfacePlaneBytesPerElement : @2, (__bridge id)kIOSurfacePlaneBytesPerRow : @(uv_row),
			  (__bridge id)kIOSurfacePlaneOffset : @(uv_offset), (__bridge id)kIOSurfacePlaneSize : @(uv_row * uh)},
		],
	};
	IOSurfaceRef yuv = make_surface(planar);
	surface_rows(yuv, 0, contents(c.buffers[@"luma"]), nil, w, height);
	surface_rows(yuv, 1, contents(c.buffers[@"chroma"]), nil, uw * 2, uh);
	id<MTLTexture> ty = surface_texture(c, yuv, 0, MTLPixelFormatR8Unorm, MTLTextureUsageShaderRead);
	id<MTLTexture> tc = surface_texture(c, yuv, 1, MTLPixelFormatRG8Unorm, MTLTextureUsageShaderRead);

	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:host_pipeline(c, @"surface_invert")];
	[e setTexture:ts atIndex:0];
	[e setTexture:td atIndex:1];
	[e dispatchThreads:MTLSizeMake(w, height, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
	[e setComputePipelineState:host_pipeline(c, @"surface_planes")];
	[e setTexture:ty atIndex:0];
	[e setTexture:tc atIndex:1];
	[e setBuffer:c.buffers[@"planes"] offset:0 atIndex:0];
	[e dispatchThreads:MTLSizeMake(w, height, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
	[e endEncoding];
	if (td.storageMode == MTLStorageModeManaged) {
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b synchronizeResource:td];
		[b endEncoding];
	}
	wait_for(cb, c.spec);
	NSMutableData *out = [NSMutableData dataWithLength:w * 4 * height];
	surface_rows(dst, 0, nil, out, w * 4, height);
	memcpy(c.buffers[@"out"].contents, out.bytes, MIN(out.length, c.buffers[@"out"].length));
	CFRelease(src);
	CFRelease(dst);
	CFRelease(yuv);
}

/* Metal's private calls for a buffer over an IOSurface's memory. */
@protocol SuiteSurfaceBuffers
- (id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)surface;
- (IOSurfaceRef)iosurface;
@end

/* A buffer over a surface's memory, which the CPU fills through the surface;
 * a kernel adds to each word in place, and the CPU reads the sums back
 * through the surface. */
static void host_iosurface_buffer(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	size_t words = number(h, @"words", 1);
	IOSurfaceRef s = make_surface(@{
		(__bridge id)kIOSurfaceWidth : @(words), (__bridge id)kIOSurfaceHeight : @1,
		(__bridge id)kIOSurfaceBytesPerElement : @4,
	});
	surface_rows(s, 0, contents(c.buffers[@"in"]), nil, words * 4, 1);
	if (![c.device respondsToSelector:@selector(newBufferWithIOSurface:)])
		@throw failure(@"the device makes no buffer of an IOSurface");
	id<MTLBuffer> b = [(id<SuiteSurfaceBuffers>)c.device newBufferWithIOSurface:s];
	if (!b)
		@throw failure(@"newBufferWithIOSurface: refused a surface of %zu bytes", IOSurfaceGetAllocSize(s));
	put(c, @"props", 0, [(id<SuiteSurfaceBuffers>)b iosurface] == s);
	put(c, @"props", 1, b.length >= words * 4);
	put(c, @"props", 2, (uint32_t)b.storageMode);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	dispatch_one(e, host_pipeline(c, @"surface_buffer_add"), @[ b ], (uint32_t)number(h, @"add", 0),
		     MTLSizeMake(words, 1, 1));
	[e endEncoding];
	wait_for(cb, c.spec);
	NSMutableData *out = [NSMutableData dataWithLength:words * 4];
	surface_rows(s, 0, nil, out, words * 4, 1);
	memcpy(c.buffers[@"out"].contents, out.bytes, MIN(out.length, c.buffers[@"out"].length));
	b = nil;
	CFRelease(s);
}

/* A surface of two planes, '444f' unless `fourcc` names another: elements
 * of `bytes` bytes, then pairs of them subsampled to cw x ch. */
static IOSurfaceRef two_plane_surface(size_t w, size_t height, size_t cw, size_t ch, size_t bytes, uint32_t fourcc)
{
	size_t row = IOSurfaceAlignProperty(kIOSurfacePlaneBytesPerRow, w * bytes);
	size_t pair_row = IOSurfaceAlignProperty(kIOSurfacePlaneBytesPerRow, cw * 2 * bytes);
	size_t pair_offset = IOSurfaceAlignProperty(kIOSurfacePlaneOffset, row * height);
	NSDictionary *properties = @{
		(__bridge id)kIOSurfaceWidth : @(w), (__bridge id)kIOSurfaceHeight : @(height),
		(__bridge id)kIOSurfacePixelFormat : @(fourcc),
		(__bridge id)kIOSurfaceAllocSize : @(pair_offset + pair_row * ch),
		(__bridge id)kIOSurfacePlaneInfo : @[
			@{(__bridge id)kIOSurfacePlaneWidth : @(w), (__bridge id)kIOSurfacePlaneHeight : @(height),
			  (__bridge id)kIOSurfacePlaneBytesPerElement : @(bytes), (__bridge id)kIOSurfacePlaneBytesPerRow : @(row),
			  (__bridge id)kIOSurfacePlaneOffset : @0, (__bridge id)kIOSurfacePlaneSize : @(row * height)},
			@{(__bridge id)kIOSurfacePlaneWidth : @(cw), (__bridge id)kIOSurfacePlaneHeight : @(ch),
			  (__bridge id)kIOSurfacePlaneBytesPerElement : @(2 * bytes), (__bridge id)kIOSurfacePlaneBytesPerRow : @(pair_row),
			  (__bridge id)kIOSurfacePlaneOffset : @(pair_offset), (__bridge id)kIOSurfacePlaneSize : @(pair_row * ch)},
		],
	};
	return make_surface(properties);
}

/* A surface of an Apple two-plane format `format`: green or luma bytes in
 * its first plane, blue and red or Cb and Cr pairs in its second,
 * subsampled `across` by `down`. A kernel reads and linearly samples it
 * through its first plane. */
static void host_two_plane(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	size_t w = number(h, @"width", 1), height = number(h, @"height", 1);
	size_t cw = w / number(h, @"across", 1), ch = height / number(h, @"down", 1);
	/* 10-bit formats keep each component in a 16-bit word. The wide kernel
	 * writes floats, which hold every 10-bit step. The surface's pixel
	 * format, whose range YCbCr conversion follows, is '444f' unless the
	 * case names another. */
	size_t bytes = number(h, @"bytes", 1);
	NSString *fourcc = h[@"fourcc"] ?: @"444f";
	const char *code = fourcc.UTF8String;
	if (strlen(code) != 4)
		@throw failure(@"fourcc %@ is not four characters", fourcc);
	uint32_t pixel = (uint32_t)code[0] << 24 | (uint32_t)code[1] << 16 | (uint32_t)code[2] << 8 | (uint32_t)code[3];
	IOSurfaceRef s = two_plane_surface(w, height, cw, ch, bytes, pixel);
	surface_rows(s, 0, contents(c.buffers[@"green"]), nil, w * bytes, height);
	surface_rows(s, 1, contents(c.buffers[@"blue_red"]), nil, cw * 2 * bytes, ch);
	id<MTLTexture> t = surface_texture(c, s, 0, (MTLPixelFormat)number(h, @"format", 0), MTLTextureUsageShaderRead);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:host_pipeline(c, number(h, @"wide", 0) ? @"surface_two_plane_wide" : @"surface_two_plane")];
	[e setTexture:t atIndex:0];
	[e setBuffer:c.buffers[@"read"] offset:0 atIndex:0];
	[e setBuffer:c.buffers[@"sampled"] offset:0 atIndex:1];
	[e dispatchThreads:MTLSizeMake(w, height, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
	[e endEncoding];
	wait_for(cb, c.spec);
	CFRelease(s);
}

/* Copies both planes of a 4:4:4 two-plane surface into `out`: the first
 * plane's bytes, then the second's pairs, tight. */
static void two_plane_out(IOSurfaceRef s, id<MTLBuffer> out, size_t w, size_t height)
{
	NSMutableData *first = [NSMutableData dataWithLength:w * height];
	NSMutableData *second = [NSMutableData dataWithLength:w * 2 * height];
	surface_rows(s, 0, nil, first, w, height);
	surface_rows(s, 1, nil, second, w * 2, height);
	if (out.length < first.length + second.length)
		@throw failure(@"two-plane output too small");
	memcpy(out.contents, first.bytes, first.length);
	memcpy((uint8_t *)out.contents + first.length, second.bytes, second.length);
}

/* Rendering into 4:4:4 two-plane surfaces of `format`: one loaded from the
 * `green` and `blue_red` planes and drawn over, blending each pixel's
 * `colors` entry by its alpha; one cleared to `clear` with its left half
 * drawn opaque. Both surfaces' planes are the outputs. */
static void host_two_plane_render(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	size_t w = number(h, @"width", 1), height = number(h, @"height", 1);
	MTLPixelFormat format = (MTLPixelFormat)number(h, @"format", 0);
	NSError *error = nil;
	id<MTLRenderPipelineState> opaque =
		host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_viewport", @"frag_pixel", format, &error);
	if (!opaque)
		@throw failure(@"two-plane pipeline: %@", error.localizedDescription);
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.colorAttachments[0].blendingEnabled = YES;
	d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
	d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	id<MTLRenderPipelineState> blended = host_render_pipeline(c, d, @"vert_viewport", @"frag_pixel", format, &error);
	if (!blended)
		@throw failure(@"blending two-plane pipeline: %@", error.localizedDescription);
	IOSurfaceRef loaded = two_plane_surface(w, height, w, height, 1, '444f');
	IOSurfaceRef cleared = two_plane_surface(w, height, w, height, 1, '444f');
	for (int i = 0; i < 2; i++) {
		IOSurfaceRef s = i ? cleared : loaded;
		surface_rows(s, 0, contents(c.buffers[@"green"]), nil, w, height);
		surface_rows(s, 1, contents(c.buffers[@"blue_red"]), nil, w * 2, height);
	}
	MTLTextureUsage usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	id<MTLTexture> targets[2] = {surface_texture(c, loaded, 0, format, usage),
				     surface_texture(c, cleared, 0, format, usage)};
	NSArray *clear = h[@"clear"];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	for (int i = 0; i < 2; i++) {
		MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.colorAttachments[0].texture = targets[i];
		pass.colorAttachments[0].loadAction = i ? MTLLoadActionClear : MTLLoadActionLoad;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake([clear[0] doubleValue], [clear[1] doubleValue],
									 [clear[2] doubleValue], [clear[3] doubleValue]);
		id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
		[e setRenderPipelineState:i ? opaque : blended];
		MTLViewport viewport = {0, 0, (double)(i ? w / 2 : w), (double)height, 0, 1};
		[e setViewport:viewport];
		[e setFragmentBuffer:c.buffers[@"colors"] offset:0 atIndex:0];
		[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
		[e endEncoding];
	}
	wait_for(cb, c.spec);
	two_plane_out(loaded, c.buffers[@"loaded"], w, height);
	two_plane_out(cleared, c.buffers[@"cleared"], w, height);
	CFRelease(loaded);
	CFRelease(cleared);
}

/* What another process does to a texture and event this one shares. */
@protocol Sharer
- (void)writeRows:(NSData *)rows into:(MTLSharedTextureHandle *)texture
	    after:(MTLSharedEventHandle *)event value:(uint64_t)value
	    reply:(void (^)(NSString *error))reply;
@end

static NSString *const share_service = @"com.stela.metal-suite.share";

/* The share service: opens the handles with its own device, waits on the
 * event, copies rows into the texture's top and signals the next value. */
@interface ShareService : NSObject <NSXPCListenerDelegate, Sharer>
@end
@implementation ShareService
- (BOOL)listener:(NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)connection
{
	connection.exportedInterface = [NSXPCInterface interfaceWithProtocol:@protocol(Sharer)];
	connection.exportedObject = self;
	[connection resume];
	return YES;
}

- (void)writeRows:(NSData *)rows into:(MTLSharedTextureHandle *)handle
	    after:(MTLSharedEventHandle *)event_handle value:(uint64_t)value
	    reply:(void (^)(NSString *error))reply
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	id<MTLTexture> t = [device newSharedTextureWithHandle:handle];
	id<MTLSharedEvent> event = [device newSharedEventWithHandle:event_handle];
	if (!t || !event) {
		reply(@"the service could not open the shared texture or event");
		return;
	}
	NSUInteger row = t.width * 4, count = rows.length / row;
	id<MTLBuffer> staging = [device newBufferWithBytes:rows.bytes length:rows.length options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [[device newCommandQueue] commandBuffer];
	[cb encodeWaitForEvent:event value:value];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:row sourceBytesPerImage:rows.length
	       sourceSize:MTLSizeMake(t.width, count, 1) toTexture:t destinationSlice:0 destinationLevel:0
	destinationOrigin:MTLOriginMake(0, 0, 0)];
	[b endEncoding];
	[cb encodeSignalEvent:event value:value + 1];
	[cb commit];
	[cb waitUntilCompleted];
	reply(cb.error ? cb.error.localizedDescription : nil);
}
@end

static int serve_shares(void)
{
	ShareService *service = [ShareService new];
	NSXPCListener *listener = [NSXPCListener serviceListener];
	listener.delegate = service;
	[listener resume];
	return 0;
}

/* This process clears a shared texture and signals 1; the share service
 * waits for 1, writes the top rows and signals 2; this process waits for
 * 2 and reads the texture back. */
static void host_share(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSUInteger w = number(h, @"width", 1), height = number(h, @"height", 1);
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
										     width:w
										    height:height
										 mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	id<MTLTexture> t = [c.device newSharedTextureWithDescriptor:d];
	id<MTLSharedEvent> event = [c.device newSharedEvent];
	if (!t || !event)
		@throw failure(@"no shared texture or event");
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = t;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	NSArray *k = h[@"clear"];
	pass.colorAttachments[0].clearColor =
		MTLClearColorMake([k[0] doubleValue], [k[1] doubleValue], [k[2] doubleValue], [k[3] doubleValue]);
	[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
	[cb encodeSignalEvent:event value:1];
	[cb commit];

	NSXPCConnection *connection = [[NSXPCConnection alloc] initWithServiceName:share_service];
	connection.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(Sharer)];
	[connection resume];
	__block NSString *problem = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	id<Sharer> sharer = [connection remoteObjectProxyWithErrorHandler:^(NSError *e) {
		problem = e.localizedDescription ?: @"connection failed";
		dispatch_semaphore_signal(done);
	}];
	[sharer writeRows:contents(c.buffers[@"rows"]) into:[t newSharedTextureHandle]
		    after:[event newSharedEventHandle] value:1
		    reply:^(NSString *error) {
			    problem = error;
			    dispatch_semaphore_signal(done);
		    }];
	long late = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC));
	[connection invalidate];
	if (late)
		@throw failure(@"the share service did not answer");
	if (problem)
		@throw failure(@"share service: %@", problem);

	id<MTLCommandBuffer> read = [queue commandBuffer];
	[read encodeWaitForEvent:event value:2];
	id<MTLBlitCommandEncoder> b = [read blitCommandEncoder];
	[b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, height, 1)
		  toBuffer:c.buffers[@"out"] destinationOffset:0 destinationBytesPerRow:w * 4
	  destinationBytesPerImage:w * 4 * height];
	[b endEncoding];
	wait_for(read, c.spec);
	put(c, @"props", 0, (uint32_t)event.signaledValue);
}

/* An archive given a compute and a render pipeline's functions, written to
 * a file and loaded again: pipelines it holds are created though a miss
 * would fail them, and one it does not hold fails. */
static void host_archive(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	NSError *error = nil;
	id<MTLBinaryArchive> archive = [c.device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
	put(c, @"props", 0, archive != nil);
	MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
	compute.computeFunction = function_named(library, @"archived", nil);
	put(c, @"props", 1, [archive addComputePipelineFunctionsWithDescriptor:compute error:&error]);
	MTLRenderPipelineDescriptor *render = [MTLRenderPipelineDescriptor new];
	render.vertexFunction = function_named(library, @"vert_left_half", nil);
	render.fragmentFunction = function_named(library, @"frag_constant", nil);
	render.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	put(c, @"props", 2, [archive addRenderPipelineFunctionsWithDescriptor:render error:&error]);
	/* A visible function, a stitched library and a mesh pipeline too. */
	MTLFunctionDescriptor *visible = [MTLFunctionDescriptor functionDescriptor];
	visible.name = @"archived_visible";
	put(c, @"props", 8, [archive addFunctionWithDescriptor:visible library:library error:&error]);
	if (@available(macOS 15.0, *)) {
		MTLFunctionStitchingGraph *graph = [[MTLFunctionStitchingGraph alloc]
			initWithFunctionName:@"archived_stitched"
				       nodes:@[]
				  outputNode:[[MTLFunctionStitchingFunctionNode alloc]
						     initWithName:@"archived_stitch"
						       arguments:@[ [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0] ]
					     controlDependencies:@[]]
				  attributes:@[]];
		MTLStitchedLibraryDescriptor *sd = [MTLStitchedLibraryDescriptor new];
		sd.functions = @[ function_named(library, @"archived_stitch", nil) ];
		sd.functionGraphs = @[ graph ];
		put(c, @"props", 9, [archive addLibraryWithDescriptor:sd error:&error]);
		MTLMeshRenderPipelineDescriptor *mesh = [MTLMeshRenderPipelineDescriptor new];
		mesh.meshFunction = function_named(library, @"archived_mesh", nil);
		mesh.fragmentFunction = render.fragmentFunction;
		mesh.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		put(c, @"props", 10, [archive addMeshRenderPipelineFunctionsWithDescriptor:mesh error:&error]);
	}
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory()
						    stringByAppendingPathComponent:[NSString stringWithFormat:@"stela-suite-%d.metallib",
													       getpid()]]];
	put(c, @"props", 3, [archive serializeToURL:url error:&error]);
	MTLBinaryArchiveDescriptor *from_file = [MTLBinaryArchiveDescriptor new];
	from_file.url = url;
	id<MTLBinaryArchive> loaded = [c.device newBinaryArchiveWithDescriptor:from_file error:&error];
	put(c, @"props", 4, loaded != nil);
	compute.binaryArchives = loaded ? @[ loaded ] : @[];
	id<MTLComputePipelineState> hit = [c.device newComputePipelineStateWithDescriptor:compute
										  options:MTLPipelineOptionFailOnBinaryArchiveMiss
									       reflection:nil
										    error:&error];
	put(c, @"props", 5, hit != nil);
	MTLComputePipelineDescriptor *other = [MTLComputePipelineDescriptor new];
	other.computeFunction = function_named(library, @"not_archived", nil);
	other.binaryArchives = compute.binaryArchives;
	put(c, @"props", 6, [c.device newComputePipelineStateWithDescriptor:other options:MTLPipelineOptionFailOnBinaryArchiveMiss
							     reflection:nil
								  error:&error] == nil);
	render.binaryArchives = compute.binaryArchives;
	put(c, @"props", 7, [c.device newRenderPipelineStateWithDescriptor:render options:MTLPipelineOptionFailOnBinaryArchiveMiss
							    reflection:nil
								 error:&error] != nil);
	[[NSFileManager defaultManager] removeItemAtURL:url error:nil];
	if (!hit)
		return;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	id<MTLBuffer> data = c.buffers[@"data"];
	dispatch_one(e, hit, @[ data ], 0, MTLSizeMake(data.length / 4, 1, 1));
	[e endEncoding];
	wait_for(cb, c.spec);
}

/* The errors creation calls report for bad input, a command buffer's
 * status through its life, and command buffers running in the order they
 * were enqueued rather than committed. */
static void host_errors(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSError *error = nil;
	id<MTLLibrary> broken = [c.device newLibraryWithSource:@"kernel void broken(" options:nil error:&error];
	put(c, @"props", 0, broken == nil);
	put(c, @"props", 1, [error.domain isEqualToString:MTLLibraryErrorDomain]);
	put(c, @"props", 2, (uint32_t)error.code);
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	put(c, @"props", 3, [library newFunctionWithName:@"absent"] == nil);
	error = nil;
	put(c, @"props", 4,
	    host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_left_half", @"frag_unwritten_input",
				 MTLPixelFormatRGBA8Unorm, &error) == nil &&
		    error != nil);

	MTLCommandBufferDescriptor *d = [MTLCommandBufferDescriptor new];
	d.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
	id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:d];
	put(c, @"props", 5, (uint32_t)cb.status);
	[cb enqueue];
	put(c, @"props", 6, (uint32_t)cb.status);
	[[cb blitCommandEncoder] endEncoding];
	wait_for(cb, c.spec);
	put(c, @"props", 7, (uint32_t)cb.status);
	put(c, @"props", 8, cb.error == nil);

	id<MTLComputePipelineState> digit = host_pipeline(c, @"append_digit");
	id<MTLCommandBuffer> first = [queue commandBuffer], second = [queue commandBuffer];
	[second enqueue];
	[first enqueue];
	for (id<MTLCommandBuffer> each in @[ first, second ]) {
		id<MTLComputeCommandEncoder> e = [each computeCommandEncoder];
		dispatch_one(e, digit, @[ c.buffers[@"order"] ], each == first ? 1 : 2, MTLSizeMake(1, 1, 1));
		[e endEncoding];
	}
	[first commit];
	wait_for(second, c.spec);
	[first waitUntilCompleted];
}

/* The state setPurgeableState: returns, which is the one before it: for a
 * buffer, a private texture and a heap, through empty and back. */
static void host_purgeable(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
										      width:64
										     height:64
										  mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 1 << 20;
	hd.storageMode = MTLStorageModePrivate;
	NSArray *resources = @[
		[c.device newBufferWithLength:1 << 16 options:MTLResourceStorageModeShared], [c.device newTextureWithDescriptor:td],
		[c.device newHeapWithDescriptor:hd]
	];
	MTLPurgeableState steps[] = {MTLPurgeableStateKeepCurrent, MTLPurgeableStateEmpty,       MTLPurgeableStateKeepCurrent,
				     MTLPurgeableStateNonVolatile, MTLPurgeableStateKeepCurrent, MTLPurgeableStateVolatile};
	NSUInteger at = 0;
	for (id r in resources)
		for (size_t i = 0; i < sizeof steps / sizeof *steps; i++)
			put(c, @"props", at++, (uint32_t)[r setPurgeableState:steps[i]]);
}

/* Many dispatches in one encoder, many command buffers in one queue,
 * threads encoding into one queue at once, and queues adding at once. */
static void host_streams(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSUInteger dispatches = number(h, @"dispatches", 1), buffers = number(h, @"command_buffers", 1);
	NSUInteger threads = number(h, @"threads", 1), each = number(h, @"per_thread", 1);
	id<MTLBuffer> counts = c.buffers[@"counts"];
	id<MTLComputePipelineState> bump = host_pipeline(c, @"bump"), add = host_pipeline(c, @"bump_atomic");
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	for (NSUInteger i = 0; i < dispatches; i++)
		dispatch_one(e, bump, @[ counts ], 0, MTLSizeMake(1, 1, 1));
	[e endEncoding];
	wait_for(cb, c.spec);
	for (NSUInteger i = 0; i < buffers; i++) {
		cb = [queue commandBuffer];
		e = [cb computeCommandEncoder];
		dispatch_one(e, bump, @[ counts ], 1, MTLSizeMake(1, 1, 1));
		[e endEncoding];
		if (i + 1 < buffers)
			[cb commit];
	}
	wait_for(cb, c.spec);
	NSMutableArray *queues = [NSMutableArray new];
	for (NSUInteger t = 0; t < threads; t++)
		[queues addObject:[c.device newCommandQueue]];
	for (int phase = 0; phase < 2; phase++) {
		dispatch_apply(threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
			id<MTLCommandQueue> q = phase ? queues[t] : queue;
			id<MTLCommandBuffer> last = nil;
			for (NSUInteger i = 0; i < each; i++) {
				last = [q commandBuffer];
				id<MTLComputeCommandEncoder> te = [last computeCommandEncoder];
				dispatch_one(te, phase ? add : bump, @[ counts ], phase ? (uint32_t)(2 + threads) : (uint32_t)(2 + t),
					     MTLSizeMake(1, 1, 1));
				[te endEncoding];
				[last commit];
			}
			[last waitUntilCompleted];
		});
	}
}

/* Buffers and textures made, used by a command buffer and dropped before
 * it runs, many times over: what each held must reach the output. */
static void host_churn(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSUInteger rounds = number(h, @"rounds", 1), size = number(h, @"size", 4096);
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
										      width:64
										     height:64
										  mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageRenderTarget;
	id<MTLCommandBuffer> cb = nil;
	for (NSUInteger i = 0; i < rounds; i++) {
		@autoreleasepool {
			id<MTLBuffer> b = [c.device newBufferWithLength:size options:MTLResourceStorageModeShared];
			uint32_t word = (uint32_t)(i * 2654435761u);
			memset_pattern4(b.contents, &word, size);
			id<MTLTexture> t = [c.device newTextureWithDescriptor:td];
			cb = [queue commandBuffer];
			MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
			pass.colorAttachments[0].texture = t;
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(i, 0, 0, 0);
			[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
			id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
			[e copyFromBuffer:b sourceOffset:(i * 4096) % size toBuffer:c.buffers[@"words"] destinationOffset:i * 4 size:4];
			[e copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(i % 64, i / 64 % 64, 0)
				sourceSize:MTLSizeMake(1, 1, 1) toBuffer:c.buffers[@"texels"] destinationOffset:i * 4
			   destinationBytesPerRow:4 destinationBytesPerImage:4];
			[e endEncoding];
			if (i + 1 < rounds)
				[cb commit];
		}
	}
	wait_for(cb, c.spec);
}

/* Bytes of `length` that are not zero, or with `written`, that are neither
 * zero nor in [first, last]. */
static uint32_t stray_bytes(const uint8_t *bytes, NSUInteger length, BOOL written, uint8_t first, uint8_t last)
{
	uint32_t count = 0;
	for (NSUInteger i = 0; i < length; i++)
		count += bytes[i] != 0 && !(written && bytes[i] >= first && bytes[i] <= last);
	return count;
}

/* Rounds of new buffers and textures of each storage mode: each is read
 * back, filled with the round's byte and freed, so later rounds take the
 * memory and recycled textures earlier ones held. A new buffer must read
 * zero; a new texture may hold only zero or bytes this case wrote. */
static void host_fresh_resources(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSUInteger rounds = number(h, @"rounds", 1), at = 0;
	uint8_t first = (uint8_t)number(h, @"first_fill", 0xa1);
	MTLResourceOptions modes[] = {MTLResourceStorageModeShared, MTLResourceStorageModeManaged,
				      MTLResourceStorageModePrivate};
	for (NSUInteger round = 0; round < rounds; round++) {
		uint8_t fill = (uint8_t)(first + round);
		for (NSNumber *size in h[@"buffer_sizes"]) {
			NSUInteger length = size.unsignedIntegerValue;
			for (size_t m = 0; m < sizeof modes / sizeof *modes; m++)
				@autoreleasepool {
					id<MTLBuffer> b = [c.device newBufferWithLength:length options:modes[m]];
					if (!b)
						@throw failure(@"no %lu-byte buffer", (unsigned long)length);
					BOOL gpu_only = modes[m] == MTLResourceStorageModePrivate;
					id<MTLBuffer> read =
						gpu_only ? [c.device newBufferWithLength:length options:MTLResourceStorageModeShared] : b;
					id<MTLCommandBuffer> cb = [queue commandBuffer];
					id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
					if (gpu_only) {
						/* The readback starts nonzero, so a copy that missed
						 * shows. */
						memset(read.contents, 0xff, length);
						[e copyFromBuffer:b sourceOffset:0 toBuffer:read destinationOffset:0 size:length];
					}
					if (modes[m] == MTLResourceStorageModeManaged)
						[e synchronizeResource:b];
					[e endEncoding];
					wait_for(cb, c.spec);
					put(c, @"props", at++, stray_bytes(read.contents, length, NO, 0, 0));
					cb = [queue commandBuffer];
					e = [cb blitCommandEncoder];
					[e fillBuffer:b range:NSMakeRange(0, length) value:fill];
					[e endEncoding];
					wait_for(cb, c.spec);
				}
		}
		/* A size every round, which recycles textures, and a size of the
		 * round's own, which takes freed memory. */
		NSUInteger widths[] = {64, 80 + 16 * round};
		for (size_t w = 0; w < sizeof widths / sizeof *widths; w++)
			for (size_t m = 0; m < sizeof modes / sizeof *modes; m++)
				@autoreleasepool {
					NSUInteger width = widths[w], height = 64, row = width * 4, length = row * height;
					MTLTextureDescriptor *td =
						[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
												   width:width
												  height:height
											       mipmapped:NO];
					td.storageMode = (MTLStorageMode)(modes[m] >> MTLResourceStorageModeShift);
					td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
					id<MTLTexture> t = [c.device newTextureWithDescriptor:td];
					if (!t)
						@throw failure(@"no %lux%lu texture", (unsigned long)width, (unsigned long)height);
					id<MTLBuffer> read = [c.device newBufferWithLength:length options:MTLResourceStorageModeShared];
					memset(read.contents, 0xff, length);
					id<MTLCommandBuffer> cb = [queue commandBuffer];
					id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
					if (td.storageMode == MTLStorageModePrivate)
						[e copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
							      sourceSize:MTLSizeMake(width, height, 1) toBuffer:read destinationOffset:0
						  destinationBytesPerRow:row destinationBytesPerImage:length];
					if (td.storageMode == MTLStorageModeManaged)
						[e synchronizeResource:t];
					[e endEncoding];
					wait_for(cb, c.spec);
					if (td.storageMode != MTLStorageModePrivate)
						[t getBytes:read.contents bytesPerRow:row fromRegion:MTLRegionMake2D(0, 0, width, height)
						mipmapLevel:0];
					put(c, @"props", at++, stray_bytes(read.contents, length, YES, first, fill));
					cb = [queue commandBuffer];
					MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
					pass.colorAttachments[0].texture = t;
					pass.colorAttachments[0].loadAction = MTLLoadActionClear;
					pass.colorAttachments[0].storeAction = MTLStoreActionStore;
					double value = fill / 255.0;
					pass.colorAttachments[0].clearColor = MTLClearColorMake(value, value, value, value);
					[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
					wait_for(cb, c.spec);
				}
	}
}

/* A private buffer past 4 GiB: kernels stamp windows below 2 GiB, across
 * 2 and 4 GiB and at its end, a fill lands past 4 GiB, and blits bring
 * each back. */
static void host_large_buffer(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSUInteger length = number(h, @"length", 0);
	if (c.device.maxBufferLength < length)
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"maxBufferLength" userInfo:nil];
	id<MTLBuffer> big = [c.device newBufferWithLength:length options:MTLResourceStorageModePrivate];
	if (!big)
		@throw failure(@"no %lu-byte buffer", (unsigned long)length);
	id<MTLComputePipelineState> stamp = host_pipeline(c, @"stamp");
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	for (NSArray *w in h[@"windows"])
		dispatch_one(e, stamp, @[ big ], [w[0] unsignedIntValue], MTLSizeMake([w[1] unsignedIntegerValue], 1, 1));
	[e endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	NSArray *fill = h[@"fill"];
	[b fillBuffer:big range:NSMakeRange([fill[0] unsignedIntegerValue], [fill[1] unsignedIntegerValue]) value:[fill[2] unsignedCharValue]];
	NSUInteger at = 0;
	for (NSArray *w in h[@"windows"]) {
		NSUInteger bytes = [w[1] unsignedIntegerValue] * 4;
		[b copyFromBuffer:big sourceOffset:[w[0] unsignedIntegerValue] * 4 toBuffer:c.buffers[@"out"] destinationOffset:at size:bytes];
		at += bytes;
	}
	[b copyFromBuffer:big sourceOffset:[fill[0] unsignedIntegerValue] toBuffer:c.buffers[@"out"] destinationOffset:at
		     size:[fill[1] unsignedIntegerValue]];
	[b endEncoding];
	wait_for(cb, c.spec);
}

/* An 8x4 RGBA8 target for host draws, and a pass over it. */
static id<MTLTexture> host_target(Case *c, MTLPixelFormat format, NSUInteger w, NSUInteger h)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
	d.usage = MTLTextureUsageRenderTarget;
	d.storageMode = MTLStorageModePrivate;
	return [c.device newTextureWithDescriptor:d];
}

static id<MTLRenderCommandEncoder> host_pass(id<MTLCommandBuffer> cb, id<MTLTexture> t, MTLStoreAction store)
{
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = t;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
	pass.colorAttachments[0].storeAction = store;
	return [cb renderCommandEncoderWithDescriptor:pass];
}

static void read_back(id<MTLCommandBuffer> cb, id<MTLTexture> t, id<MTLBuffer> out, NSUInteger texel, MTLBlitOption option)
{
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
		sourceSize:MTLSizeMake(t.width, t.height, 1) toBuffer:out destinationOffset:0
	   destinationBytesPerRow:t.width * texel destinationBytesPerImage:t.width * t.height * texel options:option];
	[b endEncoding];
}

/* An R8 texture of four levels, level k holding k * 51 everywhere. */
static id<MTLTexture> leveled_texture(Case *c)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:8 height:8 mipmapped:YES];
	d.storageMode = MTLStorageModeManaged;
	id<MTLTexture> t = [c.device newTextureWithDescriptor:d];
	uint8_t bytes[64];
	for (NSUInteger level = 0; level < t.mipmapLevelCount; level++) {
		NSUInteger side = 8 >> level;
		memset(bytes, (int)(level * 51), sizeof bytes);
		[t replaceRegion:MTLRegionMake2D(0, 0, side, side) mipmapLevel:level withBytes:bytes bytesPerRow:side];
	}
	return t;
}

static id<MTLSamplerState> nearest_mip_sampler(Case *c)
{
	MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
	d.mipFilter = MTLSamplerMipFilterNearest;
	return [c.device newSamplerStateWithDescriptor:d];
}

/* Binding changes without new bindings: offsets moved on bound buffers,
 * buffers and textures bound as ranges, and samplers bound with their own
 * level-of-detail clamps, in a kernel and in two draws. */
static void host_rebind(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLBuffer> words = c.buffers[@"words"], out = c.buffers[@"out"], levels = c.buffers[@"levels"];
	id<MTLTexture> leveled = leveled_texture(c);
	id<MTLSamplerState> sampler = nearest_mip_sampler(c);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:host_pipeline(c, @"copy_word")];
	id<MTLBuffer> pair[] = {out, words};
	NSUInteger offsets[] = {0, 0};
	[e setBuffers:pair offsets:offsets withRange:NSMakeRange(0, 2)];
	for (uint32_t k = 0; k < 3; k++) {
		if (k)
			[e setBufferOffset:k * 4 atIndex:1];
		[e setBytes:&k length:sizeof k atIndex:2];
		[e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	}
	[e setComputePipelineState:host_pipeline(c, @"sample_levels")];
	id<MTLTexture> textures[] = {leveled, leveled};
	[e setTextures:textures withRange:NSMakeRange(0, 2)];
	[e setSamplerState:sampler lodMinClamp:2 lodMaxClamp:2 atIndex:0];
	[e setBuffer:levels offset:0 atIndex:0];
	[e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[e endEncoding];

	NSError *error = nil;
	id<MTLRenderPipelineState> p = host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_offset_rect", @"frag_rebind",
							    MTLPixelFormatRGBA8Unorm, &error);
	if (!p)
		@throw failure(@"rebind pipeline: %@", error.localizedDescription);
	id<MTLTexture> t = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
	id<MTLRenderCommandEncoder> r = host_pass(cb, t, MTLStoreActionStore);
	[r setRenderPipelineState:p];
	[r setVertexBuffer:c.buffers[@"rects"] offset:0 atIndex:0];
	[r setFragmentBuffer:c.buffers[@"colors"] offset:0 atIndex:0];
	[r setFragmentTextures:textures withRange:NSMakeRange(0, 2)];
	[r setFragmentSamplerState:sampler lodMinClamp:1 lodMaxClamp:1 atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[r setVertexBufferOffset:256 atIndex:0];
	[r setFragmentBufferOffset:256 atIndex:0];
	[r setFragmentSamplerState:sampler lodMinClamp:3 lodMaxClamp:3 atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[r endEncoding];
	read_back(cb, t, c.buffers[@"image"], 4, MTLBlitOptionNone);
	wait_for(cb, c.spec);
}

/* One vertex layout of dynamic stride: a tight draw, then the same
 * attribute at twice the stride from a moved offset. */
static void host_dynamic_stride(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	if (@available(macOS 14.0, *)) {
		MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
		MTLVertexDescriptor *v = [MTLVertexDescriptor vertexDescriptor];
		v.attributes[0].format = MTLVertexFormatFloat2;
		v.attributes[0].offset = 0;
		v.attributes[0].bufferIndex = 1;
		v.layouts[1].stride = MTLBufferLayoutStrideDynamic;
		d.vertexDescriptor = v;
		NSError *error = nil;
		id<MTLRenderPipelineState> p =
			host_render_pipeline(c, d, @"vert_attribute", @"frag_constant", MTLPixelFormatRGBA8Unorm, &error);
		if (!p)
			@throw failure(@"dynamic stride pipeline: %@", error.localizedDescription);
		id<MTLTexture> t = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLRenderCommandEncoder> r = host_pass(cb, t, MTLStoreActionStore);
		[r setRenderPipelineState:p];
		NSArray *colors = h[@"colors"];
		for (NSUInteger k = 0; k < 2; k++) {
			NSArray *color = colors[k];
			float rgba[4] = {[color[0] floatValue], [color[1] floatValue], [color[2] floatValue], [color[3] floatValue]};
			if (k == 0)
				[r setVertexBuffer:c.buffers[@"corners"] offset:0 attributeStride:8 atIndex:1];
			else
				[r setVertexBufferOffset:number(h, @"second_offset", 0) attributeStride:16 atIndex:1];
			[r setFragmentBytes:rgba length:sizeof rgba atIndex:0];
			[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
		}
		[r endEncoding];
		read_back(cb, t, c.buffers[@"image"], 4, MTLBlitOptionNone);
		wait_for(cb, c.spec);
	} else {
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"dynamic strides need macOS 14" userInfo:nil];
	}
}

/* Barriers on named resources and on stages: two concurrent dispatches
 * ordered by a resource barrier, and three point draws whose fragments
 * each extend a chain the next draw's vertex function reads. */
static void host_barriers(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	if (![h[@"render"] boolValue]) {
		id<MTLBuffer> x = c.buffers[@"x"], y = c.buffers[@"y"];
		NSUInteger n = x.length / 4;
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
		dispatch_one(e, host_pipeline(c, @"write_values"), @[ x ], 3, MTLSizeMake(n, 1, 1));
		id<MTLResource> ordered[] = {x};
		[e memoryBarrierWithResources:ordered count:1];
		[e setComputePipelineState:host_pipeline(c, @"double_values")];
		[e setBuffer:x offset:0 atIndex:0];
		[e setBuffer:y offset:0 atIndex:1];
		[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(n, 64), 1, 1)];
		[e endEncoding];
		wait_for(cb, c.spec);
		return;
	}
	id<MTLBuffer> chain = c.buffers[@"chain"];
	NSError *error = nil;
	id<MTLRenderPipelineState> p = host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_chain", @"frag_chain",
							    MTLPixelFormatR8Unorm, &error);
	if (!p)
		@throw failure(@"chain pipeline: %@", error.localizedDescription);
	id<MTLRenderCommandEncoder> r = host_pass(cb, host_target(c, MTLPixelFormatR8Unorm, 1, 1), MTLStoreActionDontCare);
	[r setRenderPipelineState:p];
	[r setVertexBuffer:chain offset:0 atIndex:0];
	[r setFragmentBuffer:chain offset:0 atIndex:0];
	uint32_t links[][2] = {{3, 0}, {0, 1}, {1, 2}};
	for (int k = 0; k < 3; k++) {
		if (k == 1)
			[r memoryBarrierWithScope:MTLBarrierScopeBuffers afterStages:MTLRenderStageFragment beforeStages:MTLRenderStageVertex];
		if (k == 2)
			[r memoryBarrierWithResources:(id<MTLResource>[]){chain} count:1 afterStages:MTLRenderStageFragment
					 beforeStages:MTLRenderStageVertex];
		[r setVertexBytes:&links[k][0] length:4 atIndex:1];
		[r setFragmentBytes:&links[k][1] length:4 atIndex:1];
		[r drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
	}
	[r endEncoding];
	wait_for(cb, c.spec);
}

/* A pass whose target's earlier contents are left undefined (DontCare),
 * filled by one draw; after a render-target barrier a second draw over the
 * left half reads the target it renders to and writes one minus it. */
static void host_render_target_barrier(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSError *error = nil;
	id<MTLRenderPipelineState> fill = host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_offset_rect",
							       @"frag_constant", MTLPixelFormatRGBA8Unorm, &error);
	id<MTLRenderPipelineState> invert = host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_offset_rect",
								 @"frag_feedback", MTLPixelFormatRGBA8Unorm, &error);
	if (!fill || !invert)
		@throw failure(@"feedback pipelines: %@", error.localizedDescription);
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	d.storageMode = MTLStorageModePrivate;
	id<MTLTexture> t = [c.device newTextureWithDescriptor:d];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = t;
	pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
	float whole[5] = {-1, -1, 1, 1, 0}, left[5] = {-1, -1, 0, 1, 0}, color[4] = {0.2f, 0.4f, 0.6f, 1.0f};
	[r setRenderPipelineState:fill];
	[r setVertexBytes:whole length:sizeof whole atIndex:0];
	[r setFragmentBytes:color length:sizeof color atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	if ([h[@"barrier"] isEqualToString:@"texture"]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		[r textureBarrier];
#pragma clang diagnostic pop
	} else {
		[r memoryBarrierWithScope:MTLBarrierScopeRenderTargets afterStages:MTLRenderStageFragment beforeStages:MTLRenderStageFragment];
	}
	[r setRenderPipelineState:invert];
	[r setVertexBytes:left length:sizeof left atIndex:0];
	[r setFragmentTexture:t atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[r endEncoding];
	read_back(cb, t, c.buffers[@"image"], 4, MTLBlitOptionNone);
	wait_for(cb, c.spec);
}

/* A path filled as RenderBox fills one, in one pass over a colour target
 * and a float winding target. A fan from a pivot to each edge of a square
 * ring's outer contour and its opposite inner contour adds each triangle's
 * facing to the count with additive blending; a full-target draw reads the
 * count and the colour back through inputs a function constant enables,
 * fills where the count is nonzero and clears the count. */
static void host_winding_coverage(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSError *error = nil;
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	NSArray *reads = @[ @{@"type" : @"bool", @"index" : @0, @"value" : @1} ];
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = function_named(library, @"vert_points", nil);
	d.fragmentFunction = function_named(library, @"frag_winding", nil);
	d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	d.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
	d.colorAttachments[1].pixelFormat = MTLPixelFormatRG16Float;
	d.colorAttachments[1].blendingEnabled = YES;
	d.colorAttachments[1].sourceRGBBlendFactor = MTLBlendFactorOne;
	d.colorAttachments[1].destinationRGBBlendFactor = MTLBlendFactorOne;
	d.colorAttachments[1].sourceAlphaBlendFactor = MTLBlendFactorOne;
	d.colorAttachments[1].destinationAlphaBlendFactor = MTLBlendFactorOne;
	id<MTLRenderPipelineState> wind = [c.device newRenderPipelineStateWithDescriptor:d error:&error];
	d.vertexFunction = function_named(library, @"vert_points", nil);
	d.fragmentFunction = function_named(library, @"frag_resolve_winding", reads);
	d.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
	d.colorAttachments[1].blendingEnabled = NO;
	id<MTLRenderPipelineState> resolve = wind ? [c.device newRenderPipelineStateWithDescriptor:d error:&error] : nil;
	if (!wind || !resolve)
		@throw failure(@"winding pipelines: %@", error.localizedDescription);
	NSUInteger side = 16;
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:side height:side mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> color = [c.device newTextureWithDescriptor:td];
	td.pixelFormat = MTLPixelFormatRG16Float;
	id<MTLTexture> count = [c.device newTextureWithDescriptor:td];
	/* Outer contour counterclockwise, inner clockwise, in clip space. */
	const float outer = 0.75f, inner = 0.25f, pivot[2] = {-0.9f, -0.97f};
	const float contours[2][4][2] = {
		{{-outer, -outer}, {outer, -outer}, {outer, outer}, {-outer, outer}},
		{{-inner, -inner}, {-inner, inner}, {inner, inner}, {inner, -inner}},
	};
	float fan[2 * 4 * 3 * 2], *v = fan;
	for (int k = 0; k < 2; k++) {
		for (int e = 0; e < 4; e++) {
			const float *a = contours[k][e], *b = contours[k][(e + 1) % 4];
			float tri[6] = {pivot[0], pivot[1], a[0], a[1], b[0], b[1]};
			memcpy(v, tri, sizeof tri);
			v += 6;
		}
	}
	const float whole[12] = {-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};
	const __fp16 fill[4] = {1, 0, 0, 1};
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = color;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 1, 1);
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.colorAttachments[1].texture = count;
	pass.colorAttachments[1].loadAction = MTLLoadActionClear;
	pass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
	pass.colorAttachments[1].storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
	[r setRenderPipelineState:wind];
	[r setVertexBytes:fan length:sizeof fan atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:24];
	[r setRenderPipelineState:resolve];
	[r setVertexBytes:whole length:sizeof whole atIndex:0];
	[r setFragmentBytes:fill length:sizeof fill atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
	[r endEncoding];
	read_back(cb, color, c.buffers[@"color"], 4, MTLBlitOptionNone);
	read_back(cb, count, c.buffers[@"count"], 4, MTLBlitOptionNone);
	wait_for(cb, c.spec);
}

/* Fast resource loading. The source bytes are written to a file as they
 * are and in each compression method. A serial, high-priority IO queue
 * loads the plain file into a buffer, into memory and, past a barrier and
 * from an offset, into a texture, then signals an event. A concurrent,
 * low-priority queue waits for it, loads each compressed file into its own
 * buffer and signals again; a GPU command buffer waits for that and reads
 * the texture back. */
/* Scratch memory an IO queue asks for to decompress into, from the
 * application rather than from Metal. */
API_AVAILABLE(macos(13.0))
@interface IOScratch : NSObject <MTLIOScratchBuffer>
@property(retain) id<MTLBuffer> buffer;
@end
@implementation IOScratch
@end

API_AVAILABLE(macos(13.0))
@interface IOScratchAllocator : NSObject <MTLIOScratchBufferAllocator>
@property(retain) id<MTLDevice> device;
@end
@implementation IOScratchAllocator
- (id<MTLIOScratchBuffer>)newScratchBufferWithMinimumSize:(NSUInteger)minimumSize
{
	IOScratch *s = [IOScratch new];
	s.buffer = [self.device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared];
	return s;
}
@end

static void host_io(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	if (@available(macOS 13.0, *)) {
		id<MTLBuffer> source = c.buffers[@"source"];
		NSUInteger size = number(h, @"size", 0), texture_offset = number(h, @"texture_offset", 0);
		NSDictionary *methods = @{@"zlib" : @(MTLIOCompressionMethodZlib), @"lzfse" : @(MTLIOCompressionMethodLZFSE),
					  @"lz4" : @(MTLIOCompressionMethodLZ4), @"lzma" : @(MTLIOCompressionMethodLZMA),
					  @"lzbitmap" : @(MTLIOCompressionMethodLZBitmap)};
		NSString *stem = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"stela-io-%d", getpid()]];
		NSString *plain = [stem stringByAppendingString:@".plain"];
		NSMutableArray<NSString *> *files = [NSMutableArray arrayWithObject:plain];
		if (![[NSData dataWithBytesNoCopy:source.contents length:size freeWhenDone:NO] writeToFile:plain atomically:NO])
			@throw failure(@"writing %@", plain);
		for (NSString *name in methods) {
			NSString *path = [stem stringByAppendingFormat:@".%@", name];
			[files addObject:path];
			MTLIOCompressionContext context = MTLIOCreateCompressionContext(
				path.fileSystemRepresentation, (MTLIOCompressionMethod)[methods[name] integerValue],
				MTLIOCompressionContextDefaultChunkSize());
			if (!context)
				@throw failure(@"no %@ compression context", name);
			MTLIOCompressionContextAppendData(context, source.contents, size);
			if (MTLIOFlushAndDestroyCompressionContext(context) != MTLIOCompressionStatusComplete)
				@throw failure(@"%@ compression failed", name);
		}
		@try {
			NSError *error = nil;
			MTLIOCommandQueueDescriptor *qd = [MTLIOCommandQueueDescriptor new];
			qd.type = MTLIOCommandQueueTypeSerial;
			qd.priority = MTLIOPriorityHigh;
			id<MTLIOCommandQueue> serial = [c.device newIOCommandQueueWithDescriptor:qd error:&error];
			/* The concurrent queue's limits and its own scratch memory. */
			qd.type = MTLIOCommandQueueTypeConcurrent;
			qd.priority = MTLIOPriorityLow;
			qd.maxCommandBufferCount = 4;
			qd.maxCommandsInFlight = 8;
			IOScratchAllocator *allocator = [IOScratchAllocator new];
			allocator.device = c.device;
			qd.scratchBufferAllocator = allocator;
			id<MTLIOCommandQueue> concurrent = serial ? [c.device newIOCommandQueueWithDescriptor:qd error:&error] : nil;
			id<MTLIOFileHandle> handle = [c.device newIOFileHandleWithURL:[NSURL fileURLWithPath:plain] error:&error];
			if (!serial || !concurrent || !handle)
				@throw failure(@"IO queue or handle: %@", error.localizedDescription);
			id<MTLSharedEvent> event = [c.device newSharedEvent];
			MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
												 width:16 height:16 mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			id<MTLTexture> texture = [c.device newTextureWithDescriptor:td];
			NSMutableData *memory = [NSMutableData dataWithLength:size];
			id<MTLIOCommandBuffer> first = [serial commandBuffer];
			[first pushDebugGroup:@"plain loads"];
			[first loadBuffer:c.buffers[@"plain"] offset:0 size:size sourceHandle:handle sourceHandleOffset:0];
			[first loadBytes:memory.mutableBytes size:size sourceHandle:handle sourceHandleOffset:0];
			[first addBarrier];
			[first loadTexture:texture slice:0 level:0 size:MTLSizeMake(16, 16, 1) sourceBytesPerRow:64 sourceBytesPerImage:1024
			    destinationOrigin:MTLOriginMake(0, 0, 0) sourceHandle:handle sourceHandleOffset:texture_offset];
			[first popDebugGroup];
			[first signalEvent:event value:1];
			[first commit];
			id<MTLIOCommandBuffer> second = [concurrent commandBuffer];
			[second waitForEvent:event value:1];
			NSMutableArray *handles = [NSMutableArray new];
			for (NSString *name in methods) {
				id<MTLIOFileHandle> compressed =
					[c.device newIOFileHandleWithURL:[NSURL fileURLWithPath:[stem stringByAppendingFormat:@".%@", name]]
							 compressionMethod:(MTLIOCompressionMethod)[methods[name] integerValue]
								     error:&error];
				if (!compressed)
					@throw failure(@"%@ handle: %@", name, error.localizedDescription);
				[handles addObject:compressed];
				[second loadBuffer:c.buffers[name] offset:0 size:size sourceHandle:compressed sourceHandleOffset:0];
			}
			/* The same loads through the older handle calls. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			id<MTLIOFileHandle> legacy = [c.device newIOHandleWithURL:[NSURL fileURLWithPath:plain] error:&error];
			id<MTLIOFileHandle> legacy_zlib =
				[c.device newIOHandleWithURL:[NSURL fileURLWithPath:[stem stringByAppendingString:@".zlib"]]
					   compressionMethod:MTLIOCompressionMethodZlib
						       error:&error];
#pragma clang diagnostic pop
			if (!legacy || !legacy_zlib)
				@throw failure(@"older IO handles: %@", error.localizedDescription);
			[handles addObjectsFromArray:@[ legacy, legacy_zlib ]];
			[second loadBuffer:c.buffers[@"legacy"] offset:0 size:size sourceHandle:legacy sourceHandleOffset:0];
			[second loadBuffer:c.buffers[@"legacy_zlib"] offset:0 size:size sourceHandle:legacy_zlib sourceHandleOffset:0];
			[second copyStatusToBuffer:c.buffers[@"status"] offset:0];
			[second signalEvent:event value:2];
			[second commit];
			/* A load the barrier holds behind the others, waiting for a
			 * value the host signals only once it has asked for the
			 * load to be cancelled. */
			[concurrent enqueueBarrier];
			id<MTLIOCommandBuffer> cancelled = [concurrent commandBuffer];
			[cancelled waitForEvent:event value:3];
			[cancelled loadBuffer:c.buffers[@"plain"] offset:0 size:size sourceHandle:handle sourceHandleOffset:0];
			[cancelled commit];
			[cancelled tryCancel];
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			[cb encodeWaitForEvent:event value:2];
			read_back(cb, texture, c.buffers[@"texture"], 4, MTLBlitOptionNone);
			wait_for(cb, c.spec);
			[first waitUntilCompleted];
			[second waitUntilCompleted];
			event.signaledValue = 3;
			[cancelled waitUntilCompleted];
			for (id<MTLIOCommandBuffer> b in @[ first, second ])
				if (b.status != MTLIOStatusComplete)
					@throw failure(@"IO command buffer status %ld: %@", (long)b.status, b.error.localizedDescription);
			((uint32_t *)c.buffers[@"status"].contents)[2] = (uint32_t)cancelled.status;
			memcpy(c.buffers[@"memory"].contents, memory.bytes, size);
		} @finally {
			for (NSString *path in files)
				[[NSFileManager defaultManager] removeItemAtPath:path error:nil];
		}
	} else {
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"MTLIO needs macOS 13" userInfo:nil];
	}
}

/* A managed array texture one level and slice of which a pass clears;
 * synchronizing just that image makes the CPU see it. */
static void host_sync_texture(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:YES];
	d.textureType = MTLTextureType2DArray;
	d.arrayLength = 2;
	d.storageMode = MTLStorageModeManaged;
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	id<MTLTexture> t = [c.device newTextureWithDescriptor:d];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = t;
	pass.colorAttachments[0].level = 1;
	pass.colorAttachments[0].slice = 1;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.6, 1.0, 1.0);
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b synchronizeTexture:t slice:1 level:1];
	[b endEncoding];
	wait_for(cb, c.spec);
	[t getBytes:c.buffers[@"out"].contents bytesPerRow:8 bytesPerImage:16 fromRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 slice:1];
}

/* Two viewports, each with its own scissor: an instanced full-target
 * rectangle, one instance per viewport, keeps to its scissor. */
static void host_scissors(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSError *error = nil;
	id<MTLRenderPipelineState> p = host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_viewport", @"frag_instance",
							    MTLPixelFormatRGBA8Unorm, &error);
	if (!p)
		@throw failure(@"viewport pipeline: %@", error.localizedDescription);
	id<MTLTexture> t = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLRenderCommandEncoder> r = host_pass(cb, t, MTLStoreActionStore);
	[r setRenderPipelineState:p];
	MTLViewport viewports[2] = {{0, 0, 8, 4, 0, 1}, {0, 0, 8, 4, 0, 1}};
	MTLScissorRect scissors[2] = {{0, 0, 4, 4}, {4, 0, 4, 4}};
	[r setViewports:viewports count:2];
	[r setScissorRects:scissors count:2];
	[r setFragmentBuffer:c.buffers[@"colors"] offset:0 atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:2];
	[r endEncoding];
	read_back(cb, t, c.buffers[@"image"], 4, MTLBlitOptionNone);
	wait_for(cb, c.spec);
}

/* Store actions left unknown when the pass begins and chosen after its
 * draw, for colour, depth and stencil, with their options. */
static void host_deferred_store(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSError *error = nil;
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
	d.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
	id<MTLRenderPipelineState> p = host_render_pipeline(c, d, @"vert_offset_rect", @"frag_rebind_color", MTLPixelFormatRGBA8Unorm, &error);
	if (!p)
		@throw failure(@"deferred store pipeline: %@", error.localizedDescription);
	id<MTLTexture> color = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
	id<MTLTexture> depth = host_target(c, MTLPixelFormatDepth32Float_Stencil8, 8, 4);
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = color;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionUnknown;
	pass.depthAttachment.texture = depth;
	pass.depthAttachment.loadAction = MTLLoadActionClear;
	pass.depthAttachment.clearDepth = 1.0;
	pass.depthAttachment.storeAction = MTLStoreActionUnknown;
	pass.stencilAttachment.texture = depth;
	pass.stencilAttachment.loadAction = MTLLoadActionClear;
	pass.stencilAttachment.storeAction = MTLStoreActionUnknown;
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	dd.depthCompareFunction = MTLCompareFunctionAlways;
	dd.depthWriteEnabled = YES;
	MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
	sd.depthStencilPassOperation = MTLStencilOperationReplace;
	dd.frontFaceStencil = sd;
	dd.backFaceStencil = sd;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
	[r setRenderPipelineState:p];
	[r setDepthStencilState:[c.device newDepthStencilStateWithDescriptor:dd]];
	[r setStencilReferenceValue:7];
	[r setVertexBuffer:c.buffers[@"rects"] offset:0 atIndex:0];
	[r setFragmentBuffer:c.buffers[@"colors"] offset:0 atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[r setColorStoreAction:MTLStoreActionStore atIndex:0];
	[r setColorStoreActionOptions:MTLStoreActionOptionNone atIndex:0];
	[r setDepthStoreAction:MTLStoreActionStore];
	[r setDepthStoreActionOptions:MTLStoreActionOptionNone];
	[r setStencilStoreAction:MTLStoreActionStore];
	[r setStencilStoreActionOptions:MTLStoreActionOptionNone];
	[r endEncoding];
	read_back(cb, color, c.buffers[@"image"], 4, MTLBlitOptionNone);
	read_back(cb, depth, c.buffers[@"depth"], 4, MTLBlitOptionDepthFromDepthStencil);
	read_back(cb, depth, c.buffers[@"stencil"], 1, MTLBlitOptionStencilFromDepthStencil);
	wait_for(cb, c.spec);
}

static NSUInteger align_up(NSUInteger v, NSUInteger a)
{
	return (v + a - 1) / a * a;
}

static void write_bytes(Case *c, NSString *buffer, NSData *data)
{
	id<MTLBuffer> b = c.buffers[buffer];
	memcpy(b.contents, data.bytes, MIN(data.length, b.length));
}

/* What the device and its objects report: a library's function names, a
 * library made from a metallib's bytes, the alignments of textures over
 * buffers (then used), heap sizes and use (then placed), timestamps and
 * their frequency, event waits with timeouts and pipeline thread limits. */
static void host_queries(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	NSArray *names = [library.functionNames sortedArrayUsingSelector:@selector(compare:)];
	write_bytes(c, @"names", [[names componentsJoinedByString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding]);

	NSData *bytes = read_file(c.plan[@"libraries"][c.spec[@"library"]]);
	dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	NSError *error = nil;
	id<MTLLibrary> from_data = [c.device newLibraryWithData:data error:&error];
	if (!from_data)
		@throw failure(@"library from data: %@", error.localizedDescription);
	id<MTLComputePipelineState> archived =
		[c.device newComputePipelineStateWithFunction:function_named(from_data, @"archived", nil) error:&error];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	dispatch_one(e, archived, @[ c.buffers[@"data"] ], 0, MTLSizeMake(c.buffers[@"data"].length / 4, 1, 1));

	NSUInteger linear = [c.device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm];
	NSUInteger texel_buffer = [c.device minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm];
	put(c, @"props", 0, power_of_two(linear));
	put(c, @"props", 1, power_of_two(texel_buffer));
	NSUInteger w = 16, rows = 4, row = align_up(w * 4, linear), tb_offset = align_up(linear + row * rows, texel_buffer);
	id<MTLBuffer> backing = [c.device newBufferWithLength:tb_offset + w * 4 options:MTLResourceStorageModePrivate];
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:rows mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageShaderWrite;
	id<MTLTexture> over = [backing newTextureWithDescriptor:td offset:linear bytesPerRow:row];
	MTLTextureDescriptor *bd = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w
										 resourceOptions:MTLResourceStorageModePrivate
											   usage:MTLTextureUsageShaderWrite];
	id<MTLTexture> texels = [backing newTextureWithDescriptor:bd offset:tb_offset bytesPerRow:w * 4];
	[e setComputePipelineState:host_pipeline(c, @"fill_linear")];
	[e setTexture:over atIndex:0];
	[e setTexture:texels atIndex:1];
	[e dispatchThreads:MTLSizeMake(w, rows, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
	[e endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	for (NSUInteger y = 0; y < rows; y++)
		[b copyFromBuffer:backing sourceOffset:linear + y * row toBuffer:c.buffers[@"linear"] destinationOffset:y * w * 4 size:w * 4];
	[b copyFromBuffer:backing sourceOffset:tb_offset toBuffer:c.buffers[@"texels"] destinationOffset:0 size:w * 4];
	[b endEncoding];
	wait_for(cb, c.spec);

	MTLSizeAndAlign bs = [c.device heapBufferSizeAndAlignWithLength:4096 options:MTLResourceStorageModePrivate];
	MTLTextureDescriptor *ht = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64 mipmapped:NO];
	ht.storageMode = MTLStorageModePrivate;
	ht.usage = MTLTextureUsageRenderTarget;
	MTLSizeAndAlign ts = [c.device heapTextureSizeAndAlignWithDescriptor:ht];
	put(c, @"props", 2, bs.size >= 4096 && power_of_two(bs.align));
	put(c, @"props", 3, ts.size >= 64 * 64 * 4 && power_of_two(ts.align));
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.type = MTLHeapTypePlacement;
	hd.storageMode = MTLStorageModePrivate;
	NSUInteger texture_at = align_up(bs.size, ts.align);
	hd.size = texture_at + ts.size;
	id<MTLHeap> placed = [c.device newHeapWithDescriptor:hd];
	id<MTLBuffer> pb = [placed newBufferWithLength:4096 options:MTLResourceStorageModePrivate offset:0];
	id<MTLTexture> pt = [placed newTextureWithDescriptor:ht offset:texture_at];
	put(c, @"props", 4, pb && pt);
	cb = [queue commandBuffer];
	b = [cb blitCommandEncoder];
	[b fillBuffer:pb range:NSMakeRange(0, 4096) value:0x5a];
	[b copyFromBuffer:pb sourceOffset:4092 toBuffer:c.buffers[@"placed"] destinationOffset:0 size:4];
	[b endEncoding];
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = pt;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 1);
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
	b = [cb blitCommandEncoder];
	[b copyFromTexture:pt sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(63, 63, 0) sourceSize:MTLSizeMake(1, 1, 1)
		  toBuffer:c.buffers[@"placed"] destinationOffset:4 destinationBytesPerRow:4 destinationBytesPerImage:4];
	[b endEncoding];
	wait_for(cb, c.spec);

	MTLHeapDescriptor *ad = [MTLHeapDescriptor new];
	ad.size = 1 << 20;
	ad.storageMode = MTLStorageModePrivate;
	id<MTLHeap> automatic = [c.device newHeapWithDescriptor:ad];
	NSUInteger empty = automatic.usedSize;
	id<MTLBuffer> held = [automatic newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
	NSUInteger used = automatic.usedSize, available = [automatic maxAvailableSizeWithAlignment:256];
	put(c, @"props", 5, held && empty == 0 && used >= 4096);
	put(c, @"props", 6, automatic.currentAllocatedSize >= used && automatic.currentAllocatedSize >= ad.size);
	put(c, @"props", 7, available > 0 && available <= ad.size - used);
	NSUInteger before = c.device.currentAllocatedSize;
	id<MTLBuffer> large = [c.device newBufferWithLength:64 << 20 options:MTLResourceStorageModePrivate];
	put(c, @"props", 8, large && c.device.currentAllocatedSize >= before + (64 << 20));

	MTLTimestamp cpu[2], gpu[2];
	[c.device sampleTimestamps:&cpu[0] gpuTimestamp:&gpu[0]];
	usleep(2000);
	[c.device sampleTimestamps:&cpu[1] gpuTimestamp:&gpu[1]];
	memcpy(c.buffers[@"cpu_times"].contents, cpu, sizeof cpu);
	memcpy(c.buffers[@"gpu_times"].contents, gpu, sizeof gpu);
	/* The rate of the GPU timestamps counter heaps hold, in ticks per
	 * second; the header ties it to no other timestamps. */
	if (@available(macOS 26.0, *)) {
		put(c, @"props", 19, [c.device queryTimestampFrequency] > 0);
	} else {
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"timestamp frequencies need macOS 26" userInfo:nil];
	}

	if (@available(macOS 12.0, *)) {
		id<MTLSharedEvent> event = [c.device newSharedEvent];
		cb = [queue commandBuffer];
		[cb encodeSignalEvent:event value:3];
		[cb commit];
		put(c, @"props", 9, [event waitUntilSignaledValue:3 timeoutMS:5000]);
		put(c, @"props", 10, ![event waitUntilSignaledValue:10 timeoutMS:10]);
		event.signaledValue = 12;
		put(c, @"props", 11, [event waitUntilSignaledValue:10 timeoutMS:0]);
		[cb waitUntilCompleted];
	}

	id<MTLComputePipelineState> limited = host_pipeline(c, @"limited");
	put(c, @"props", 12, limited.maxTotalThreadsPerThreadgroup <= 128 && limited.maxTotalThreadsPerThreadgroup > 0);
	put(c, @"props", 13, power_of_two(limited.threadExecutionWidth) &&
				     limited.maxTotalThreadsPerThreadgroup % limited.threadExecutionWidth == 0);
	MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
	pd.computeFunction = function_named(library, @"archived", nil);
	pd.maxTotalThreadsPerThreadgroup = 64;
	id<MTLComputePipelineState> capped = [c.device newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone
										   reflection:nil
											error:&error];
	put(c, @"props", 14, capped && capped.maxTotalThreadsPerThreadgroup <= 64);

	/* What functions report of themselves. */
	id<MTLFunction> attribute = function_named(library, @"vert_attribute", nil);
	put(c, @"props", 15, function_named(library, @"archived", nil).functionType == MTLFunctionTypeKernel);
	put(c, @"props", 16, attribute.functionType == MTLFunctionTypeVertex);
	put(c, @"props", 17, attribute.vertexAttributes.count == 1 && attribute.vertexAttributes[0].attributeIndex == 0 &&
				     attribute.vertexAttributes[0].attributeType == MTLDataTypeFloat2);
	put(c, @"props", 18, function_named(library, @"frag_constant", nil).functionType == MTLFunctionTypeFragment);
}

/* A dynamic library compiled at run time under an install name, which it
 * then reports. */
static void host_dynamic_library(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	MTLCompileOptions *o = [MTLCompileOptions new];
	o.libraryType = MTLLibraryTypeDynamic;
	o.installName = h[@"install_name"];
	NSError *error = nil;
	id<MTLLibrary> source = [c.device newLibraryWithSource:@"int stela_twice(int x) { return 2 * x; }" options:o error:&error];
	if (!source)
		@throw failure(@"dynamic library source: %@", error.localizedDescription);
	id<MTLDynamicLibrary> dynamic = [c.device newDynamicLibrary:source error:&error];
	if (!dynamic)
		@throw failure(@"dynamic library: %@", error.localizedDescription);
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory()
						    stringByAppendingPathComponent:[NSString stringWithFormat:@"stela-suite-%d-dynamic.metallib",
													       getpid()]]];
	if (![dynamic serializeToURL:url error:&error])
		@throw failure(@"dynamic library serialize: %@", error.localizedDescription);
	id<MTLDynamicLibrary> loaded = [c.device newDynamicLibraryWithURL:url error:&error];
	[[NSFileManager defaultManager] removeItemAtURL:url error:nil];
	if (!loaded)
		@throw failure(@"dynamic library load: %@", error.localizedDescription);
	write_bytes(c, @"name", [[NSString stringWithFormat:@"%@\n%@", dynamic.installName, loaded.installName]
					dataUsingEncoding:NSUTF8StringEncoding]);
}

/* Timestamps sampled at a pass's boundaries into a private buffer and
 * resolved by a blit, and sampled between dispatches. */
static void host_resolve_counters(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	BOOL dispatches = [h[@"between_dispatches"] boolValue];
	MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
	d.counterSet = counter_set(c.device, MTLCommonCounterSetTimestamp);
	d.sampleCount = 2;
	d.storageMode = dispatches ? MTLStorageModeShared : MTLStorageModePrivate;
	NSError *error = nil;
	id<MTLCounterSampleBuffer> samples = [c.device newCounterSampleBufferWithDescriptor:d error:&error];
	if (!samples)
		@throw failure(@"counter sample buffer: %@", error.localizedDescription);
	id<MTLComputePipelineState> p = host_pipeline(c, @"write_values");
	id<MTLBuffer> x = c.buffers[@"x"];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e;
	if (dispatches) {
		e = [cb computeCommandEncoder];
		[e sampleCountersInBuffer:samples atSampleIndex:0 withBarrier:YES];
		dispatch_one(e, p, @[ x ], 3, MTLSizeMake(x.length / 4, 1, 1));
		[e sampleCountersInBuffer:samples atSampleIndex:1 withBarrier:YES];
	} else {
		MTLComputePassDescriptor *cp = [MTLComputePassDescriptor computePassDescriptor];
		cp.sampleBufferAttachments[0].sampleBuffer = samples;
		cp.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
		cp.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
		e = [cb computeCommandEncoderWithDescriptor:cp];
		dispatch_one(e, p, @[ x ], 3, MTLSizeMake(x.length / 4, 1, 1));
	}
	[e endEncoding];
	if (!dispatches) {
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b resolveCounters:samples inRange:NSMakeRange(0, 2) destinationBuffer:c.buffers[@"stamps"] destinationOffset:0];
		[b endEncoding];
	}
	wait_for(cb, c.spec);
	if (dispatches) {
		NSData *resolved = [samples resolveCounterRange:NSMakeRange(0, 2)];
		write_bytes(c, @"stamps", resolved);
	}
}

/* An argument buffer laid out from argument descriptors, not a function:
 * two pointers a kernel then reads through. */
static void host_argument_descriptors(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	MTLArgumentDescriptor *data = [MTLArgumentDescriptor argumentDescriptor], *k = [MTLArgumentDescriptor argumentDescriptor];
	data.dataType = MTLDataTypePointer;
	data.index = 0;
	data.access = MTLBindingAccessReadWrite;
	k.dataType = MTLDataTypePointer;
	k.index = 1;
	k.access = MTLBindingAccessReadOnly;
	/* A constant after the pointers the kernel reads, in a block of its
	 * own alignment. */
	MTLArgumentDescriptor *tail = [MTLArgumentDescriptor argumentDescriptor];
	tail.dataType = MTLDataTypeUInt;
	tail.index = 2;
	tail.constantBlockAlignment = 16;
	id<MTLArgumentEncoder> encoder = [c.device newArgumentEncoderWithArguments:@[ data, k, tail ]];
	id<MTLBuffer> arguments = [c.device newBufferWithLength:MAX(encoder.encodedLength, 16) options:MTLResourceStorageModeShared];
	[encoder setArgumentBuffer:arguments offset:0];
	[encoder setBuffer:c.buffers[@"data"] offset:0 atIndex:0];
	[encoder setBuffer:c.buffers[@"k"] offset:0 atIndex:1];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:host_pipeline(c, @"through_arguments")];
	[e setBuffer:arguments offset:0 atIndex:0];
	[e useResource:c.buffers[@"data"] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
	[e useResource:c.buffers[@"k"] usage:MTLResourceUsageRead];
	NSUInteger n = c.buffers[@"data"].length / 4;
	[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(n, 64), 1, 1)];
	[e endEncoding];
	wait_for(cb, c.spec);
}

/* The heap fits Core Animation and SwiftUI depend on, as a Mac answers
 * them: the size and alignment answers for small textures, how many 8x8
 * R8 glyphs a 128 KiB heap holds and its usedSize and
 * maxAvailableSizeWithAlignment: then, one more after makeAliasable, a heap
 * of 100 bytes holding a page, two 188x77 R16Float masks in 128 KiB, a
 * 68x67 RGBA16Float layer in 64 KiB, and three of four 64x64 BGR10A2
 * textures in a heap of three answers; then the 100-byte heap's size and the
 * mask and layer sequences below. */
static void host_heap_fit(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	__block NSUInteger at = 0;
	void (^fact)(NSUInteger) = ^(NSUInteger value) {
		put(c, @"facts", at++, (uint32_t)value);
	};
	MTLTextureDescriptor *(^desc)(MTLPixelFormat, NSUInteger, NSUInteger, MTLTextureUsage) =
		^(MTLPixelFormat f, NSUInteger w, NSUInteger hh, MTLTextureUsage u) {
			MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:hh mipmapped:NO];
			d.storageMode = MTLStorageModePrivate;
			d.usage = u;
			return d;
		};
	id<MTLHeap> (^heap)(NSUInteger) = ^(NSUInteger size) {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = size;
		d.storageMode = MTLStorageModePrivate;
		d.type = MTLHeapTypeAutomatic;
		return [c.device newHeapWithDescriptor:d];
	};
	MTLTextureUsage read = MTLTextureUsageShaderRead, drawn = read | MTLTextureUsageRenderTarget;
	MTLTextureDescriptor *answered[] = {
		desc(MTLPixelFormatR8Unorm, 8, 8, read),	   desc(MTLPixelFormatRG8Unorm, 16, 16, read),
		desc(MTLPixelFormatBGRA8Unorm, 32, 32, read),	   desc(MTLPixelFormatR16Float, 188, 77, drawn),
		desc(MTLPixelFormatRGBA16Float, 68, 67, drawn),	   desc(MTLPixelFormatRGBA16Float, 122, 67, drawn),
		desc(MTLPixelFormatBGR10A2Unorm, 64, 64, drawn), desc(MTLPixelFormatBGRA8Unorm, 2560, 1440, drawn),
	};
	for (size_t i = 0; i < sizeof answered / sizeof *answered; i++) {
		MTLSizeAndAlign a = [c.device heapTextureSizeAndAlignWithDescriptor:answered[i]];
		fact(a.size);
		fact(a.align);
	}
	id<MTLHeap> glyphs = heap(131072);
	NSMutableArray *held = [NSMutableArray new];
	for (int i = 0; i < 2048; i++) {
		id<MTLTexture> t = [glyphs newTextureWithDescriptor:answered[0]];
		if (!t)
			break;
		[held addObject:t];
	}
	fact(held.count);
	fact(glyphs.usedSize);
	fact([glyphs maxAvailableSizeWithAlignment:128]);
	[held[0] makeAliasable];
	fact(glyphs.usedSize);
	fact([glyphs newTextureWithDescriptor:answered[0]] != nil);
	id<MTLHeap> small = heap(100);
	fact([small maxAvailableSizeWithAlignment:128]);
	fact(small.currentAllocatedSize);
	id<MTLHeap> masks = heap(131072);
	id<MTLTexture> first = [masks newTextureWithDescriptor:answered[3]];
	id<MTLTexture> second = [masks newTextureWithDescriptor:answered[3]];
	fact(first != nil);
	fact(second != nil);
	fact(masks.usedSize);
	id<MTLHeap> layer = heap(65536);
	fact([layer newTextureWithDescriptor:answered[4]] != nil);
	MTLSizeAndAlign ten = [c.device heapTextureSizeAndAlignWithDescriptor:answered[6]];
	id<MTLHeap> three = heap(3 * ten.size);
	NSMutableArray *tens = [NSMutableArray new];
	for (int i = 0; i < 4; i++) {
		id<MTLTexture> t = [three newTextureWithDescriptor:answered[6]];
		if (t)
			[tens addObject:t];
	}
	fact(tens.count);
	fact(three.usedSize);
	fact(small.size);
	/* The masks and layers Core Animation filled on the guest: each heap's
	 * size, then per texture whether it fit, usedSize and
	 * maxAvailableSizeWithAlignment: 128 and 16384. */
	struct {
		NSUInteger heap, format, width, height, count;
	} seqs[] = {
		{131072, MTLPixelFormatR16Float, 72, 151, 2},
		{131072, MTLPixelFormatR16Float, 113, 153, 2},
		{196608, MTLPixelFormatR16Float, 248, 81, 3},
		{65536, MTLPixelFormatRGBA16Float, 36, 35, 2},
	};
	for (size_t i = 0; i < sizeof seqs / sizeof *seqs; i++) {
		id<MTLHeap> into = heap(seqs[i].heap);
		MTLTextureDescriptor *d = desc(seqs[i].format, seqs[i].width, seqs[i].height, drawn);
		NSMutableArray *kept = [NSMutableArray new];
		fact(into.currentAllocatedSize);
		for (NSUInteger n = 0; n < seqs[i].count; n++) {
			id<MTLTexture> t = [into newTextureWithDescriptor:d];
			fact(t != nil);
			if (t)
				[kept addObject:t];
			fact(into.usedSize);
			fact([into maxAvailableSizeWithAlignment:128]);
			fact([into maxAvailableSizeWithAlignment:16384]);
		}
	}
}

/* Textures over CPU memory after the CPU clears it, as Core Animation's
 * backing stores are cleared and redrawn: a no-copy buffer over
 * vm_allocate memory, a shared buffer's contents, an IOSurface, a texture
 * replaceRegion fills from vm_allocate memory and a private texture a blit
 * fills from a shared buffer. Each 256x128 R32Uint texture (128 KiB, so
 * bzero zeroes with DC ZVA) gets a dense pattern, is read by a kernel, then
 * its memory is cleared by bzero of all of it, by bzero from an unaligned
 * offset, by memset_pattern4, by madvise(MADV_FREE_REUSABLE) and
 * MADV_FREE_REUSE, by vm_deallocate and vm_allocate at the same address,
 * or on the GPU by a buffer fill or a render pass clear, gets a sparse
 * pattern and is read again. The facts are, per source and clear, the
 * words the kernel read that differ from the memory after the first
 * pattern, the nonzero words a GPU clear left in the memory, and the words
 * that differ after the second pattern. */
static void host_cpu_clear(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	enum { W = 256, H = 128, ROW = W * 4, BYTES = ROW * H, WORDS = W * H };
	enum { BZERO, BZERO_PART, MEMSET, REUSABLE, REALLOC, GPU_FILL, CLEARS };
	enum { NOCOPY, SHARED, SURFACE, REPLACE, BLIT, SOURCES };
	static const int allowed[SOURCES][CLEARS] = {
		[NOCOPY] = {1, 1, 1, 1, 1, 1}, [SHARED] = {1, 1, 1, 0, 0, 1}, [SURFACE] = {1, 1, 1, 0, 0, 1},
		[REPLACE] = {1, 1, 1, 1, 1, 0}, [BLIT] = {1, 1, 1, 0, 0, 0},
	};
	id<MTLComputePipelineState> read = host_pipeline(c, @"texel_words");
	id<MTLBuffer> out = [c.device newBufferWithLength:BYTES options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
										     width:W
										    height:H
										 mipmapped:NO];
	NSUInteger at = 0;
	for (int source = 0; source < SOURCES; source++)
		for (int clear = 0; clear < CLEARS; clear++) {
			if (!allowed[source][clear])
				continue;
			mach_vm_address_t vm = 0;
			id<MTLBuffer> buffer = nil;
			id<MTLTexture> t = nil;
			IOSurfaceRef surface = NULL;
			uint32_t *memory = NULL;
			if (source == NOCOPY || source == REPLACE) {
				if (mach_vm_allocate(mach_task_self(), &vm, BYTES, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
					@throw failure(@"vm_allocate");
				memory = (uint32_t *)vm;
			}
			d.storageMode = MTLStorageModeShared;
			d.usage = MTLTextureUsageShaderRead | (source == SURFACE ? MTLTextureUsageRenderTarget : 0);
			if (source == NOCOPY) {
				buffer = [c.device newBufferWithBytesNoCopy:memory
								     length:BYTES
								    options:MTLResourceStorageModeShared
								deallocator:nil];
				t = [buffer newTextureWithDescriptor:d offset:0 bytesPerRow:ROW];
			} else if (source == SHARED || source == BLIT) {
				buffer = [c.device newBufferWithLength:BYTES options:MTLResourceStorageModeShared];
				memory = buffer.contents;
				if (source == SHARED) {
					t = [buffer newTextureWithDescriptor:d offset:0 bytesPerRow:ROW];
				} else {
					d.storageMode = MTLStorageModePrivate;
					t = [c.device newTextureWithDescriptor:d];
				}
			} else if (source == SURFACE) {
				surface = make_surface(@{
					(__bridge id)kIOSurfaceWidth : @(W),
					(__bridge id)kIOSurfaceHeight : @(H),
					(__bridge id)kIOSurfaceBytesPerElement : @4,
					(__bridge id)kIOSurfaceBytesPerRow : @(ROW),
					(__bridge id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
				});
				IOSurfaceLock(surface, 0, NULL);
				memory = IOSurfaceGetBaseAddress(surface);
				t = [c.device newTextureWithDescriptor:d iosurface:surface plane:0];
			} else {
				t = [c.device newTextureWithDescriptor:d];
			}
			if (!t)
				@throw failure(@"no texture for source %d", source);
			for (int pass = 0; pass < 2; pass++) {
				if (pass == 0) {
					for (uint32_t i = 0; i < WORDS; i++)
						memory[i] = i * 2654435761u | 1;
				} else {
					uint8_t *bytes = (uint8_t *)memory;
					static const uint32_t zero = 0;
					switch (clear) {
					case BZERO:
						bzero(bytes, BYTES);
						break;
					case BZERO_PART:
						bzero(bytes + 4160, BYTES - 4160 - 100);
						break;
					case MEMSET:
						memset_pattern4(bytes, &zero, BYTES);
						break;
					case REUSABLE:
						madvise(bytes, BYTES, MADV_FREE_REUSABLE);
						madvise(bytes, BYTES, MADV_FREE_REUSE);
						break;
					case REALLOC:
						mach_vm_deallocate(mach_task_self(), vm, BYTES);
						if (mach_vm_allocate(mach_task_self(), &vm, BYTES, VM_FLAGS_FIXED) != KERN_SUCCESS)
							@throw failure(@"vm_allocate at %llx", (unsigned long long)vm);
						break;
					case GPU_FILL: {
						id<MTLCommandBuffer> fill = [queue commandBuffer];
						if (buffer) {
							id<MTLBlitCommandEncoder> b = [fill blitCommandEncoder];
							[b fillBuffer:buffer range:NSMakeRange(0, BYTES) value:0];
							[b endEncoding];
						} else {
							MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
							pass.colorAttachments[0].texture = t;
							pass.colorAttachments[0].loadAction = MTLLoadActionClear;
							pass.colorAttachments[0].storeAction = MTLStoreActionStore;
							pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
							[[fill renderCommandEncoderWithDescriptor:pass] endEncoding];
						}
						wait_for(fill, c.spec);
						uint32_t kept = 0;
						for (uint32_t i = 0; i < WORDS; i++)
							kept += memory[i] != 0;
						put(c, @"facts", at++, kept);
						break;
					}
					}
					for (uint32_t i = 0; i < WORDS; i += 37)
						memory[i] = (i ^ 0xa5a5a5a5u) | 2;
				}
				id<MTLCommandBuffer> cb = [queue commandBuffer];
				if (source == REPLACE) {
					[t replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:memory bytesPerRow:ROW];
				} else if (source == BLIT) {
					id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
					[b copyFromBuffer:buffer
						   sourceOffset:0
					      sourceBytesPerRow:ROW
					    sourceBytesPerImage:BYTES
						     sourceSize:MTLSizeMake(W, H, 1)
						      toTexture:t
						destinationSlice:0
						destinationLevel:0
					       destinationOrigin:MTLOriginMake(0, 0, 0)];
					[b endEncoding];
				}
				id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
				[e setComputePipelineState:read];
				[e setTexture:t atIndex:0];
				[e setBuffer:out offset:0 atIndex:0];
				[e dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
				[e endEncoding];
				wait_for(cb, c.spec);
				const uint32_t *seen = out.contents;
				uint32_t differ = 0;
				for (uint32_t i = 0; i < WORDS; i++)
					differ += seen[i] != memory[i];
				put(c, @"facts", at++, differ);
			}
			if (surface) {
				IOSurfaceUnlock(surface, 0, NULL);
				t = nil;
				CFRelease(surface);
			}
			t = nil;
			buffer = nil;
			if (vm)
				mach_vm_deallocate(mach_task_self(), vm, BYTES);
		}
}

/* A sub-rectangle cleared as Core Animation's OpenGL-style renderer clears
 * one in WindowServer: its own path_blit_vert_lph and
 * attachment_clear_frag_lph from QuartzCore's library, without blending,
 * on a target filled with opaque red, then 5% white blended over the
 * rectangle, three times. Targets: a private BGRA8 texture and one over a
 * BGRA IOSurface, or a private texture of `format` (RGB10A8_2P_XR10, whose
 * first plane a blit copies out). The facts are each target's texel at the
 * rectangle's centre and one outside it, as words: one layer of white over
 * transparent inside, red outside. */
static void host_region_clear(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	enum { W = 64, H = 32 };
	MTLPixelFormat format = h[@"format"] ? (MTLPixelFormat)number(h, @"format", 0) : MTLPixelFormatBGRA8Unorm;
	int count = format == MTLPixelFormatBGRA8Unorm ? 2 : 1;
	NSError *error = nil;
	NSBundle *quartz = [NSBundle bundleWithPath:@"/System/Library/Frameworks/QuartzCore.framework"];
	id<MTLLibrary> ca = [c.device newDefaultLibraryWithBundle:quartz error:&error];
	if (!ca)
		@throw failure(@"QuartzCore library: %@", error.localizedDescription);
	MTLRenderPipelineDescriptor *cd = [MTLRenderPipelineDescriptor new];
	cd.vertexFunction = [ca newFunctionWithName:@"path_blit_vert_lph"];
	cd.fragmentFunction = [ca newFunctionWithName:@"attachment_clear_frag_lph"];
	if (!cd.vertexFunction || !cd.fragmentFunction)
		@throw failure(@"QuartzCore's region clear functions");
	cd.colorAttachments[0].pixelFormat = format;
	id<MTLRenderPipelineState> clear = [c.device newRenderPipelineStateWithDescriptor:cd error:&error];
	MTLRenderPipelineDescriptor *td = [MTLRenderPipelineDescriptor new];
	td.colorAttachments[0].blendingEnabled = YES;
	td.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
	td.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
	td.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	td.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	id<MTLRenderPipelineState> tint =
		clear ? host_render_pipeline(c, td, @"vert_left_half", @"frag_constant", format, &error) : nil;
	if (!clear || !tint)
		@throw failure(@"region clear pipelines: %@", error.localizedDescription);
	/* Pixels to clip space, as CA's mvp_matrix maps them; rectangles by
	 * instance, the first off the target. */
	float mvp[16] = {2.0f / W, 0, 0, 0, 0, -2.0f / H, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1};
	struct {
		float min[2], max[2];
		uint16_t color[4];
	} rects[2] = {{{-16, -16}, {-8, -8}, {0, 0, 0, 0}}, {{8, 4}, {24, 20}, {0, 0, 0, 0}}};
	float white[4] = {0.05f, 0.05f, 0.05f, 0.05f};
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
										     width:W
										    height:H
										 mipmapped:NO];
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	IOSurfaceRef surface = make_surface(@{
		(__bridge id)kIOSurfaceWidth : @(W),
		(__bridge id)kIOSurfaceHeight : @(H),
		(__bridge id)kIOSurfaceBytesPerElement : @4,
		(__bridge id)kIOSurfaceBytesPerRow : @(IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, W * 4)),
		(__bridge id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	});
	d.storageMode = MTLStorageModePrivate;
	id<MTLTexture> targets[2] = {[c.device newTextureWithDescriptor:d], nil};
	d.storageMode = c.device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
	if (count == 2)
		targets[1] = [c.device newTextureWithDescriptor:d iosurface:surface plane:0];
	for (int i = 0; i < count; i++)
		if (!targets[i])
			@throw failure(@"no region clear target %d of format %lu", i, (unsigned long)format);
	id<MTLBuffer> out = [c.device newBufferWithLength:W * H * 4 * 2 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	for (int i = 0; i < count; i++) {
		for (int round = 0; round < 4; round++) {
			MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
			pass.colorAttachments[0].texture = targets[i];
			pass.colorAttachments[0].loadAction = round ? MTLLoadActionLoad : MTLLoadActionClear;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
			if (round) {
				[e setRenderPipelineState:clear];
				[e setVertexBytes:mvp length:sizeof mvp atIndex:2];
				[e setVertexBytes:rects length:sizeof rects atIndex:1];
				[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:2];
				[e setRenderPipelineState:tint];
				[e setScissorRect:(MTLScissorRect){8, 4, 16, 16}];
				[e setFragmentBytes:white length:sizeof white atIndex:0];
				[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
			}
			[e endEncoding];
		}
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b copyFromTexture:targets[i]
			     sourceSlice:0
			     sourceLevel:0
			    sourceOrigin:MTLOriginMake(0, 0, 0)
			      sourceSize:MTLSizeMake(W, H, 1)
				toBuffer:out
		       destinationOffset:i * W * H * 4
		  destinationBytesPerRow:W * 4
		destinationBytesPerImage:W * H * 4];
		[b endEncoding];
	}
	wait_for(cb, c.spec);
	const uint32_t *texels = out.contents;
	for (int i = 0; i < count; i++) {
		put(c, @"facts", i * 2, texels[i * W * H + 12 * W + 16]);
		put(c, @"facts", i * 2 + 1, texels[i * W * H + 28 * W + 40]);
	}
	CFRelease(surface);
}

/* A heap exactly one buffer large: a second buffer fits only once the
 * first is made aliasable. */
static void host_aliasable(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	MTLSizeAndAlign sa = [c.device heapBufferSizeAndAlignWithLength:32768 options:MTLResourceStorageModePrivate];
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = sa.size;
	d.storageMode = MTLStorageModePrivate;
	id<MTLHeap> heap = [c.device newHeapWithDescriptor:d];
	id<MTLBuffer> first = [heap newBufferWithLength:32768 options:MTLResourceStorageModePrivate];
	id<MTLBuffer> refused = [heap newBufferWithLength:32768 options:MTLResourceStorageModePrivate];
	put(c, @"props", 0, first != nil);
	put(c, @"props", 1, refused == nil);
	put(c, @"props", 2, !first.isAliasable);
	[first makeAliasable];
	put(c, @"props", 3, first.isAliasable);
	id<MTLBuffer> second = [heap newBufferWithLength:32768 options:MTLResourceStorageModePrivate];
	put(c, @"props", 4, second != nil);
	if (!second)
		return;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b fillBuffer:second range:NSMakeRange(0, 32768) value:0x3c];
	[b copyFromBuffer:second sourceOffset:32764 toBuffer:c.buffers[@"out"] destinationOffset:0 size:4];
	[b endEncoding];
	wait_for(cb, c.spec);
}

/* Rasterization rate maps: the layer counts the device takes; a two-layer
 * full-quality map's sizes, granularity, parameter data size and mappings; a
 * lower-quality map's physical size within its screen, its corner mapped to
 * that size and a coordinate mapped there and back; a map of no layers; a
 * decoder over the full map's parameter data; and a pass with a map drawing
 * the left half red. */
/* A map's defaults and reset, then draws through maps that would send
 * logical outputs to other attachments: columns 0-1 to attachment 2, 2-3 to
 * 1, 4-5 through no map, and 6-7 swapping two outputs. Pipelines from an
 * MTLRenderPipelineDescriptor map each output to its own attachment
 * whatever the encoder's map. */
API_AVAILABLE(macos(26.0))
static void color_attachment_map(Case *c, id<MTLCommandQueue> queue)
{
	MTLLogicalToPhysicalColorAttachmentMap *fresh = [MTLLogicalToPhysicalColorAttachmentMap new];
	for (NSUInteger i = 0; i < 8; i++)
		put(c, @"facts", i, (uint32_t)[fresh getPhysicalIndexForLogicalIndex:i]);
	[fresh setPhysicalIndex:5 forLogicalIndex:3];
	put(c, @"facts", 8, (uint32_t)[fresh getPhysicalIndexForLogicalIndex:3]);
	MTLLogicalToPhysicalColorAttachmentMap *copy = [fresh copy];
	put(c, @"facts", 9, (uint32_t)[copy getPhysicalIndexForLogicalIndex:3]);
	[fresh reset];
	put(c, @"facts", 10, (uint32_t)[fresh getPhysicalIndexForLogicalIndex:3]);
	put(c, @"facts", 11, (uint32_t)[copy getPhysicalIndexForLogicalIndex:3]);
	put(c, @"facts", 12, [MTLRenderPassDescriptor new].supportColorAttachmentMapping);
	put(c, @"facts", 13, [MTLIndirectCommandBufferDescriptor new].supportColorAttachmentMapping);

	/* Pipelines name every attachment of the pass; those their functions
	 * leave unwritten keep their contents. */
	NSError *error = nil;
	MTLRenderPipelineDescriptor *pd[2] = {[MTLRenderPipelineDescriptor new], [MTLRenderPipelineDescriptor new]};
	for (NSUInteger p = 0; p < 2; p++)
		for (NSUInteger i = 0; i < 3; i++) {
			pd[p].colorAttachments[i].pixelFormat = MTLPixelFormatRGBA8Unorm;
			if (i > p)
				pd[p].colorAttachments[i].writeMask = MTLColorWriteMaskNone;
		}
	id<MTLRenderPipelineState> one =
		host_render_pipeline(c, pd[0], @"vert_offset_rect", @"frag_constant", MTLPixelFormatRGBA8Unorm, &error);
	id<MTLRenderPipelineState> two =
		one ? host_render_pipeline(c, pd[1], @"vert_offset_rect", @"frag_two_colors", MTLPixelFormatRGBA8Unorm, &error) : nil;
	if (!two)
		@throw failure(@"attachment map pipelines: %@", error.localizedDescription);
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.supportColorAttachmentMapping = YES;
	id<MTLTexture> targets[3];
	for (NSUInteger i = 0; i < 3; i++) {
		targets[i] = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
		pass.colorAttachments[i].texture = targets[i];
		pass.colorAttachments[i].loadAction = MTLLoadActionClear;
		pass.colorAttachments[i].clearColor = MTLClearColorMake(i == 0, i == 1, i == 2, 1);
		pass.colorAttachments[i].storeAction = MTLStoreActionStore;
	}
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
	float gray[] = {0.5f, 0.5f, 0.5f, 1};
	MTLLogicalToPhysicalColorAttachmentMap *map = [MTLLogicalToPhysicalColorAttachmentMap new];
	[r setRenderPipelineState:one];
	[r setFragmentBytes:gray length:sizeof gray atIndex:0];
	for (NSUInteger column = 0; column < 4; column++) {
		float rect[] = {column * 0.5f - 1, -1, column * 0.5f - 0.5f, 1, 0};
		if (column == 0)
			[map setPhysicalIndex:2 forLogicalIndex:0];
		if (column == 1)
			[map setPhysicalIndex:1 forLogicalIndex:0];
		if (column == 3) {
			[map setPhysicalIndex:1 forLogicalIndex:0];
			[map setPhysicalIndex:0 forLogicalIndex:1];
			[r setRenderPipelineState:two];
		}
		[r setColorAttachmentMap:column == 2 ? nil : map];
		[r setVertexBytes:rect length:sizeof rect atIndex:0];
		[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	}
	[r endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	for (NSUInteger i = 0; i < 3; i++)
		[b copyFromTexture:targets[i] sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			sourceSize:MTLSizeMake(8, 4, 1) toBuffer:c.buffers[@"images"] destinationOffset:i * 8 * 4 * 4
		   destinationBytesPerRow:8 * 4 destinationBytesPerImage:8 * 4 * 4];
	[b endEncoding];
	wait_for(cb, c.spec);
}

/* A pool's views of a texture, a swizzled view, a buffer's view and a
 * copy into another pool, read through their resource IDs; facts on its
 * counts and IDs. */
API_AVAILABLE(macos(26.0))
static void texture_view_pools(Case *c, id<MTLCommandQueue> queue)
{
	id<MTLDevice> d = c.device;
	MTLResourceViewPoolDescriptor *pd = [MTLResourceViewPoolDescriptor new];
	pd.resourceViewCount = 4;
	pd.label = @"pool";
	NSError *error = nil;
	id<MTLTextureViewPool> pool = [d newTextureViewPoolWithDescriptor:pd error:&error];
	id<MTLTextureViewPool> other = pool ? [d newTextureViewPoolWithDescriptor:pd error:&error] : nil;
	if (!other)
		@throw failure(@"texture view pool: %@", error.localizedDescription);
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
										       width:1 height:1 mipmapped:NO];
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	td.storageMode = MTLStorageModeShared;
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	uint8_t texel[] = {64, 128, 192, 255};
	[t replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:texel bytesPerRow:4];
	id<MTLBuffer> backing = [d newBufferWithLength:256 options:MTLResourceStorageModeShared];
	uint8_t red[] = {255, 0, 0, 255};
	memcpy(backing.contents, red, sizeof red);
	MTLTextureViewDescriptor *vd = [MTLTextureViewDescriptor new];
	vd.pixelFormat = MTLPixelFormatRGBA8Unorm;
	vd.textureType = MTLTextureType2D;
	vd.levelRange = NSMakeRange(0, 1);
	vd.sliceRange = NSMakeRange(0, 1);
	vd.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue, MTLTextureSwizzleGreen, MTLTextureSwizzleRed,
						   MTLTextureSwizzleOne);
	MTLResourceID plain = [pool setTextureView:t atIndex:0];
	MTLResourceID swizzled = [pool setTextureView:t descriptor:vd atIndex:1];
	MTLResourceID buffered = [pool setTextureViewFromBuffer:backing descriptor:td offset:0 bytesPerRow:256 atIndex:2];
	MTLResourceID copied = [other copyResourceViewsFromPool:pool sourceRange:NSMakeRange(0, 3) destinationIndex:1];
	MTLResourceID base = pool.baseResourceID;
	put(c, @"facts", 0, (uint32_t)pool.resourceViewCount);
	/* An entry keeps its ID as its view changes. */
	MTLResourceID first = [pool setTextureView:t descriptor:vd atIndex:3];
	put(c, @"facts", 1, other.resourceViewCount == 4 && [pool setTextureView:t atIndex:3]._impl == first._impl);
	put(c, @"facts", 2, pool.device == d);
	put(c, @"facts", 3, plain._impl == base._impl);
	put(c, @"facts", 4, swizzled._impl == base._impl + 1 && buffered._impl == base._impl + 2);
	put(c, @"facts", 5, copied._impl == other.baseResourceID._impl + 1);
	put(c, @"facts", 6, plain._impl != 0 && swizzled._impl != plain._impl && buffered._impl != plain._impl);

	id<MTLBuffer> table = [d newBufferWithLength:4 * sizeof(MTLResourceID) options:MTLResourceStorageModeShared];
	MTLResourceID *ids = table.contents;
	ids[0] = plain;
	ids[1] = swizzled;
	ids[2] = buffered;
	ids[3] = copied;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:host_pipeline(c, @"pool_views")];
	[e setBuffer:table offset:0 atIndex:0];
	[e setBuffer:c.buffers[@"texels"] offset:0 atIndex:1];
	[e useResource:t usage:MTLResourceUsageRead];
	[e useResource:backing usage:MTLResourceUsageRead];
	[e dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
	[e endEncoding];
	wait_for(cb, c.spec);
}

static void host_texture_view_pools(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	(void)h;
	if (@available(macOS 26.0, *))
		texture_view_pools(c, queue);
	else
		@throw failure(@"texture view pools need macOS 26");
}

static void host_color_attachment_map(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	(void)h;
	if (@available(macOS 26.0, *))
		color_attachment_map(c, queue);
	else
		@throw failure(@"color attachment maps need macOS 26");
}

/* A sparse texture of 2 x 2 tiles on a sparse heap: tile sizes and region
 * conversions, then probes of each tile's residency and contents after
 * mapping tile 0 by region and tile 1 through an indirect buffer and
 * clearing the texture, after unmapping tile 1 by a region list, and after moving tile 0's memory to a second texture, whose
 * probes follow. */
static void host_sparse_textures(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	(void)h;
	id<MTLDevice> d = c.device;
	/* Tile sizes are the device's own; they agree with its page sizes. */
	MTLSize fallback = [d sparseTileSizeWithTextureType:MTLTextureType2D pixelFormat:MTLPixelFormatRGBA8Unorm sampleCount:1];
	MTLSize tile = [d sparseTileSizeWithTextureType:MTLTextureType2D
					    pixelFormat:MTLPixelFormatRGBA8Unorm
					    sampleCount:1
					 sparsePageSize:MTLSparsePageSize64];
	NSUInteger page = [d sparseTileSizeInBytesForSparsePageSize:MTLSparsePageSize64];
	put(c, @"facts", 0, tile.width * tile.height * 4 == page && page == 65536 && tile.depth == 1);
	put(c, @"facts", 1, fallback.width * fallback.height * 4 == d.sparseTileSizeInBytes);
	MTLRegion pixels = MTLRegionMake2D(1, 1, tile.width, tile.height), outward, inward, back;
	[d convertSparsePixelRegions:&pixels toTileRegions:&outward withTileSize:tile
		       alignmentMode:MTLSparseTextureRegionAlignmentModeOutward numRegions:1];
	[d convertSparsePixelRegions:&pixels toTileRegions:&inward withTileSize:tile
		       alignmentMode:MTLSparseTextureRegionAlignmentModeInward numRegions:1];
	MTLRegion tiles = MTLRegionMake2D(1, 0, 1, 1);
	[d convertSparseTileRegions:&tiles toPixelRegions:&back withTileSize:tile numRegions:1];
	NSUInteger values[] = {outward.origin.x, outward.origin.y, outward.size.width, outward.size.height,
			       inward.origin.x, inward.origin.y, inward.size.width, inward.size.height,
			       back.origin.x / tile.width, back.origin.y, back.size.width / tile.width,
			       back.size.height / tile.height};
	for (NSUInteger i = 0; i < 12; i++)
		put(c, @"facts", 2 + i, (uint32_t)values[i]);

	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.type = MTLHeapTypeSparse;
	hd.storageMode = MTLStorageModePrivate;
	hd.sparsePageSize = MTLSparsePageSize64;
	hd.size = 4 * page;
	id<MTLHeap> heap = [d newHeapWithDescriptor:hd];
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
										       width:tile.width * 2
										      height:tile.height * 2
										   mipmapped:NO];
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> t = [heap newTextureWithDescriptor:td];
	id<MTLTexture> moved = t ? [heap newTextureWithDescriptor:td] : nil;
	if (!moved)
		@throw failure(@"no sparse textures");
	put(c, @"facts", 14, heap.type == MTLHeapTypeSparse && t.isSparse && moved.isSparse);
	put(c, @"facts", 15, (uint32_t)t.firstMipmapInTail);
	put(c, @"facts", 16, (uint32_t)t.tailSizeInBytes);
	if (@available(macOS 26.0, *))
		put(c, @"facts", 17, t.sparseTextureTier >= MTLTextureSparseTier1 &&
					     [d newBufferWithLength:16 options:0].sparseBufferTier == MTLBufferSparseTierNone);

	uint32_t extents[] = {(uint32_t)tile.width, (uint32_t)tile.height}, *extent = extents;
	id<MTLComputePipelineState> probe = host_pipeline(c, @"sparse_probe");
	void (^probes)(id<MTLCommandBuffer>, id<MTLTexture>, NSUInteger) = ^(id<MTLCommandBuffer> cb, id<MTLTexture> texture,
									     NSUInteger round) {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:probe];
		[e setTexture:texture atIndex:0];
		[e setBytes:extent length:sizeof extents atIndex:0];
		[e setBuffer:c.buffers[@"probes"] offset:round * 4 * 16 atIndex:1];
		[e dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
		[e endEncoding];
	};
	/* The indirect buffer counts its mappings before them. */
	struct {
		uint32_t count;
		MTLMapIndirectArguments mapping;
	} last = {1, {.regionOriginX = 1, .regionOriginY = 0, .regionOriginZ = 0, .regionSizeWidth = 1,
		      .regionSizeHeight = 1, .regionSizeDepth = 1, .mipMapLevel = 0, .sliceId = 0}};
	id<MTLBuffer> indirect = [d newBufferWithBytes:&last length:sizeof last options:MTLResourceStorageModeShared];
	/* Heap textures are untracked: each encoder waits for the last. */
	void (^step)(void (^)(id<MTLCommandBuffer>)) = ^(void (^encode)(id<MTLCommandBuffer>)) {
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		encode(cb);
		wait_for(cb, c.spec);
	};
	step(^(id<MTLCommandBuffer> cb) {
		id<MTLResourceStateCommandEncoder> r = [cb resourceStateCommandEncoder];
		[r updateTextureMapping:t mode:MTLSparseTextureMappingModeMap region:MTLRegionMake2D(0, 0, 1, 1) mipLevel:0 slice:0];
		[r updateTextureMapping:t mode:MTLSparseTextureMappingModeMap indirectBuffer:indirect indirectBufferOffset:0];
		[r endEncoding];
	});
	step(^(id<MTLCommandBuffer> cb) {
		MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.colorAttachments[0].texture = t;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(128 / 255.0, 64 / 255.0, 192 / 255.0, 1);
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
	});
	step(^(id<MTLCommandBuffer> cb) { probes(cb, t, 0); });
	/* Every sparse function over the cleared texture and a depth texture
	 * of 0.25. */
	MTLTextureDescriptor *zd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
										       width:td.width
										      height:td.height
										   mipmapped:NO];
	zd.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	zd.storageMode = MTLStorageModePrivate;
	id<MTLTexture> z = [d newTextureWithDescriptor:zd];
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	id<MTLSamplerState> nearest = [d newSamplerStateWithDescriptor:sd];
	id<MTLComputePipelineState> functions = host_pipeline(c, @"sparse_functions");
	step(^(id<MTLCommandBuffer> cb) {
		MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.depthAttachment.texture = z;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.clearDepth = 0.25;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:functions];
		[e setTexture:t atIndex:0];
		[e setTexture:z atIndex:1];
		[e setSamplerState:nearest atIndex:0];
		[e setBytes:extent length:sizeof extents atIndex:0];
		[e setBuffer:c.buffers[@"functions"] offset:0 atIndex:1];
		[e dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
		[e endEncoding];
	});
	step(^(id<MTLCommandBuffer> cb) {
		id<MTLResourceStateCommandEncoder> r =
			[cb resourceStateCommandEncoderWithDescriptor:[MTLResourceStatePassDescriptor resourceStatePassDescriptor]];
		MTLRegion last = MTLRegionMake2D(1, 0, 1, 1);
		NSUInteger level = 0, slice = 0;
		[r updateTextureMappings:t mode:MTLSparseTextureMappingModeUnmap regions:&last mipLevels:&level slices:&slice numRegions:1];
		[r endEncoding];
	});
	step(^(id<MTLCommandBuffer> cb) { probes(cb, t, 1); });
	step(^(id<MTLCommandBuffer> cb) {
		id<MTLResourceStateCommandEncoder> r = [cb resourceStateCommandEncoder];
		[r moveTextureMappingsFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
				       sourceSize:MTLSizeMake(1, 1, 1) toTexture:moved destinationSlice:0
				 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
		[r endEncoding];
	});
	step(^(id<MTLCommandBuffer> cb) {
		probes(cb, t, 2);
		probes(cb, moved, 3);
	});
}

/* Stencil resolve without stencil export: four draws each cover one sample
 * of a multisampled depth and stencil texture, at depth 0.1 * (k + 1), and
 * replace its stencil with k + 1; passes resolve it by sample 0, and by the
 * sample whose depth resolves by min and by max. */
static void host_stencil_resolve(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	(void)h;
	id<MTLDevice> d = c.device;
	id<MTLLibrary> library = library_named(d, c.plan, c.spec[@"library"], c.libraries);
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	pd.vertexFunction = function_named(library, @"vert_offset_rect", nil);
	pd.fragmentFunction = function_named(library, @"frag_one_sample", nil);
	pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
	pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
	pd.rasterSampleCount = 4;
	NSError *error = nil;
	id<MTLRenderPipelineState> p = [d newRenderPipelineStateWithDescriptor:pd error:&error];
	if (!p)
		@throw failure(@"one-sample pipeline: %@", error.localizedDescription);
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	dd.depthCompareFunction = MTLCompareFunctionAlways;
	dd.depthWriteEnabled = YES;
	MTLStencilDescriptor *stencil = [MTLStencilDescriptor new];
	stencil.stencilCompareFunction = MTLCompareFunctionAlways;
	stencil.depthStencilPassOperation = MTLStencilOperationReplace;
	dd.frontFaceStencil = stencil;
	dd.backFaceStencil = stencil;
	id<MTLDepthStencilState> ds = [d newDepthStencilStateWithDescriptor:dd];
	MTLTextureDescriptor *md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
										       width:8 height:8 mipmapped:NO];
	md.textureType = MTLTextureType2DMultisample;
	md.sampleCount = 4;
	md.usage = MTLTextureUsageRenderTarget;
	md.storageMode = MTLStorageModePrivate;
	MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
										       width:8 height:8 mipmapped:NO];
	rd.usage = MTLTextureUsageRenderTarget;
	rd.storageMode = MTLStorageModePrivate;
	MTLMultisampleDepthResolveFilter depths[] = {MTLMultisampleDepthResolveFilterSample0, MTLMultisampleDepthResolveFilterMin,
						     MTLMultisampleDepthResolveFilterMax};
	MTLMultisampleStencilResolveFilter stencils[] = {MTLMultisampleStencilResolveFilterSample0,
							 MTLMultisampleStencilResolveFilterDepthResolvedSample,
							 MTLMultisampleStencilResolveFilterDepthResolvedSample};
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	for (NSUInteger i = 0; i < 3; i++) {
		id<MTLTexture> ms = [d newTextureWithDescriptor:md], resolved = [d newTextureWithDescriptor:rd];
		MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.depthAttachment.texture = ms;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.clearDepth = 1;
		pass.depthAttachment.storeAction = MTLStoreActionMultisampleResolve;
		pass.depthAttachment.resolveTexture = resolved;
		pass.depthAttachment.depthResolveFilter = depths[i];
		pass.stencilAttachment.texture = ms;
		pass.stencilAttachment.loadAction = MTLLoadActionClear;
		pass.stencilAttachment.clearStencil = 0;
		pass.stencilAttachment.storeAction = MTLStoreActionMultisampleResolve;
		pass.stencilAttachment.resolveTexture = resolved;
		pass.stencilAttachment.stencilResolveFilter = stencils[i];
		id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
		float rect[] = {-1, -1, 1, 1, 0};
		[r setRenderPipelineState:p];
		[r setDepthStencilState:ds];
		[r setVertexBytes:rect length:sizeof rect atIndex:0];
		for (uint32_t k = 0; k < 4; k++) {
			[r setStencilReferenceValue:k + 1];
			[r setFragmentBytes:&k length:sizeof k atIndex:0];
			[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
		}
		[r endEncoding];
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b copyFromTexture:resolved sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			sourceSize:MTLSizeMake(8, 8, 1) toBuffer:c.buffers[@"stencils"] destinationOffset:i * 64
		   destinationBytesPerRow:8 destinationBytesPerImage:64 options:MTLBlitOptionStencilFromDepthStencil];
		[b copyFromTexture:resolved sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			sourceSize:MTLSizeMake(8, 8, 1) toBuffer:c.buffers[@"depths"] destinationOffset:i * 256
		   destinationBytesPerRow:32 destinationBytesPerImage:256 options:MTLBlitOptionDepthFromDepthStencil];
		[b endEncoding];
	}
	wait_for(cb, c.spec);
}

static void host_rate_maps(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLDevice> d = c.device;
	for (NSUInteger i = 0; i < 4; i++)
		put(c, @"facts", i, [d supportsRasterizationRateMapWithLayerCount:i]);
	float one = 1, horizontal[] = {1, 0.5f, 1}, vertical[] = {1, 0.25f};
	MTLRasterizationRateLayerDescriptor *full =
		[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(1, 1, 0) horizontal:&one vertical:&one];
	MTLRasterizationRateLayerDescriptor *low =
		[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(3, 2, 0) horizontal:horizontal vertical:vertical];
	MTLRasterizationRateMapDescriptor *two = [MTLRasterizationRateMapDescriptor new];
	two.screenSize = MTLSizeMake(640, 480, 0);
	[two setLayer:full atIndex:0];
	[two setLayer:full atIndex:1];
	id<MTLRasterizationRateMap> map = [d newRasterizationRateMapWithDescriptor:two];
	put(c, @"facts", 4, map != nil);
	if (!map)
		@throw failure(@"no rasterization rate map");
	MTLSize screen = map.screenSize, granularity = map.physicalGranularity;
	MTLSizeAndAlign parameters = map.parameterBufferSizeAndAlign;
	uint32_t sizes[] = {(uint32_t)screen.width, (uint32_t)screen.height, (uint32_t)map.layerCount,
			    (uint32_t)granularity.width, (uint32_t)granularity.height, (uint32_t)granularity.depth,
			    (uint32_t)parameters.size, (uint32_t)parameters.align};
	for (NSUInteger i = 0; i < 8; i++)
		put(c, @"facts", 5 + i, sizes[i]);
	for (NSUInteger layer = 0; layer < 2; layer++) {
		MTLSize physical = [map physicalSizeForLayer:layer];
		put(c, @"facts", 13 + layer * 2, (uint32_t)physical.width);
		put(c, @"facts", 14 + layer * 2, (uint32_t)physical.height);
	}
	MTLCoordinate2D point = MTLCoordinate2DMake(100.5f, 7.25f);
	MTLCoordinate2D to = [map mapScreenToPhysicalCoordinates:point forLayer:1];
	MTLCoordinate2D from = [map mapPhysicalToScreenCoordinates:point forLayer:1];
	float mapped[] = {to.x, to.y, from.x, from.y};
	for (NSUInteger i = 0; i < 4; i++) {
		uint32_t bits;
		memcpy(&bits, &mapped[i], sizeof bits);
		put(c, @"facts", 17 + i, bits);
	}
	id<MTLRasterizationRateMap> lower = [d newRasterizationRateMapWithDescriptor:
		[MTLRasterizationRateMapDescriptor rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(640, 480, 0) layer:low]];
	MTLSize shrunk = [lower physicalSizeForLayer:0];
	MTLCoordinate2D corner = [lower mapScreenToPhysicalCoordinates:MTLCoordinate2DMake(640, 480) forLayer:0];
	MTLCoordinate2D back = [lower mapPhysicalToScreenCoordinates:[lower mapScreenToPhysicalCoordinates:point forLayer:0] forLayer:0];
	put(c, @"facts", 21, lower && shrunk.width <= 640 && shrunk.height <= 480 && shrunk.width && shrunk.height);
	put(c, @"facts", 22, fabsf(corner.x - shrunk.width) < 0.01f && fabsf(corner.y - shrunk.height) < 0.01f);
	put(c, @"facts", 23, fabsf(back.x - point.x) < 0.01f && fabsf(back.y - point.y) < 0.01f);
	MTLRasterizationRateMapDescriptor *empty = [MTLRasterizationRateMapDescriptor new];
	empty.screenSize = MTLSizeMake(64, 64, 0);
	put(c, @"facts", 24, [d newRasterizationRateMapWithDescriptor:empty] != nil);

	id<MTLBuffer> data = [d newBufferWithLength:parameters.size options:MTLResourceStorageModeShared];
	[map copyParameterDataToBuffer:data offset:0];
	float coordinates[16];
	for (NSUInteger i = 0; i < 8; i++) {
		coordinates[i * 2] = i * 10.5f;
		coordinates[i * 2 + 1] = i * 3.25f;
	}
	id<MTLBuffer> in = [d newBufferWithBytes:coordinates length:sizeof coordinates options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:host_pipeline(c, @"rate_map_decode")];
	[e setBuffer:data offset:0 atIndex:0];
	[e setBuffer:in offset:0 atIndex:1];
	[e setBuffer:c.buffers[@"mapped"] offset:0 atIndex:2];
	[e dispatchThreads:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
	[e endEncoding];

	NSError *error = nil;
	id<MTLRenderPipelineState> p =
		host_render_pipeline(c, [MTLRenderPipelineDescriptor new], @"vert_left_half", @"frag_constant", MTLPixelFormatRGBA8Unorm, &error);
	if (!p)
		@throw failure(@"rate map pipeline: %@", error.localizedDescription);
	id<MTLTexture> t = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = t;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.rasterizationRateMap = [d newRasterizationRateMapWithDescriptor:
		[MTLRasterizationRateMapDescriptor rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(8, 4, 0) layer:full]];
	id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
	float red[] = {1, 0, 0, 1};
	[r setRenderPipelineState:p];
	[r setFragmentBytes:red length:sizeof red atIndex:0];
	[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[r endEncoding];
	read_back(cb, t, c.buffers[@"image"], 4, MTLBlitOptionNone);
	wait_for(cb, c.spec);
}

/* A left half drawn at depth 0.25, then the whole target drawn with the
 * depth bounds test passing only stored depths in [0.5, 1]: the second
 * colour reaches the right half alone. */
static void host_depth_bounds(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	if (@available(macOS 26.0, *)) {
		MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
		d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
		NSError *error = nil;
		id<MTLRenderPipelineState> p = host_render_pipeline(c, d, @"vert_offset_rect", @"frag_constant", MTLPixelFormatRGBA8Unorm, &error);
		if (!p)
			@throw failure(@"depth bounds pipeline: %@", error.localizedDescription);
		id<MTLTexture> t = host_target(c, MTLPixelFormatRGBA8Unorm, 8, 4);
		MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:8 height:4 mipmapped:NO];
		dd.usage = MTLTextureUsageRenderTarget;
		dd.storageMode = MTLStorageModePrivate;
		MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.colorAttachments[0].texture = t;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.depthAttachment.texture = [c.device newTextureWithDescriptor:dd];
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.clearDepth = 1.0;
		MTLDepthStencilDescriptor *write = [MTLDepthStencilDescriptor new], *keep = [MTLDepthStencilDescriptor new];
		write.depthCompareFunction = MTLCompareFunctionAlways;
		write.depthWriteEnabled = YES;
		keep.depthCompareFunction = MTLCompareFunctionAlways;
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:pass];
		[r setRenderPipelineState:p];
		float left[] = {-1, -1, 0, 1, 0.25f}, whole[] = {-1, -1, 1, 1, 0}, first[] = {1, 0, 0, 1}, second[] = {0, 0, 1, 1};
		[r setDepthStencilState:[c.device newDepthStencilStateWithDescriptor:write]];
		[r setVertexBytes:left length:sizeof left atIndex:0];
		[r setFragmentBytes:first length:sizeof first atIndex:0];
		[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
		[r setDepthStencilState:[c.device newDepthStencilStateWithDescriptor:keep]];
		[r setDepthTestMinBound:0.5f maxBound:1.0f];
		[r setVertexBytes:whole length:sizeof whole atIndex:0];
		[r setFragmentBytes:second length:sizeof second atIndex:0];
		[r drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
		[r endEncoding];
		read_back(cb, t, c.buffers[@"image"], 4, MTLBlitOptionNone);
		wait_for(cb, c.spec);
	} else {
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"depth bounds need macOS 26" userInfo:nil];
	}
}

/* What textures report about how they were made: a cube from its class
 * method, a view's parent, level and slice, a texture over a buffer, a
 * descriptor's swizzle, layout and compression, shareable textures,
 * resources' heap offsets and sizes, ownership, a view's minimum LOD,
 * views from a descriptor and from a pixel format alone. */
static void host_texture_properties(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLDevice> device = c.device;
	MTLTextureDescriptor *cd = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm size:8 mipmapped:YES];
	id<MTLTexture> cube = [device newTextureWithDescriptor:cd];
	uint32_t cube_words[] = {(uint32_t)cube.textureType, (uint32_t)cube.width, (uint32_t)cube.height,
				 (uint32_t)cube.mipmapLevelCount, (uint32_t)cube.arrayLength, (uint32_t)cube.usage};
	for (NSUInteger k = 0; k < 6; k++)
		put(c, @"props", k, cube_words[k]);
	MTLTextureDescriptor *ad = [MTLTextureDescriptor new];
	ad.textureType = MTLTextureType2DArray;
	ad.pixelFormat = MTLPixelFormatRGBA8Unorm;
	ad.width = ad.height = 8;
	ad.mipmapLevelCount = 3;
	ad.arrayLength = 4;
	ad.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	id<MTLTexture> array = [device newTextureWithDescriptor:ad];
	id<MTLTexture> view = [array newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm textureType:MTLTextureType2DArray
							   levels:NSMakeRange(1, 2) slices:NSMakeRange(1, 2)];
	uint32_t view_words[] = {view.parentTexture == array, (uint32_t)view.parentRelativeLevel, (uint32_t)view.parentRelativeSlice,
				 (uint32_t)view.width, (uint32_t)view.mipmapLevelCount, (uint32_t)view.arrayLength};
	for (NSUInteger k = 0; k < 6; k++)
		put(c, @"props", 6 + k, view_words[k]);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	put(c, @"props", 12, view.rootResource == array);
#pragma clang diagnostic pop
	id<MTLBuffer> backing = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *bd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:4 mipmapped:NO];
	bd.storageMode = MTLStorageModeShared;
	id<MTLTexture> linear = [backing newTextureWithDescriptor:bd offset:256 bytesPerRow:64];
	put(c, @"props", 13, linear.buffer == backing);
	put(c, @"props", 14, (uint32_t)linear.bufferOffset);
	put(c, @"props", 15, (uint32_t)linear.bufferBytesPerRow);
	MTLTextureDescriptor *sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	sd.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue, MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleOne);
	sd.allowGPUOptimizedContents = NO;
	sd.compressionType = MTLTextureCompressionTypeLossless;
	id<MTLTexture> swizzled = [device newTextureWithDescriptor:sd];
	put(c, @"props", 16, swizzled.swizzle.red);
	put(c, @"props", 17, swizzled.swizzle.green);
	put(c, @"props", 18, swizzled.swizzle.blue);
	put(c, @"props", 19, swizzled.swizzle.alpha);
	put(c, @"props", 20, swizzled.allowGPUOptimizedContents);
	put(c, @"props", 21, (uint32_t)swizzled.compressionType);
	MTLTextureDescriptor *pd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	pd.storageMode = MTLStorageModePrivate;
	put(c, @"props", 22, [device newSharedTextureWithDescriptor:pd].shareable);
	put(c, @"props", 23, [device newTextureWithDescriptor:pd].shareable);
	/* Placement: a buffer one alignment in, a texture after it. */
	MTLSizeAndAlign ba = [device heapBufferSizeAndAlignWithLength:4096 options:MTLResourceStorageModePrivate];
	MTLSizeAndAlign ta = [device heapTextureSizeAndAlignWithDescriptor:pd];
	NSUInteger texture_at = (ba.align + ba.size + ta.align - 1) / ta.align * ta.align;
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.type = MTLHeapTypePlacement;
	hd.storageMode = MTLStorageModePrivate;
	hd.size = texture_at + ta.size;
	id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
	id<MTLBuffer> placed = [heap newBufferWithLength:4096 options:MTLResourceStorageModePrivate offset:ba.align];
	id<MTLTexture> placed_texture = [heap newTextureWithDescriptor:pd offset:texture_at];
	put(c, @"props", 24, placed.heapOffset == ba.align);
	put(c, @"props", 25, placed_texture.heapOffset == texture_at);
	put(c, @"props", 26, backing.allocatedSize >= backing.length && heap.size >= hd.size);
	/* Memory charged to this process's identity. */
	if (@available(macOS 14.4, *)) {
		task_id_token_t token;
		kern_return_t made = task_create_identity_token(mach_task_self(), &token);
		put(c, @"props", 27, made == KERN_SUCCESS && [backing setOwnerWithIdentity:token] == KERN_SUCCESS);
		if (made == KERN_SUCCESS)
			mach_port_deallocate(mach_task_self(), token);
	}
	if (@available(macOS 26.0, *)) {
		MTLTextureViewDescriptor *vd = [MTLTextureViewDescriptor new];
		vd.pixelFormat = MTLPixelFormatRGBA8Unorm;
		vd.textureType = MTLTextureType2D;
		vd.levelRange = NSMakeRange(1, 1);
		vd.sliceRange = NSMakeRange(2, 1);
		vd.swizzle = MTLTextureSwizzleChannelsDefault;
		id<MTLTexture> described = [array newTextureViewWithDescriptor:vd];
		put(c, @"props", 28, (uint32_t)described.width);
		put(c, @"props", 29, (uint32_t)described.arrayLength);
		put(c, @"props", 30, (uint32_t)described.parentRelativeSlice);
		if (@available(macOS 27.0, *)) {
			vd.levelRange = NSMakeRange(0, 3);
			vd.minLOD = 1.5f;
			put(c, @"props", 31, (uint32_t)([array newTextureViewWithDescriptor:vd].minLOD * 2.0f));
		}
	}
	id<MTLTexture> srgb = [array newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
	put(c, @"props", 32, srgb.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
	put(c, @"props", 33, (uint32_t)srgb.mipmapLevelCount);
}

/* A residency set's allocations as they are added and removed, then the
 * set making an argument buffer's targets resident: from the queue, then
 * from the command buffer alone. */
static void host_residency(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	if (@available(macOS 15.0, *)) {
		id<MTLBuffer> data = c.buffers[@"data"], k = c.buffers[@"k"], spare = c.buffers[@"spare"];
		MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
		rd.initialCapacity = 2;
		NSError *error = nil;
		id<MTLResidencySet> set = [c.device newResidencySetWithDescriptor:rd error:&error];
		if (!set)
			@throw failure(@"residency set: %@", error.localizedDescription);
		[set addAllocation:data];
		__unsafe_unretained id<MTLAllocation> more[] = {k, spare};
		[set addAllocations:more count:2];
		put(c, @"props", 0, (uint32_t)set.allocationCount);
		put(c, @"props", 1, [set containsAllocation:spare]);
		[set removeAllocation:spare];
		put(c, @"props", 2, [set containsAllocation:spare]);
		__unsafe_unretained id<MTLAllocation> again[] = {spare};
		[set addAllocations:again count:1];
		[set removeAllocations:again count:1];
		put(c, @"props", 3, (uint32_t)set.allAllocations.count);
		[set commit];
		[set requestResidency];
		put(c, @"props", 4, set.allocatedSize >= data.length + k.length);
		id<MTLBuffer> arguments = c.buffers[@"arguments"];
		uint64_t addresses[] = {data.gpuAddress, k.gpuAddress};
		memcpy(arguments.contents, addresses, sizeof addresses);
		id<MTLComputePipelineState> p = host_pipeline(c, @"through_arguments");
		NSUInteger n = data.length / 4;
		void (^pass)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:p];
			[e setBuffer:arguments offset:0 atIndex:0];
			[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(n, 64), 1, 1)];
			[e endEncoding];
			wait_for(cb, c.spec);
		};
		[queue addResidencySet:set];
		pass([queue commandBuffer]);
		[queue removeResidencySet:set];
		__unsafe_unretained id<MTLResidencySet> sets[] = {set};
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		[cb useResidencySets:sets count:1];
		pass(cb);
		[queue addResidencySets:sets count:1];
		[queue removeResidencySets:sets count:1];
		[set endResidency];
		[set removeAllAllocations];
		[set commit];
		put(c, @"props", 5, (uint32_t)set.allocationCount);
	} else {
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:@"residency sets need macOS 15" userInfo:nil];
	}
}

/* Command queues of at most two command buffers, made by count and by
 * descriptor, each running four in order; what command buffers report
 * of their queue, references and the CPU's scheduling times. */
static void host_queues(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLComputePipelineState> digit = host_pipeline(c, @"append_digit");
	NSMutableArray<id<MTLCommandQueue>> *queues = [NSMutableArray arrayWithObject:[c.device newCommandQueueWithMaxCommandBufferCount:2]];
	if (@available(macOS 15.0, *)) {
		MTLCommandQueueDescriptor *qd = [MTLCommandQueueDescriptor new];
		qd.maxCommandBufferCount = 2;
		put(c, @"props", 0, (uint32_t)qd.maxCommandBufferCount);
		[queues addObject:[c.device newCommandQueueWithDescriptor:qd]];
	}
	for (NSUInteger q = 0; q < queues.count; q++) {
		id<MTLBuffer> order = c.buffers[q ? @"order_descriptor" : @"order_count"];
		id<MTLCommandBuffer> last = nil;
		for (uint32_t d = 1; d <= 4; d++) {
			last = [queues[q] commandBuffer];
			id<MTLComputeCommandEncoder> e = [last computeCommandEncoder];
			dispatch_one(e, digit, @[ order ], d, MTLSizeMake(1, 1, 1));
			[e endEncoding];
			[last commit];
		}
		[last waitUntilScheduled];
		put(c, @"props", 5 + q, last.status >= MTLCommandBufferStatusScheduled);
		[last waitUntilCompleted];
	}
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	put(c, @"props", 1, cb.commandQueue == queue);
	put(c, @"props", 2, cb.retainedReferences);
	put(c, @"props", 3, [queue commandBufferWithUnretainedReferences].retainedReferences);
	MTLCommandBufferDescriptor *cd = [MTLCommandBufferDescriptor new];
	cd.retainedReferences = NO;
	put(c, @"props", 4, [queue commandBufferWithDescriptor:cd].retainedReferences);
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	dispatch_one(e, digit, @[ c.buffers[@"order_count"] ], 5, MTLSizeMake(1, 1, 1));
	[e endEncoding];
	wait_for(cb, c.spec);
	uint64_t *times = c.buffers[@"times"].contents;
	times[0] = (uint64_t)(cb.kernelStartTime * 1e9);
	times[1] = (uint64_t)(cb.kernelEndTime * 1e9);
}

/* A function's reflection, one line each: its type, patch type and
 * control points; vertex attributes and stage-in attributes; function
 * constants by name; and, from the library, its bindings and annotation. */
static void host_function_reflection(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	NSMutableArray *lines = [NSMutableArray new];
	for (NSString *name in h[@"functions"]) {
		id<MTLFunction> f = function_named(library, name, nil);
		[lines addObject:[NSString stringWithFormat:@"%@ type=%ld patch=%ld points=%ld", name, (long)f.functionType, (long)f.patchType,
							    (long)f.patchControlPointCount]];
		for (MTLVertexAttribute *a in f.vertexAttributes)
			[lines addObject:[NSString stringWithFormat:@" vertex %@#%lu:%ld active=%d patch=%d control=%d", a.name,
								    (unsigned long)a.attributeIndex, (long)a.attributeType, a.active,
								    a.patchData, a.patchControlPointData]];
		for (MTLAttribute *a in f.stageInputAttributes)
			[lines addObject:[NSString stringWithFormat:@" input %@#%lu:%ld active=%d patch=%d control=%d", a.name,
								    (unsigned long)a.attributeIndex, (long)a.attributeType, a.active,
								    a.patchData, a.patchControlPointData]];
		NSDictionary<NSString *, MTLFunctionConstant *> *constants = f.functionConstantsDictionary;
		for (NSString *k in [constants.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
			MTLFunctionConstant *fc = constants[k];
			[lines addObject:[NSString stringWithFormat:@" constant %@#%lu:%ld required=%d", fc.name, (unsigned long)fc.index,
								    (long)fc.type, fc.required]];
		}
		if (@available(macOS 26.0, *)) {
			MTLFunctionReflection *r = [library reflectionForFunctionWithName:name];
			[lines addObject:[NSString stringWithFormat:@" library bindings=%lu annotation=%@", (unsigned long)r.bindings.count,
								    r.userAnnotation ?: @"-"]];
		}
	}
	write_text(c, @"text", [[lines componentsJoinedByString:@"\n"] stringByAppendingString:@"\n"]);
}

/* The case's library loaded again from its path, as the default library
 * of a bundle holding it and as the runner's own: each with the same
 * functions. */
static void host_library_loading(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLLibrary> library = library_named(c.device, c.plan, c.spec[@"library"], c.libraries);
	NSString *path = [plan_dir stringByAppendingPathComponent:c.plan[@"libraries"][c.spec[@"library"]]];
	NSArray *names = [library.functionNames sortedArrayUsingSelector:@selector(compare:)];
	NSError *error = nil;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	id<MTLLibrary> from_file = [c.device newLibraryWithFile:path error:&error];
#pragma clang diagnostic pop
	put(c, @"props", 0, [[from_file.functionNames sortedArrayUsingSelector:@selector(compare:)] isEqualToArray:names]);
	NSString *bundle = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"stela-%d.bundle", getpid()]];
	NSFileManager *fm = [NSFileManager defaultManager];
	[fm removeItemAtPath:bundle error:nil];
	[fm createDirectoryAtPath:[bundle stringByAppendingPathComponent:@"Contents/Resources"] withIntermediateDirectories:YES attributes:nil error:nil];
	[fm copyItemAtPath:path toPath:[bundle stringByAppendingPathComponent:@"Contents/Resources/default.metallib"] error:nil];
	@try {
		id<MTLLibrary> from_bundle = [c.device newDefaultLibraryWithBundle:[NSBundle bundleWithPath:bundle] error:&error];
		put(c, @"props", 1, [[from_bundle.functionNames sortedArrayUsingSelector:@selector(compare:)] isEqualToArray:names]);
	} @finally {
		[fm removeItemAtPath:bundle error:nil];
	}
	/* The default library the runner's bundle ships, this one. */
	id<MTLLibrary> from_main = [c.device newDefaultLibrary];
	put(c, @"props", 2, [[from_main.functionNames sortedArrayUsingSelector:@selector(compare:)] isEqualToArray:names]);
}

/* Metal's SPI for a library's extern functions. */
@protocol StelaExternFunctions
- (NSArray<NSString *> *)externFunctionNames;
@end

/* A library of extern functions, which Core Image declares for the kernels
 * it compiles and finds its stitchable functions by: it names no function
 * of its own and lists them as externFunctionNames. */
static void host_extern_functions(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	NSString *path = @"/System/Library/Frameworks/CoreImage.framework/Resources/ci_stdlib.metallib";
	NSError *error = nil;
	id<MTLLibrary> library = [c.device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
	if (!library) @throw failure(@"%@: %@", path, error);
	NSArray *externs = [library respondsToSelector:@selector(externFunctionNames)]
				   ? [(id<StelaExternFunctions>)library externFunctionNames]
				   : nil;
	put(c, @"props", 0, library.functionNames.count == 0);
	put(c, @"props", 1, [externs containsObject:@"_ci_premul"]);
}

/* Debug groups, signposts and markers around real work, which they leave
 * alone, and a barrier between two encoders' dispatches by stage. */
static void host_debug(Case *c, id<MTLCommandQueue> queue, NSDictionary *h)
{
	id<MTLBuffer> values = c.buffers[@"values"];
	[values addDebugMarker:@"values" range:NSMakeRange(0, values.length)];
	id<MTLComputePipelineState> write = host_pipeline(c, @"write_values"), twice = host_pipeline(c, @"double_values");
	NSUInteger n = values.length / 4;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	[cb pushDebugGroup:@"frame"];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e pushDebugGroup:@"write"];
	[e insertDebugSignpost:@"before"];
	dispatch_one(e, write, @[ values ], 3, MTLSizeMake(n, 1, 1));
	[e popDebugGroup];
	[e endEncoding];
	e = [cb computeCommandEncoder];
	if (@available(macOS 26.0, *))
		[e barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch];
	[e setComputePipelineState:twice];
	[e setBuffer:values offset:0 atIndex:0];
	[e setBuffer:c.buffers[@"doubled"] offset:0 atIndex:1];
	[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(n, 64), 1, 1)];
	[e endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b pushDebugGroup:@"copy"];
	[b insertDebugSignpost:@"copy"];
	[b copyFromBuffer:c.buffers[@"doubled"] sourceOffset:0 toBuffer:c.buffers[@"copied"] destinationOffset:0 size:values.length];
	[b popDebugGroup];
	[b endEncoding];
	[cb popDebugGroup];
	wait_for(cb, c.spec);
	[values removeAllDebugMarkers];
}

static void run_host(Case *c, id<MTLCommandQueue> queue, NSDictionary *spec)
{
	NSDictionary *h = spec[@"host"];
	NSDictionary *work = @{
		@"drawables" : [NSValue valueWithPointer:host_drawables],
		@"iosurface" : [NSValue valueWithPointer:host_iosurface],
		@"iosurface_buffer" : [NSValue valueWithPointer:host_iosurface_buffer],
		@"two_plane" : [NSValue valueWithPointer:host_two_plane],
		@"two_plane_render" : [NSValue valueWithPointer:host_two_plane_render],
		@"share" : [NSValue valueWithPointer:host_share],
		@"archive" : [NSValue valueWithPointer:host_archive],
		@"errors" : [NSValue valueWithPointer:host_errors],
		@"purgeable" : [NSValue valueWithPointer:host_purgeable],
		@"streams" : [NSValue valueWithPointer:host_streams],
		@"churn" : [NSValue valueWithPointer:host_churn],
		@"fresh_resources" : [NSValue valueWithPointer:host_fresh_resources],
		@"large_buffer" : [NSValue valueWithPointer:host_large_buffer],
		@"rebind" : [NSValue valueWithPointer:host_rebind],
		@"dynamic_stride" : [NSValue valueWithPointer:host_dynamic_stride],
		@"render_target_barrier" : [NSValue valueWithPointer:host_render_target_barrier],
		@"winding_coverage" : [NSValue valueWithPointer:host_winding_coverage],
		@"io" : [NSValue valueWithPointer:host_io],
		@"barriers" : [NSValue valueWithPointer:host_barriers],
		@"sync_texture" : [NSValue valueWithPointer:host_sync_texture],
		@"scissors" : [NSValue valueWithPointer:host_scissors],
		@"deferred_store" : [NSValue valueWithPointer:host_deferred_store],
		@"queries" : [NSValue valueWithPointer:host_queries],
		@"dynamic_library" : [NSValue valueWithPointer:host_dynamic_library],
		@"resolve_counters" : [NSValue valueWithPointer:host_resolve_counters],
		@"argument_descriptors" : [NSValue valueWithPointer:host_argument_descriptors],
		@"aliasable" : [NSValue valueWithPointer:host_aliasable],
		@"heap_fit" : [NSValue valueWithPointer:host_heap_fit],
		@"cpu_clear" : [NSValue valueWithPointer:host_cpu_clear],
		@"region_clear" : [NSValue valueWithPointer:host_region_clear],
		@"texture_properties" : [NSValue valueWithPointer:host_texture_properties],
		@"residency" : [NSValue valueWithPointer:host_residency],
		@"queues" : [NSValue valueWithPointer:host_queues],
		@"function_reflection" : [NSValue valueWithPointer:host_function_reflection],
		@"library_loading" : [NSValue valueWithPointer:host_library_loading],
		@"extern_functions" : [NSValue valueWithPointer:host_extern_functions],
		@"debug" : [NSValue valueWithPointer:host_debug],
		@"depth_bounds" : [NSValue valueWithPointer:host_depth_bounds],
		@"rate_maps" : [NSValue valueWithPointer:host_rate_maps],
		@"color_attachment_map" : [NSValue valueWithPointer:host_color_attachment_map],
		@"texture_view_pools" : [NSValue valueWithPointer:host_texture_view_pools],
		@"sparse_textures" : [NSValue valueWithPointer:host_sparse_textures],
		@"stencil_resolve" : [NSValue valueWithPointer:host_stencil_resolve],
	};
	NSValue *v = work[h[@"kind"]];
	if (!v)
		@throw failure(@"unknown host work %@", h[@"kind"]);
	((void (*)(Case *, id<MTLCommandQueue>, NSDictionary *))v.pointerValue)(c, queue, h);
}

static void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue, NSDictionary *plan, NSDictionary *spec,
		     NSString *dir, NSMutableDictionary *libraries)
{
	Case *c = [Case new];
	c.spec = spec;
	c.name = spec[@"name"];
	c.dir = dir;
	c.device = device;
	c.buffers = [NSMutableDictionary new];
	c.textures = [NSMutableDictionary new];
	c.samplers = [NSMutableDictionary new];
	c.resident = [NSMutableArray new];
	c.fences = [NSMutableDictionary new];
	c.events = [NSMutableDictionary new];
	c.icbs = [NSMutableDictionary new];
	c.heaps = [NSMutableDictionary new];
	c.libraries = libraries;
	c.plan = plan;
	c.queue = queue;
	NSString *missing = unmet(device, spec[@"requires"]);
	if (missing)
		@throw [NSException exceptionWithName:@"CaseUnsupported" reason:missing userInfo:nil];
	for (NSDictionary *h in spec[@"heaps"])
		c.heaps[h[@"name"]] = make_heap(c, h, spec[@"textures"]);
	for (NSDictionary *b in spec[@"buffers"])
		c.buffers[b[@"name"]] = make_buffer(c, b);
	for (NSDictionary *t in spec[@"textures"]) {
		id<MTLTexture> texture = make_texture(c, t, queue);
		if (texture)
			c.textures[t[@"name"]] = texture;
	}
	for (NSDictionary *b in spec[@"buffers"])
		if (b[@"size_and_align_of"])
			write_heap_answers(c, b, spec[@"textures"]);
	for (NSDictionary *s in spec[@"samplers"])
		c.samplers[s[@"name"]] = make_sampler(c, s);
	for (NSDictionary *b in spec[@"icbs"])
		c.icbs[b[@"name"]] = make_icb(c, b);
	for (NSDictionary *a in spec[@"arguments"])
		encode_arguments(c, a);
	NSString *kind = spec[@"kind"];
	if ([kind isEqualToString:@"device"]) {
		write_json([dir stringByAppendingPathComponent:@"properties.json"], device_properties(device));
		return;
	}
	if ([kind isEqualToString:@"blit"]) {
		run_blits(c, queue, spec);
	} else if ([kind isEqualToString:@"framework"]) {
		run_framework(c, queue, spec);
	} else if ([kind isEqualToString:@"host"]) {
		run_host(c, queue, spec);
	} else if ([kind isEqualToString:@"sequence"]) {
		run_sequence(c, queue, plan, spec, libraries);
	} else {
		id<MTLLibrary> library = library_named(device, plan, spec[@"library"], libraries);
		if ([kind isEqualToString:@"compute"])
			run_compute(c, queue, library, spec);
		else if ([kind isEqualToString:@"render"])
			run_render(c, queue, library, spec);
		else
			@throw failure(@"unknown case kind %@", kind);
	}
	store_outputs(c, queue, spec);
}

int main(int argc, const char **argv)
{
	@autoreleasepool {
		case_lookups = [NSMutableDictionary new];
		lookup_tables = [NSMutableDictionary new];
		/* launchd names every job it starts, not only this service: a
		 * console shell has "0". Only the service's own name selects it. */
		const char *service = getenv("XPC_SERVICE_NAME");
		if (service && !strcmp(service, share_service.UTF8String))
			return serve_shares();
		if (argc < 3) {
			fprintf(stderr, "usage: %s PLAN_DIR RESULTS_DIR [--only SUBSTRING]... [--skip-done]\n", argv[0]);
			return 64;
		}
		plan_dir = [NSString stringWithUTF8String:argv[1]];
		NSString *results = [NSString stringWithUTF8String:argv[2]];
		NSMutableArray *only = [NSMutableArray new];
		BOOL skip_done = NO;
		for (int i = 3; i < argc; i++) {
			if (!strcmp(argv[i], "--only") && i + 1 < argc)
				[only addObject:[NSString stringWithUTF8String:argv[++i]]];
			else if (!strcmp(argv[i], "--skip-done"))
				skip_done = YES;
			else {
				fprintf(stderr, "unknown argument %s\n", argv[i]);
				return 64;
			}
		}
		NSData *plan_bytes = [NSData dataWithContentsOfFile:[plan_dir stringByAppendingPathComponent:@"plan.json"]];
		NSDictionary *plan = plan_bytes ? [NSJSONSerialization JSONObjectWithData:plan_bytes options:0 error:nil] : nil;
		if (!plan) {
			fprintf(stderr, "cannot read %s/plan.json\n", argv[1]);
			return 66;
		}
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (!device) {
			fprintf(stderr, "no Metal device\n");
			return 69;
		}
		id<MTLCommandQueue> queue = [device newCommandQueue];
		NSFileManager *fm = [NSFileManager defaultManager];
		[fm createDirectoryAtPath:results withIntermediateDirectories:YES attributes:nil error:nil];
		write_json([results stringByAppendingPathComponent:@"device.json"], @{
			@"name" : device.name,
			@"os" : [[NSProcessInfo processInfo] operatingSystemVersionString],
			@"plan" : plan[@"digest"] ?: @"",
		});
		NSMutableDictionary *libraries = [NSMutableDictionary new];
		unsigned ok = 0, failed = 0, skipped = 0;
		for (NSDictionary *spec in plan[@"cases"]) {
			NSString *name = spec[@"name"];
			if (only.count) {
				BOOL match = NO;
				for (NSString *o in only)
					match |= [name containsString:o];
				if (!match)
					continue;
			}
			NSString *dir = [results stringByAppendingPathComponent:name];
			NSString *status_path = [dir stringByAppendingPathComponent:@"status.json"];
			if (skip_done && [fm fileExistsAtPath:status_path])
				continue;
			[fm removeItemAtPath:dir error:nil];
			[fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
			write_json(status_path, @{@"status" : @"running"});
			@synchronized(lookup_tables) {
				[case_lookups removeAllObjects];
			}
			creation = spec[@"creation"];
			NSDate *start = [NSDate date];
			gpu_seconds = 0;
			timing = YES;
			@try {
				@autoreleasepool {
					run_case(device, queue, plan, spec, dir, libraries);
				}
				NSMutableDictionary *used = [NSMutableDictionary new];
				@synchronized(lookup_tables) {
					for (NSString *kind in case_lookups)
						used[kind] = [case_lookups[kind].allObjects sortedArrayUsingSelector:@selector(compare:)];
				}
				write_json(status_path, @{@"status" : @"ok", @"seconds" : @(-[start timeIntervalSinceNow]),
							  @"gpu_ms" : @(gpu_seconds * 1000.0), @"lookups" : used});
				ok++;
			} @catch (NSException *e) {
				BOOL timeout = [e.name isEqualToString:@"CaseTimeout"];
				BOOL unsupported = [e.name isEqualToString:@"CaseUnsupported"];
				write_json(status_path, @{@"status" : timeout ? @"timeout" : unsupported ? @"unsupported" : @"error",
							  @"error" : e.reason ?: e.name});
				if (unsupported) {
					skipped++;
					continue;
				}
				fprintf(stderr, "%s: %s\n", name.UTF8String, (e.reason ?: e.name).UTF8String);
				failed++;
				if (timeout)
					return 2;
			}
		}
		/* The tables' names, merged with those of earlier runs into this
		 * directory. */
		NSString *tables_path = [results stringByAppendingPathComponent:@"lookups.json"];
		NSData *earlier = [NSData dataWithContentsOfFile:tables_path];
		NSDictionary *before = earlier ? [NSJSONSerialization JSONObjectWithData:earlier options:0 error:nil] : @{};
		NSMutableDictionary *tables = [NSMutableDictionary new];
		@synchronized(lookup_tables) {
			for (NSString *kind in lookup_tables)
				tables[kind] = [NSMutableSet setWithSet:lookup_tables[kind]];
		}
		for (NSString *kind in before) {
			if (!tables[kind])
				tables[kind] = [NSMutableSet new];
			[tables[kind] addObjectsFromArray:before[kind]];
		}
		NSMutableDictionary *sorted = [NSMutableDictionary new];
		for (NSString *kind in tables)
			sorted[kind] = [[tables[kind] allObjects] sortedArrayUsingSelector:@selector(compare:)];
		write_json(tables_path, sorted);
		printf("%u ok, %u failed, %u unsupported\n", ok, failed, skipped);
	}
	return 0;
}

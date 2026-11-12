/* Buffers, textures, samplers, heaps, events and counters. */
#include "Bundle.h"

#define PAGE 0x4000ull

static uint64_t page_aligned(uint64_t length)
{
	return (length + PAGE - 1) & ~(PAGE - 1);
}

static NSUInteger storage_mode(NSUInteger options)
{
	return (options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift;
}

static bool cpu_visible(NSUInteger options)
{
	NSUInteger mode = storage_mode(options);
	return mode == MTLStorageModeShared || mode == MTLStorageModeManaged;
}

/* The public bits of MTLTextureUsage; the rest are Apple's drivers' own hints. */
#define TEXTURE_USAGE_KNOWN 0x37ull
/* The usage Apple's drivers report for a texture made with Unknown usage. */
#define TEXTURE_USAGE_UNKNOWN 0x17ull

void stela_texture_record(id descriptor, StelaTextureDescriptorRecord *record)
{
	MTLTextureSwizzleChannels swizzle = [descriptor swizzle];
	NSUInteger usage = [descriptor usage];
	if (usage & ~TEXTURE_USAGE_KNOWN) {
		StelaFailure failure = {.reason = "a private hint this device drops"};
		stela_report([[NSString stringWithFormat:@"texture usage %#lx", (unsigned long)usage] UTF8String], &failure);
	}
	*record = (StelaTextureDescriptorRecord){
		.texture_type = [descriptor textureType],
		.pixel_format = [descriptor pixelFormat],
		.width = [descriptor width],
		.height = [descriptor height],
		.depth = [descriptor depth],
		.mipmap_level_count = [descriptor mipmapLevelCount],
		.sample_count = [descriptor sampleCount],
		.array_length = [descriptor arrayLength],
		.options = [descriptor resourceOptions],
		.usage = usage & TEXTURE_USAGE_KNOWN,
		.swizzle = {swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha},
	};
}

static void stencil_record(id stencil, StelaStencilRecord *record)
{
	*record = (StelaStencilRecord){
		.compare_function = stencil ? [stencil stencilCompareFunction] : MTLCompareFunctionAlways,
		.stencil_failure_operation = stencil ? [stencil stencilFailureOperation] : MTLStencilOperationKeep,
		.depth_failure_operation = stencil ? [stencil depthFailureOperation] : MTLStencilOperationKeep,
		.depth_stencil_pass_operation = stencil ? [stencil depthStencilPassOperation] : MTLStencilOperationKeep,
		.read_mask = stencil ? [stencil readMask] : UINT32_MAX,
		.write_mask = stencil ? (uint32_t)[stencil writeMask] : UINT32_MAX,
	};
}

void stela_depth_stencil_record(id descriptor, StelaDepthStencilRecord *record)
{
	*record = (StelaDepthStencilRecord){
		.depth_compare_function = descriptor ? [descriptor depthCompareFunction] : MTLCompareFunctionAlways,
		.depth_write_enabled = descriptor ? [descriptor isDepthWriteEnabled] : NO,
	};
	stencil_record(descriptor ? [descriptor frontFaceStencil] : nil, &record->front_face_stencil);
	stencil_record(descriptor ? [descriptor backFaceStencil] : nil, &record->back_face_stencil);
}

@implementation StelaGPUObject

- (instancetype)initWithDevice:(StelaGPUDevice *)device handle:(uint64_t)handle
{
	self = [super init];
	if (self) {
		_device = device;
		_handle = handle;
	}
	return self;
}

- (StelaGPUDevice *)device
{
	return _device;
}

- (void)dealloc
{
	if (_handle) {
		[_device releaseHandle:_handle];
	}
}

@end

NSUInteger stela_purgeable_state(NSUInteger *slot, NSUInteger state)
{
	NSUInteger previous;
	if (state == MTLPurgeableStateNonVolatile || state == MTLPurgeableStateVolatile ||
	    state == MTLPurgeableStateEmpty) {
		previous = __atomic_exchange_n(slot, state == MTLPurgeableStateNonVolatile ? 0 : state,
					       __ATOMIC_RELAXED);
	} else {
		previous = __atomic_load_n(slot, __ATOMIC_RELAXED);
	}
	return previous ? previous : MTLPurgeableStateNonVolatile;
}

@implementation StelaGPUResource

- (NSUInteger)resourceOptions
{
	return _options;
}

- (NSUInteger)cpuCacheMode
{
	return (_options & MTLResourceCPUCacheModeMask) >> MTLResourceCPUCacheModeShift;
}

- (NSUInteger)storageMode
{
	return storage_mode(_options);
}

- (NSUInteger)hazardTrackingMode
{
	return (_options & MTLResourceHazardTrackingModeMask) >> MTLResourceHazardTrackingModeShift;
}

/* Memory stays resident: everything is nonvolatile, as a query or a request. */
/* A heap's resources take its state, which they cannot set apart from it. */
- (NSUInteger)setPurgeableState:(NSUInteger)state
{
	if (_heap) {
		return [(StelaGPUHeap *)_heap setPurgeableState:MTLPurgeableStateKeepCurrent];
	}
	return stela_purgeable_state(&_purgeable, state);
}

- (NSUInteger)allocatedSize
{
	return _allocatedSize;
}

/* Memory attribution to another task, which only the kernel's ledgers read. */
- (int)setOwnerWithIdentity:(unsigned int)identity
{
	return 0;
}

- (NSInteger)sparseBufferTier
{
	return 0;
}

- (NSUInteger)heapOffset
{
	return _heapOffset;
}

- (id)heap
{
	return _heap;
}

- (void)makeAliasable
{
	StelaHandle request = {.handle = _handle};
	StelaEmpty nothing;
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, MakeAliasable, &request, &nothing, &failure) != StelaStatusOk) {
		stela_report("makeAliasable", &failure);
	} else {
		_aliasable = YES;
	}
}

- (BOOL)isAliasable
{
	return _aliasable;
}

@end

@implementation StelaGPUBuffer

- (instancetype)initWithDevice:(StelaGPUDevice *)device
		      response:(const StelaBufferResponse *)response
		       options:(NSUInteger)options
			length:(NSUInteger)length
{
	self = [super initWithDevice:device handle:response->handle];
	if (self) {
		_options = options;
		_length = length;
		_gpuAddress = response->gpu_address;
		_allocatedSize = response->allocated_size;
		_heapOffset = response->heap_offset;
	}
	return self;
}

- (void)dealloc
{
	if (_slab) {
		/* Commands committed before this may still reach its place. */
		[_device releaseHandleAfterWork:_handle slab:_slab];
		_handle = 0;
	} else if (_deallocator) {
		_deallocator(_contents, _length);
	} else if (_mappedBytes) {
		mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)_contents, _mappedBytes);
	}
}

- (NSUInteger)length
{
	return _length;
}

- (void *)contents
{
	return _contents;
}

- (uint64_t)gpuAddress
{
	return _gpuAddress;
}

- (void)didModifyRange:(NSRange)range
{
	StelaBufferRangeRequest request = {.buffer = _handle, .offset = range.location, .length = range.length};
	StelaEmpty nothing;
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, BufferDidModifyRange, &request, &nothing, &failure) != StelaStatusOk) {
		stela_report("didModifyRange:", &failure);
	}
}

- (id)newTextureWithDescriptor:(id)descriptor offset:(NSUInteger)offset bytesPerRow:(NSUInteger)bytesPerRow
{
	return [self newTextureWithDescriptor:descriptor
				       offset:offset
				  bytesPerRow:bytesPerRow
				 secondOffset:0
			    secondBytesPerRow:0
				surfaceFormat:0];
}

/* A two-plane format's texture also names its second plane's bytes, and
 * the surface's pixel format, whose range YCbCr converts in. */
- (id)newTextureWithDescriptor:(id)descriptor
			offset:(NSUInteger)offset
		   bytesPerRow:(NSUInteger)bytesPerRow
		  secondOffset:(NSUInteger)secondOffset
	     secondBytesPerRow:(NSUInteger)secondBytesPerRow
		 surfaceFormat:(uint32_t)surfaceFormat
{
	StelaNewBufferTextureRequest request = {
		.buffer = _handle,
		.offset = offset,
		.bytes_per_row = bytesPerRow,
		.second_offset = secondOffset,
		.second_bytes_per_row = secondBytesPerRow,
		.surface_format = surfaceFormat,
	};
	stela_texture_record(descriptor, &request.descriptor);
	StelaTextureResponse response;
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, NewBufferTexture, &request, &response, &failure) != StelaStatusOk) {
		stela_report("a texture of a buffer", &failure);
		return nil;
	}
	StelaGPUTexture *texture = [[StelaGPUTexture alloc] initWithDevice:_device
								  response:&response
								descriptor:descriptor];
	texture->_buffer = self;
	texture->_bufferOffset = offset;
	texture->_bufferBytesPerRow = bytesPerRow;
	return texture;
}

/* Metal's private spelling of the above: a 2D image's bytes follow from its rows. */
- (id)newLinearTextureWithDescriptor:(id)descriptor
			      offset:(NSUInteger)offset
			 bytesPerRow:(NSUInteger)bytesPerRow
			bytesPerImage:(NSUInteger)bytesPerImage
{
	return [self newTextureWithDescriptor:descriptor offset:offset bytesPerRow:bytesPerRow];
}

- (void)addDebugMarker:(NSString *)marker range:(NSRange)range
{
	(void)marker;
	(void)range;
}

- (void)removeAllDebugMarkers
{
}

- (id)remoteStorageBuffer
{
	return nil;
}

- (id)newRemoteBufferViewForDevice:(id)device
{
	(void)device;
	return nil;
}

@end

@implementation StelaGPUTexture

- (instancetype)initWithDevice:(StelaGPUDevice *)device
		      response:(const StelaTextureResponse *)response
		    descriptor:(id)descriptor
{
	StelaTextureDescriptorRecord record;
	stela_texture_record(descriptor, &record);
	self = [self initWithDevice:device response:response record:&record];
	if (self && [descriptor respondsToSelector:@selector(allowGPUOptimizedContents)]) {
		_gpuOptimized = [descriptor allowGPUOptimizedContents];
	}
	/* Core Animation marks the descriptors of its layers' drawables. */
	if (self && [descriptor respondsToSelector:@selector(isDrawable)]) {
		_drawable = [descriptor isDrawable];
	}
	return self;
}

- (instancetype)initWithDevice:(StelaGPUDevice *)device
		      response:(const StelaTextureResponse *)response
			record:(const StelaTextureDescriptorRecord *)record
{
	self = [super initWithDevice:device handle:response->handle];
	if (self) {
		_record = *record;
		_options = record->options;
		_allocatedSize = response->allocated_size;
		_heapOffset = response->heap_offset;
		_resourceID = response->resource_id;
		_sparse = response->sparse != 0;
		_firstMipmapInTail = response->first_mipmap_in_tail;
		_tailSizeInBytes = response->tail_size;
		_textureType = record->texture_type;
		_pixelFormat = record->pixel_format;
		_width = record->width;
		_height = record->height;
		_depth = record->depth;
		_mipmapLevelCount = record->mipmap_level_count;
		_sampleCount = record->sample_count;
		_arrayLength = record->array_length;
		/* Unknown usage admits every use, and Apple's drivers report it so:
		 * shader reads and writes, rendering and pixel format views. */
		_usage = record->usage ? record->usage : TEXTURE_USAGE_UNKNOWN;
		_swizzle = (MTLTextureSwizzleChannels){record->swizzle[0], record->swizzle[1],
						       record->swizzle[2], record->swizzle[3]};
		_gpuOptimized = YES;
	}
	return self;
}

- (NSUInteger)textureType
{
	return _textureType;
}

- (NSUInteger)pixelFormat
{
	return _pixelFormat;
}

- (NSUInteger)width
{
	return _width;
}

- (NSUInteger)height
{
	return _height;
}

- (NSUInteger)depth
{
	return _depth;
}

- (NSUInteger)mipmapLevelCount
{
	return _mipmapLevelCount;
}

- (NSUInteger)sampleCount
{
	return _sampleCount;
}

- (NSUInteger)arrayLength
{
	return _arrayLength;
}

- (NSUInteger)usage
{
	return _usage;
}

- (BOOL)isShareable
{
	return _shareable;
}

- (BOOL)isFramebufferOnly
{
	return NO;
}

- (BOOL)allowGPUOptimizedContents
{
	return _gpuOptimized;
}

- (NSUInteger)compressionType
{
	return 0;
}

- (MTLResourceID)gpuResourceID
{
	return (MTLResourceID){_resourceID};
}

- (MTLTextureSwizzleChannels)swizzle
{
	return _swizzle;
}

- (id)rootResource
{
	return _parent ?: self;
}

- (id)remoteStorageTexture
{
	return nil;
}

- (float)minLOD
{
	return _minLOD;
}

/* Tier 1: unmapped tiles read zero and drop writes. Tier 2's per-tile
 * activity counters are left out. */
- (NSInteger)sparseTextureTier
{
	return _sparse ? 1 : 0;
}

static StelaGPUTexture *new_view(StelaGPUTexture *texture, NSUInteger format, NSUInteger type, NSRange levels,
    NSRange slices, MTLTextureSwizzleChannels swizzle, float min_lod);

- (id)newTextureViewWithDescriptor:(id)descriptor
{
	float min_lod = [descriptor respondsToSelector:@selector(minLOD)] ? [descriptor minLOD] : 0;
	return new_view(self, [descriptor pixelFormat], [descriptor textureType], [descriptor levelRange],
	    [descriptor sliceRange], [descriptor swizzle], min_lod);
}

- (id)parentTexture
{
	return _parent;
}

- (NSUInteger)parentRelativeLevel
{
	return _parentLevel;
}

- (NSUInteger)parentRelativeSlice
{
	return _parentSlice;
}

- (id)buffer
{
	return _buffer;
}

- (NSUInteger)bufferOffset
{
	return _bufferOffset;
}

- (NSUInteger)bufferBytesPerRow
{
	return _bufferBytesPerRow;
}

- (NSUInteger)firstMipmapInTail
{
	return _firstMipmapInTail;
}

- (NSUInteger)tailSizeInBytes
{
	return _tailSizeInBytes;
}

- (BOOL)isSparse
{
	return _sparse;
}

- (IOSurfaceRef)iosurface
{
	return _surface;
}

- (NSUInteger)iosurfacePlane
{
	return _surfacePlane;
}

- (StelaTextureBytesRequest)bytesRequest:(const void *)bytes
			     bytesPerRow:(NSUInteger)bytesPerRow
			   bytesPerImage:(NSUInteger)bytesPerImage
				  region:(MTLRegion)region
				   level:(NSUInteger)level
				   slice:(NSUInteger)slice
{
	if (!_block.block_width) {
		_block = [_device alignmentsOfFormat:_pixelFormat];
	}
	/* Metal reaches the last row's texels, not its stride: a caller may end
	 * its memory there, and the span is what the frontend touches when the
	 * service finds a page unmapped. */
	NSUInteger images = region.size.depth ? region.size.depth : 1;
	NSUInteger rows = _block.block_height ? (region.size.height + _block.block_height - 1) / _block.block_height : 0;
	NSUInteger rowBytes = _block.block_width
		? (region.size.width + _block.block_width - 1) / _block.block_width * _block.block_bytes
		: 0;
	NSUInteger image = bytesPerImage ? bytesPerImage : bytesPerRow * rows;
	NSUInteger length = rows && rowBytes ? (images - 1) * image + (rows - 1) * bytesPerRow + rowBytes : 0;
	return (StelaTextureBytesRequest){
		.texture = _handle,
		.region = {.slice = slice,
			   .level = level,
			   .origin = {region.origin.x, region.origin.y, region.origin.z},
			   .size = {region.size.width, region.size.height, region.size.depth}},
		.bytes = stela_span(bytes, length),
		.bytes_per_row = bytesPerRow,
		.bytes_per_image = bytesPerImage,
	};
}

- (void)getBytes:(void *)bytes
      bytesPerRow:(NSUInteger)bytesPerRow
    bytesPerImage:(NSUInteger)bytesPerImage
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
	    slice:(NSUInteger)slice
{
	StelaTextureBytesRequest request = [self bytesRequest:bytes
						  bytesPerRow:bytesPerRow
						bytesPerImage:bytesPerImage
						       region:region
							level:level
							slice:slice];
	StelaEmpty nothing;
	StelaFailure failure;
	StelaSpans spans = {.writes = &request.bytes, .write_count = 1};
	if (STELA_CALL_SPANS(&_device->_gate, GetBytes, &request, &nothing, spans, &failure) != StelaStatusOk) {
		stela_report("getBytes:", &failure);
	}
}

- (void)getBytes:(void *)bytes
    bytesPerRow:(NSUInteger)bytesPerRow
     fromRegion:(MTLRegion)region
    mipmapLevel:(NSUInteger)level
{
	[self getBytes:bytes bytesPerRow:bytesPerRow bytesPerImage:0 fromRegion:region mipmapLevel:level slice:0];
}

- (void)replaceRegion:(MTLRegion)region
	  mipmapLevel:(NSUInteger)level
		slice:(NSUInteger)slice
	    withBytes:(const void *)bytes
	  bytesPerRow:(NSUInteger)bytesPerRow
	bytesPerImage:(NSUInteger)bytesPerImage
{
	StelaTextureBytesRequest request = [self bytesRequest:bytes
						  bytesPerRow:bytesPerRow
						bytesPerImage:bytesPerImage
						       region:region
							level:level
							slice:slice];
	StelaEmpty nothing;
	StelaFailure failure;
	StelaSpans spans = {.reads = &request.bytes, .read_count = 1};
	if (STELA_CALL_SPANS(&_device->_gate, ReplaceRegion, &request, &nothing, spans, &failure) != StelaStatusOk) {
		stela_report("replaceRegion:", &failure);
	}
}

- (void)replaceRegion:(MTLRegion)region
	  mipmapLevel:(NSUInteger)level
	    withBytes:(const void *)bytes
	  bytesPerRow:(NSUInteger)bytesPerRow
{
	[self replaceRegion:region mipmapLevel:level slice:0 withBytes:bytes bytesPerRow:bytesPerRow bytesPerImage:0];
}

- (id)newTextureViewWithPixelFormat:(NSUInteger)format
			textureType:(NSUInteger)type
			     levels:(NSRange)levels
			     slices:(NSRange)slices
			    swizzle:(MTLTextureSwizzleChannels)swizzle
{
	return new_view(self, format, type, levels, slices, swizzle, 0);
}

- (id)newTextureViewWithPixelFormat:(NSUInteger)format
			textureType:(NSUInteger)type
			     levels:(NSRange)levels
			     slices:(NSRange)slices
{
	return [self newTextureViewWithPixelFormat:format
				       textureType:type
					    levels:levels
					    slices:slices
					   swizzle:_swizzle];
}

- (id)newTextureViewWithPixelFormat:(NSUInteger)format
{
	return [self newTextureViewWithPixelFormat:format
				       textureType:_textureType
					    levels:(NSRange){0, _mipmapLevelCount}
					    slices:(NSRange){0, _arrayLength}
					   swizzle:_swizzle];
}

@end

@implementation StelaGPUSampler

- (MTLResourceID)gpuResourceID
{
	return (MTLResourceID){_resourceID};
}

@end

@implementation StelaGPUDepthStencil
- (id)device
{
	return _device;
}
@end

@implementation StelaGPUFence
- (id)device
{
	return _device;
}
@end

@implementation StelaGPUHeap

- (void)dealloc
{
	if (_mappedBytes) {
		mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)_contents, _mappedBytes);
	}
}

- (NSUInteger)size
{
	return _size;
}

- (NSUInteger)resourceOptions
{
	return _options;
}

- (NSUInteger)cpuCacheMode
{
	return (_options & MTLResourceCPUCacheModeMask) >> MTLResourceCPUCacheModeShift;
}

- (NSUInteger)storageMode
{
	return storage_mode(_options);
}

- (NSUInteger)hazardTrackingMode
{
	return (_options & MTLResourceHazardTrackingModeMask) >> MTLResourceHazardTrackingModeShift;
}

- (NSUInteger)type
{
	return _type;
}

- (NSUInteger)allocatedSize
{
	return _size;
}

- (NSUInteger)setPurgeableState:(NSUInteger)state
{
	return stela_purgeable_state(&_purgeable, state);
}

- (StelaHeapUsageResponse)usageForAlignment:(NSUInteger)alignment
{
	StelaHeapUsageRequest request = {.heap = _handle, .alignment = alignment};
	StelaHeapUsageResponse usage = {};
	STELA_CALL(&_device->_gate, HeapUsage, &request, &usage, NULL);
	return usage;
}

- (NSUInteger)usedSize
{
	return [self usageForAlignment:1].used_size;
}

- (NSUInteger)currentAllocatedSize
{
	return _size;
}

- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment
{
	return [self usageForAlignment:alignment].max_available_size;
}

- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options offset:(uint64_t)offset
{
	StelaNewHeapBufferRequest request = {
		.heap = _handle, .length = length, .options = options, .offset = offset};
	StelaBufferResponse response;
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, NewHeapBuffer, &request, &response, &failure) != StelaStatusOk) {
		return nil;
	}
	StelaGPUBuffer *buffer = [[StelaGPUBuffer alloc] initWithDevice:_device
							       response:&response
								options:options
								 length:length];
	buffer->_heap = self;
	buffer->_contents = _contents ? (uint8_t *)_contents + response.heap_offset : NULL;
	return buffer;
}

- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{
	return [self newBufferWithLength:length options:options offset:STELA_AUTOMATIC];
}

- (id)newTextureWithDescriptor:(id)descriptor offset:(uint64_t)offset
{
	StelaNewHeapTextureRequest request = {.heap = _handle, .offset = offset};
	stela_texture_record(descriptor, &request.descriptor);
	StelaTextureResponse response;
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, NewHeapTexture, &request, &response, &failure) != StelaStatusOk) {
		return nil;
	}
	StelaGPUTexture *texture = [[StelaGPUTexture alloc] initWithDevice:_device
								  response:&response
								descriptor:descriptor];
	texture->_heap = self;
	return texture;
}

- (id)newTextureWithDescriptor:(id)descriptor
{
	return [self newTextureWithDescriptor:descriptor offset:STELA_AUTOMATIC];
}

- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options placementOffset:(NSUInteger)offset
{
	return [self newBufferWithLength:length options:options offset:offset];
}

- (id)newTextureWithDescriptor:(id)descriptor placementOffset:(NSUInteger)offset
{
	return [self newTextureWithDescriptor:descriptor offset:offset];
}

@end

/* Where kernel signals a GPU wait listens for reach the service. */
static MTLSharedEventListener *forwarding(void)
{
	static MTLSharedEventListener *listener;
	static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
	os_unfair_lock_lock(&lock);
	if (!listener) {
		listener = [[MTLSharedEventListener alloc]
			initWithDispatchQueue:dispatch_queue_create("com.stela.gpu.event-forwarding", NULL)];
	}
	MTLSharedEventListener *found = listener;
	os_unfair_lock_unlock(&lock);
	return found;
}

@implementation StelaGPUEvent

- (instancetype)initWithDevice:(StelaGPUDevice *)device handle:(uint64_t)handle port:(mach_port_t)port
{
	self = port ? [super initWithMachPort:port] : [super initWithOptions:0];
	if (self) {
		_device = device;
		_handle = handle;
	} else if (handle) {
		[device releaseHandle:handle];
	}
	return self;
}

- (StelaGPUDevice *)device
{
	return _device;
}

- (void)dealloc
{
	if (_handle) {
		[_device releaseHandle:_handle];
	}
}

- (uint64_t)labelTraceID
{
	return 0;
}

/* The service's value; 0 when it cannot answer, which it reports. */
- (uint64_t)serviceValue
{
	StelaHandle request = {.handle = _handle};
	StelaValue value = {};
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, EventSignaledValue, &request, &value, &failure) != StelaStatusOk) {
		stela_report("signaledValue", &failure);
	}
	return value.value;
}

/* Raises the service's value; it never lowers. */
- (void)raiseService:(uint64_t)value
{
	StelaEventValueRequest request = {.event = _handle, .value = value};
	StelaEmpty nothing;
	StelaFailure failure;
	if (STELA_CALL(&_device->_gate, SetEventSignaledValue, &request, &nothing, &failure) != StelaStatusOk) {
		stela_report("setSignaledValue:", &failure);
	}
}

/* The later of the two values, which both then hold. */
- (uint64_t)signaledValue
{
	uint64_t service = [self serviceValue];
	uint64_t kernel = [super signaledValue];
	if (kernel > service) {
		[self raiseService:kernel];
		return kernel;
	}
	if (service > kernel) {
		[super setSignaledValue:service];
	}
	return service;
}

- (void)setSignaledValue:(uint64_t)value
{
	[self raiseService:value];
	[super setSignaledValue:value];
	[_device wakeWaiters];
}

- (void)publish
{
	[self publishReaching:0];
}

- (BOOL)publishReaching:(uint64_t)value
{
	uint64_t service = [self serviceValue];
	if (service > [super signaledValue]) {
		[super setSignaledValue:service];
	}
	return service >= value;
}

- (void)forwardAt:(uint64_t)value
{
	if ([super signaledValue] >= value) {
		[self raiseService:value];
		return;
	}
	[super notifyListener:forwarding()
		      atValue:value
			block:^(id event, uint64_t reached) {
				[(StelaGPUEvent *)event raiseService:reached];
			}];
}

/*
 * A GPU signal is seen at the completion notification that follows it, so the
 * waiter sleeps between notifications and asks the service once after each.
 * A CPU waiter may also be what the GPU waits for in the same command buffer as
 * the signal, so no completion need ever arrive, and another process's CPU
 * signal reaches only the kernel's value: both values are also read when a
 * sleep times out, at intervals doubling from 1 ms to FALLBACK_MS, each
 * service read also submitting what the signal released.
 */
#define FALLBACK_MS 16u

- (BOOL)waitUntilSignaledValue:(uint64_t)value timeoutMS:(uint64_t)milliseconds
{
	mach_timebase_info_data_t timebase;
	mach_timebase_info(&timebase);
	uint64_t start = mach_absolute_time();
	uint64_t interval = 1;
	for (;;) {
		uint64_t wakes = [_device wakes];
		StelaWaitEventRequest request = {.event = _handle, .value = value};
		StelaWaitResponse response = {};
		StelaFailure failure;
		if (STELA_CALL(&_device->_gate, WaitEvent, &request, &response, &failure) != StelaStatusOk) {
			stela_report("waitUntilSignaledValue:", &failure);
			return NO;
		}
		if (response.reached) {
			[self publish];
			return YES;
		}
		uint64_t kernel = [super signaledValue];
		if (kernel >= value) {
			[self raiseService:kernel];
			return YES;
		}
		uint64_t waited = (mach_absolute_time() - start) * timebase.numer / timebase.denom / 1000000;
		if (waited >= milliseconds) {
			return NO;
		}
		[_device waitForWakeAfter:wakes
			     milliseconds:interval < milliseconds - waited ? interval : milliseconds - waited];
		/* A notification restarts the fallback's intervals; a timeout doubles them. */
		interval = [_device wakes] != wakes ? 1 : (interval * 2 < FALLBACK_MS ? interval * 2 : FALLBACK_MS);
	}
}

- (void)notifyListener:(id)listener atValue:(uint64_t)value block:(void (^)(id, uint64_t))block
{
	dispatch_async(dispatch_get_global_queue(0, 0), ^{
		[self waitUntilSignaledValue:value timeoutMS:UINT64_MAX];
		dispatch_async([listener dispatchQueue], ^{
			block(self, [self signaledValue]);
		});
	});
}

@end

/* Each foreign IOSurface event's twin on this device, by association. */
static char twin;

StelaGPUEvent *stela_event(StelaGPUDevice *device, id event)
{
	if ([event isKindOfClass:[StelaGPUEvent class]] && ((StelaGPUEvent *)event)->_device == device) {
		return event;
	}
	if (![event isKindOfClass:[IOSurfaceSharedEvent class]]) {
		return nil;
	}
	static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
	os_unfair_lock_lock(&lock);
	StelaGPUEvent *found = objc_getAssociatedObject(event, &twin);
	os_unfair_lock_unlock(&lock);
	if (found) {
		return found;
	}
	StelaGPUEvent *made = [device newSharedEventWithMachPort:[event eventPort]];
	os_unfair_lock_lock(&lock);
	found = objc_getAssociatedObject(event, &twin);
	if (!found && made) {
		objc_setAssociatedObject(event, &twin, made, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
		found = made;
	}
	os_unfair_lock_unlock(&lock);
	return found;
}

@implementation StelaGPUCounter
@end

@implementation StelaGPUCounterSet

/* The statistics a sample holds: every MTLCounterResultStatistic member but
 * fragments passed, which Vulkan's pipeline statistics do not count. */
#define STATISTIC_COUNTERS 7

+ (instancetype)setWithCode:(uint64_t)code
{
	StelaGPUCounterSet *set = [StelaGPUCounterSet new];
	set->_code = code;
	BOOL timestamp = code == StelaCounterSetCodeTimestamp;
	set.name = timestamp ? MTLCommonCounterSetTimestamp : MTLCommonCounterSetStatistic;
	NSString *names[STATISTIC_COUNTERS] = {
		MTLCommonCounterTessellationInputPatches,
		MTLCommonCounterVertexInvocations,
		MTLCommonCounterPostTessellationVertexInvocations,
		MTLCommonCounterClipperInvocations,
		MTLCommonCounterClipperPrimitivesOut,
		MTLCommonCounterFragmentInvocations,
		MTLCommonCounterComputeKernelInvocations,
	};
	NSUInteger count = timestamp ? 1 : STATISTIC_COUNTERS;
	NSMutableArray *counters = [NSMutableArray arrayWithCapacity:count];
	for (NSUInteger i = 0; i < count; i++) {
		StelaGPUCounter *counter = [StelaGPUCounter new];
		counter.name = timestamp ? MTLCommonCounterTimestamp : names[i];
		[counters addObject:counter];
	}
	set.counters = counters;
	return set;
}

@end

@implementation StelaGPUCounterSampleBuffer

- (NSUInteger)sampleCount
{
	return _sampleCount;
}

- (NSData *)resolveCounterRange:(NSRange)range
{
	NSUInteger length = range.length * sizeof(uint64_t) * _valuesPerSample;
	NSMutableData *data = [NSMutableData dataWithLength:length];
	StelaResolveCountersRequest request = {
		.buffer = _handle,
		.first = range.location,
		.count = range.length,
		.out = stela_span(data.mutableBytes, length),
	};
	StelaResolveCountersResponse response = {};
	StelaFailure failure;
	StelaSpans spans = {.writes = &request.out, .write_count = 1};
	uint32_t status = STELA_CALL_SPANS(&_device->_gate, ResolveCounters, &request, &response, spans, &failure);
	if (status != StelaStatusOk) {
		stela_report("resolveCounterRange:", &failure);
		return nil;
	}
	return response.resolved ? data : nil;
}

@end

/* The device's resource factories. */

@implementation StelaGPUDevice (Resources)

- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{
	StelaBufferResponse response;
	StelaFailure failure;
	if (!cpu_visible(options)) {
		StelaNewBufferRequest request = {.length = length, .options = options};
		if (STELA_CALL(&_gate, NewBuffer, &request, &response, &failure) != StelaStatusOk) {
			return nil;
		}
		return [[StelaGPUBuffer alloc] initWithDevice:self response:&response options:options length:length];
	}
	StelaGPUBuffer *pooled = [self pooledBufferWithLength:length options:options];
	if (pooled) {
		return pooled;
	}
	uint64_t mapped = page_aligned(length);
	void *contents = NULL;
	uint64_t memory = [self importLength:mapped contents:&contents failure:&failure];
	if (memory == 0) {
		return nil;
	}
	StelaNewImportedBufferRequest request = {.memory = memory, .length = length, .options = options};
	if (STELA_CALL(&_gate, NewImportedBuffer, &request, &response, &failure) != StelaStatusOk) {
		[self releaseHandle:memory];
		mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)contents, mapped);
		return nil;
	}
	StelaGPUBuffer *buffer = [[StelaGPUBuffer alloc] initWithDevice:self
							       response:&response
								options:options
								 length:length];
	buffer->_contents = contents;
	buffer->_mappedBytes = mapped;
	return buffer;
}

- (id)newBufferWithBytes:(const void *)bytes length:(NSUInteger)length options:(NSUInteger)options
{
	StelaGPUBuffer *buffer = [self newBufferWithLength:length options:options];
	if (buffer && !buffer->_contents) {
		StelaFailure failure = {.reason = "private memory is out of the CPU's reach"};
		stela_report("newBufferWithBytes:", &failure);
		return nil;
	}
	if (buffer) {
		memcpy(buffer->_contents, bytes, length);
	}
	return buffer;
}

- (id)newBufferWithBytesNoCopy:(void *)bytes
			length:(NSUInteger)length
		       options:(NSUInteger)options
		   deallocator:(void (^)(void *, NSUInteger))deallocator
{
	/* Metal answers nil for an empty buffer, as Apple GPUs do (measured on
	 * an M1 Pro); it is no failure to report. */
	if (length == 0) {
		return nil;
	}
	StelaFailure failure;
	uint64_t memory = [self importAddress:bytes length:page_aligned(length) failure:&failure];
	if (memory == 0) {
		stela_report([[NSString stringWithFormat:@"newBufferWithBytesNoCopy:%p length:%lu", bytes, length] UTF8String],
			     &failure);
		return nil;
	}
	StelaNewImportedBufferRequest request = {.memory = memory, .length = length, .options = options};
	StelaBufferResponse response;
	if (STELA_CALL(&_gate, NewImportedBuffer, &request, &response, &failure) != StelaStatusOk) {
		stela_report("newBufferWithBytesNoCopy:", &failure);
		[self releaseHandle:memory];
		return nil;
	}
	StelaGPUBuffer *buffer = [[StelaGPUBuffer alloc] initWithDevice:self
							       response:&response
								options:options
								 length:length];
	buffer->_contents = bytes;
	buffer->_deallocator = deallocator;
	return buffer;
}

/* A buffer over the surface's own pages, all of its allocation, shared with
 * the CPU as any imported buffer is. It keeps the surface; wiring the pages
 * outlasts the lock that maps them. */
- (id)newBufferWithIOSurface:(IOSurfaceRef)surface
{
	if (!surface) {
		return nil;
	}
	if (IOSurfaceLock(surface, 0, NULL) != 0) {
		StelaFailure failure = {.reason = "the surface does not lock"};
		stela_report("newBufferWithIOSurface:", &failure);
		return nil;
	}
	CFRetain(surface);
	StelaGPUBuffer *buffer = [self newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(surface)
							 length:IOSurfaceGetAllocSize(surface)
							options:MTLStorageModeShared << 4
						    deallocator:^(void *contents, NSUInteger length) {
							    CFRelease(surface);
						    }];
	IOSurfaceUnlock(surface, 0, NULL);
	if (!buffer) {
		CFRelease(surface);
		return nil;
	}
	buffer->_surface = surface;
	return buffer;
}

/* The texture reads the surface's own pages, through a buffer over them that
 * keeps the surface. Wiring them outlasts the lock that maps them. A surface
 * counts rows of its elements, which can be taller than the format's blocks. */
- (id)newTextureWithDescriptor:(id)descriptor iosurface:(IOSurfaceRef)surface plane:(NSUInteger)plane
{
	if (IOSurfaceLock(surface, 0, NULL) != 0) {
		StelaFailure failure = {.reason = "the surface does not lock"};
		stela_report("newTextureWithDescriptor:iosurface:plane:", &failure);
		return nil;
	}
	uint8_t *base = IOSurfaceGetBaseAddress(surface);
	NSUInteger offset = (uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, plane) - base;
	StelaFormatAlignmentsResponse format = [self alignmentsOfFormat:[descriptor pixelFormat]];
	NSUInteger bytesPerRow = IOSurfaceGetBytesPerRowOfPlane(surface, plane) * format.block_height
		/ IOSurfaceGetElementHeightOfPlane(surface, plane);
	/* A lossless surface's plane says little of its rows, which this device
	 * stores plain. */
	NSUInteger row = [descriptor width] * format.block_bytes;
	if (IOSurfaceGetPixelFormat(surface) >> 24 == '&' && bytesPerRow < row) {
		bytesPerRow = row;
	}
	CFRetain(surface);
	StelaGPUBuffer *buffer = [self newBufferWithBytesNoCopy:base
							 length:IOSurfaceGetAllocSize(surface)
							options:MTLStorageModeShared << 4
						    deallocator:^(void *contents, NSUInteger length) {
							    CFRelease(surface);
						    }];
	IOSurfaceUnlock(surface, 0, NULL);
	if (!buffer) {
		CFRelease(surface);
		return nil;
	}
	/* Apple's two-plane formats take both of a surface's planes from its
	 * first; other formats ignore the second, whose texels are pairs of the
	 * first's elements. Its rows count as the first's do. */
	NSUInteger secondOffset = 0, secondBytesPerRow = 0;
	if (plane == 0 && IOSurfaceGetPlaneCount(surface) >= 2) {
		secondOffset = (uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, 1) - base;
		secondBytesPerRow = IOSurfaceGetBytesPerRowOfPlane(surface, 1) / IOSurfaceGetElementHeightOfPlane(surface, 1);
		/* A pair of 8- or 16-bit components, as the format's first plane's
		 * element is one; RGB10A8_2P_XR10's (550, 551) second plane is a
		 * byte of alpha. */
		NSUInteger pixelFormat = [descriptor pixelFormat];
		NSUInteger secondTexel = pixelFormat == 550 || pixelFormat == 551 ? 1 : 2 * format.block_bytes;
		NSUInteger secondRow = IOSurfaceGetWidthOfPlane(surface, 1) * secondTexel;
		if (IOSurfaceGetPixelFormat(surface) >> 24 == '&' && secondBytesPerRow < secondRow) {
			secondBytesPerRow = secondRow;
		}
	}
	StelaGPUTexture *texture = [buffer newTextureWithDescriptor:descriptor
							     offset:offset
							bytesPerRow:bytesPerRow
						       secondOffset:secondOffset
						  secondBytesPerRow:secondBytesPerRow
						      surfaceFormat:IOSurfaceGetPixelFormat(surface)];
	if (texture) {
		texture->_surface = surface;
		texture->_surfacePlane = plane;
	} else {
		uint32_t fourcc = IOSurfaceGetPixelFormat(surface);
		StelaFailure failure = {.reason = "no texture of this surface"};
		stela_report([[NSString stringWithFormat:@"pixel format %lu of surface %c%c%c%c, %zux%zu, "
							 "%zu bytes an element, %zu a row, %zu planes, %zu in all; "
							 "plane %lu at %lu, %lu a row; second plane at %lu, %lu a row",
							 (unsigned long)[descriptor pixelFormat], fourcc >> 24,
							 (fourcc >> 16) & 0xff, (fourcc >> 8) & 0xff, fourcc & 0xff,
							 IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface),
							 IOSurfaceGetBytesPerElement(surface),
							 IOSurfaceGetBytesPerRow(surface), IOSurfaceGetPlaneCount(surface),
							 IOSurfaceGetAllocSize(surface), (unsigned long)plane,
							 (unsigned long)offset, (unsigned long)bytesPerRow,
							 (unsigned long)secondOffset, (unsigned long)secondBytesPerRow] UTF8String],
			     &failure);
	}
	return texture;
}

- (id)newTextureWithDescriptor:(id)descriptor
{
	StelaTextureDescriptorRecord request;
	stela_texture_record(descriptor, &request);
	StelaTextureResponse response;
	StelaFailure failure;
	if (STELA_CALL(&_gate, NewTexture, &request, &response, &failure) != StelaStatusOk) {
		stela_report([[NSString stringWithFormat:@"a texture of format %llu", request.pixel_format] UTF8String], &failure);
		return nil;
	}
	return [[StelaGPUTexture alloc] initWithDevice:self response:&response descriptor:descriptor];
}

- (id)newSamplerStateWithDescriptor:(id)descriptor
{
	StelaSamplerDescriptorRecord request = {
		.min_filter = [descriptor minFilter],
		.mag_filter = [descriptor magFilter],
		.mip_filter = [descriptor mipFilter],
		.max_anisotropy = [descriptor maxAnisotropy],
		.address_modes = {[descriptor sAddressMode], [descriptor tAddressMode], [descriptor rAddressMode]},
		.border_color = [descriptor borderColor],
		.reduction_mode = [descriptor reductionMode],
		.compare_function = [descriptor compareFunction],
		.lod_min_clamp = [descriptor lodMinClamp],
		.lod_max_clamp = [descriptor lodMaxClamp],
		.normalized_coordinates = [descriptor normalizedCoordinates],
	};
	StelaSamplerResponse response;
	StelaFailure failure;
	if (STELA_CALL(&_gate, NewSampler, &request, &response, &failure) != StelaStatusOk) {
		stela_report("a sampler", &failure);
		return nil;
	}
	StelaGPUSampler *sampler = [[StelaGPUSampler alloc] initWithDevice:self handle:response.handle];
	sampler->_resourceID = response.resource_id;
	sampler.label = [descriptor label];
	return sampler;
}

- (id)newHeapWithDescriptor:(id)descriptor
{
	StelaHeapDescriptorRecord record = {
		.size = [descriptor size],
		.options = [descriptor resourceOptions],
		.heap_type = [descriptor type],
		.sparse_page_size = [descriptor sparsePageSize],
	};
	return [self newHeapWithRecord:record];
}

- (id)newHeapWithRecord:(StelaHeapDescriptorRecord)request
{
	NSUInteger options = request.options;
	void *contents = NULL;
	uint64_t mapped = 0;
	StelaFailure failure;
	if (cpu_visible(options)) {
		mapped = page_aligned(request.size);
		request.memory = [self importLength:mapped contents:&contents failure:&failure];
		if (request.memory == 0) {
			return nil;
		}
	}
	StelaHandle response;
	if (STELA_CALL(&_gate, NewHeap, &request, &response, &failure) != StelaStatusOk) {
		if (mapped) {
			[self releaseHandle:request.memory];
			mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)contents, mapped);
		}
		return nil;
	}
	StelaGPUHeap *heap = [[StelaGPUHeap alloc] initWithDevice:self handle:response.handle];
	heap->_contents = contents;
	heap->_mappedBytes = mapped;
	heap->_options = options;
	/* The service answers the capacity, whole pages as on a Mac. */
	heap->_size = [heap usageForAlignment:1].size;
	if (heap->_size == 0) {
		heap->_size = request.size;
	}
	heap->_type = request.heap_type;
	return heap;
}

- (MTLSizeAndAlign)heapBufferSizeAndAlignWithLength:(NSUInteger)length options:(NSUInteger)options
{
	StelaHeapBufferSizeRequest request = {.length = length, .options = options};
	StelaSizeAndAlignRecord size = {};
	STELA_CALL(&_gate, HeapBufferSizeAndAlign, &request, &size, NULL);
	return (MTLSizeAndAlign){size.size, size.align};
}

- (MTLSizeAndAlign)heapTextureSizeAndAlignWithDescriptor:(id)descriptor
{
	StelaTextureDescriptorRecord request;
	stela_texture_record(descriptor, &request);
	StelaSizeAndAlignRecord size = {};
	STELA_CALL(&_gate, HeapTextureSizeAndAlign, &request, &size, NULL);
	return (MTLSizeAndAlign){size.size, size.align};
}

- (id)newEvent
{
	return [self newSharedEvent];
}

- (id)newSharedEvent
{
	return [self newSharedEventWithMachPort:0];
}

/* A new event on a new kernel event, or on the one `port` names. */
- (id)newSharedEventWithMachPort:(mach_port_t)port
{
	StelaEmpty nothing = {};
	StelaHandle response;
	StelaFailure failure;
	if (STELA_CALL(&_gate, NewEvent, &nothing, &response, &failure) != StelaStatusOk) {
		stela_report("an event", &failure);
		return nil;
	}
	return [[StelaGPUEvent alloc] initWithDevice:self handle:response.handle port:port];
}

- (id)newSharedEventWithOptions:(uint64_t)options
{
	return [self newSharedEvent];
}

- (id)newCounterSampleBufferWithDescriptor:(id)descriptor error:(NSError **)error
{
	StelaGPUCounterSet *set = [descriptor counterSet];
	StelaNewCounterBufferRequest request = {.counter_set = set->_code, .sample_count = [descriptor sampleCount]};
	StelaHandle response;
	StelaFailure failure;
	if (STELA_CALL(&_gate, NewCounterBuffer, &request, &response, &failure) != StelaStatusOk) {
		if (error) {
			*error = stela_error(MTLCounterErrorDomain, MTLCounterSampleBufferErrorInternal, &failure);
		}
		return nil;
	}
	StelaGPUCounterSampleBuffer *buffer = [[StelaGPUCounterSampleBuffer alloc] initWithDevice:self
											   handle:response.handle];
	buffer->_sampleCount = request.sample_count;
	buffer->_valuesPerSample = set->_code == StelaCounterSetCodeStatistic ? STELA_STATISTIC_SAMPLE_VALUES : 1;
	buffer.label = [descriptor label];
	return buffer;
}

@end

static StelaGPUTexture *new_view(StelaGPUTexture *texture, NSUInteger format, NSUInteger type, NSRange levels,
    NSRange slices, MTLTextureSwizzleChannels swizzle, float min_lod)
{
	StelaNewTextureViewRequest request = {
		.texture = texture->_handle,
		.pixel_format = format,
		.texture_type = type,
		.levels = {levels.location, levels.length},
		.slices = {slices.location, slices.length},
		.swizzle = {swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha},
		.min_lod = min_lod,
	};
	StelaTextureResponse response;
	StelaFailure failure;
	if (STELA_CALL(&texture->_device->_gate, NewTextureView, &request, &response, &failure) != StelaStatusOk) {
		stela_report("a texture view", &failure);
		return nil;
	}
	StelaGPUTexture *view = [StelaGPUTexture alloc];
	view = [view initWithDevice:texture->_device handle:response.handle];
	view->_options = texture->_options;
	view->_allocatedSize = response.allocated_size;
	view->_resourceID = response.resource_id;
	view->_textureType = type;
	view->_pixelFormat = format;
	view->_width = texture->_width >> levels.location ?: 1;
	view->_height = texture->_height >> levels.location ?: 1;
	view->_depth = texture->_depth >> levels.location ?: 1;
	view->_mipmapLevelCount = levels.length;
	view->_sampleCount = texture->_sampleCount;
	view->_arrayLength = slices.length;
	view->_usage = texture->_usage;
	view->_swizzle = swizzle;
	view->_minLOD = min_lod;
	view->_gpuOptimized = texture->_gpuOptimized;
	view->_drawable = texture->_drawable;
	view->_parent = texture;
	view->_parentLevel = levels.location;
	view->_parentSlice = slices.location;
	return view;
}

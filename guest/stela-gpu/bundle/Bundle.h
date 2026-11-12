/*
 * The Metal plugin: Metal's objects over the native service, through the gate
 * page the kext maps into the task.
 *
 * Objects hold the service's handles; encoders write the command stream a
 * commit replays, and nothing else crosses. What a CPU reaches is the task's
 * own memory, which the kext wires and the GPU imports.
 */
#pragma once

#include "Metal.h"
#include <StelaGPU/Interface.h>
#include <StelaGPU/Wire.h>
#include <stela/gate.h>

/* The gate page of one device in this task. */
typedef struct {
	io_connect_t connect;
	volatile uint64_t *page;
} StelaGate;

/* Why a call failed: its status, the service's reason, and the gate's fault for
 * a call the service never completed. */
typedef struct {
	uint32_t status;
	uint64_t fault;
	char reason[160];
} StelaFailure;

/* The memory a call's records name, which the call maps before it runs. */
typedef struct {
	const StelaSpan *reads, *writes;
	size_t read_count, write_count;
} StelaSpans;

/*
 * Runs one operation and answers its status. A page may be away when the service
 * copies it, so the call touches `response` and `spans`, and touches them again
 * if the service found one gone. `failure` may be NULL.
 */
uint32_t stela_gate_call(StelaGate *gate, uint32_t operation, const void *request, size_t request_bytes,
			 void *response, size_t response_bytes, StelaSpans spans, StelaFailure *failure);

/* The Mach-O CPU type of a universal library's AIR slice, the code this device
 * compiles and its target architecture. */
#define CPU_TYPE_AIR64 0x01000017u

/* A call of operation `name`, whose records' types the compiler checks. */
#define STELA_CALL_SPANS(gate, name, request, response, spans, failure)                            \
	({                                                                                         \
		const Stela##name##Call *call_request = (request);                                 \
		Stela##name##Reply *call_response = (response);                                    \
		stela_gate_call((gate), StelaOperation##name, call_request, sizeof(*call_request), \
				call_response, sizeof(*call_response), (spans), (failure));        \
	})
#define STELA_CALL(gate, name, request, response, failure)                                         \
	STELA_CALL_SPANS(gate, name, request, response, (StelaSpans){0}, failure)

static inline StelaSpan stela_span(const void *address, size_t length)
{
	return (StelaSpan){.address = (uint64_t)address, .length = length};
}

/* An NSError of `domain` from a failed call. */
NSError *stela_error(NSString *domain, NSInteger code, const StelaFailure *failure);
/* Says in the system log and the host's, once for each, that `what` could not
 * be done. Until `stela_report_to` names a gate, only in the system log. */
void stela_report_to(StelaGate *gate);
void stela_report(const char *what, const StelaFailure *failure);

/* A growing command stream. */
typedef struct {
	uint8_t *bytes;
	size_t length, capacity;
} StelaStream;

void stela_stream_entry(StelaStream *stream, uint32_t code, const void *record, size_t record_bytes,
			const void *data, size_t data_bytes);
void stela_stream_append(StelaStream *stream, const StelaStream *other);
void stela_stream_free(StelaStream *stream);

/* The descriptors' records; a nil descriptor is the default one. */
void stela_texture_record(id descriptor, StelaTextureDescriptorRecord *record);
void stela_depth_stencil_record(id descriptor, StelaDepthStencilRecord *record);

@class StelaGPUCommandBuffer, StelaGPUEvent, StelaGPUFunction, StelaGPUIndirectCommandBuffer;

/* MTLAddDevice asserts both protocols on the device itself. */
@interface StelaGPUDevice : _MTLDevice <MTLDevice, MTLDeviceSPI> {
@public
	StelaGate _gate;
	StelaBlock *_capabilities;
	BOOL _metalAssertionsEnabled;
}
- (instancetype)initWithAcceleratorPort:(io_service_t)port;
- (uint64_t)registryID;
/* `length` bytes of new memory at *contents wired for the GPU; 0 on failure. */
- (uint64_t)importLength:(uint64_t)length contents:(void **)contents failure:(StelaFailure *)failure;
/* Wires the task's memory at `address`; 0 on failure. */
- (uint64_t)importAddress:(void *)address length:(uint64_t)length failure:(StelaFailure *)failure;
/* Completes `buffer` once the service reaches its serial. */
- (void)track:(StelaGPUCommandBuffer *)buffer;
/* Completes `buffer`, whose commit `failure` refused. */
- (void)failed:(StelaGPUCommandBuffer *)buffer with:(StelaFailure)failure;
/* Publishes `event` at each completion until the service reaches `value`,
 * which a refused command buffer's replayed signal applies. */
- (void)settle:(StelaGPUEvent *)event at:(uint64_t)value;
- (void)releaseHandle:(uint64_t)handle;
/* Releases `handle` once the GPU completes everything committed so far,
 * keeping `slab` until then. */
- (void)releaseHandleAfterWork:(uint64_t)handle slab:(id)slab;
/* A shared buffer placed in a heap of the device's own, or nil. */
- (id)pooledBufferWithLength:(NSUInteger)length options:(NSUInteger)options;
/* The serial of a commit the service accepted. */
- (void)noteSubmitted:(uint64_t)serial;
/* The host seconds of a GPU timestamp, as `GPUStartTime` counts them. */
- (CFTimeInterval)secondsOfTick:(uint64_t)tick;
/* Indirect command buffers with encoding the next commit takes. */
- (void)noteIndirect:(StelaGPUIndirectCommandBuffer *)buffer;
- (void)forgetIndirect:(StelaGPUIndirectCommandBuffer *)buffer;
- (void)flushIndirect;
- (StelaFormatAlignmentsResponse)alignmentsOfFormat:(NSUInteger)format;
/* Event waiters sleep between completion notifications: `wakes` counts the
 * notifications and the CPU signals of this process's events, and a waiter
 * sleeps until the count passes the one it read, or for `milliseconds`. */
- (uint64_t)wakes;
- (void)wakeWaiters;
- (void)waitForWakeAfter:(uint64_t)wakes milliseconds:(uint64_t)milliseconds;
@end

@interface StelaGPUDevice (Resources)
- (id)newTextureWithDescriptor:(id)descriptor;
- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options;
/* A heap of `record`, which imports the task's memory for a CPU-visible one. */
- (id)newHeapWithRecord:(StelaHeapDescriptorRecord)record;
- (id)newSharedEventWithMachPort:(mach_port_t)port;
@end

@interface StelaGPUDevice (Pipelines)
/* A pipeline of `descriptor`; a nonzero `archive` is the one it compiles into. */
- (id)renderPipeline:(id)descriptor
	     options:(NSUInteger)options
	  reflection:(id *)reflection
	     archive:(uint64_t)archive
	       error:(NSError **)error;
- (id)meshPipeline:(id)descriptor
	   options:(NSUInteger)options
	reflection:(id *)reflection
	   archive:(uint64_t)archive
	     error:(NSError **)error;
- (id)computePipeline:(id)descriptor
	      options:(NSUInteger)options
	   reflection:(id *)reflection
	      archive:(uint64_t)archive
		error:(NSError **)error;
- (id)tilePipeline:(id)descriptor
	   options:(NSUInteger)options
	reflection:(id *)reflection
	   archive:(uint64_t)archive
	     error:(NSError **)error;
@end

/* Library.m: the dynamic library in the file at path, which the libraries
 * linking it load by install name. */
@interface StelaGPUDevice (Libraries)
- (id)dynamicLibraryAtPath:(NSString *)path error:(NSError **)error;
@end

/* Stitching.m; an archive checks the libraries added to it. */
@interface StelaGPUDevice (Stitching)
- (id)newLibraryWithStitchedDescriptor:(id)descriptor error:(NSError **)error;
@end

/* What holds a service handle. */
@interface StelaGPUObject : NSObject {
@public
	StelaGPUDevice *_device;
	uint64_t _handle;
}
@property(copy) NSString *label;
- (instancetype)initWithDevice:(StelaGPUDevice *)device handle:(uint64_t)handle;
- (StelaGPUDevice *)device;
@end

@interface StelaGPUResource : StelaGPUObject <MTLResource, MTLAllocation> {
@public
	NSUInteger _options;
	NSUInteger _allocatedSize, _heapOffset;
	id _heap;
	BOOL _aliasable;
	/* Its MTLPurgeableState, zero for the NonVolatile it starts in. */
	NSUInteger _purgeable;
	/* The process its memory counts toward, as `setResponsibleProcess:` names it. */
	int _responsibleProcess;
}
@end

@interface StelaGPUBuffer : StelaGPUResource <MTLBuffer> {
@public
	void *_contents;
	NSUInteger _length;
	uint64_t _gpuAddress;
	/* Memory of the bundle's own, unmapped with the buffer. */
	uint64_t _mappedBytes;
	void (^_deallocator)(void *, NSUInteger);
	/* The device's own heap this buffer was placed in, which the app never
	 * sees; its place is given back once the GPU is past it. */
	id _slab;
	/* The IOSurface the buffer lies in, which its deallocator keeps. */
	IOSurfaceRef _surface;
}
- (instancetype)initWithDevice:(StelaGPUDevice *)device
		      response:(const StelaBufferResponse *)response
		       options:(NSUInteger)options
			length:(NSUInteger)length;
- (id)newTextureWithDescriptor:(id)descriptor
			offset:(NSUInteger)offset
		   bytesPerRow:(NSUInteger)bytesPerRow
		  secondOffset:(NSUInteger)secondOffset
	     secondBytesPerRow:(NSUInteger)secondBytesPerRow
		 surfaceFormat:(uint32_t)surfaceFormat;
@end

@interface StelaGPUTexture : StelaGPUResource <MTLTexture> {
@public
	uint64_t _resourceID;
	NSUInteger _textureType, _pixelFormat, _width, _height, _depth;
	NSUInteger _mipmapLevelCount, _sampleCount, _arrayLength, _usage;
	NSUInteger _bufferOffset, _bufferBytesPerRow, _firstMipmapInTail, _tailSizeInBytes;
	NSUInteger _parentLevel, _parentSlice;
	MTLTextureSwizzleChannels _swizzle;
	/* A view's minLOD, in its levels. */
	float _minLOD;
	/* _drawable: made for a layer's drawable (a descriptor's private
	 * isDrawable), as a view of one is. */
	BOOL _sparse, _shareable, _gpuOptimized, _drawable;
	id _buffer, _parent;
	IOSurfaceRef _surface;
	NSUInteger _surfacePlane;
	StelaTextureDescriptorRecord _record;
	/* The pixel format's block, asked for at the first CPU transfer: a zero
	 * width until then. */
	StelaFormatAlignmentsResponse _block;
}
- (instancetype)initWithDevice:(StelaGPUDevice *)device
		      response:(const StelaTextureResponse *)response
			record:(const StelaTextureDescriptorRecord *)record;
- (instancetype)initWithDevice:(StelaGPUDevice *)device
		      response:(const StelaTextureResponse *)response
		    descriptor:(id)descriptor;
@end

@interface StelaGPUSampler : StelaGPUObject <MTLSamplerState> {
@public
	uint64_t _resourceID;
}
@end

/* Encoded by value at each use: the service keeps no depth-stencil objects. */
@interface StelaGPUDepthStencil : NSObject <MTLDepthStencilState> {
@public
	StelaDepthStencilRecord _record;
	StelaGPUDevice *_device;
}
@property(copy) NSString *label;
@end

@interface StelaGPUHeap : StelaGPUObject <MTLHeap> {
@public
	void *_contents;
	NSUInteger _size, _options, _type;
	uint64_t _mappedBytes;
	/* The device's pooled buffers placed in it that are not yet released. */
	NSUInteger _pooled;
	/* Its MTLPurgeableState, zero for the NonVolatile it starts in. */
	NSUInteger _purgeable;
}
- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options;
@end

/* setPurgeableState: of what `slot` holds: the state it was in. The device
 * never reclaims memory, so a volatile resource stays volatile until set
 * otherwise; an emptied one's contents are undefined. */
NSUInteger stela_purgeable_state(NSUInteger *slot, NSUInteger state);

@interface StelaGPUFence : NSObject <MTLFence> {
@public
	StelaGPUDevice *_device;
}
@property(copy) NSString *label;
@end

/* A texture view pool: a buffer of its entries' view IDs, ViewPool.m. */
@interface StelaGPUTextureViewPool : NSObject <MTLTextureViewPool> {
@public
	StelaGPUDevice *_device;
	NSUInteger _count;
	/* The shared buffer of each entry's view ID, or zero. */
	StelaGPUBuffer *_table;
	/* Each entry's view, which the pool keeps. */
	__strong id *_views;
	os_unfair_lock _lock;
}
@property(copy) NSString *label;
@end

/* A rasterization rate map, which renders at full quality: RateMap.m. */
@interface StelaGPURateMap : NSObject <MTLRasterizationRateMap> {
@public
	StelaGPUDevice *_device;
	MTLSize _screenSize;
	NSUInteger _layerCount;
}
@property(copy) NSString *label;
@end

/*
 * A shared event is a kernel IOSurface event, as Apple's are, so its port
 * reaches IOSurface clients, the display and other processes, and a service
 * event the GPU signals and waits on. The two meet at what each side can
 * observe: a command buffer's completion raises the kernel event to the
 * service's value, a GPU wait the GPU alone cannot satisfy listens for the
 * kernel event, and CPU reads, sets and waits take both.
 */
@interface StelaGPUEvent : IOSurfaceSharedEvent <MTLSharedEvent> {
@public
	StelaGPUDevice *_device;
	uint64_t _handle;
}
@property(copy) NSString *label;
/* A new kernel event, or the one `port` names, beside service event `handle`. */
- (instancetype)initWithDevice:(StelaGPUDevice *)device handle:(uint64_t)handle port:(mach_port_t)port;
- (StelaGPUDevice *)device;
/* After a command buffer that signals it completes. */
- (void)publish;
/* Publishes, and whether the service's value reached `value`. */
- (BOOL)publishReaching:(uint64_t)value;
/* Before the GPU waits for `value`. */
- (void)forwardAt:(uint64_t)value;
@end

/* This device's event for `event`, any IOSurface event; nil for another kind. */
StelaGPUEvent *stela_event(StelaGPUDevice *device, id event);

@interface StelaGPULibrary : StelaGPUObject <MTLLibrary, MTLDynamicLibrary> {
@public
	BOOL _identified;
	NSUInteger _type;
	NSString *_installName;
	/* The path of the file it was read from, whose directory `@loader_path`
	 * names. */
	NSString *_path;
	/* The dynamic libraries it links, by install name, loaded as pipelines
	 * first need them. */
	NSArray *_linkedNames;
	NSMutableDictionary *_linked;
	os_unfair_lock _linkLock;
}
- (NSUInteger)type;
/* Libraries to `preloaded`, the dynamic libraries this one links that none
 * of them provides: nil, with *error set, when one does not load. */
- (NSArray *)libraries:(NSArray *)preloaded withLinked:(NSError **)error;
- (BOOL)link:(NSMutableArray *)libraries error:(NSError **)error;
@end

/* The AIR Metal `source` compiles to under `MTLCompileOptions`. */
NSData *stela_compile_source(NSString *source, id options, NSError **error);

/* `MTLBinaryArchive`: the service's pipeline cache. */
@interface StelaGPUBinaryArchive : StelaGPUObject <MTLBinaryArchive>
@end

/* `MTLVertexAttribute` and `MTLAttribute`. */
@interface StelaGPUAttribute : NSObject
@property(copy) NSString *name;
@property NSUInteger attributeIndex, attributeType;
@property(getter=isActive) BOOL active;
@property(getter=isPatchData) BOOL patchData;
@property(getter=isPatchControlPointData) BOOL patchControlPointData;
@end

/* `MTLFunctionConstant`. */
@interface StelaGPUFunctionConstant : NSObject
@property(copy) NSString *name;
@property NSUInteger type, index;
@property BOOL required;
@end

/* Every `MTLType`: which selectors mean something follows from `dataType`. */
@interface StelaGPUType : NSObject {
@public
	NSUInteger _dataType;
	StelaGPUType *_element;
	NSArray *_members;
	/* How far an array's elements are apart in an argument buffer. */
	NSUInteger _stride;
}
@property NSUInteger elementType, arrayLength, argumentIndexStride;
@property NSUInteger access, alignment, dataSize, textureType, textureDataType;
@property BOOL elementIsArgumentBuffer, isDepthTexture;
- (NSUInteger)dataType;
- (NSUInteger)stride;
@end

/* `MTLStructMember`. */
@interface StelaGPUMember : NSObject {
@public
	StelaGPUType *_type;
}
@property(copy) NSString *name;
@property NSUInteger dataType, offset, argumentIndex;
@end

/* Every `MTLBinding`: buffers', textures', threadgroup memory's and payloads'. */
@interface StelaGPUBinding : NSObject <MTLBufferBinding, MTLTextureBinding, MTLThreadgroupBinding, MTLObjectPayloadBinding> {
@public
	BOOL _used;
	StelaGPUType *_typeInfo;
}
@property(copy) NSString *name;
@property NSUInteger type, access, index, arrayLength;
@property NSUInteger bufferAlignment, bufferDataSize, bufferDataType;
@property NSUInteger threadgroupMemoryAlignment, threadgroupMemoryDataSize;
@property NSUInteger objectPayloadAlignment, objectPayloadDataSize;
@property NSUInteger textureType, textureDataType;
@property BOOL isDepthTexture;
@end

@interface StelaGPUReflection : NSObject
+ (NSArray *)bindingsOfFunction:(StelaGPUFunction *)function;
@end

/* `MTLRenderPipelineReflection`. */
@interface StelaGPURenderReflection : NSObject
@property(copy) NSArray *vertexBindings, *fragmentBindings, *objectBindings, *meshBindings, *tileBindings;
@end

/* `MTLFunctionReflection`. */
@interface StelaGPUFunctionReflection : NSObject
@property(copy) NSArray *bindings;
@property(copy) NSString *userAnnotation;
@end

/* `MTLComputePipelineReflection`. */
@interface StelaGPUComputeReflection : NSObject
@property(copy) NSArray *bindings;
@end

@interface StelaGPUArgumentEncoder : NSObject <MTLArgumentEncoder> {
@public
	NSUInteger _encodedLength, _alignment;
}
@property(copy) NSString *label;
- (instancetype)initWithDevice:(StelaGPUDevice *)device
		 encodedLength:(NSUInteger)length
		     alignment:(NSUInteger)alignment;
- (void)addSlotAt:(NSUInteger)index offset:(NSUInteger)offset;
- (void)addType:(StelaGPUType *)type offset:(NSUInteger)offset index:(NSUInteger)index;
@end

@interface StelaGPUFunction : StelaGPUObject <MTLFunction> {
@public
	StelaGPULibrary *_library;
	NSString *_name;
	NSUInteger _functionType, _patchType;
	NSInteger _patchControlPointCount;
	NSArray *_attributes;
	NSDictionary *_constants;
	/* `bitCodeHash`: the SHA-256 of its AIR module. */
	uint8_t _bitCodeHash[32];
}
- (instancetype)initWithLibrary:(StelaGPULibrary *)library handle:(uint64_t)handle name:(NSString *)name;
@end

@interface StelaGPURenderPipeline : StelaGPUObject <MTLRenderPipelineState> {
@public
	id _descriptor;
	uint64_t _resourceID;
	NSUInteger _maxTotalThreadsPerObjectThreadgroup, _maxTotalThreadsPerMeshThreadgroup;
	/* The most threadgroups a mesh grid launches, as the descriptor fixes it. */
	NSUInteger _maxTotalThreadgroupsPerMeshGrid;
	/* The object and mesh threadgroup sizes it fixes, zero for none. */
	MTLSize _requiredObjectThreads, _requiredMeshThreads;
	NSUInteger _objectThreadExecutionWidth, _meshThreadExecutionWidth;
	BOOL _supportIndirectCommandBuffers;
	/* Each stage's linked functions, which handles are found by name in. */
	id _vertexLinked, _fragmentLinked, _objectLinked, _meshLinked;
	/* A tile pipeline, which the service holds as a kernel: its most
	 * threads per tile, whether they match the tile, and its linked
	 * functions. */
	BOOL _tile, _threadgroupSizeMatchesTileSize;
	NSUInteger _maxTotalThreadsPerTileThreadgroup;
	MTLSize _requiredTileThreads;
	id _tileLinked;
	/* The bytes of each sample of its fragment function's imageblock data,
	 * or of its tile function's explicit imageblock. */
	NSUInteger _imageblockSampleLength;
}
@end

@interface StelaGPUComputePipeline : StelaGPUObject <MTLComputePipelineState> {
@public
	id _descriptor;
	uint64_t _resourceID;
	NSUInteger _maxTotalThreadsPerThreadgroup, _threadExecutionWidth;
	NSUInteger _staticThreadgroupMemoryLength;
	/* The bytes of each element of the kernel's explicit imageblock. */
	NSUInteger _imageblockBytes;
	BOOL _supportIndirectCommandBuffers;
	/* What the descriptor fixes: the threadgroup size, zero for none, and
	 * the forward progress usage. */
	MTLSize _requiredThreads;
	NSInteger _forwardProgress;
	id _linked;
}
@end

@interface StelaGPUFunctionHandle : NSObject <MTLFunctionHandle> {
@public
	uint64_t _value;
	StelaGPUFunction *_function;
}
@end

@interface StelaGPUFunctionTable : StelaGPUBuffer <MTLVisibleFunctionTable> {
@public
	NSUInteger _functionCount;
}
@end

@interface StelaGPUIndirectCommandBuffer : StelaGPUResource <MTLIndirectCommandBuffer> {
@public
	NSUInteger _size;
	uint64_t _resourceID;
}
- (void)flush;
/* The address shaders encode commands through, which makes the buffer
 * GPU-resident when first asked for. */
- (MTLResourceID)gpuResourceID;
@end

/* A command of an indirect command buffer, which encoding addresses. */
@interface StelaGPUIndirectRenderCommand : NSObject <MTLIndirectRenderCommand> {
@public
	StelaGPUIndirectCommandBuffer *_buffer;
	NSUInteger _index;
}
@end

@interface StelaGPUIndirectComputeCommand : NSObject <MTLIndirectComputeCommand> {
@public
	StelaGPUIndirectCommandBuffer *_buffer;
	NSUInteger _index;
}
@end

@interface StelaGPUCounter : NSObject <MTLCounter>
@property(copy) NSString *name;
@end

@interface StelaGPUCounterSet : NSObject <MTLCounterSet> {
@public
	uint64_t _code;
}
@property(copy) NSString *name;
@property(copy) NSArray *counters;
+ (instancetype)setWithCode:(uint64_t)code;
@end

@interface StelaGPUCounterSampleBuffer : StelaGPUObject <MTLCounterSampleBuffer> {
@public
	NSUInteger _sampleCount, _valuesPerSample;
}
@end

/* What `Logging.m` reads messages from once no command buffer using it runs. */
@interface StelaGPULogState : NSObject <MTLLogState>
- (uint64_t)bufferHandle;
- (void)commandBufferStarted;
- (void)commandBufferEnded;
@end

@interface StelaGPUDevice (Logging)
- (StelaGPULogState *)environmentLogState;
- (StelaGPULogState *)logStateWithCapacity:(NSInteger)capacity level:(NSInteger)level error:(NSError **)error;
@end

@interface StelaGPUQueue : NSObject <MTLCommandQueue> {
@public
	StelaGPUDevice *_device;
	StelaGPULogState *_logState;
	/* Names the queue to the service, which keeps each queue's order. */
	uint64_t _identity;
	/* The priorities callers set, which the device's one priority serves:
	 * Apple's queues start at 1 and, in the background, 2. */
	_Atomic NSUInteger _gpuPriority, _backgroundGPUPriority;
}
@property(copy) NSString *label;
/* Where completion handlers run, in place of the device's own queue. */
@property(strong) id completionQueue;
- (instancetype)initWithDevice:(StelaGPUDevice *)device;
- (void)enqueue:(StelaGPUCommandBuffer *)buffer;
- (void)flush;
@end

/* Error records use only execution state known at the service boundary. */
@interface StelaGPUEncoderInfo : NSObject <MTLCommandBufferEncoderInfo>
@property(copy) NSString *label;
@property(copy) NSArray *debugSignposts;
@property NSInteger errorState;
@end

NSError *stela_command_error(const StelaFailure *failure, NSArray *encoders, BOOL submitted);
id stela_empty_logs(void);

@interface StelaGPUCommandBuffer : NSObject <MTLCommandBuffer> {
@public
	StelaGPUDevice *_device;
	StelaGPUQueue *_queue;
	StelaStream _stream;
	NSMutableArray *_retained;
	StelaGPULogState *_logState;
	uint64_t _serial;
	_Atomic(bool) _committed;
	/* `userDictionary`, made at its first ask. */
	NSMutableDictionary *_userDictionary;
}
@property(copy) NSString *label;
@property NSUInteger errorOptions;
/* Held for its reader: no memory here is protected. */
@property NSUInteger protectionOptions;
- (instancetype)initWithQueue:(StelaGPUQueue *)queue
	    retainsReferences:(BOOL)retains
		     logState:(StelaGPULogState *)logState;
/* Keeps `object` until the command buffer completes, unless it retains none. */
- (void)use:(id)object;
- (StelaGPUEncoderInfo *)recordEncoder;
- (void)submit;
/* On the device's queue, ahead of completion. */
- (void)didSchedule;
- (void)beginRender:(id)descriptor;
/* The service finished the command buffer, or `failure` ended it. */
- (void)didCompleteWith:(const StelaFailure *)failure;
/* Metal's synchronization notifications (SPI of `_MTLCommandBuffer`, probed
 * on macOS 27.2): `addSynchronizationNotification:` keeps a block, which is
 * called with the command buffer and a StelaSynchronization at each memory
 * barrier an encoder encodes (phase 0, with the barrier's scope or
 * resources), at each encoder's end (phase 1) and at commit (phase 2). A
 * block that answers YES is done and dropped; phases 2 and 3 drop every
 * block. MetalFX waits for them. */
- (void)addSynchronizationNotification:(BOOL (^)(id, const void *))block;
- (void)executeSynchronizationNotifications:(int)phase;
- (void)executeSynchronizationNotifications:(int)phase
				      scope:(uint64_t)scope
				  resources:(const id __unsafe_unretained *)resources
				      count:(uint64_t)count;
@end

/* The commands an encoder writes, into its command buffer's stream or its own. */
@interface StelaGPUEncoder : NSObject <MTLCommandEncoder> {
@public
	StelaGPUCommandBuffer *_commands;
	StelaStream *_stream;
}
@property(copy) NSString *label;
- (instancetype)initWithCommandBuffer:(StelaGPUCommandBuffer *)commands stream:(StelaStream *)stream;
/* Samples a descriptor's `sampleBufferAttachments` ask at the encoder's two ends. */
- (void)sampleAttachments:(id)attachments;
- (void)endEncoding;
- (void)sampleCountersInBuffer:(StelaGPUCounterSampleBuffer *)buffer
		 atSampleIndex:(NSUInteger)index
		   withBarrier:(BOOL)barrier;
@end

@interface StelaGPUBlitEncoder : StelaGPUEncoder <MTLBlitCommandEncoder>
@end

@interface StelaGPUComputeEncoder : StelaGPUEncoder <MTLComputeCommandEncoder> {
@public
	NSUInteger _dispatchType;
}
@end

@interface StelaGPURenderEncoder : StelaGPUEncoder <MTLRenderCommandEncoder> {
@public
	/* The pass's tile size, which tile dispatches cover the targets with. */
	NSUInteger _tileWidth, _tileHeight;
}
- (void)setCullMode:(NSUInteger)mode;
- (void)setFrontFacingWinding:(NSUInteger)winding;
- (void)setTriangleFillMode:(NSUInteger)mode;
- (void)setDepthClipMode:(NSUInteger)mode;
- (void)setDepthBias:(float)bias slopeScale:(float)scale clamp:(float)clamp;
- (void)setDepthStencilState:(StelaGPUDepthStencil *)state;
- (void)setStencilReferenceValue:(uint32_t)value;
- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha;
- (void)setVisibilityResultMode:(NSUInteger)mode offset:(NSUInteger)offset;
@end

@interface StelaGPUResourceStateEncoder : StelaGPUEncoder <MTLResourceStateCommandEncoder>
@end

@interface StelaGPUParallelEncoder : StelaGPUEncoder <MTLParallelRenderCommandEncoder> {
@public
	NSUInteger _tileWidth, _tileHeight;
}
@end

/* A render pass descriptor's tile size as Apple's GPUs take it: 32 by 32
 * pixels by default, a height of at least 16 and at most the width. */
void stela_tile_size(id descriptor, NSUInteger *width, NSUInteger *height);

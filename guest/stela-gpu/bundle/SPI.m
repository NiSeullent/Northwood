/* What Metal's private SPI asks of a driver that has a real answer here. */
#include "Bundle.h"

@interface NSObject (StelaSPI)
- (void)commit;
- (void)waitUntilScheduled;
- (void)encodeSignalEvent:(id)event value:(uint64_t)value;
- (void)setFragmentTexture:(id)texture atIndex:(NSUInteger)index;
- (void)setFragmentSamplerState:(id)sampler atIndex:(NSUInteger)index;
- (void)setDepthClipMode:(NSUInteger)mode;
- (NSUInteger)resourceOptions;
- (void)setCpuType:(uint32_t)type;
- (void)setSubType:(uint32_t)type;
@end

@implementation StelaGPUDevice (SPI)

/* The code the device runs, by which Metal picks a universal library's slice
 * and keys binary archives: AIR, which every library carries for a device
 * other than Apple's own, and no slice of Apple's GPU code. */
- (id)targetDeviceArchitecture
{
	id architecture = [NSClassFromString(@"MTLTargetDeviceArchitecture") new];
	[architecture setCpuType:CPU_TYPE_AIR64];
	[architecture setSubType:0];
	return architecture;
}

@end

@implementation StelaGPUTexture (SPI)

/* The CPU changed the texture's memory behind Metal. An IOSurface or buffer
 * texture here is a linear view of that coherent memory, which every later
 * command reads as it is, so nothing the GPU holds needs dropping. */
- (void)didModifyData
{
}

/* Whether a layer's drawable holds the texture: made from a descriptor Core
 * Animation marks drawable, or a view of such a texture, as Apple's drivers
 * answer. */
- (BOOL)isDrawable
{
	return _drawable;
}

@end

@implementation StelaGPUBuffer (SPI)

/* The IOSurface the buffer lies in, NULL for other buffers. */
- (IOSurfaceRef)iosurface
{
	return _surface;
}

@end

@implementation StelaGPUQueue (SPI)

/* Every queue here executes what it is given. */
- (BOOL)executionEnabled
{
	return YES;
}

/* A commit runs on its caller, with no queue of its own to replace. */
- (void)setSubmissionQueue:(id)queue
{
}

/* Every queue here runs at the device's one priority, and answers the
 * priorities its callers set. */
- (BOOL)setGPUPriority:(NSUInteger)priority
{
	_gpuPriority = priority;
	return YES;
}

- (BOOL)setGPUPriority:(NSUInteger)priority offset:(NSUInteger)offset
{
	return [self setGPUPriority:priority];
}

- (NSUInteger)getGPUPriority
{
	return _gpuPriority;
}

- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority
{
	_backgroundGPUPriority = priority;
	return YES;
}

- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(NSUInteger)offset
{
	return [self setBackgroundGPUPriority:priority];
}

- (NSUInteger)getBackgroundGPUPriority
{
	return _backgroundGPUPriority;
}

@end

@implementation StelaGPUCommandBuffer (SPI)

/* The device has no deadlines: work runs as it arrives. */
- (void)commitWithDeadline:(uint64_t)deadline
{
	[(id)self commit];
}

- (BOOL)commitAndWaitUntilSubmitted
{
	[(id)self commit];
	[(id)self waitUntilScheduled];
	return YES;
}

- (BOOL)commitAndWaitUntilSubmittedWithDeadline:(uint64_t)deadline
{
	return [self commitAndWaitUntilSubmitted];
}

/* What frameworks keep with a command buffer until it is released, such as
 * the work MPS has encoded into it: one dictionary for its lifetime. */
static os_unfair_lock userDictionaries = OS_UNFAIR_LOCK_INIT;

- (NSMutableDictionary *)userDictionary
{
	os_unfair_lock_lock(&userDictionaries);
	if (!_userDictionary) {
		_userDictionary = [NSMutableDictionary dictionary];
	}
	NSMutableDictionary *dictionary = _userDictionary;
	os_unfair_lock_unlock(&userDictionaries);
	return dictionary;
}

/* Every agent sees a signal the device makes. */
- (void)encodeSignalEvent:(id)event value:(uint64_t)value agentMask:(uint64_t)mask
{
	[(id)self encodeSignalEvent:event value:value];
}

@end

@implementation StelaGPUFunction (SPI)

/* The SHA-256 of the function's AIR module, which callers key caches by
 * and read through the pointer. */
- (const void *)bitCodeHash
{
	return _bitCodeHash;
}

@end

@implementation StelaGPUResource (SPI)

/* Work on a resource finishes with its command buffer, which callers wait on. */
- (BOOL)isComplete
{
	return YES;
}

- (BOOL)isWriteComplete
{
	return YES;
}

- (NSUInteger)unfilteredResourceOptions
{
	return [(id)self resourceOptions];
}

/* No memory here is protected. */
- (NSUInteger)protectionOptions
{
	return 0;
}

/* The process whose footprint the resource counts toward. Apple's kernel
 * driver charges the memory to it; this device's service charges each
 * connection its own, so the resource keeps the answer for its callers. */
- (void)setResponsibleProcess:(int)process
{
	_responsibleProcess = process;
}

- (int)responsibleProcess
{
	return _responsibleProcess;
}

@end

@implementation StelaGPURenderEncoder (SPI)

- (void)setDepthClipModeSPI:(NSUInteger)mode
{
	[(id)self setDepthClipMode:mode];
}

- (void)setFragmentTexture:(id)texture
	    atTextureIndex:(NSUInteger)textureIndex
	      samplerState:(id)sampler
	    atSamplerIndex:(NSUInteger)samplerIndex
{
	[(id)self setFragmentTexture:texture atIndex:textureIndex];
	[(id)self setFragmentSamplerState:sampler atIndex:samplerIndex];
}

@end

/* Metal's tracing names each device, queue, command buffer, encoder and heap
 * by a process-wide number that never repeats, which MPS reads. Apple's
 * driver numbers them as it makes them; here each takes the next number
 * when first asked. */
static uint64_t stela_trace_object_id(id object)
{
	static uint64_t next = 1;
	static char key;
	static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
	os_unfair_lock_lock(&lock);
	NSNumber *number = objc_getAssociatedObject(object, &key);
	if (!number) {
		number = @(next++);
		objc_setAssociatedObject(object, &key, number, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	}
	os_unfair_lock_unlock(&lock);
	return number.unsignedLongLongValue;
}

@implementation StelaGPUDevice (Trace)
- (uint64_t)globalTraceObjectID
{
	return stela_trace_object_id(self);
}
@end

@implementation StelaGPUQueue (Trace)
- (uint64_t)globalTraceObjectID
{
	return stela_trace_object_id(self);
}
@end

@implementation StelaGPUCommandBuffer (Trace)
- (uint64_t)globalTraceObjectID
{
	return stela_trace_object_id(self);
}

/* No work here is protected. */
- (NSUInteger)protectionOptions
{
	return 0;
}
@end

@implementation StelaGPUEncoder (Trace)
- (uint64_t)globalTraceObjectID
{
	return stela_trace_object_id(self);
}
@end

@implementation StelaGPUHeap (Trace)
- (uint64_t)globalTraceObjectID
{
	return stela_trace_object_id(self);
}

/* No memory here is protected. */
- (NSUInteger)protectionOptions
{
	return 0;
}
@end

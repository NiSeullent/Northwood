/*
 * The device: what the capability block answers, the objects that need nothing
 * of their own, and completion delivery. Metal's own _MTLDevice keeps the
 * registration bookkeeping; every query it would answer from a hardware hook
 * is answered here.
 */
#include "Bundle.h"

#include <stdatomic.h>

#define FLAG(selector, flag)                                                                       \
	-(BOOL)selector                                                                            \
	{                                                                                          \
		return (_capabilities->flags & StelaFlags_##flag) != 0;                            \
	}
#define LIMIT(selector, field)                                                                     \
	-(NSUInteger)selector                                                                      \
	{                                                                                          \
		return _capabilities->limits.field;                                                \
	}
#define FIELD(selector, field)                                                                     \
	-(NSUInteger)selector                                                                      \
	{                                                                                          \
		return _capabilities->field;                                                       \
	}

/* Vulkan's standard sample locations, the GPU's own, in 1/16ths of a pixel. */
static const uint8_t PATTERN_2[][2] = {{12, 12}, {4, 4}};
static const uint8_t PATTERN_4[][2] = {{6, 2}, {14, 6}, {2, 10}, {10, 14}};
static const uint8_t PATTERN_8[][2] = {{9, 5},	{7, 11}, {13, 9}, {5, 3},
				       {3, 13}, {1, 7},	 {11, 15}, {15, 1}};
static const uint8_t PATTERN_16[][2] = {{9, 9},	 {7, 5},  {5, 10}, {12, 7}, {3, 6},  {10, 13},
					{13, 11}, {11, 3}, {6, 14}, {8, 1},  {4, 2},  {2, 12},
					{0, 8},	 {15, 4}, {14, 15}, {1, 0}};

@interface StelaGPUArchitecture : NSObject
@property(copy) NSString *name;
@end

@implementation StelaGPUArchitecture
@end

static void completed(void *reference, int result, void *count);

/* Small shared buffers share heaps of the device's own, as a GPU driver
 * suballocates them: each new heap imports memory once, while a buffer of its
 * own would import, map and unmap its pages through the host's kernel. */
#define POOLED_MAX_BYTES (256u << 10)
#define SLAB_BYTES (4u << 20)

/* Metal's values for the modes a pool is kept by: CPU cache mode (default,
 * write-combined) and hazard tracking mode (default, untracked, tracked). */
enum {
	POOL_CACHE_MODES = 2,
	POOL_HAZARD_DEFAULT = 0,
	POOL_HAZARD_TRACKED = 2,
	POOL_HAZARD_MODES = 3,
	POOL_KINDS = POOL_CACHE_MODES * POOL_HAZARD_MODES,
	/* MTLHeapDescriptor's defaults. */
	POOL_HEAP_AUTOMATIC = 0,
	POOL_SPARSE_PAGE_16 = MTLSparsePageSize16,
};

/* A handle whose release waits for the GPU, and the heap kept until then. */
@interface StelaDeferredRelease : NSObject {
@public
	uint64_t _handle, _serial;
	id _slab;
}
@end

@implementation StelaDeferredRelease
@end

/* The tiles a pixel span covers, or lies across whole for an inward mode. */
static void tile_span(NSUInteger start, NSUInteger length, NSUInteger tile, NSUInteger mode, NSUInteger *first,
		      NSUInteger *count)
{
	NSUInteger end = start + length;
	BOOL inward = mode == MTLSparseTextureRegionAlignmentModeInward;
	NSUInteger low = (inward ? start + tile - 1 : start) / tile;
	NSUInteger high = (inward ? end : end + tile - 1) / tile;
	*first = low;
	*count = high > low ? high - low : 0;
}

@implementation StelaGPUDevice {
	io_service_t _port;
	id _acceleratorService;
	__weak id _deviceWrapper;
	dispatch_queue_t _completions;
	IONotificationPortRef _notifications;
	os_unfair_lock _lock;
	/* Committed command buffers, by serial. */
	NSMutableArray *_inFlight;
	/* What refused command buffers signal, as [event, value]: the service
	 * still applies it, in order, and each completion publishes it. */
	NSMutableArray *_settling;
	NSMutableArray *_indirect;
	uint64_t _seen;
	/* Whether the kext holds this device's notification; only `_completions`
	 * raises it, and the kext's delivery ends it. */
	_Atomic(bool) _armed;
	bool _lost;
	/* A moment both clocks read: the GPU's in ticks, the CPU's in seconds. */
	uint64_t _anchorTick;
	double _anchorSeconds;
	BOOL _shouldMaximizeConcurrentCompilation;
	/* The newest serial the service accepted a commit at, and the newest it
	 * reached; releases waiting for the GPU, in serial order. */
	_Atomic(uint64_t) _submitted;
	uint64_t _reached;
	NSMutableArray *_deferred;
	os_unfair_lock _deferLock;
	/* The pooled buffers' heaps by cache and hazard tracking mode, newest
	 * last. */
	NSMutableArray *_slabs[POOL_KINDS];
	os_unfair_lock _poolLock;
	/* Completion notifications and CPU signals seen, and the event waiters
	 * sleeping on `_wake` until the next. */
	os_unfair_lock _wakeLock;
	dispatch_semaphore_t _wake;
	uint64_t _wakes;
	long _sleepers;
}

- (instancetype)initWithAcceleratorPort:(io_service_t)port
{
	self = [super init];
	if (!self) {
		return nil;
	}
	_port = port;
	IOObjectRetain(port);
	_lock = OS_UNFAIR_LOCK_INIT;
	_inFlight = [NSMutableArray arrayWithCapacity:64];
	_settling = [NSMutableArray arrayWithCapacity:2];
	_indirect = [NSMutableArray arrayWithCapacity:4];
	_deferred = [NSMutableArray arrayWithCapacity:64];
	for (NSUInteger i = 0; i < POOL_KINDS; i++) {
		_slabs[i] = [NSMutableArray arrayWithCapacity:4];
	}
	_deferLock = OS_UNFAIR_LOCK_INIT;
	_poolLock = OS_UNFAIR_LOCK_INIT;
	_wakeLock = OS_UNFAIR_LOCK_INIT;
	_wake = dispatch_semaphore_create(0);
	mach_vm_address_t page = 0;
	mach_vm_size_t bytes = 0;
	if (IOServiceOpen(port, mach_task_self(), kStelaGPUUserClientType, &_gate.connect) != KERN_SUCCESS) {
		return nil;
	}
	if (IOConnectMapMemory64(_gate.connect, kStelaGPUGateMemory, mach_task_self(), &page, &bytes,
				 kIOMapAnywhere | kIOMapDefaultCache)
	    != KERN_SUCCESS) {
		IOServiceClose(_gate.connect);
		return nil;
	}
	_gate.page = (volatile uint64_t *)page;

	StelaNegotiateRequest versions = {.major = STELA_MAJOR, .minor = STELA_MINOR};
	StelaNegotiateResponse agreed;
	StelaFailure failure;
	if (STELA_CALL(&_gate, Negotiate, &versions, &agreed, &failure) != StelaStatusOk) {
		stela_report("the service's protocol", &failure);
		return nil;
	}
	stela_report_to(&_gate);
	StelaEmpty nothing = {};
	if (agreed.capability_bytes != sizeof(*_capabilities)) {
		NSLog(@"StelaGPU: the capability block is %u bytes, not %zu", agreed.capability_bytes,
		      sizeof(*_capabilities));
		return nil;
	}
	_capabilities = malloc(sizeof(*_capabilities));
	if (!_capabilities) {
		return nil;
	}
	if (STELA_CALL(&_gate, Capabilities, &nothing, _capabilities, &failure) != StelaStatusOk) {
		stela_report("the capability block", &failure);
		return nil;
	}

	mach_timebase_info_data_t timebase;
	mach_timebase_info(&timebase);
	uint64_t now;
	[self sampleTimestamps:&now gpuTimestamp:&_anchorTick];
	_anchorSeconds = (double)now * timebase.numer / timebase.denom / 1e9;

	_completions = dispatch_queue_create("com.stela.StelaGPU.completions", NULL);
	_notifications = IONotificationPortCreate(kIOMainPortDefault);
	IONotificationPortSetDispatchQueue(_notifications, _completions);
	/* Armed once something waits: nothing does yet. */
	return self;
}

- (uint32_t)acceleratorPort
{
	return _port;
}

- (void)_setAcceleratorService:(id)service
{
	_acceleratorService = service;
}

- (id)acceleratorService
{
	return _acceleratorService;
}

/* The base class leaves these to the driver. Metal's lookup by registry ID
 * answers with the wrapper, which is this device until a layer names itself. */
- (void)_setDeviceWrapper:(id)wrapper
{
	_deviceWrapper = wrapper;
}

- (id)_deviceWrapper
{
	return _deviceWrapper ?: self;
}

/* The block answers these; there is nothing to gather. */
- (void)initLimits
{
}

- (void)initFeatureQueries
{
}

- (void)initGPUFamilySupport
{
}

/* An unknown profile: no code path of Metal's may take this for an AGX GPU. */
- (uint32_t)featureProfile
{
	return 'STLA';
}

/*
 * Waking on completions. The kext wakes every armed device at each completion
 * interrupt, whichever process's work completed, so a device stays armed only
 * while something of its own waits for one: command buffers in flight,
 * releases waiting for the GPU, refused buffers' signals to publish, or event
 * waiters asleep. Otherwise each process using Metal would wake and ask the
 * service at every other process's completion.
 */

/* On `_completions`. */
- (void)arm
{
	atomic_store(&_armed, true);
	uint64_t reference[kOSAsyncRef64Count] = {
		[kIOAsyncCalloutFuncIndex] = (uint64_t)completed,
		[kIOAsyncCalloutRefconIndex] = (uint64_t)(__bridge void *)self,
	};
	mach_port_t port = IONotificationPortGetMachPort(_notifications);
	IOConnectCallAsyncScalarMethod(_gate.connect, kStelaGPUNotifyCompletion, port, reference,
				       kOSAsyncRef64Count, &_seen, 1, NULL, NULL);
}

/* Whether anything of this device waits for a completion notification. */
- (bool)awaitsCompletion
{
	os_unfair_lock_lock(&_lock);
	bool waiting = _inFlight.count != 0 || _settling.count != 0;
	os_unfair_lock_unlock(&_lock);
	os_unfair_lock_lock(&_deferLock);
	waiting = waiting || _deferred.count != 0;
	os_unfair_lock_unlock(&_deferLock);
	os_unfair_lock_lock(&_wakeLock);
	waiting = waiting || _sleepers != 0;
	os_unfair_lock_unlock(&_wakeLock);
	return waiting;
}

/*
 * On `_completions`, after a delivery ended the kext's notification: arms the
 * next while anything waits. It reads as disarmed before the look, so whoever
 * starts waiting after the look finds it disarmed and asks (`armIfNeeded`).
 */
- (void)rearm
{
	atomic_store(&_armed, false);
	atomic_thread_fence(memory_order_seq_cst);
	if ([self awaitsCompletion]) {
		[self arm];
	}
}

/* From any thread, once something waits: arms the notification unless armed. */
- (void)armIfNeeded
{
	atomic_thread_fence(memory_order_seq_cst);
	if (atomic_load(&_armed)) {
		return;
	}
	dispatch_async(_completions, ^{
		if (!atomic_load(&self->_armed)) {
			[self arm];
		}
	});
}

- (void)completionsReached:(uint64_t)count
{
	_seen = count;
	StelaEmpty nothing = {};
	StelaPollResponse reached = {};
	StelaFailure failure;
	bool lost = STELA_CALL(&_gate, Poll, &nothing, &reached, &failure) != StelaStatusOk;
	NSMutableArray *done = [NSMutableArray arrayWithCapacity:8];
	os_unfair_lock_lock(&_lock);
	_lost = _lost || lost;
	while (_inFlight.count) {
		StelaGPUCommandBuffer *buffer = [_inFlight objectAtIndex:0];
		if (!lost && buffer->_serial > reached.value) {
			break;
		}
		[done addObject:buffer];
		[_inFlight removeObjectAtIndex:0];
	}
	/* Another queue's held work delays the serial order, not this queue's
	 * completions: the service names those that completed past it. */
	uint64_t after = reached.value;
	while (!lost && [self inFlightSpansQueues]) {
		os_unfair_lock_unlock(&_lock);
		StelaValue request = {.value = after}, next = {};
		bool answered = STELA_CALL(&_gate, CompletedAfter, &request, &next, NULL) == StelaStatusOk;
		os_unfair_lock_lock(&_lock);
		if (!answered || next.value == 0) {
			break;
		}
		after = next.value;
		for (NSUInteger i = 0; i < _inFlight.count; i++) {
			StelaGPUCommandBuffer *buffer = [_inFlight objectAtIndex:i];
			if (buffer->_serial == after) {
				[done addObject:buffer];
				[_inFlight removeObjectAtIndex:i];
				break;
			}
		}
	}
	NSArray *settling = [_settling copy];
	os_unfair_lock_unlock(&_lock);
	for (NSUInteger i = 0; i < done.count; i++) {
		StelaGPUCommandBuffer *buffer = [done objectAtIndex:i];
		/* The service names the oldest completed buffer whose submission
		 * the driver refused: it and later ones answer whether theirs was. */
		StelaFailure refusal;
		bool refused = false;
		if (!lost && reached.refused != 0 && buffer->_serial >= reached.refused) {
			StelaValue serial = {.value = buffer->_serial};
			refused = STELA_CALL(&_gate, CommandFailure, &serial, &nothing, &refusal) !=
				  StelaStatusOk;
		}
		[buffer didCompleteWith:lost ? &failure : refused ? &refusal : NULL];
	}
	[self publishSettling:settling lost:lost];
	[self releaseDeferredReaching:lost ? UINT64_MAX : reached.value];
	[self wakeWaiters];
	[self rearm];
}

/* Under `_lock`: whether the command buffers in flight are of more than one
 * queue, so one may complete before an older one. */
- (bool)inFlightSpansQueues
{
	StelaGPUQueue *first = nil;
	for (NSUInteger i = 0; i < _inFlight.count; i++) {
		StelaGPUQueue *queue = ((StelaGPUCommandBuffer *)[_inFlight objectAtIndex:i])->_queue;
		if (first && queue != first) {
			return true;
		}
		first = queue;
	}
	return false;
}

- (uint64_t)wakes
{
	os_unfair_lock_lock(&_wakeLock);
	uint64_t wakes = _wakes;
	os_unfair_lock_unlock(&_wakeLock);
	return wakes;
}

/* Wakes every sleeper counted since the last wake, one signal each. */
- (void)wakeWaiters
{
	os_unfair_lock_lock(&_wakeLock);
	_wakes++;
	long sleepers = _sleepers;
	_sleepers = 0;
	os_unfair_lock_unlock(&_wakeLock);
	while (sleepers-- > 0) {
		dispatch_semaphore_signal(_wake);
	}
}

- (void)waitForWakeAfter:(uint64_t)wakes milliseconds:(uint64_t)milliseconds
{
	os_unfair_lock_lock(&_wakeLock);
	if (_wakes != wakes) {
		os_unfair_lock_unlock(&_wakeLock);
		return;
	}
	_sleepers++;
	os_unfair_lock_unlock(&_wakeLock);
	[self armIfNeeded];
	if (dispatch_semaphore_wait(_wake, dispatch_time(DISPATCH_TIME_NOW, (int64_t)milliseconds * 1000000)) == 0) {
		return;
	}
	/* Timed out: still counted unless a wake came since. That wake's signal
	 * for this sleeper stays, and wakes a later one early, which then only
	 * looks again. */
	os_unfair_lock_lock(&_wakeLock);
	if (_wakes == wakes) {
		_sleepers--;
	}
	os_unfair_lock_unlock(&_wakeLock);
}

- (void)noteSubmitted:(uint64_t)serial
{
	uint64_t seen = atomic_load(&_submitted);
	while (seen < serial && !atomic_compare_exchange_weak(&_submitted, &seen, serial)) {
	}
}

- (void)releaseHandleAfterWork:(uint64_t)handle slab:(id)slab
{
	StelaDeferredRelease *entry = [StelaDeferredRelease new];
	entry->_handle = handle;
	entry->_slab = slab;
	os_unfair_lock_lock(&_deferLock);
	entry->_serial = atomic_load(&_submitted);
	bool now = entry->_serial <= _reached;
	if (!now) {
		[_deferred addObject:entry];
	}
	os_unfair_lock_unlock(&_deferLock);
	if (now) {
		[self released:entry];
	} else {
		[self armIfNeeded];
	}
}

/* Releases what waited for the GPU to reach `reached`. */
- (void)releaseDeferredReaching:(uint64_t)reached
{
	NSMutableArray *ready = [NSMutableArray array];
	os_unfair_lock_lock(&_deferLock);
	_reached = reached > _reached ? reached : _reached;
	while (_deferred.count) {
		StelaDeferredRelease *entry = [_deferred objectAtIndex:0];
		if (entry->_serial > _reached) {
			break;
		}
		[ready addObject:entry];
		[_deferred removeObjectAtIndex:0];
	}
	os_unfair_lock_unlock(&_deferLock);
	for (NSUInteger i = 0; i < ready.count; i++) {
		[self released:[ready objectAtIndex:i]];
	}
}

/* A pooled buffer's place is free: its heap goes once no buffer holds it and
 * a newer one takes the pool's allocations. */
- (void)released:(StelaDeferredRelease *)entry
{
	[self releaseHandle:entry->_handle];
	StelaGPUHeap *slab = entry->_slab;
	if (!slab) {
		return;
	}
	os_unfair_lock_lock(&_poolLock);
	if (--slab->_pooled == 0) {
		for (NSUInteger kind = 0; kind < POOL_KINDS; kind++) {
			NSMutableArray *slabs = _slabs[kind];
			NSUInteger index = [slabs indexOfObjectIdenticalTo:slab];
			if (index != NSNotFound && index + 1 < slabs.count) {
				[slabs removeObjectAtIndex:index];
			}
		}
	}
	os_unfair_lock_unlock(&_poolLock);
}

- (id)pooledBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{
	NSUInteger mode = (options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift;
	NSUInteger cache = (options & MTLResourceCPUCacheModeMask) >> MTLResourceCPUCacheModeShift;
	NSUInteger hazard = (options & MTLResourceHazardTrackingModeMask) >> MTLResourceHazardTrackingModeShift;
	/* Other modes and options take the path that reports what they are. */
	NSUInteger kept = MTLResourceCPUCacheModeMask | MTLResourceStorageModeMask | MTLResourceHazardTrackingModeMask;
	if (mode != MTLStorageModeShared || (options & ~kept) || cache >= POOL_CACHE_MODES ||
	    hazard >= POOL_HAZARD_MODES || length == 0 || length > POOLED_MAX_BYTES) {
		return nil;
	}
	NSMutableArray *slabs = _slabs[cache * POOL_HAZARD_MODES + hazard];
	StelaGPUBuffer *buffer = nil;
	StelaGPUHeap *slab = nil;
	os_unfair_lock_lock(&_poolLock);
	/* Only the newest heap takes buffers; older ones are kept only while
	 * buffers remain in them, so each drains and goes. */
	if (slabs.count) {
		slab = [slabs objectAtIndex:slabs.count - 1];
		buffer = [slab newBufferWithLength:length options:options];
	}
	if (!buffer) {
		/* A buffer of its own is tracked unless the app says otherwise. */
		NSUInteger tracking = hazard == POOL_HAZARD_DEFAULT ? POOL_HAZARD_TRACKED : hazard;
		StelaHeapDescriptorRecord record = {
			.size = SLAB_BYTES,
			.options = (cache << MTLResourceCPUCacheModeShift) |
				   ((NSUInteger)MTLStorageModeShared << MTLResourceStorageModeShift) |
				   (tracking << MTLResourceHazardTrackingModeShift),
			.heap_type = POOL_HEAP_AUTOMATIC,
			.sparse_page_size = POOL_SPARSE_PAGE_16,
		};
		slab = [self newHeapWithRecord:record];
		buffer = [slab newBufferWithLength:length options:options];
		if (buffer) {
			[slabs addObject:slab];
		}
	}
	if (buffer) {
		slab->_pooled++;
		/* An app's buffer of its own: no heap, at no heap offset. */
		buffer->_heap = nil;
		buffer->_heapOffset = 0;
		buffer->_slab = slab;
	}
	os_unfair_lock_unlock(&_poolLock);
	/* Metal zero-fills a new buffer. A heap range is not: it may hold the
	 * bytes of a buffer freed from it before. */
	if (buffer && buffer->_contents) {
		memset(buffer->_contents, 0, length);
	}
	return buffer;
}

static void completed(void *reference, int result, void *count)
{
	(void)result;
	[(__bridge StelaGPUDevice *)reference completionsReached:(uint64_t)count];
}

/* A completion can precede this, so the queue looks again once it is noted. */
- (void)track:(StelaGPUCommandBuffer *)buffer
{
	os_unfair_lock_lock(&_lock);
	[_inFlight addObject:buffer];
	os_unfair_lock_unlock(&_lock);
	dispatch_async(_completions, ^{
		[buffer didSchedule];
		[self completionsReached:self->_seen];
	});
}

- (void)settle:(StelaGPUEvent *)event at:(uint64_t)value
{
	os_unfair_lock_lock(&_lock);
	[_settling addObject:@[ event, @(value) ]];
	os_unfair_lock_unlock(&_lock);
	[self armIfNeeded];
}

/* Publishes what refused command buffers signal, dropping each value the
 * kernel event reached, or all of them once the device is lost. */
- (void)publishSettling:(NSArray *)settling lost:(bool)lost
{
	NSMutableArray *reached = [NSMutableArray arrayWithCapacity:settling.count];
	for (NSUInteger i = 0; i < settling.count; i++) {
		NSArray *pair = [settling objectAtIndex:i];
		StelaGPUEvent *event = [pair objectAtIndex:0];
		if (lost || [event publishReaching:[[pair objectAtIndex:1] unsignedLongLongValue]]) {
			[reached addObject:pair];
		}
	}
	if (reached.count) {
		os_unfair_lock_lock(&_lock);
		for (NSUInteger i = 0; i < reached.count; i++) {
			NSUInteger index = [_settling indexOfObjectIdenticalTo:[reached objectAtIndex:i]];
			if (index != NSNotFound) {
				[_settling removeObjectAtIndex:index];
			}
		}
		os_unfair_lock_unlock(&_lock);
	}
}

- (void)failed:(StelaGPUCommandBuffer *)buffer with:(StelaFailure)failure
{
	dispatch_async(_completions, ^{
		[buffer didCompleteWith:&failure];
	});
}

- (CFTimeInterval)secondsOfTick:(uint64_t)tick
{
	if (tick == 0) {
		return 0;
	}
	return _anchorSeconds + (double)(int64_t)(tick - _anchorTick) / (double)_capabilities->timestamp_frequency;
}

- (void)noteIndirect:(StelaGPUIndirectCommandBuffer *)buffer
{
	os_unfair_lock_lock(&_lock);
	[_indirect addObject:[NSValue valueWithNonretainedObject:buffer]];
	os_unfair_lock_unlock(&_lock);
}

- (void)forgetIndirect:(StelaGPUIndirectCommandBuffer *)buffer
{
	os_unfair_lock_lock(&_lock);
	for (NSUInteger i = _indirect.count; i > 0; i--) {
		if ([[_indirect objectAtIndex:i - 1] nonretainedObjectValue] == buffer) {
			[_indirect removeObjectAtIndex:i - 1];
		}
	}
	os_unfair_lock_unlock(&_lock);
}

- (void)flushIndirect
{
	os_unfair_lock_lock(&_lock);
	NSArray *noted = _indirect;
	_indirect = [NSMutableArray arrayWithCapacity:4];
	for (NSUInteger i = 0; i < noted.count; i++) {
		[(StelaGPUIndirectCommandBuffer *)[[noted objectAtIndex:i] nonretainedObjectValue] flush];
	}
	os_unfair_lock_unlock(&_lock);
}

- (void)releaseHandle:(uint64_t)handle
{
	StelaHandle request = {.handle = handle};
	StelaEmpty nothing;
	STELA_CALL(&_gate, Release, &request, &nothing, NULL);
}

/* What a CPU reaches is the task's own memory, wired for the GPU. */

- (uint64_t)importAddress:(void *)address length:(uint64_t)length failure:(StelaFailure *)failure
{
	uint64_t input[2] = {(uint64_t)address, length};
	uint64_t handle = 0;
	uint32_t outputs = 1;
	kern_return_t result =
	    IOConnectCallScalarMethod(_gate.connect, kStelaGPUImportMemory, input, 2, &handle, &outputs);
	if (result != KERN_SUCCESS) {
		*failure = (StelaFailure){.status = StelaStatusOutOfMemory, .reason = "out of wired memory"};
		return 0;
	}
	return handle;
}

- (uint64_t)importLength:(uint64_t)length contents:(void **)contents failure:(StelaFailure *)failure
{
	mach_vm_address_t address = 0;
	if (mach_vm_allocate(mach_task_self(), &address, length, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) {
		*failure = (StelaFailure){.status = StelaStatusOutOfMemory, .reason = "out of memory"};
		return 0;
	}
	uint64_t memory = [self importAddress:(void *)address length:length failure:failure];
	if (memory == 0) {
		mach_vm_deallocate(mach_task_self(), address, length);
		return 0;
	}
	*contents = (void *)address;
	return memory;
}

/* Identity. */

- (NSString *)name
{
	return [[NSString alloc] initWithBytes:_capabilities->name
					length:_capabilities->name_bytes
				      encoding:NSUTF8StringEncoding];
}

- (id)architecture
{
	StelaGPUArchitecture *architecture = [StelaGPUArchitecture new];
	architecture.name = [[NSString alloc] initWithBytes:_capabilities->architecture
						     length:_capabilities->architecture_bytes
						   encoding:NSUTF8StringEncoding];
	return architecture;
}

- (uint64_t)registryID
{
	uint64_t identifier = 0;
	IORegistryEntryGetRegistryEntryID(_port, &identifier);
	return identifier;
}

- (uint64_t)peerGroupID
{
	return 0;
}

- (uint32_t)peerIndex
{
	return 0;
}

- (uint32_t)peerCount
{
	return 0;
}

FIELD(location, location)
FIELD(locationNumber, location_number)
FIELD(argumentBuffersSupport, argument_buffers_tier)
FIELD(readWriteTextureSupport, read_write_texture_tier)

- (uint64_t)maxTransferRate
{
	return _capabilities->max_transfer_rate;
}

- (uint64_t)recommendedMaxWorkingSetSize
{
	return _capabilities->recommended_max_working_set_size;
}

- (uint64_t)sharedMemorySize
{
	return [self hasUnifiedMemory] ? [self recommendedMaxWorkingSetSize] : 0;
}

- (uint64_t)dedicatedMemorySize
{
	return [self hasUnifiedMemory] ? 0 : [self recommendedMaxWorkingSetSize];
}

/* Metal's own default: nothing here runs at a raised GPU priority band. */
- (BOOL)supportPriorityBand
{
	return NO;
}

- (BOOL)metalAssertionsEnabled
{
	return _metalAssertionsEnabled;
}

- (void)setMetalAssertionsEnabled:(BOOL)enabled
{
	_metalAssertionsEnabled = enabled;
}

- (BOOL)supportsSampleCount:(NSUInteger)count
{
	return [self supportsTextureSampleCount:count];
}

- (NSUInteger)currentAllocatedSize
{
	StelaEmpty nothing = {};
	StelaValue size = {};
	STELA_CALL(&_gate, AllocatedSize, &nothing, &size, NULL);
	return size.value;
}

FLAG(isHeadless, HEADLESS)
FLAG(isLowPower, LOW_POWER)
FLAG(isRemovable, REMOVABLE)
FLAG(hasUnifiedMemory, UNIFIED_MEMORY)
FLAG(supports32BitFloatFiltering, FLOAT32_FILTERING)
FLAG(supports32BitMSAA, FLOAT32_MSAA)
FLAG(supportsBCTextureCompression, BC_TEXTURE_COMPRESSION)
FLAG(isDepth24Stencil8PixelFormatSupported, DEPTH24_STENCIL8)
FLAG(supportsQueryTextureLOD, QUERY_TEXTURE_LOD)
FLAG(supportsPullModelInterpolation, PULL_MODEL_INTERPOLATION)
FLAG(supportsShaderBarycentricCoordinates, SHADER_BARYCENTRIC_COORDINATES)
FLAG(areBarycentricCoordsSupported, SHADER_BARYCENTRIC_COORDINATES)
FLAG(areProgrammableSamplePositionsSupported, PROGRAMMABLE_SAMPLE_POSITIONS)
FLAG(areRasterOrderGroupsSupported, RASTER_ORDER_GROUPS)
FLAG(supportsRaytracing, RAYTRACING)
FLAG(supportsRaytracingFromRender, RAYTRACING_FROM_RENDER)
FLAG(supportsPrimitiveMotionBlur, PRIMITIVE_MOTION_BLUR)
FLAG(supportsFunctionPointers, FUNCTION_POINTERS)
FLAG(supportsFunctionPointersFromRender, FUNCTION_POINTERS_FROM_RENDER)
FLAG(supportsDynamicLibraries, DYNAMIC_LIBRARIES)
FLAG(supportsRenderDynamicLibraries, RENDER_DYNAMIC_LIBRARIES)
FLAG(supportsPlacementSparse, PLACEMENT_SPARSE)
FLAG(supportsMemorylessRenderTargets, MEMORYLESS_RENDER_TARGETS)
FLAG(isFramebufferReadSupported, FRAMEBUFFER_READS)

LIMIT(maxThreadgroupMemoryLength, max_threadgroup_memory_length)
LIMIT(maxBufferLength, max_buffer_length)
LIMIT(maxArgumentBufferSamplerCount, max_argument_buffer_sampler_count)
LIMIT(sparseTileSizeInBytes, sparse_tile_bytes)

/* Metal's base class answers these from a table only Apple's drivers fill, and
 * zero from it makes every texture too large for Core Animation. */
LIMIT(maxTextureWidth1D, max_texture_dimension_1d)
LIMIT(maxTextureWidth2D, max_texture_dimension_2d)
LIMIT(maxTextureHeight2D, max_texture_dimension_2d)
LIMIT(maxTextureWidth3D, max_texture_dimension_3d)
LIMIT(maxTextureHeight3D, max_texture_dimension_3d)
LIMIT(maxTextureDepth3D, max_texture_dimension_3d)
LIMIT(maxTextureDimensionCube, max_texture_dimension_cube)
LIMIT(maxTextureLayers, max_texture_array_length)
LIMIT(maxTextureBufferWidth, max_texture_buffer_width)
LIMIT(maxColorAttachments, max_color_attachments)
LIMIT(maxViewportCount, max_viewports)
LIMIT(maxVertexAmplificationCount, max_vertex_amplification_count)
LIMIT(maxVertexAmplificationFactor, max_vertex_amplification_count)
LIMIT(maxTessellationFactor, max_tessellation_factor)
LIMIT(maxTotalComputeThreadsPerThreadgroup, max_total_threads_per_threadgroup)
LIMIT(maxComputeThreadgroupMemory, max_threadgroup_memory_length)
LIMIT(minConstantBufferAlignmentBytes, buffer_alignment_for_uniforms)

- (float)maxPointSize
{
	return _capabilities->limits.max_point_size;
}

/* The argument table's sizes (`capabilities.rs`). */
#define TABLE(selector, count)                                                                     \
	-(NSUInteger)selector                                                                      \
	{                                                                                          \
		return count;                                                                      \
	}
TABLE(maxVertexBuffers, 31)
TABLE(maxFragmentBuffers, 31)
TABLE(maxComputeBuffers, 31)
TABLE(maxVertexTextures, 128)
TABLE(maxFragmentTextures, 128)
TABLE(maxComputeTextures, 128)
TABLE(maxVertexSamplers, 16)
TABLE(maxFragmentSamplers, 16)
TABLE(maxComputeSamplers, 16)
TABLE(maxVertexAttributes, 31)

/* MTLPixelFormatRGBA32Float: the strictest format, for callers that ask of the device as a whole. */
#define WIDEST_FORMAT 125

- (NSUInteger)linearTextureAlignmentBytes
{
	return [self minimumLinearTextureAlignmentForPixelFormat:WIDEST_FORMAT];
}

- (NSUInteger)deviceLinearTextureAlignmentBytes
{
	return [self linearTextureAlignmentBytes];
}

- (NSUInteger)deviceLinearReadOnlyTextureAlignmentBytes
{
	return [self linearTextureAlignmentBytes];
}

/* Core Animation holds a surface's rows to this as well as its base. A base is
 * always a page, so only a row's alignment limits a surface. */
- (NSUInteger)iosurfaceTextureAlignmentBytes
{
	return _capabilities->linear_row_alignment;
}

- (NSUInteger)iosurfaceReadOnlyTextureAlignmentBytes
{
	return _capabilities->linear_row_alignment;
}

/* What wiring a client's pages needs of them. */
- (NSUInteger)minBufferNoCopyAlignmentBytes
{
	return 0x4000;
}

- (MTLSize)maxThreadsPerThreadgroup
{
	const uint64_t *size = _capabilities->limits.max_threads_per_threadgroup;
	return (MTLSize){size[0], size[1], size[2]};
}

- (NSUInteger)sparseTileSizeInBytesForSparsePageSize:(NSInteger)page
{
	NSUInteger bytes = _capabilities->limits.sparse_tile_bytes;
	/* MTLSparsePageSize's raw values are not its sizes. */
	NSUInteger kib;
	switch (page) {
	case MTLSparsePageSize16:
		kib = 16;
		break;
	case MTLSparsePageSize64:
		kib = 64;
		break;
	case MTLSparsePageSize256:
		kib = 256;
		break;
	default:
		return 0;
	}
	return kib * 1024 == bytes ? bytes : 0;
}

- (MTLSize)sparseTileSizeWithTextureType:(NSUInteger)type
			     pixelFormat:(NSUInteger)format
			     sampleCount:(NSUInteger)count
{
	StelaSparseTileRequest request = {.texture_type = type, .pixel_format = format, .sample_count = count};
	StelaSparseTileResponse tile = {};
	STELA_CALL(&_gate, SparseTileSize, &request, &tile, NULL);
	return (MTLSize){tile.size[0], tile.size[1], tile.size[2]};
}

- (MTLSize)sparseTileSizeWithTextureType:(NSUInteger)type
			     pixelFormat:(NSUInteger)format
			     sampleCount:(NSUInteger)count
			  sparsePageSize:(NSInteger)page
{
	if (![self sparseTileSizeInBytesForSparsePageSize:page]) {
		return (MTLSize){0, 0, 0};
	}
	return [self sparseTileSizeWithTextureType:type pixelFormat:format sampleCount:count];
}

- (void)convertSparsePixelRegions:(const MTLRegion[])pixels
		    toTileRegions:(MTLRegion[])tiles
		     withTileSize:(MTLSize)tile
		    alignmentMode:(NSUInteger)mode
		       numRegions:(NSUInteger)count
{
	for (NSUInteger i = 0; i < count; i++) {
		MTLOrigin origin = pixels[i].origin;
		MTLSize size = pixels[i].size;
		tile_span(origin.x, size.width, tile.width, mode, &tiles[i].origin.x, &tiles[i].size.width);
		tile_span(origin.y, size.height, tile.height, mode, &tiles[i].origin.y, &tiles[i].size.height);
		tile_span(origin.z, size.depth, tile.depth, mode, &tiles[i].origin.z, &tiles[i].size.depth);
	}
}

- (void)convertSparseTileRegions:(const MTLRegion[])tiles
		   toPixelRegions:(MTLRegion[])pixels
		     withTileSize:(MTLSize)tile
		       numRegions:(NSUInteger)count
{
	for (NSUInteger i = 0; i < count; i++) {
		MTLOrigin origin = tiles[i].origin;
		MTLSize size = tiles[i].size;
		pixels[i].origin = (MTLOrigin){origin.x * tile.width, origin.y * tile.height, origin.z * tile.depth};
		pixels[i].size = (MTLSize){size.width * tile.width, size.height * tile.height, size.depth * tile.depth};
	}
}

- (BOOL)shouldMaximizeConcurrentCompilation
{
	return _shouldMaximizeConcurrentCompilation;
}

- (void)setShouldMaximizeConcurrentCompilation:(BOOL)maximize
{
	_shouldMaximizeConcurrentCompilation = maximize;
}

/* The service takes one call at a time. */
- (NSUInteger)maximumConcurrentCompilationTaskCount
{
	return 1;
}

- (StelaFormatAlignmentsResponse)alignmentsOfFormat:(NSUInteger)format
{
	StelaPixelFormatRequest request = {.pixel_format = format};
	StelaFormatAlignmentsResponse alignments = {};
	StelaFailure failure;
	if (STELA_CALL(&_gate, FormatAlignments, &request, &alignments, &failure) != StelaStatusOk) {
		stela_report([[NSString stringWithFormat:@"the alignment of pixel format %lu", (unsigned long)format]
			      UTF8String],
			     &failure);
	}
	return alignments;
}

- (NSUInteger)minimumLinearTextureAlignmentForPixelFormat:(NSUInteger)format
{
	return [self alignmentsOfFormat:format].linear_texture;
}

- (NSUInteger)minimumTextureBufferAlignmentForPixelFormat:(NSUInteger)format
{
	return [self alignmentsOfFormat:format].texture_buffer;
}

/* Metal's base class checks the descriptor against limits only Apple's drivers
 * state; a row here is its blocks, rounded up to the format's alignment. */
- (NSUInteger)requiredLinearTextureBytesPerRowForDescriptor:(id)descriptor
{
	StelaFormatAlignmentsResponse format = [self alignmentsOfFormat:[descriptor pixelFormat]];
	NSUInteger blocks = format.block_width ? ([descriptor width] + format.block_width - 1) / format.block_width : 0;
	NSUInteger alignment = format.linear_texture ? format.linear_texture : 1;
	return (blocks * format.block_bytes + alignment - 1) / alignment * alignment;
}

/* Call sites whose unclaimed family was reported, open addressed by the
 * family and the caller; more than fit report again through `stela_report`,
 * which keeps each message once. */
#define ASKED 1024
static _Atomic uintptr_t asked[ASKED];

/* Reports once per call site a family or feature set this device does not
 * claim, naming the image and offset that asked, so the path a caller takes
 * without it stays visible. */
static void report_unclaimed(const char *kind, NSInteger value, void *caller)
{
	uintptr_t key = (uintptr_t)caller ^ ((uintptr_t)value << 48);
	size_t at = key >> 2;
	for (size_t probe = 0; probe < ASKED; probe++, at++) {
		uintptr_t expected = 0;
		if (atomic_compare_exchange_strong(&asked[at % ASKED], &expected, key)) {
			break;
		}
		if (expected == key) {
			return;
		}
	}
	char named[24];
	if (value > 1000 && value < 1100) {
		snprintf(named, sizeof(named), "Apple%ld", (long)(value - 1000));
	} else if (value > 2000 && value < 2100) {
		snprintf(named, sizeof(named), "Mac%ld", (long)(value - 2000));
	} else if (value > 3000 && value < 3100) {
		snprintf(named, sizeof(named), "Common%ld", (long)(value - 3000));
	} else if (value > 4000 && value < 4100) {
		snprintf(named, sizeof(named), "MacCatalyst%ld", (long)(value - 4000));
	} else if (value > 5000 && value < 5100) {
		snprintf(named, sizeof(named), "Metal%ld", (long)(value - 4998));
	} else {
		snprintf(named, sizeof(named), "%ld", (long)value);
	}
	Dl_info image = {};
	const char *name = "an unknown image";
	uintptr_t offset = (uintptr_t)caller;
	if (dladdr(caller, &image) && image.dli_fname) {
		name = image.dli_fname;
		offset -= (uintptr_t)image.dli_fbase;
	}
	char what[320];
	snprintf(what, sizeof(what), "%s %s, asked by %s+0x%lx", kind, named, name, (unsigned long)offset);
	StelaFailure failure = {.reason = "this device does not claim it"};
	stela_report(what, &failure);
}

static BOOL claims(const StelaBlock *capabilities, NSInteger family)
{
	for (uint32_t i = 0; i < capabilities->family_count; i++) {
		if (capabilities->families[i] == (uint64_t)family) {
			return YES;
		}
	}
	return NO;
}

- (BOOL)supportsFamily:(NSInteger)family
{
	BOOL claimed = claims(_capabilities, family);
	if (!claimed) {
		report_unclaimed("GPU family", family, __builtin_return_address(0));
	}
	return claimed;
}

/* Private queries Apple's drivers answer from the GPU family and Metal's base
 * class answers NO. Core Animation sizes its work by them: a glass backdrop's
 * capture margin, for one. Each family claim already checks its feature:
 * imageblocks, tile functions and raster order groups for Apple4, SIMD-group
 * functions for Apple6 and their reductions for Apple7. */
- (BOOL)supportsTileShaders
{
	return claims(_capabilities, MTLGPUFamilyApple4);
}

- (BOOL)supportsRasterOrderGroups
{
	return (_capabilities->flags & StelaFlags_RASTER_ORDER_GROUPS) != 0;
}

- (BOOL)supportsSIMDGroup
{
	return claims(_capabilities, MTLGPUFamilyApple6);
}

- (BOOL)supportsSIMDReduction
{
	return claims(_capabilities, MTLGPUFamilyApple7);
}

/* Half-precision arithmetic runs at its own width: every Apple family and the
 * NVIDIA GPUs the driver presents as one. */
- (BOOL)supportsNativeHardwareFP16
{
	return claims(_capabilities, MTLGPUFamilyApple4);
}

/* The deprecated macOS sets; iOS and tvOS sets are unknown here. */
- (BOOL)supportsFeatureSet:(NSUInteger)set
{
	BOOL claimed = NO;
	if (set == MTLFeatureSet_macOS_ReadWriteTextureTier2) {
		claimed = _capabilities->read_write_texture_tier >= 2;
	} else if (set >= MTLFeatureSet_macOS_GPUFamily1_v1 && set <= MTLFeatureSet_macOS_GPUFamily1_v4) {
		claimed = claims(_capabilities, MTLGPUFamilyMac1);
	} else if (set == MTLFeatureSet_macOS_GPUFamily2_v1) {
		claimed = claims(_capabilities, MTLGPUFamilyMac2);
	}
	if (!claimed) {
		report_unclaimed("feature set", (NSInteger)set, __builtin_return_address(0));
	}
	return claimed;
}

- (BOOL)supportsTextureSampleCount:(NSUInteger)count
{
	/* VkSampleCountFlags: each bit is its count. */
	return count != 0 && (count & (count - 1)) == 0 && (_capabilities->texture_sample_counts & count) != 0;
}

- (BOOL)supportsVertexAmplificationCount:(NSUInteger)count
{
	return count != 0 && count <= _capabilities->limits.max_vertex_amplification_count;
}

- (BOOL)supportsCounterSampling:(NSUInteger)point
{
	return point < 32 && (_capabilities->counter_sampling >> point & 1) != 0;
}

- (void)getDefaultSamplePositions:(MTLSamplePosition *)positions count:(NSUInteger)count
{
	const uint8_t(*pattern)[2];
	switch (count) {
	case 1:
		positions[0] = (MTLSamplePosition){0.5f, 0.5f};
		return;
	case 2:
		pattern = PATTERN_2;
		break;
	case 4:
		pattern = PATTERN_4;
		break;
	case 8:
		pattern = PATTERN_8;
		break;
	case 16:
		pattern = PATTERN_16;
		break;
	default:
		return;
	}
	for (NSUInteger i = 0; i < count; i++) {
		positions[i] = (MTLSamplePosition){pattern[i][0] / 16.0f, pattern[i][1] / 16.0f};
	}
}

- (void)sampleTimestamps:(MTLTimestamp *)cpu gpuTimestamp:(MTLTimestamp *)gpu
{
	StelaEmpty nothing = {};
	StelaTimestampsResponse times = {};
	StelaFailure failure;
	/* The guest's clock, not the Vulkan host's, is the CPU's. */
	*cpu = mach_absolute_time();
	if (STELA_CALL(&_gate, SampleTimestamps, &nothing, &times, &failure) != StelaStatusOk) {
		stela_report("a timestamp", &failure);
	}
	*gpu = times.gpu;
}

/* The rate of the ticks counter heaps' timestamps hold; zero without them. */
- (uint64_t)queryTimestampFrequency
{
	return _capabilities->timestamp_frequency;
}

- (NSArray *)counterSets
{
	StelaGPUCounterSet *sets[2];
	NSUInteger count = 0;
	sets[count++] = [StelaGPUCounterSet setWithCode:StelaCounterSetCodeTimestamp];
	if (_capabilities->flags & StelaFlags_STATISTIC_COUNTERS) {
		sets[count++] = [StelaGPUCounterSet setWithCode:StelaCounterSetCodeStatistic];
	}
	return [NSArray arrayWithObjects:sets count:count];
}

/* Queues. */

- (id)newCommandQueue
{
	StelaGPUQueue *queue = [[StelaGPUQueue alloc] initWithDevice:self];
	queue->_logState = [self environmentLogState];
	return queue;
}

- (id)newCommandQueueWithMaxCommandBufferCount:(NSUInteger)count
{
	(void)count;
	return [self newCommandQueue];
}

- (id)newCommandQueueWithDescriptor:(id)descriptor
{
	StelaGPUQueue *queue = [self newCommandQueue];
	id logState = [descriptor logState];
	if (logState) {
		queue->_logState = logState;
	}
	return queue;
}

/* What has no service object. */

- (id)newFence
{
	StelaGPUFence *fence = [StelaGPUFence new];
	fence->_device = self;
	return fence;
}

- (id)newDepthStencilStateWithDescriptor:(id)descriptor
{
	StelaGPUDepthStencil *state = [StelaGPUDepthStencil new];
	state->_device = self;
	state.label = [descriptor label];
	stela_depth_stencil_record(descriptor, &state->_record);
	return state;
}

@end

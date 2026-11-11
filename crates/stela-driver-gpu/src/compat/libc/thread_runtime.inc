/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Allocation, clock and TLS shared with the firmware pthread probe. */

/* Each block keeps its size and alignment just before itself. */
struct block {
	size_t size, alignment;
};

void *aligned_alloc(size_t alignment, size_t size)
{
	if (alignment < _Alignof(max_align_t))
		alignment = _Alignof(max_align_t);
	size_t header = (sizeof(struct block) + alignment - 1) & ~(alignment - 1);
	if (size > SIZE_MAX - header) {
		errno = ENOMEM;
		return NULL;
	}
	char *base = stela_libc_allocate(header + size, alignment);
	if (!base) {
		errno = ENOMEM;
		return NULL;
	}
	struct block *block = (struct block *)(base + header) - 1;
	block->size = header + size;
	block->alignment = alignment;
	return base + header;
}

void free(void *pointer)
{
	if (!pointer)
		return;
	struct block *block = (struct block *)pointer - 1;
	size_t header = (sizeof(*block) + block->alignment - 1) & ~(block->alignment - 1);
	stela_libc_release((char *)pointer - header, block->size, block->alignment);
}

int clock_gettime(clockid_t clock, struct timespec *time)
{
	if (!time) {
		errno = EINVAL;
		return -1;
	}
	if (clock != CLOCK_MONOTONIC) {
		/* Deterministic failed output also protects optional Mesa profiling
		 * callers that neglect the error; it is never a successful clock. */
		*time = (struct timespec){0};
		errno = clock == CLOCK_REALTIME || clock == CLOCK_PROCESS_CPUTIME_ID || clock == CLOCK_THREAD_CPUTIME_ID ? ENOTSUP : EINVAL;
		return -1;
	}
	uint64_t now = stela_libc_now_ns();
	time->tv_sec = (time_t)(now / 1000000000);
	time->tv_nsec = (long)(now % 1000000000);
	return 0;
}

/* emutls's per-variable control, as compilers lay it out. */
struct emutls_control {
	size_t size, alignment;
	size_t index;
	const void *initial;
};

#define TLS_VARIABLES 256
static void *tls_values[STELA_LIBC_CORES][TLS_VARIABLES];
static size_t tls_count;
static volatile int tls_lock;

void stela_libc_thread_cleanup(void)
{
	unsigned int core = stela_libc_core();
	for (size_t index = 0; index < TLS_VARIABLES; index++) {
		void *value = tls_values[core][index];
		tls_values[core][index] = NULL;
		free(value);
	}
}

/* A value per thread, and every caller of Mesa on a host processor is one. */
void *__emutls_get_address(struct emutls_control *control)
{
	size_t index = __atomic_load_n(&control->index, __ATOMIC_ACQUIRE);
	if (!index) {
		while (__atomic_exchange_n(&tls_lock, 1, __ATOMIC_ACQUIRE))
			__builtin_ia32_pause();
		index = control->index;
		if (!index) {
			index = ++tls_count;
			if (index >= TLS_VARIABLES)
				stela_libc_fatal("too many thread-local variables");
			__atomic_store_n(&control->index, index, __ATOMIC_RELEASE);
		}
		__atomic_store_n(&tls_lock, 0, __ATOMIC_RELEASE);
	}
	void **slot = &tls_values[stela_libc_core()][index];
	if (!*slot) {
		*slot = aligned_alloc(control->alignment, control->size);
		if (!*slot)
			stela_libc_fatal("out of memory for a thread-local variable");
		if (control->initial)
			memcpy(*slot, control->initial, control->size);
		else
			memset(*slot, 0, control->size);
	}
	return *slot;
}

int *__errno_location(void)
{
	/* Use the same per-thread storage even on C compilers that reject
	 * _Thread_local for the firmware COFF target. */
	static struct emutls_control error = { .size = sizeof(int), .alignment = _Alignof(int) };
	return __emutls_get_address(&error);
}

// SPDX-License-Identifier: GPL-2.0-or-later
/* The C library Mesa runs on inside `stela.efi`: memory, numbers, sorting, time
 * and the few descriptors NVK opens, which are all DRM render nodes. There is
 * no process, so a request for one's environment, signals or files answers as
 * a process without any would. */
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <poll.h>
#include <regex.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <syslog.h>
#include <sys/epoll.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/auxv.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <stela/host.h>
#include "thread_runtime.inc"

static int fail(int error)
{
	errno = error;
	return -1;
}

_Noreturn void __assert_fail(const char *condition, const char *file, unsigned int line,
			     const char *function)
{
	char number[16];
	int length = snprintf(number, sizeof(number), ":%u: ", line);
	stela_libc_log(file, strlen(file));
	stela_libc_log(number, (size_t)length);
	stela_libc_log(function, strlen(function));
	stela_libc_log(": ", 2);
	stela_libc_log(condition, strlen(condition));
	stela_libc_log("\n", 1);
	stela_libc_fatal("a Mesa assertion failed");
}

void *malloc(size_t size)
{
	return aligned_alloc(0, size);
}

void *calloc(size_t count, size_t size)
{
	size_t bytes;
	if (__builtin_mul_overflow(count, size, &bytes)) {
		errno = ENOMEM;
		return NULL;
	}
	void *pointer = malloc(bytes);
	if (pointer)
		memset(pointer, 0, bytes);
	return pointer;
}

void *realloc(void *pointer, size_t size)
{
	if (!pointer)
		return malloc(size);
	struct block *block = (struct block *)pointer - 1;
	size_t header = (sizeof(*block) + block->alignment - 1) & ~(block->alignment - 1);
	size_t old = block->size - header;
	if (size <= old)
		return pointer;
	void *grown = aligned_alloc(block->alignment, size);
	if (grown) {
		memcpy(grown, pointer, old);
		free(pointer);
	}
	return grown;
}

int posix_memalign(void **pointer, size_t alignment, size_t size)
{
	*pointer = aligned_alloc(alignment, size);
	return *pointer ? 0 : ENOMEM;
}

_Noreturn void abort(void)
{
	stela_libc_fatal("Mesa aborted");
}

_Noreturn void exit(int status)
{
	(void)status;
	stela_libc_fatal("Mesa exited");
}

_Noreturn void _Exit(int status)
{
	exit(status);
}

/* Nothing here ends before the host does. */
int atexit(void (*function)(void))
{
	(void)function;
	return 0;
}

char *getenv(const char *name)
{
	(void)name;
	return NULL;
}

char *secure_getenv(const char *name)
{
	return getenv(name);
}

int setenv(const char *name, const char *value, int overwrite)
{
	(void)name;
	(void)value;
	(void)overwrite;
	errno = ENOMEM;
	return -1;
}

int unsetenv(const char *name)
{
	(void)name;
	return 0;
}

/* There are no files to resolve a path through. */
char *realpath(const char *path, char *resolved)
{
	(void)path;
	(void)resolved;
	errno = ENOENT;
	return NULL;
}

ssize_t readlink(const char *path, char *buffer, size_t size)
{
	(void)path;
	(void)buffer;
	(void)size;
	return fail(ENOENT);
}

/* One process with one kernel: `os_same_file_description` asks for an answer
 * only when two descriptors may differ, and "unknown" is a valid one. */
int epoll_create1(int flags)
{
	(void)flags;
	return fail(ENOSYS);
}

int epoll_ctl(int instance, int operation, int fd, struct epoll_event *event)
{
	(void)instance;
	(void)operation;
	(void)fd;
	(void)event;
	return fail(ENOSYS);
}

/* Mesa's log goes to the host's, whichever logger it picks. */
void openlog(const char *name, int option, int facility)
{
	(void)name;
	(void)option;
	(void)facility;
}

void vsyslog(int priority, const char *format, va_list arguments)
{
	(void)priority;
	vfprintf(stderr, format, arguments);
}

void syslog(int priority, const char *format, ...)
{
	va_list arguments;
	va_start(arguments, format);
	vsyslog(priority, format, arguments);
	va_end(arguments);
}

static unsigned long long parse_unsigned(const char *text, char **end, int base, int *negative,
					 int *overflow)
{
	const char *at = text;
	while (isspace((unsigned char)*at))
		at++;
	*negative = *at == '-';
	if (*at == '-' || *at == '+')
		at++;
	if ((base == 0 || base == 16) && at[0] == '0' && (at[1] | 0x20) == 'x' &&
	    isxdigit((unsigned char)at[2])) {
		at += 2;
		base = 16;
	} else if (base == 0) {
		base = *at == '0' ? 8 : 10;
	}
	unsigned long long value = 0;
	const char *first = at;
	for (;; at++) {
		int digit;
		if (isdigit((unsigned char)*at))
			digit = *at - '0';
		else if (isalpha((unsigned char)*at))
			digit = (*at | 0x20) - 'a' + 10;
		else
			break;
		if (digit >= base)
			break;
		*overflow |= __builtin_mul_overflow(value, (unsigned long long)base, &value);
		*overflow |= __builtin_add_overflow(value, (unsigned long long)digit, &value);
	}
	if (end)
		*end = (char *)(at == first ? text : at);
	return value;
}

unsigned long long strtoull(const char *text, char **end, int base)
{
	int negative, overflow = 0;
	unsigned long long value = parse_unsigned(text, end, base, &negative, &overflow);
	if (overflow) {
		errno = ERANGE;
		return ULLONG_MAX;
	}
	return negative ? -value : value;
}

long long strtoll(const char *text, char **end, int base)
{
	int negative, overflow = 0;
	unsigned long long value = parse_unsigned(text, end, base, &negative, &overflow);
	unsigned long long limit = (unsigned long long)LLONG_MAX + (negative ? 1 : 0);
	if (overflow || value > limit) {
		errno = ERANGE;
		return negative ? LLONG_MIN : LLONG_MAX;
	}
	return negative ? (long long)-value : (long long)value;
}

unsigned long strtoul(const char *text, char **end, int base)
{
	return strtoull(text, end, base);
}

long strtol(const char *text, char **end, int base)
{
	return strtoll(text, end, base);
}

int atoi(const char *text)
{
	return (int)strtol(text, NULL, 10);
}

long atol(const char *text)
{
	return strtol(text, NULL, 10);
}

double strtod(const char *text, char **end)
{
	const char *at = text;
	while (isspace((unsigned char)*at))
		at++;
	long double sign = 1;
	if (*at == '-' || *at == '+')
		sign = *at++ == '-' ? -1 : 1;
	if (!strncasecmp(at, "inf", 3) || !strncasecmp(at, "nan", 3)) {
		if (end)
			*end = (char *)at + 3;
		return (double)(sign * ((at[0] | 0x20) == 'i' ? INFINITY : NAN));
	}
	long double value = 0;
	int digits = 0, exponent = 0;
	for (; isdigit((unsigned char)*at); at++, digits++)
		value = value * 10 + (*at - '0');
	if (*at == '.') {
		for (at++; isdigit((unsigned char)*at); at++, digits++, exponent--)
			value = value * 10 + (*at - '0');
	}
	if (!digits) {
		if (end)
			*end = (char *)text;
		return 0;
	}
	if ((*at | 0x20) == 'e') {
		char *after;
		long parsed = strtol(at + 1, &after, 10);
		if (after != at + 1) {
			exponent += (int)parsed;
			at = after;
		}
	}
	if (end)
		*end = (char *)at;
	long double scale = 1, power = 10;
	for (int left = exponent < 0 ? -exponent : exponent; left; left >>= 1, power *= power) {
		if (left & 1)
			scale *= power;
	}
	return (double)(sign * (exponent < 0 ? value / scale : value * scale));
}

/* To a double's precision: nothing in the image parses long doubles. */
long double strtold(const char *text, char **end)
{
	return strtod(text, end);
}

float strtof(const char *text, char **end)
{
	return (float)strtod(text, end);
}

double atof(const char *text)
{
	return strtod(text, NULL);
}

/* Heapsort: no allocation or recursion, in a library any stack may call. */
static void swap_bytes(char *a, char *b, size_t size)
{
	while (size--) {
		char byte = *a;
		*a++ = *b;
		*b++ = byte;
	}
}

void qsort_r(void *base, size_t count, size_t size,
	     int (*compare)(const void *, const void *, void *), void *context)
{
	char *items = base;
	for (size_t start = count / 2, end = count; end > 1;) {
		if (start > 0)
			start--;
		else
			swap_bytes(items, items + --end * size, size);
		for (size_t root = start, child; (child = root * 2 + 1) < end; root = child) {
			if (child + 1 < end &&
			    compare(items + child * size, items + (child + 1) * size, context) < 0)
				child++;
			if (compare(items + root * size, items + child * size, context) >= 0)
				break;
			swap_bytes(items + root * size, items + child * size, size);
		}
	}
}

static int plain_compare(const void *a, const void *b, void *context)
{
	return ((int (*)(const void *, const void *))context)(a, b);
}

void qsort(void *base, size_t count, size_t size, int (*compare)(const void *, const void *))
{
	qsort_r(base, count, size, plain_compare, (void *)compare);
}

void *bsearch(const void *key, const void *base, size_t count, size_t size,
	      int (*compare)(const void *, const void *))
{
	const char *items = base;
	while (count) {
		const char *middle = items + (count / 2) * size;
		int order = compare(key, middle);
		if (!order)
			return (void *)middle;
		if (order > 0) {
			items = middle + size;
			count -= count / 2 + 1;
		} else {
			count /= 2;
		}
	}
	return NULL;
}

int abs(int value)
{
	return value < 0 ? -value : value;
}

long labs(long value)
{
	return value < 0 ? -value : value;
}

div_t div(int number, int divisor)
{
	return (div_t){ number / divisor, number % divisor };
}

ldiv_t ldiv(long number, long divisor)
{
	return (ldiv_t){ number / divisor, number % divisor };
}

lldiv_t lldiv(long long number, long long divisor)
{
	return (lldiv_t){ number / divisor, number % divisor };
}

long long llabs(long long value)
{
	return value < 0 ? -value : value;
}

static unsigned int random_state = 1;

void srand(unsigned int seed)
{
	random_state = seed ? seed : 1;
}

int rand(void)
{
	unsigned int state = random_state;
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	random_state = state;
	return (int)(state & RAND_MAX);
}

void *memmem(const void *haystack, size_t haystack_size, const void *needle, size_t needle_size)
{
	const char *at = haystack;
	for (; haystack_size >= needle_size; at++, haystack_size--) {
		if (!memcmp(at, needle, needle_size))
			return (void *)at;
	}
	return NULL;
}

char *strncpy(char *to, const char *from, size_t size)
{
	size_t at = 0;
	for (; at < size && from[at]; at++)
		to[at] = from[at];
	memset(to + at, 0, size - at);
	return to;
}

char *strndup(const char *text, size_t limit)
{
	size_t size = strnlen(text, limit);
	char *copy = malloc(size + 1);
	if (copy) {
		memcpy(copy, text, size);
		copy[size] = 0;
	}
	return copy;
}

char *strdup(const char *text)
{
	return strndup(text, SIZE_MAX);
}

char *strtok_r(char *text, const char *delimiters, char **state)
{
	if (!text)
		text = *state;
	text += strspn(text, delimiters);
	if (!*text) {
		*state = text;
		return NULL;
	}
	char *end = text + strcspn(text, delimiters);
	*state = *end ? end + 1 : end;
	*end = 0;
	return text;
}

char *strtok(char *text, const char *delimiters)
{
	static _Thread_local char *state;
	return strtok_r(text, delimiters, &state);
}

char *strerror(int error)
{
	(void)error;
	return (char *)"host error";
}

int strerror_r(int error, char *buffer, size_t size)
{
	const char *text = strerror(error);
	size_t length = strlen(text);
	if (length >= size)
		return ERANGE;
	memcpy(buffer, text, length + 1);
	return 0;
}

int timespec_get(struct timespec *time, int base)
{
	if (base != TIME_UTC) {
		errno = EINVAL;
		return 0;
	}
	return clock_gettime(CLOCK_REALTIME, time) ? 0 : base;
}

time_t time(time_t *result)
{
	if (result)
		*result = (time_t)-1;
	errno = ENOTSUP;
	return (time_t)-1;
}

int gettimeofday(struct timeval *time, void *zone)
{
	(void)zone;
	if (time)
		*time = (struct timeval){0};
	errno = ENOTSUP;
	return -1;
}

/* Calendar conversion of a supplied Unix timestamp; no host wall-clock claim. */
struct tm *localtime_r(const time_t *time, struct tm *result)
{
	long days = (long)(*time / 86400), seconds = (long)(*time % 86400);
	/* Hinnant's civil-from-days, on years starting in March. */
	long shifted = days + 719468;
	long era = shifted / 146097;
	long day_of_era = shifted - era * 146097;
	long year_of_era =
		(day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
	long day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
	long month = (5 * day_of_year + 2) / 153;
	result->tm_mday = (int)(day_of_year - (153 * month + 2) / 5 + 1);
	result->tm_mon = (int)(month < 10 ? month + 2 : month - 10);
	result->tm_year = (int)(year_of_era + era * 400 + (month >= 10) - 1900);
	result->tm_hour = (int)(seconds / 3600);
	result->tm_min = (int)(seconds / 60 % 60);
	result->tm_sec = (int)(seconds % 60);
	result->tm_wday = (int)((days + 4) % 7);
	result->tm_yday = 0;
	result->tm_isdst = 0;
	return result;
}

struct tm *localtime(const time_t *time)
{
	static _Thread_local struct tm result;
	return localtime_r(time, &result);
}

static int nanoseconds(const struct timespec *time, uint64_t *result)
{
	if (!time || time->tv_sec < 0 || time->tv_nsec < 0 || time->tv_nsec >= 1000000000)
		return EINVAL;
	uint64_t seconds = (uint64_t)time->tv_sec;
	uint64_t nanos = (uint64_t)time->tv_nsec;
	if (seconds > (UINT64_MAX - 1 - nanos) / 1000000000)
		return EOVERFLOW;
	*result = seconds * 1000000000 + nanos;
	return 0;
}

/* What a sleep waits on. */
static const unsigned int never_changes;

int clock_nanosleep(clockid_t clock, int flags, const struct timespec *time,
		    struct timespec *remaining)
{
	if (flags & ~TIMER_ABSTIME)
		return EINVAL;
	if (clock != CLOCK_MONOTONIC)
		return clock == CLOCK_REALTIME || clock == CLOCK_PROCESS_CPUTIME_ID || clock == CLOCK_THREAD_CPUTIME_ID ? ENOTSUP : EINVAL;
	uint64_t until;
	int error = nanoseconds(time, &until);
	if (error)
		return error;
	if (!(flags & TIMER_ABSTIME)) {
		uint64_t now = stela_libc_now_ns();
		if (until >= UINT64_MAX - now)
			return EOVERFLOW;
		until += now;
	}
	while (stela_libc_now_ns() < until)
		stela_libc_wait(&never_changes, 0, until);
	if (remaining && !(flags & TIMER_ABSTIME))
		*remaining = (struct timespec){0};
	return 0;
}

int nanosleep(const struct timespec *time, struct timespec *remaining)
{
	int error = clock_nanosleep(CLOCK_MONOTONIC, 0, time, remaining);
	if (error)
		errno = error;
	return error ? -1 : 0;
}

int usleep(unsigned int micros)
{
	struct timespec time = {micros / 1000000, (long)(micros % 1000000) * 1000};
	return nanosleep(&time, NULL);
}

unsigned int sleep(unsigned int seconds)
{
	struct timespec time = {seconds, 0};
	nanosleep(&time, NULL);
	return 0;
}

int sched_getcpu(void)
{
	return (int)stela_libc_core();
}

int sched_yield(void)
{
	stela_libc_wait(&never_changes, 0, stela_libc_now_ns());
	return 0;
}

/* A thread-free process has nothing to rank against itself. */
int setpriority(int which, int who, int priority)
{
	(void)which;
	(void)who;
	(void)priority;
	return 0;
}

long syscall(long number, ...)
{
	if (number == SYS_gettid)
		return gettid();
	return fail(ENOSYS);
}

unsigned long getauxval(unsigned long type)
{
	(void)type;
	return 0;
}

long sysconf(int name)
{
	switch (name) {
	case _SC_PAGESIZE:
		return 4096;
	case _SC_NPROCESSORS_ONLN:
		return 1;
	case _SC_PHYS_PAGES:
		return (long)(stela_libc_memory_bytes() / 4096);
	default:
		errno = EINVAL;
		return -1;
	}
}

int getpagesize(void)
{
	return 4096;
}

pid_t getpid(void)
{
	return 1;
}

pid_t gettid(void)
{
	return (pid_t)stela_libc_core() + 1;
}

uid_t getuid(void)
{
	return 0;
}

uid_t geteuid(void)
{
	return 0;
}

gid_t getgid(void)
{
	return 0;
}

gid_t getegid(void)
{
	return 0;
}

int sigfillset(sigset_t *set)
{
	set->bits = ~0ul;
	return 0;
}

int sigdelset(sigset_t *set, int signal)
{
	set->bits &= ~(1ul << signal);
	return 0;
}

/* Signals are never delivered, so every mask is equivalent. */
int pthread_sigmask(int how, const sigset_t *set, sigset_t *saved)
{
	(void)how;
	(void)set;
	if (saved)
		saved->bits = 0;
	return 0;
}

int raise(int signal)
{
	(void)signal;
	stela_libc_fatal("Mesa raised a signal");
}

int getrlimit(int resource, struct rlimit *limit)
{
	(void)resource;
	limit->rlim_cur = limit->rlim_max = 1024;
	return 0;
}

/* Every descriptor is the Linux environment's, so one a DRM ioctl returns
 * works as any other. */
static int checked(int result)
{
	return result < 0 ? fail(-result) : result;
}

/* Linux's /proc/meminfo, the one procfs file Mesa reads, as a snapshot per
 * descriptor from TEXT_BASE, below the anonymous files. */
#define TEXT_BASE (1 << 23)
#define TEXT_FILES 16
static struct text_file {
	int used;
	size_t size, offset;
	char data[128];
} text_files[TEXT_FILES];

static struct text_file *text_file(int fd)
{
	unsigned int index = (unsigned int)(fd - TEXT_BASE);
	if (fd < TEXT_BASE || index >= TEXT_FILES || !__atomic_load_n(&text_files[index].used, __ATOMIC_ACQUIRE))
		return NULL;
	return &text_files[index];
}

static int open_meminfo(void)
{
	for (unsigned int index = 0; index < TEXT_FILES; index++) {
		struct text_file *file = &text_files[index];
		if (__atomic_exchange_n(&file->used, 1, __ATOMIC_ACQUIRE))
			continue;
		unsigned long long total = stela_libc_memory_bytes() >> 10;
		unsigned long long available = stela_libc_available_memory_bytes() >> 10;
		int size = snprintf(file->data, sizeof(file->data),
				    "MemTotal: %llu kB\nMemFree: %llu kB\nMemAvailable: %llu kB\n", total,
				    available, available);
		file->size = (size_t)size < sizeof(file->data) ? (size_t)size : sizeof(file->data) - 1;
		file->offset = 0;
		return TEXT_BASE + (int)index;
	}
	return fail(EMFILE);
}

int open(const char *path, int flags, ...)
{
	(void)flags;
	if (!strcmp(path, "/proc/meminfo"))
		return open_meminfo();
	static const char prefix[] = "/dev/dri/renderD";
	if (strncmp(path, prefix, sizeof(prefix) - 1))
		return fail(ENOENT);
	return checked(stela_libc_drm_open((unsigned int)strtoul(path + sizeof(prefix) - 1, NULL, 10)));
}

int close(int fd)
{
	struct text_file *text = text_file(fd);
	if (text) {
		__atomic_store_n(&text->used, 0, __ATOMIC_RELEASE);
		return 0;
	}
	if (fd >= STELA_LIBC_FILE_BASE)
		return checked(stela_libc_file_close(fd));
	return checked(stela_libc_drm_close(fd));
}

int dup(int fd)
{
	return checked(stela_libc_drm_dup(fd));
}

int dup2(int from, int to)
{
	(void)from;
	(void)to;
	return fail(EBADF);
}

int fcntl(int fd, int command, ...)
{
	if (command == F_DUPFD || command == F_DUPFD_CLOEXEC)
		return dup(fd);
	return 0;
}

int ioctl(int fd, unsigned long request, ...)
{
	va_list arguments;
	va_start(arguments, request);
	void *argument = va_arg(arguments, void *);
	va_end(arguments);
	return checked(stela_libc_drm_ioctl(fd, request, argument));
}

static void *mapped_or_failed(void *mapped)
{
	if ((uintptr_t)mapped >= (uintptr_t)-4095) {
		errno = -(int)(intptr_t)mapped;
		return MAP_FAILED;
	}
	return mapped;
}

int memfd_create(const char *name, unsigned int flags)
{
	(void)name;
	(void)flags;
	return checked(stela_libc_file_create());
}

void *mmap(void *address, size_t size, int protection, int flags, int fd, off_t offset)
{
	if (protection & PROT_EXEC) {
		errno = ENOTSUP;
		return MAP_FAILED;
	}
	int anonymous = flags & MAP_ANONYMOUS;
	if (flags & MAP_FIXED || (anonymous && flags & MAP_SHARED) || fd >= STELA_LIBC_FILE_BASE)
		return mapped_or_failed(stela_libc_file_mmap(anonymous ? -1 : fd, (uint64_t)offset, size,
							     flags & MAP_FIXED ? address : NULL));
	if (anonymous) {
		void *pointer = aligned_alloc(4096, size);
		if (pointer)
			memset(pointer, 0, size);
		return pointer ? pointer : MAP_FAILED;
	}
	return mapped_or_failed(stela_libc_drm_mmap(fd, (uint64_t)offset, size));
}

int munmap(void *address, size_t size)
{
	int file = stela_libc_file_munmap(address, size);
	if (file)
		return file < 0 ? checked(file) : 0;
	int mapped = stela_libc_drm_munmap(address, size);
	if (mapped < 0)
		return checked(mapped);
	if (!mapped)
		free(address);
	return 0;
}

int mprotect(void *address, size_t size, int protection)
{
	(void)address;
	(void)size;
	(void)protection;
	/* These mappings have fixed permissions. In particular, a JIT may only
	 * enter generated code after its writable pages have become executable. */
	return fail(ENOTSUP);
}

int madvise(void *address, size_t size, int advice)
{
	(void)address;
	(void)size;
	(void)advice;
	return 0;
}

int fstat(int fd, struct stat *result)
{
	struct text_file *text = text_file(fd);
	if (text) {
		memset(result, 0, sizeof(*result));
		result->st_mode = S_IFREG | 0444;
		result->st_size = (off_t)text->size;
		return 0;
	}
	long device = stela_libc_drm_rdev(fd);
	if (device < 0)
		return fail((int)-device);
	memset(result, 0, sizeof(*result));
	result->st_mode = S_IFCHR;
	/* Linux keeps the major above MINORBITS; a process sees it encoded. */
	result->st_rdev = makedev((unsigned int)(device >> 20), (unsigned int)(device & 0xfffff));
	return 0;
}

/* The only named files are DRM's nodes and /proc/meminfo, which open answers. */
int stat(const char *path, struct stat *result)
{
	int fd = open(path, O_RDWR);
	if (fd < 0)
		return -1;
	int status = fstat(fd, result);
	close(fd);
	return status;
}

int lstat(const char *path, struct stat *result)
{
	return stat(path, result);
}

int mkdir(const char *path, mode_t mode)
{
	(void)path;
	(void)mode;
	return fail(EACCES);
}

int access(const char *path, int mode)
{
	(void)path;
	(void)mode;
	return fail(ENOENT);
}

int unlink(const char *path)
{
	(void)path;
	return fail(ENOENT);
}

int isatty(int fd)
{
	(void)fd;
	return 0;
}

int mkstemps(char *path, int suffix)
{
	(void)path;
	(void)suffix;
	return fail(EACCES);
}

int ftruncate(int fd, off_t size)
{
	if (fd >= STELA_LIBC_FILE_BASE)
		return checked(stela_libc_file_resize(fd, (uint64_t)size));
	return fail(EINVAL);
}

int mkstemp(char *path)
{
	return mkstemps(path, 0);
}

ssize_t read(int fd, void *buffer, size_t size)
{
	struct text_file *text = text_file(fd);
	if (!text)
		return fail(EBADF);
	size_t count = text->size - text->offset < size ? text->size - text->offset : size;
	memcpy(buffer, text->data + text->offset, count);
	text->offset += count;
	return (ssize_t)count;
}

ssize_t write(int fd, const void *buffer, size_t size)
{
	if (fd != STDOUT_FILENO && fd != STDERR_FILENO)
		return fail(EBADF);
	stela_libc_log(buffer, size);
	return (ssize_t)size;
}

off_t lseek(int fd, off_t offset, int whence)
{
	(void)fd;
	(void)offset;
	(void)whence;
	return fail(EBADF);
}

int flock(int fd, int operation)
{
	(void)fd;
	(void)operation;
	return 0;
}

/* Nothing but render nodes polls, and their events are never waited on. */
/* Mesa polls only sync files and window systems, which nothing here has: an
 * answer of "nothing ready" would have its caller wait for good. */
int poll(struct pollfd *fds, nfds_t count, int timeout)
{
	(void)fds;
	(void)count;
	(void)timeout;
	errno = ENOSYS;
	return -1;
}

int ppoll(struct pollfd *fds, nfds_t count, const struct timespec *timeout, const sigset_t *mask)
{
	(void)fds;
	(void)count;
	(void)timeout;
	(void)mask;
	errno = ENOSYS;
	return -1;
}

int sysinfo(struct sysinfo *info)
{
	memset(info, 0, sizeof(*info));
	info->totalram = stela_libc_memory_bytes();
	info->mem_unit = 1;
	return 0;
}

void sync(void)
{
}

int regcomp(regex_t *regex, const char *pattern, int flags)
{
	return llvm_regcomp(regex, pattern, flags);
}

int regexec(const regex_t *regex, const char *text, size_t count, regmatch_t *matches, int flags)
{
	return llvm_regexec(regex, text, count, matches, flags);
}

void regfree(regex_t *regex)
{
	llvm_regfree(regex);
}

size_t regerror(int code, const regex_t *regex, char *buffer, size_t size)
{
	return llvm_regerror(code, regex, buffer, size);
}

void *dlopen(const char *file, int flags)
{
	(void)file;
	(void)flags;
	return NULL;
}

void *dlsym(void *handle, const char *name)
{
	(void)handle;
	(void)name;
	return NULL;
}

int dlclose(void *handle)
{
	(void)handle;
	return 0;
}

const char *dlerror(void)
{
	return "the image is linked whole";
}

int dladdr(const void *address, Dl_info *info)
{
	(void)address;
	(void)info;
	return 0;
}

char *setlocale(int category, const char *locale)
{
	(void)category;
	(void)locale;
	return (char *)"C";
}

locale_t newlocale(int mask, const char *locale, locale_t base)
{
	(void)mask;
	(void)locale;
	(void)base;
	return (locale_t)1;
}

void freelocale(locale_t locale)
{
	(void)locale;
}

locale_t uselocale(locale_t locale)
{
	(void)locale;
	return (locale_t)1;
}

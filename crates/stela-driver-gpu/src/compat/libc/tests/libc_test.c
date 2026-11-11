// SPDX-License-Identifier: GPL-2.0-or-later
/* The C library against the C standard's answers, as a static Linux program:
 * `cargo xtask gpu nvk` builds and runs it. */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <pwd.h>
#include <regex.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stela/host.h>
#include "host.h"

static int failures;

#define CHECK(condition)                                                                          \
	do {                                                                                      \
		if (!(condition)) {                                                               \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);           \
			failures++;                                                               \
		}                                                                                 \
	} while (0)

#define FORMATS(expected, ...)                                                                    \
	do {                                                                                      \
		char text[128];                                                                   \
		int length = snprintf(text, sizeof(text), __VA_ARGS__);                           \
		CHECK(!strcmp(text, expected));                                                   \
		CHECK(length == (int)strlen(expected));                                           \
	} while (0)

static int close_to(double value, double expected)
{
	double difference = fabs(value - expected);
	return difference <= 1e-12 * (fabs(expected) + 1e-300) || difference < 1e-300;
}

static void test_formatting(void)
{
	FORMATS("7", "%d", 7);
	FORMATS("-42|   -42|-42   |-0042", "%d|%6d|%-6d|%05d", -42, -42, -42, -42);
	FORMATS("ff FF 0xff 377", "%x %X %#x %o", 255u, 255u, 255u, 255u);
	FORMATS("18446744073709551615", "%llu", (unsigned long long)-1);
	FORMATS("-9223372036854775807", "%lld", -9223372036854775807ll);
	FORMATS("12345 12", "%zu %hhd", (size_t)12345, 268);
	FORMATS("a %", "%c %%", 'a');
	FORMATS("abc|  ab|ab  ", "%s|%4.2s|%-4.2s", "abc", "abc", "abc");
	FORMATS("(null)", "%s", (char *)NULL);
	FORMATS("3.141590", "%f", 3.14159);
	FORMATS("3.14|   3.1|0.5000", "%.2f|%6.1f|%.4f", 3.14159, 3.14159, 0.5);
	FORMATS("1.500000e+00", "%e", 1.5);
	FORMATS("1.5 100000 1e+06 0.0001 1e-05", "%g %g %g %g %g", 1.5, 100000.0, 1e6, 1e-4, 1e-5);
	FORMATS("inf -inf nan", "%f %f %f", (double)INFINITY, -(double)INFINITY, (double)NAN);
	FORMATS(" 5", "%*d", 2, 5);
	char small[4];
	CHECK(snprintf(small, sizeof(small), "%s", "abcdef") == 6);
	CHECK(!strcmp(small, "abc"));

	char *printed = NULL;
	CHECK(asprintf(&printed, "%d-%s", 1, "x") == 3);
	CHECK(!strcmp(printed, "1-x"));
	free(printed);

	char *stream_bytes = NULL;
	size_t stream_size = 0;
	FILE *stream = open_memstream(&stream_bytes, &stream_size);
	CHECK(stream != NULL);
	for (int i = 0; i < 1000; i++)
		fprintf(stream, "%03d", i);
	fclose(stream);
	CHECK(stream_size == 3000);
	CHECK(!memcmp(stream_bytes + 2997, "999", 3));
	free(stream_bytes);
}

static void test_parsing(void)
{
	char *end;
	CHECK(strtol("  -123x", &end, 10) == -123 && *end == 'x');
	CHECK(strtol("0x1f", NULL, 0) == 31);
	CHECK(strtol("077", NULL, 0) == 63);
	CHECK(strtoul("ff", NULL, 16) == 255);
	CHECK(strtoull("18446744073709551615", NULL, 10) == UINT64_MAX);
	CHECK(atoi("12") == 12);
	CHECK(close_to(strtod("1.5", NULL), 1.5));
	CHECK(close_to(strtod("-2.5e3", &end), -2500.0) && !*end);
	CHECK(close_to(strtod("1e10", NULL), 1e10));
	CHECK(close_to(strtod(".25", NULL), 0.25));
	CHECK(strtod("0.1", NULL) == 0.1);
	CHECK(strtod("123456789.125", NULL) == 123456789.125);
	CHECK(isinf(strtod("inf", NULL)));
	CHECK(isnan(strtod("nan", NULL)));
	CHECK(strtof("0.5", NULL) == 0.5f);
	int number = 0;
	char word[8];
	float real = 0;
	CHECK(sscanf("12 ab 2.5", "%d %7s %f", &number, word, &real) == 3);
	CHECK(number == 12 && !strcmp(word, "ab") && real == 2.5f);
	int consumed = 0;
	CHECK(sscanf("  7 x", "%d%n", &number, &consumed) == 1 && number == 7 && consumed == 3);
	CHECK(sscanf("abcdef", "%3s", word) == 1 && !strcmp(word, "abc"));
	unsigned int high, low;
	CHECK(sscanf("10de:2c05", "%x:%x", &high, &low) == 2);
	CHECK(high == 0x10de && low == 0x2c05);
}

static int ascending(const void *a, const void *b)
{
	int left = *(const int *)a, right = *(const int *)b;
	return (left > right) - (left < right);
}

static void test_sorting(void)
{
	int values[1024];
	unsigned int state = 1;
	for (int i = 0; i < 1024; i++) {
		state = state * 1103515245 + 12345;
		values[i] = (int)(state >> 16);
	}
	qsort(values, 1024, sizeof(int), ascending);
	for (int i = 1; i < 1024; i++)
		CHECK(values[i - 1] <= values[i]);
	int key = values[512];
	int *found = bsearch(&key, values, 1024, sizeof(int), ascending);
	CHECK(found && *found == key);
	key = values[0] - 1;
	CHECK(!bsearch(&key, values, 1024, sizeof(int), ascending));
}

static void test_strings(void)
{
	CHECK(strlen("abc") == 3);
	CHECK(strcmp("a", "b") < 0);
	CHECK(!strncmp("abcd", "abce", 3));
	CHECK(!strcasecmp("AbC", "aBc"));
	CHECK(!strcmp(strchr("a.b.c", '.'), ".b.c"));
	CHECK(!strcmp(strrchr("a.b.c", '.'), ".c"));
	CHECK(!strcmp(strstr("needle in", "in"), "in"));
	CHECK(strspn("aab", "a") == 2);
	CHECK(strcspn("aab", "b") == 2);
	char *copy = strdup("x,y");
	char *rest = NULL;
	CHECK(!strcmp(strtok_r(copy, ",", &rest), "x"));
	CHECK(!strcmp(strtok_r(NULL, ",", &rest), "y"));
	CHECK(!strtok_r(NULL, ",", &rest));
	free(copy);
	CHECK(strerror(ENOMEM)[0]);
	char text[32];
	CHECK(strerror_r(ENOMEM, text, sizeof(text)) == 0 && !strcmp(text, strerror(ENOMEM)));
	CHECK(strerror_r(ENOMEM, text, 1) == ERANGE);
}

static void test_allocation(void)
{
	size_t before = live_blocks;
	char *block = malloc(16);
	memset(block, 7, 16);
	block = realloc(block, 4096);
	CHECK(block[15] == 7);
	free(block);
	unsigned char *zero = calloc(32, 4);
	for (int i = 0; i < 128; i++)
		CHECK(zero[i] == 0);
	free(zero);
	void *aligned = aligned_alloc(4096, 8192);
	CHECK(!((uintptr_t)aligned & 4095));
	free(aligned);
	void *pointer = NULL;
	CHECK(!posix_memalign(&pointer, 64, 100));
	CHECK(!((uintptr_t)pointer & 63));
	free(pointer);
	CHECK(live_blocks == before);
	free(NULL);
}

static void test_unmap_refusal(void)
{
	extern void *drm_unmap_address;
	extern int drm_unmap_result;
	/* Initialize errno's compiler TLS before counting the mapping's owner. */
	errno = 0;
	size_t before = live_blocks;
	unsigned char *mapping = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
				      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	CHECK(mapping != MAP_FAILED && live_blocks == before + 1);
	mapping[0] = 0x5a;
	drm_unmap_address = mapping;
	drm_unmap_result = -ENOMEM;
	CHECK(munmap(mapping, 4096) == -1 && errno == ENOMEM);
	CHECK(live_blocks == before + 1 && mapping[0] == 0x5a);
	/* A successful DRM unmap is handled by the host, not the libc heap. */
	drm_unmap_result = 1;
	CHECK(!munmap(mapping, 4096) && live_blocks == before + 1);
	/* The ordinary anonymous mapping path still releases its heap owner. */
	drm_unmap_address = NULL;
	CHECK(!munmap(mapping, 4096) && live_blocks == before);
}

static void test_unavailable_mapping_protection(void)
{
	const int flags = MAP_PRIVATE | MAP_ANONYMOUS;
	CHECK(mmap(NULL, 4096, PROT_READ | PROT_EXEC, flags, -1, 0) == MAP_FAILED &&
	      errno == ENOTSUP);
	unsigned char *mapping = mmap(NULL, 4096, PROT_READ | PROT_WRITE, flags, -1, 0);
	CHECK(mapping != MAP_FAILED);
	if (mapping == MAP_FAILED)
		return;

	const int protections[] = { PROT_READ | PROT_EXEC, PROT_READ, PROT_NONE };
	for (size_t index = 0; index < sizeof(protections) / sizeof(protections[0]); index++) {
		errno = 0;
		CHECK(mprotect(mapping, 4096, protections[index]) == -1 && errno == ENOTSUP);
	}
	mapping[0] = 0x5a;
	CHECK(mapping[0] == 0x5a);
	CHECK(!munmap(mapping, 4096));
}

static void test_anonymous_files(void)
{
	extern int file_mapped_fd;
	extern void *file_mapped_address;
	extern uint64_t file_bytes;
	void *place = (void *)0x2000;
	int file = memfd_create("test", MFD_CLOEXEC);
	CHECK(file >= STELA_LIBC_FILE_BASE);
	CHECK(!ftruncate(file, 8192) && file_bytes == 8192);
	void *mapping = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
	CHECK(mapping == (void *)0x1000 && file_mapped_fd == file && !file_mapped_address);
	CHECK(!munmap(mapping, 4096));
	/* Addresses that are bound later are the file layer's, not the heap's. */
	CHECK(mmap(NULL, 4096, PROT_READ, MAP_SHARED | MAP_ANONYMOUS, -1, 0) == (void *)0x1000);
	CHECK(file_mapped_fd == -1);
	CHECK(mmap(place, 4096, PROT_READ, MAP_SHARED | MAP_FIXED, file, 0) == place);
	CHECK(file_mapped_fd == file && file_mapped_address == place);
	CHECK(mmap(place, 4096, PROT_READ, MAP_SHARED | MAP_FIXED | MAP_ANONYMOUS, -1, 0) == place);
	CHECK(file_mapped_fd == -1 && file_mapped_address == place);
	CHECK(!close(file));
}

static void test_meminfo(void)
{
	char text[256];
	int file = open("/proc/meminfo", O_RDONLY);
	CHECK(file >= 0);
	struct stat status;
	CHECK(!fstat(file, &status) && S_ISREG(status.st_mode) && status.st_size > 0);
	ssize_t size = read(file, text, sizeof(text) - 1);
	CHECK(size == status.st_size && read(file, text, 1) == 0);
	text[size] = 0;
	unsigned long long total, available;
	CHECK(sscanf(strstr(text, "MemTotal:"), "MemTotal: %llu", &total) == 1 && total == 4194304);
	CHECK(sscanf(strstr(text, "MemAvailable:"), "MemAvailable: %llu", &available) == 1);
	CHECK(available == 3145728);
	CHECK(!close(file) && read(file, text, 1) == -1 && errno == EBADF);
	errno = 0;
	CHECK(open("/proc/cpuinfo", O_RDONLY) == -1 && errno == ENOENT);
}

static void test_refusals(void)
{
	struct pollfd file = { .fd = 0, .events = POLLIN };
	errno = 0;
	CHECK(poll(&file, 1, 0) == -1);
	CHECK(errno == ENOSYS);
}

static void test_division(void)
{
	div_t small = div(-7, 2);
	CHECK(small.quot == -3 && small.rem == -1);
	ldiv_t wide = ldiv(7L, -2L);
	CHECK(wide.quot == -3 && wide.rem == 1);
	lldiv_t widest = lldiv(1LL << 40, 3);
	CHECK(widest.quot == 366503875925LL && widest.rem == 1);
}

static void test_system(void)
{
	char path[16];
	errno = 0;
	CHECK(chdir("/") == -1 && errno == ENOSYS);
	CHECK(getcwd(path, sizeof(path)) == NULL);
	CHECK(opendir("/") == NULL);
	struct passwd entry, *found = &entry;
	CHECK(getpwuid_r(0, &entry, path, sizeof(path), &found) == 0 && found == NULL);
}

static void test_wide(void)
{
	static const wchar_t number[] = L"-42 rest";
	wchar_t *end;
	CHECK(wcstol(number, &end, 10) == -42 && end == number + 3);
	CHECK(wcstoull(L"0x10", NULL, 16) == 16);
	CHECK(close_to(wcstod(L"2.5", NULL), 2.5));
	CHECK(wcslen(number) == 8);
	CHECK(wmemchr(number, L'r', 8) == number + 4 && !wmemchr(number, L'z', 8));
	CHECK(wmemcmp(L"ab", L"ac", 2) < 0 && wmemcmp(L"ab", L"ab", 2) == 0);
	wchar_t text[8];
	CHECK(swprintf(text, 8, L"%d", 123) == 3 && !wmemcmp(text, L"123", 4));
	CHECK(swprintf(text, 2, L"%d", 123) == -1);
}

static void test_math(void)
{
	CHECK(close_to(sin(1.0), 0.8414709848078965));
	CHECK(close_to(cos(1.0), 0.5403023058681398));
	CHECK(close_to(tan(0.5), 0.5463024898437905));
	CHECK(close_to(atan2(1.0, -1.0), 2.356194490192345));
	CHECK(close_to(asin(0.5), 0.5235987755982989));
	CHECK(close_to(acos(0.5), 1.0471975511965979));
	CHECK(close_to(exp(1.5), 4.4816890703380645));
	CHECK(close_to(log(10.0), 2.302585092994046));
	CHECK(close_to(log2(1024.0), 10.0));
	CHECK(close_to(log10(1000.0), 3.0));
	CHECK(close_to(pow(2.0, 0.5), 1.4142135623730951));
	CHECK(pow(2.0, 10.0) == 1024.0);
	CHECK(pow(-2.0, 3.0) == -8.0);
	CHECK(pow(0.0, 0.0) == 1.0);
	CHECK(close_to(cbrt(27.0), 3.0));
	CHECK(sqrt(9.0) == 3.0);
	CHECK(fmod(7.5, 2.0) == 1.5);
	CHECK(fmod(-7.5, 2.0) == -1.5);
	CHECK(remainder(7.0, 2.0) == -1.0);
	CHECK(floor(-1.5) == -2.0 && ceil(-1.5) == -1.0);
	CHECK(trunc(-1.5) == -1.0 && round(2.5) == 3.0);
	CHECK(lround(-2.5) == -3);
	CHECK(lrint(2.5) == 2);
	CHECK(llrint(3.5) == 4);
	CHECK(ldexp(3.0, 4) == 48.0);
	int exponent;
	CHECK(frexp(48.0, &exponent) == 0.75 && exponent == 6);
	double whole;
	CHECK(modf(-1.25, &whole) == -0.25 && whole == -1.0);
	CHECK(copysign(2.0, -0.0) == -2.0);
	CHECK(fma(2.0, 3.0, 1.0) == 7.0);
	CHECK(nextafter(1.0, 2.0) > 1.0);
	CHECK(powf(2.0f, 3.0f) == 8.0f);
	CHECK(sqrtf(16.0f) == 4.0f);
	CHECK(isnan(sqrt(-1.0)));
	CHECK(logb(8.0) == 3.0 && logb(0.75) == -1.0 && isinf(logb(0.0)));
	CHECK(fabs(erf(0.5) - 0.5204998778130465) < 1e-15);
	CHECK(fabs(erf(3.0) - 0.9999779095030014) < 1e-15);
	CHECK(erf(-3.0) == -erf(3.0) && erf(100.0) == 1.0);
	CHECK(fesetround(FE_UPWARD) == 0);
	CHECK(fegetround() == FE_UPWARD);
	CHECK(lrint(2.1) == 3);
	CHECK(fesetround(FE_TONEAREST) == 0);
	feclearexcept(FE_ALL_EXCEPT);
	volatile double zero = 0.0;
	CHECK(isinf(1.0 / zero));
	CHECK(fetestexcept(FE_DIVBYZERO));
	feclearexcept(FE_ALL_EXCEPT);
	CHECK(!fetestexcept(FE_DIVBYZERO));
}

static void test_time(void)
{
	time_t epoch = 86400 * 365;
	struct tm parts;
	CHECK(localtime_r(&epoch, &parts) == &parts);
	CHECK(parts.tm_year == 71 && parts.tm_mon == 0 && parts.tm_mday == 1);
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	clock_gettime(CLOCK_MONOTONIC, &end);
	CHECK(end.tv_sec >= start.tv_sec);
	clockid_t unsupported[] = {CLOCK_REALTIME, CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID};
	for (unsigned int index = 0; index < sizeof(unsupported) / sizeof(unsupported[0]); index++) {
		end = (struct timespec){1, 2};
		CHECK(clock_gettime(unsupported[index], &end) == -1 && errno == ENOTSUP);
		CHECK(!end.tv_sec && !end.tv_nsec);
	}
	CHECK(clock_gettime(-1, &end) == -1 && errno == EINVAL);
	CHECK(clock_gettime(CLOCK_MONOTONIC, NULL) == -1 && errno == EINVAL);
	CHECK(!timespec_get(&end, TIME_UTC) && errno == ENOTSUP);
	CHECK(!timespec_get(&end, -1) && errno == EINVAL);
	CHECK(time(&epoch) == (time_t)-1 && epoch == (time_t)-1 && errno == ENOTSUP);
	struct timeval wall = {1, 2};
	CHECK(gettimeofday(&wall, NULL) == -1 && errno == ENOTSUP);
	CHECK(!wall.tv_sec && !wall.tv_usec);
	struct timespec invalid[] = {{-1, 0}, {0, -1}, {0, 1000000000}, {INT64_MAX, 0}};
	for (unsigned int index = 0; index < sizeof(invalid) / sizeof(invalid[0]); index++) {
		int error = index == 3 ? EOVERFLOW : EINVAL;
		CHECK(clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &invalid[index], NULL) == error);
		CHECK(nanosleep(&invalid[index], NULL) == -1 && errno == error);
	}
	struct timespec zero = {0}, remaining = {1, 2};
	CHECK(!clock_nanosleep(CLOCK_MONOTONIC, 0, &zero, &remaining));
	CHECK(!remaining.tv_sec && !remaining.tv_nsec);
	CHECK(clock_nanosleep(CLOCK_REALTIME, 0, &zero, NULL) == ENOTSUP);
	CHECK(clock_nanosleep(-1, 0, &zero, NULL) == EINVAL);
	CHECK(clock_nanosleep(CLOCK_MONOTONIC, 2, &zero, NULL) == EINVAL);
	/* A representable duration can still overflow when added to now. */
	struct timespec too_late = {(time_t)((UINT64_MAX - 1) / 1000000000),
		(long)((UINT64_MAX - 1) % 1000000000)};
	CHECK(clock_nanosleep(CLOCK_MONOTONIC, 0, &too_late, NULL) == EOVERFLOW);
	CHECK(!getenv("HOME"));
}

static _Thread_local int per_thread = 5;
static int once_runs;

static void once_body(void)
{
	once_runs++;
}

static void *thread_body(void *argument)
{
	CHECK(pthread_self() == STELA_LIBC_PROCESSORS + 1);
	CHECK(per_thread == 5);
	per_thread = 6;
	return (char *)argument + 1;
}

/* A compiler takes a thread-local object's address once per function. */
__attribute__((noinline)) static int *per_thread_on(unsigned int core)
{
	current_core = core;
	return &per_thread;
}

static void test_threads(void)
{
	/* A thread and each processor have their own thread-local object. */
	*per_thread_on(0) = 7;
	CHECK(*per_thread_on(3) == 5);
	CHECK(*per_thread_on(0) == 7);

	char argument[2];
	pthread_t thread;
	CHECK(!pthread_create(&thread, NULL, thread_body, argument));
	void *result = NULL;
	CHECK(!pthread_join(thread, &result));
	CHECK(result == argument + 1);
	CHECK(*per_thread_on(0) == 7);
    CHECK(pthread_join(pthread_self(), NULL) == EDEADLK);
	CHECK(pthread_setname_np(pthread_self(), "mesa-worker") == ENOTSUP);
	clockid_t cpu_clock = -1;
	CHECK(pthread_getcpuclockid(pthread_self(), &cpu_clock) == ENOTSUP);
	struct timespec cpu_time = {1, 2};
	CHECK(clock_gettime(cpu_clock, &cpu_time) == -1 && errno == ENOTSUP);
	CHECK(!cpu_time.tv_sec && !cpu_time.tv_nsec);

	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
	unsigned int woken = wakes;
	CHECK(!pthread_mutex_lock(&mutex));
	CHECK(pthread_mutex_trylock(&mutex) == EBUSY);
	/* A waiter goes through the host, where the holder lets go. */
	waited_word = (volatile unsigned int *)&mutex.locked;
	waited_value = 0;
	unsigned int waited = waits;
	CHECK(!pthread_mutex_lock(&mutex));
	CHECK(waits == waited + 1);
	waited_word = NULL;
	CHECK(!pthread_mutex_unlock(&mutex));
	CHECK(wakes == woken + 1);

	pthread_mutexattr_t attribute;
	pthread_mutexattr_init(&attribute);
	CHECK(pthread_mutexattr_settype(&attribute, -1) == EINVAL);
	CHECK(attribute.kind == PTHREAD_MUTEX_NORMAL);
	CHECK(!pthread_mutexattr_settype(&attribute, PTHREAD_MUTEX_RECURSIVE));
	pthread_mutex_t recursive;
	pthread_mutex_init(&recursive, &attribute);
	CHECK(!pthread_mutex_lock(&recursive));
	CHECK(!pthread_mutex_lock(&recursive));
	CHECK(!pthread_mutex_unlock(&recursive));
	CHECK(recursive.locked);
	CHECK(!pthread_mutex_unlock(&recursive));
	CHECK(!recursive.locked);

	pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	pthread_mutex_lock(&mutex);
	unsigned int before_wait = waits;
	CHECK(pthread_cond_timedwait(&cond, &mutex, &now) == ENOTSUP);
	CHECK(pthread_cond_timedwait(&cond, &mutex, NULL) == EINVAL);
	CHECK(mutex.locked && waits == before_wait);
	CHECK(pthread_mutex_timedlock(&mutex, &now) == ENOTSUP);
	pthread_condattr_t condattr;
	CHECK(!pthread_condattr_init(&condattr) && condattr.clock == CLOCK_REALTIME);
	CHECK(pthread_condattr_setclock(&condattr, -1) == EINVAL);
	CHECK(condattr.clock == CLOCK_REALTIME);
	CHECK(!pthread_condattr_setclock(&condattr, CLOCK_MONOTONIC));
	CHECK(pthread_condattr_setclock(&condattr, CLOCK_REALTIME) == ENOTSUP);
	CHECK(condattr.clock == CLOCK_MONOTONIC);
	pthread_cond_t monotonic;
	CHECK(!pthread_cond_init(&monotonic, &condattr));
	CHECK(monotonic.clock == CLOCK_MONOTONIC);
	CHECK(!pthread_condattr_destroy(&condattr));
	struct timespec invalid_deadlines[] = {{-1, 0}, {0, -1}, {0, 1000000000}, {INT64_MAX, 0}};
	for (unsigned int index = 0; index < sizeof(invalid_deadlines) / sizeof(invalid_deadlines[0]); index++) {
		int error = index == 3 ? EOVERFLOW : EINVAL;
		CHECK(pthread_cond_timedwait(&monotonic, &mutex, &invalid_deadlines[index]) == error);
		CHECK(mutex.locked && waits == before_wait);
		CHECK(pthread_mutex_timedlock(&mutex, &invalid_deadlines[index]) == error);
	}
	CHECK(pthread_cond_timedwait(&monotonic, &mutex, &now) == ETIMEDOUT);
	CHECK(mutex.locked);
	/* Another processor signals while this one waits. */
	waited_word = &cond.generation;
	waited_value = 1;
	CHECK(!pthread_cond_wait(&cond, &mutex));
	waited_word = NULL;
	pthread_mutex_unlock(&mutex);

	pthread_once_t once = PTHREAD_ONCE_INIT;
	CHECK(!pthread_once(&once, once_body));
	CHECK(!pthread_once(&once, once_body));
	CHECK(once_runs == 1);

	pthread_rwlock_t lock = PTHREAD_RWLOCK_INITIALIZER;
	CHECK(!pthread_rwlock_rdlock(&lock));
	CHECK(!pthread_rwlock_rdlock(&lock));
	CHECK(lock.state == 2);
	CHECK(!pthread_rwlock_unlock(&lock));
	CHECK(!pthread_rwlock_unlock(&lock));
	CHECK(!pthread_rwlock_wrlock(&lock));
	CHECK(lock.state == -1);
	CHECK(!pthread_rwlock_unlock(&lock));
	CHECK(!lock.state);

	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, NULL, 1);
	CHECK(pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD);

	pthread_key_t key;
	CHECK(!pthread_key_create(&key, NULL));
	CHECK(!pthread_setspecific(key, argument));
	CHECK(pthread_getspecific(key) == argument);
	current_core = 3;
	CHECK(!pthread_getspecific(key));
	current_core = 0;
	CHECK(!pthread_key_delete(key));
}

static pthread_key_t destructor_key;
static unsigned int destructor_calls;

static void key_destructor(void *value)
{
	CHECK(pthread_getspecific(destructor_key) == NULL);
	destructor_calls++;
	CHECK(!pthread_setspecific(destructor_key, value));
}

static void *exiting_thread(void *value)
{
	CHECK(per_thread == 5);
	per_thread = 9;
	CHECK(!pthread_setspecific(destructor_key, value));
	pthread_exit(value);
}

static void test_thread_lifetimes(void)
{
	/* More completed threads than the live-thread limit, with fresh TLS. */
	char argument[2];
	for (unsigned int i = 0; i < STELA_LIBC_THREADS * 3; i++) {
		pthread_t thread;
		CHECK(!pthread_create(&thread, NULL, thread_body, argument));
		void *result = NULL;
		CHECK(!pthread_join(thread, &result));
		CHECK(result == argument + 1);
		CHECK(pthread_join(thread, NULL) == EINVAL);
	}
	/* Failed starts must not consume slots either. */
	extern int thread_start_error;
	thread_start_error = -ENOMEM;
	for (unsigned int i = 0; i < STELA_LIBC_THREADS * 2; i++) {
		pthread_t thread;
		CHECK(pthread_create(&thread, NULL, thread_body, argument) == ENOMEM);
	}
	thread_start_error = 0;
	CHECK(!pthread_key_create(&destructor_key, key_destructor));
	pthread_t thread;
	void *result = NULL;
	CHECK(!pthread_create(&thread, NULL, exiting_thread, argument));
	CHECK(!pthread_join(thread, &result));
	CHECK(result == argument);
	CHECK(destructor_calls == PTHREAD_DESTRUCTOR_ITERATIONS);
	CHECK(!pthread_key_delete(destructor_key));
	CHECK(pthread_setspecific(destructor_key, argument) == EINVAL);
	CHECK(pthread_getspecific(destructor_key) == NULL);
	/* A detached thread reclaims its slot without a join. */
	pthread_attr_t attr;
	CHECK(!pthread_attr_init(&attr));
	CHECK(pthread_attr_setdetachstate(&attr, -1) == EINVAL);
	CHECK(!pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED));
	for (unsigned int i = 0; i < STELA_LIBC_THREADS * 2; i++)
		CHECK(!pthread_create(&thread, &attr, thread_body, argument));
	CHECK(pthread_join(thread, NULL) == EINVAL);
	CHECK(!pthread_attr_destroy(&attr));
	CHECK(!pthread_create(&thread, NULL, thread_body, argument));
	CHECK(!pthread_detach(thread));
	CHECK(pthread_detach(thread) == EINVAL);
	/* Key deletion and reuse neither call a destructor nor retain its value. */
	for (unsigned int i = 0; i < 128; i++) {
		pthread_key_t key;
		CHECK(!pthread_key_create(&key, key_destructor));
		CHECK(!pthread_setspecific(key, argument));
		CHECK(!pthread_key_delete(key));
		CHECK(pthread_key_delete(key) == EINVAL);
	}
	CHECK(destructor_calls == PTHREAD_DESTRUCTOR_ITERATIONS);
}

static void test_regex(void)
{
	regex_t regex;
	regmatch_t matches[3];
	CHECK(!regcomp(&regex, "^([a-z]+)-([[:digit:]]+)$", REG_EXTENDED | REG_ICASE));
	CHECK(!regexec(&regex, "Mesa-123", 3, matches, 0));
	CHECK(matches[0].rm_so == 0 && matches[0].rm_eo == 8);
	CHECK(matches[1].rm_so == 0 && matches[1].rm_eo == 4);
	CHECK(matches[2].rm_so == 5 && matches[2].rm_eo == 8);
	CHECK(regexec(&regex, "Mesa-abc", 0, NULL, 0) == REG_NOMATCH);
	regfree(&regex);
	CHECK(!regcomp(&regex, "^(red|blue)$", REG_EXTENDED | REG_NOSUB));
	CHECK(!regexec(&regex, "blue", 0, NULL, 0));
	CHECK(regexec(&regex, "green", 0, NULL, 0) == REG_NOMATCH);
	regfree(&regex);
	int code = regcomp(&regex, "(", REG_EXTENDED);
	CHECK(code != 0);
	char message[64], short_message[2];
	size_t bytes = regerror(code, &regex, message, sizeof(message));
	CHECK(bytes > 1 && bytes == strlen(message) + 1);
	CHECK(regerror(code, &regex, short_message, sizeof(short_message)) == bytes);
	CHECK(short_message[1] == 0);
}

static void test_nonlocal_return(void)
{
	jmp_buf saved;
	unsigned int before, after;
	unsigned short x87_before, x87_after;
	__asm__ volatile("stmxcsr %0" : "=m"(before));
	__asm__ volatile("fnstcw %0" : "=m"(x87_before));
	int value = setjmp(saved);
	if (!value) {
		unsigned int changed = before ^ 0x2000;
		unsigned short x87_changed = x87_before ^ 0x400;
		__asm__ volatile("ldmxcsr %0" : : "m"(changed));
		__asm__ volatile("fldcw %0" : : "m"(x87_changed));
		longjmp(saved, 0);
	}
	CHECK(value == 1);
	__asm__ volatile("stmxcsr %0" : "=m"(after));
	__asm__ volatile("fnstcw %0" : "=m"(x87_after));
	CHECK(before == after && x87_before == x87_after);
}

int main(void)
{
	test_formatting();
	test_parsing();
	test_sorting();
	test_strings();
	test_allocation();
	test_unmap_refusal();
	test_unavailable_mapping_protection();
	test_anonymous_files();
	test_meminfo();
	test_refusals();
	test_division();
	test_system();
	test_wide();
	test_math();
	test_time();
	test_threads();
	test_thread_lifetimes();
	test_regex();
	test_nonlocal_return();
	if (failures)
		fprintf(stderr, "%d of the C library's checks failed\n", failures);
	return failures != 0;
}

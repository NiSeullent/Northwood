/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stela/host.h>

#ifdef STELA_PTHREAD_STANDALONE
#include <thread_runtime.inc>
#endif

extern bool stela_pthread_probe_waiting(const volatile unsigned int *word);
extern bool stela_pthread_probe_retired(void);

#define NS_PER_SECOND UINT64_C(1000000000)

/* The runtime retains the worker if a probe cannot prove its return. */
static struct {
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	unsigned int phase;
	bool signal;
} probe;

static uint64_t now(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_MONOTONIC, &time) == 0
		? (uint64_t)time.tv_sec * NS_PER_SECOND + (uint64_t)time.tv_nsec : 0;
}

static struct timespec deadline(uint64_t ns)
{
	return (struct timespec){ns / NS_PER_SECOND, ns % NS_PER_SECOND};
}

static void *worker(void *argument)
{
	void *result = NULL;
	pthread_mutex_lock(&probe.mutex);
	uint64_t until = now() + 20000000;
	struct timespec timeout = deadline(until);
	if (pthread_cond_timedwait(&probe.condition, &probe.mutex, &timeout) == ETIMEDOUT && now() >= until) {
		__atomic_store_n(&probe.phase, 1, __ATOMIC_RELEASE);
		timeout = deadline(now() + 5 * NS_PER_SECOND);
		int waited = 0;
		while (!probe.signal && !waited)
			waited = pthread_cond_timedwait(&probe.condition, &probe.mutex, &timeout);
		if (probe.signal && !waited)
			result = argument;
	}
	pthread_mutex_unlock(&probe.mutex);
	__atomic_store_n(&probe.phase, 2, __ATOMIC_RELEASE);
	return result;
}

int stela_pthread_runtime_probe(void)
{
	pthread_condattr_t attr;
	if (!now() || pthread_mutex_init(&probe.mutex, NULL) || pthread_condattr_init(&attr))
		return 1;
	int status = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
	if (!status)
		status = pthread_cond_init(&probe.condition, &attr);
	pthread_condattr_destroy(&attr);
	if (status)
		return 2;

	pthread_t thread;
	if (pthread_create(&thread, NULL, worker, &probe))
		return 3;
	const uint64_t until = now() + 5 * NS_PER_SECOND;
	while (now() < until) {
		bool parked = stela_pthread_probe_waiting(&probe.condition.generation);
		unsigned int phase = __atomic_load_n(&probe.phase, __ATOMIC_ACQUIRE);
		if (parked && phase == 1) {
			pthread_mutex_lock(&probe.mutex);
			probe.signal = true;
			pthread_cond_signal(&probe.condition);
			pthread_mutex_unlock(&probe.mutex);
			break;
		}
		if (phase == 2)
			break;
	}
	while (__atomic_load_n(&probe.phase, __ATOMIC_ACQUIRE) != 2 && now() < until)
		stela_pthread_probe_waiting(&probe.condition.generation);
	if (__atomic_load_n(&probe.phase, __ATOMIC_ACQUIRE) != 2)
		return 4;
	/* Confirm native return before joining, so the probe can time out even
	 * if pthread's completion has not been published. */
	const uint64_t retirement = now() + 5 * NS_PER_SECOND;
	while (!stela_pthread_probe_retired()) {
		if (now() >= retirement)
			return 7;
	}
	void *result = NULL;
	if (pthread_join(thread, &result))
		return 5;
	pthread_cond_destroy(&probe.condition);
	pthread_mutex_destroy(&probe.mutex);
	return result == &probe ? 0 : 6;
}

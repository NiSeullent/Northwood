// SPDX-License-Identifier: GPL-2.0-only
/* Parts are adapted from the Linux files named below, whose notices are:
 * Copyright (C) 2010 Red Hat, Inc., Peter Zijlstra
 */
/* Linux's irq_work, with the claiming of kernel/irq_work.c. Stela's drivers
 * have no self-interrupt, so queued work runs on a worker. It keeps what hard
 * interrupt context promises the function: no sleeping, and an RCU read side
 * that outlives the function's final reference to its own irq_work. */
#include <linux/irq_work.h>
#include <linux/hardirq.h>
#include <linux/llist.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/workqueue.h>
#include <stela/atomic_callback.h>
#include <stela/linux.h>

static_assert(!IS_ENABLED(CONFIG_SMP));
static_assert(!IS_ENABLED(CONFIG_PREEMPT_RT));
static_assert(IS_ENABLED(CONFIG_IRQ_WORK));

static LLIST_HEAD(queued);

void irq_work_single(void *arg)
{
	/* This is a delivery primitive, never a process/ISR admission route.
	 * Our worker establishes the checked callback scope before using it. */
	if (!stela_linux_atomic_callback_active() || !in_hardirq() || !irqs_disabled())
		__builtin_trap();
	struct irq_work *work = arg;
	int flags = atomic_read(&work->node.a_flags) & ~IRQ_WORK_PENDING;

	atomic_set(&work->node.a_flags, flags);
	smp_mb();
	rcu_read_lock();
	work->func(work);
	atomic_cmpxchg(&work->node.a_flags, flags, flags & ~IRQ_WORK_BUSY);
	/* The no-self-interrupt profile waits for completion with rcuwait,
	 * as kernel/irq_work.c does. Keep this final access inside RCU. */
	rcuwait_wake_up(&work->irqwait);
	rcu_read_unlock();
}

static void run_queued(struct work_struct *runner)
{
	(void)runner;
	struct stela_linux_atomic_callback scope;
	stela_linux_atomic_callback_begin(&scope);
	struct llist_node *first = llist_del_all(&queued);
	struct irq_work *work, *next;

	llist_for_each_entry_safe(work, next, first, node.llist)
		irq_work_single(work);
	stela_linux_atomic_callback_end(&scope);
}

static DECLARE_WORK(runner, run_queued);

bool irq_work_needs_cpu(void)
{
	return !llist_empty(&queued);
}

/* Whether work raised now runs at the raising worker's own next quiescent
 * point instead. On a processor with a self-interrupt Linux runs it as soon
 * as interrupts are enabled again, which in a workqueue worker's process
 * context is before the work item returns; every item of a worker with a
 * task ends in cond_resched() (process_one_work), and an item that sleeps
 * first parks: both run it (stela_linux_irq_work_run_owed), with the
 * function's hard interrupt context, after the raising code and before any
 * other task runs. That needs no runner step of its own. */
static bool owed_to_raising_worker(void)
{
	return in_task() && !stela_linux_atomic_callback_active() &&
	       (current->flags & PF_WQ_WORKER);
}

/* The host supplies no self-interrupt or native tick entry to Linux. Raised
 * and lazy work runs at the raising worker's next quiescent point (above) or
 * on the retained worker; callers never receive an inline callback or an
 * invented interrupt entry. */
void arch_irq_work_raise(void)
{
	if (irq_work_needs_cpu() && !owed_to_raising_worker())
		schedule_work(&runner);
}

/* Runs queued work in its hard interrupt context from a quiescent task, as
 * the self-interrupt that work raised there would have. Nothing otherwise. */
void stela_linux_irq_work_run_owed(void)
{
	if (irq_work_needs_cpu() && stela_linux_quiescent())
		run_queued(NULL);
}

void irq_work_run(void)
{
	arch_irq_work_raise();
}

void irq_work_tick(void)
{
	arch_irq_work_raise();
}

bool irq_work_queue(struct irq_work *work)
{
	if (atomic_fetch_or(IRQ_WORK_CLAIMED | CSD_TYPE_IRQ_WORK, &work->node.a_flags) & IRQ_WORK_PENDING)
		return false;
	llist_add(&work->node.llist, &queued);
	arch_irq_work_raise();
	return true;
}

bool irq_work_queue_on(struct irq_work *work, int cpu)
{
	/* The original !SMP branch queues locally regardless of cpu. */
	(void)cpu;
	return irq_work_queue(work);
}

void irq_work_sync(struct irq_work *work)
{
	lockdep_assert_irqs_enabled();
	might_sleep();
	rcuwait_wait_event(&work->irqwait, !irq_work_is_busy(work), TASK_UNINTERRUPTIBLE);
	/* BUSY is cleared before the runner's final irqwait access. A grace
	 * period closes that RCU read side before the caller can free work. */
	synchronize_rcu();
}

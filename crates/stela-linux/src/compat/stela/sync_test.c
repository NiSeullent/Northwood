// SPDX-License-Identifier: GPL-2.0-or-later
/* Linux sleeping locks, kthread workers, and irq_work on original tasks. */
#include <linux/completion.h>
#include <linux/irq_work.h>
#include <linux/kthread.h>
#include <linux/rwsem.h>
#include <linux/sched.h>
#include <linux/srcu.h>
#include <linux/workqueue.h>
#include <stela/atomic_callback.h>
#include <stela/linux.h>
#include <stela/test.h>

void stela_linux_sync_test(void);

static DECLARE_RWSEM(lock);
static struct work_struct writer_work;
static DECLARE_COMPLETION(writer_started);
static bool readers_left, writer_done;

static void writer(struct work_struct *work)
{
	(void)work;
	complete(&writer_started);
	assert(!down_write_killable(&lock));
	assert(readers_left);
	writer_done = true;
	up_write(&lock);
}

static void rwsem(void)
{
	down_read(&lock);
	assert(down_read_trylock(&lock) && !down_write_trylock(&lock));
	INIT_WORK(&writer_work, writer);
	assert(schedule_work(&writer_work));
	wait_for_completion(&writer_started);
	up_read(&lock);
	readers_left = true;
	up_read(&lock);
	assert(flush_work(&writer_work) && writer_done);

	assert(down_write_trylock(&lock) && !down_read_trylock(&lock));
	downgrade_write(&lock);
	assert(down_read_trylock(&lock) && !down_write_trylock(&lock));
	up_read(&lock);
	up_read(&lock);
}

static struct kthread_work steps[2], blocked;
static char order[3];
static unsigned int order_length;
static DECLARE_COMPLETION(blocked_started);
static DECLARE_COMPLETION(blocked_release);

static void step(struct kthread_work *work)
{
	order[order_length++] = (char)('0' + (work - steps));
}

static void block(struct kthread_work *work)
{
	(void)work;
	complete(&blocked_started);
	wait_for_completion(&blocked_release);
}

static void workers(void)
{
	struct kthread_worker *worker = kthread_run_worker(0, "test-%d", 1);

	assert(!IS_ERR(worker));
	sched_set_fifo(worker->task);
	kthread_init_work(&blocked, block);
	kthread_init_work(&steps[0], step);
	kthread_init_work(&steps[1], step);
	assert(kthread_queue_work(worker, &blocked));
	wait_for_completion(&blocked_started);

	/* While the worker is busy: a queued item, and one cancelled unrun. */
	assert(kthread_queue_work(worker, &steps[0]));
	assert(!kthread_queue_work(worker, &steps[0]));
	assert(kthread_queue_work(worker, &steps[1]));
	assert(kthread_cancel_work_sync(&steps[1]));
	assert(!kthread_cancel_work_sync(&steps[1]));
	assert(kthread_queue_work(worker, &steps[1]));
	complete(&blocked_release);
	kthread_flush_work(&steps[1]);
	assert(order_length == 2 && !memcmp(order, "01", 2));
	kthread_flush_worker(worker);
	kthread_destroy_worker(worker);
}

static DECLARE_COMPLETION(deferred_ran);
static unsigned int deferred_calls;
static bool deferred_requeue;

static void deferred(struct irq_work *work)
{
	assert(irq_work_is_busy(work) && !irq_work_is_pending(work));
	assert(atomic_read(&work->node.a_flags) & CSD_TYPE_IRQ_WORK);
	assert(in_hardirq() && irqs_disabled() && !in_task());
	assert(stela_linux_atomic_callback_active() && !stela_linux_quiescent());
	deferred_calls++;
	if (deferred_requeue && deferred_calls < 3) {
		assert(irq_work_queue_on(work, 0));
		assert(irq_work_is_pending(work) && !irq_work_queue(work));
		return;
	}
	complete(&deferred_ran);
}

static void irq_works(void)
{
	/* Synchronization must finish every final runner access before this
	 * real task's stack storage leaves scope. */
	struct irq_work work;

	assert(IS_ENABLED(CONFIG_IRQ_WORK) && !arch_irq_work_has_interrupt());
	assert(!irq_work_needs_cpu());
	init_irq_work(&work, deferred);
	assert(irq_work_queue(&work) && !irq_work_queue(&work));
	assert(irq_work_needs_cpu());
	irq_work_run();
	irq_work_tick();
	arch_irq_work_raise();
	assert(!deferred_calls && irq_work_needs_cpu());
	wait_for_completion(&deferred_ran);
	irq_work_sync(&work);
	assert(deferred_calls == 1 && !irq_work_is_busy(&work));
	/* Reuse after synchronization, then requeue from inside the callback:
	 * BUSY must survive the earlier callback's final compare/exchange. */
	deferred_calls = 0;
	deferred_requeue = true;
	assert(irq_work_queue(&work));
	irq_work_sync(&work);
	wait_for_completion(&deferred_ran);
	assert(deferred_calls == 3 && !irq_work_is_busy(&work) && !irq_work_is_pending(&work));
	assert(!irq_work_needs_cpu());
	assert(in_task() && !irqs_disabled() && stela_linux_quiescent());
	assert(!stela_linux_atomic_callback_active());
}

/* irq_work a workqueue item raises in process context runs in its hard
 * interrupt context once the item returns, before the worker's next item and
 * before a task the item woke: the order the retained runner gave it, with no
 * runner step. An item that waits for its own irq_work runs it at its park. */
static char trace[8];
static unsigned int trace_length;
static struct irq_work raised;
static struct work_struct raising, queued_after, self_waiting;
static DECLARE_COMPLETION(raising_done);

static void note(char mark)
{
	assert(trace_length < sizeof(trace));
	trace[trace_length++] = mark;
}

static void run_raised(struct irq_work *work)
{
	assert(work == &raised && in_hardirq() && irqs_disabled() && !in_task());
	assert(stela_linux_atomic_callback_active());
	note('I');
}

static void run_queued_after(struct work_struct *work)
{
	assert(work == &queued_after);
	note('B');
}

static void run_raising(struct work_struct *work)
{
	unsigned long flags;

	assert(work == &raising && (current->flags & PF_WQ_WORKER));
	note('A');
	local_irq_save(flags);
	assert(irq_work_queue(&raised));
	local_irq_restore(flags);
	assert(queue_work(system_wq, &queued_after));
	complete(&raising_done);
	note('a');
}

static void run_self_waiting(struct work_struct *work)
{
	assert(work == &self_waiting);
	assert(irq_work_queue(&raised));
	irq_work_sync(&raised);
	assert(!irq_work_is_busy(&raised));
	note('S');
}

static void worker_irq_works(void)
{
	struct workqueue_struct *unbound = alloc_ordered_workqueue("stela-irq-work", 0);

	assert(unbound);
	init_irq_work(&raised, run_raised);
	INIT_WORK(&raising, run_raising);
	INIT_WORK(&queued_after, run_queued_after);
	INIT_WORK(&self_waiting, run_self_waiting);
	assert(queue_work(unbound, &raising));
	wait_for_completion(&raising_done);
	note('W');
	flush_work(&raising);
	flush_work(&queued_after);
	irq_work_sync(&raised);
	assert(trace_length == 5 && !memcmp(trace, "AaI", 3));
	/* The woken task and the later item both follow the irq_work. */
	assert(memchr(trace + 3, 'B', 2) && memchr(trace + 3, 'W', 2));
	assert(!irq_work_needs_cpu());
	trace_length = 0;
	assert(queue_work(unbound, &self_waiting));
	flush_work(&self_waiting);
	assert(trace_length == 2 && !memcmp(trace, "IS", 2));
	assert(!irq_work_needs_cpu() && !irq_work_is_busy(&raised));
	destroy_workqueue(unbound);
}

void stela_linux_sync_test(void)
{
	rwsem();
	workers();
	irq_works();
	worker_irq_works();
}

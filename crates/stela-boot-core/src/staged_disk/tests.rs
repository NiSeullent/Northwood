use super::*;
use alloc::{rc::Rc, string::String, vec};
use core::cell::RefCell;
use stela_drivers::block::{Features, Request, Submission, Transport};

const BLOCK: usize = 512;

std::thread_local! {
    /// The calls of this test's [`wake`].
    static WAKES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

fn wake() {
    WAKES.with(|wakes| wakes.set(wakes.get() + 1));
}

fn wakes() -> usize {
    WAKES.with(core::cell::Cell::get)
}

/// A host disk with a write cache whose commands complete only when a test
/// says so: completed writes are cached until a completed flush makes them
/// durable.
#[derive(Default)]
struct Disk {
    /// 0: it runs each command where it is asked.
    depth: usize,
    /// Complete every running command at each finish call.
    auto: bool,
    /// Refuse every start.
    refuse: bool,
    /// Show no posted completion, as a disk whose signal was lost.
    hidden: bool,
    /// Run at each finish call before it looks, as a driver's own work
    /// (an interrupt, a lock's release) may.
    during_finish: Option<Box<dyn Fn()>>,
    /// A finish collects every completion; `finished` takes the rest.
    together: bool,
    /// Finish calls that reached the disk.
    looks: usize,
    running: Vec<(u32, Command)>,
    completed: BTreeMap<u32, Finished>,
    next: u32,
    cached: BTreeMap<u64, u8>,
    durable: BTreeMap<u64, u8>,
    /// Each command as the disk took it.
    taken: Vec<&'static str>,
    /// Calls the disk saw, to tell a model's from the host side's.
    calls: usize,
}

impl Disk {
    fn run(&mut self, command: Command) -> Finished {
        match command {
            Command::Write { lba, data, .. } => {
                for (index, block) in data.chunks(BLOCK).enumerate() {
                    self.cached.insert(lba + index as u64, block[0]);
                }
                Ok(None)
            }
            Command::Read { lba, bytes } => {
                let mut data = vec![0; bytes];
                for (index, block) in data.chunks_mut(BLOCK).enumerate() {
                    let at = lba + index as u64;
                    let value = self.cached.get(&at).or(self.durable.get(&at));
                    block.fill(value.copied().unwrap_or(0));
                }
                Ok(Some(data))
            }
            Command::Flush => {
                let cached = core::mem::take(&mut self.cached);
                self.durable.extend(cached);
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn name(command: &Command) -> &'static str {
        match command {
            Command::Read { .. } => "read",
            Command::Write { .. } => "write",
            Command::Flush => "flush",
            Command::Discard { .. } => "discard",
            Command::WriteZeroes { .. } => "zeroes",
        }
    }

    /// Complete the oldest running command.
    fn complete_oldest(&mut self) {
        let (tag, command) = self.running.remove(0);
        let result = self.run(command);
        self.completed.insert(tag, result);
    }

    fn complete_all(&mut self) {
        while !self.running.is_empty() {
            self.complete_oldest();
        }
    }
}

struct Device {
    disk: Rc<RefCell<Disk>>,
    identity: Identity,
}

impl BlockDevice for Device {
    fn identity(&self) -> &Identity {
        &self.identity
    }
    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        let mut disk = self.disk.borrow_mut();
        disk.calls += 1;
        let data = disk.run(Command::Read {
            lba,
            bytes: buffer.len(),
        })?;
        buffer.copy_from_slice(&data.unwrap());
        Ok(())
    }
    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        let mut disk = self.disk.borrow_mut();
        disk.calls += 1;
        disk.run(Command::Write {
            lba,
            data: data.to_vec(),
            fua: false,
        })
        .map(drop)
    }
    fn flush_cache(&mut self) -> Result<(), BlockError> {
        let mut disk = self.disk.borrow_mut();
        disk.calls += 1;
        disk.taken.push("flush");
        disk.run(Command::Flush).map(drop)
    }
    fn execute(&mut self, command: Command) -> Finished {
        let mut disk = self.disk.borrow_mut();
        disk.calls += 1;
        disk.taken.push(Disk::name(&command));
        disk.run(command)
    }
    fn depth(&self) -> usize {
        self.disk.borrow().depth
    }
    fn room(&self) -> bool {
        let disk = self.disk.borrow();
        disk.running.len() + disk.completed.len() < disk.depth
    }
    fn start(&mut self, command: Command) -> Result<u32, BlockError> {
        let mut disk = self.disk.borrow_mut();
        disk.calls += 1;
        if disk.refuse {
            return Err(BlockError::Protocol("refused"));
        }
        disk.taken.push(Disk::name(&command));
        let tag = disk.next;
        disk.next += 1;
        disk.running.push((tag, command));
        Ok(tag)
    }
    /// A completion waits, or one would at the next finish.
    fn posted(&self) -> bool {
        let disk = self.disk.borrow();
        !disk.hidden && (!disk.completed.is_empty() || disk.auto && !disk.running.is_empty())
    }
    fn finish(&mut self, tag: u32) -> Option<Finished> {
        let during = self.disk.borrow_mut().during_finish.take();
        if let Some(during) = &during {
            during();
        }
        let mut disk = self.disk.borrow_mut();
        disk.during_finish = during;
        disk.calls += 1;
        disk.looks += 1;
        if disk.auto {
            disk.complete_all();
        }
        disk.completed.remove(&tag)
    }
    fn reaps_together(&self) -> bool {
        self.disk.borrow().together
    }
    fn finished(&mut self, tag: u32) -> Option<Finished> {
        let mut disk = self.disk.borrow_mut();
        disk.calls += 1;
        disk.together.then(|| disk.completed.remove(&tag)).flatten()
    }
}

/// A staged disk whose host side a separate processor steps.
fn staged(depth: usize) -> (StagedDisk, DiskWorker, Rc<RefCell<Disk>>) {
    let (model, worker, disk) = unstepped(depth);
    worker.set_staged(true);
    (model, worker, disk)
}

/// A staged disk before any processor steps it.
fn unstepped(depth: usize) -> (StagedDisk, DiskWorker, Rc<RefCell<Disk>>) {
    let disk = Rc::new(RefCell::new(Disk {
        depth,
        ..Disk::default()
    }));
    let identity = Identity {
        transport: Transport::Ram,
        location: String::new(),
        model: String::new(),
        serial: String::from("TEST"),
        blocks: 1024,
        block_size: BLOCK as u32,
        read_only: false,
        features: Features {
            write_cache: true,
            ..Features::default()
        },
    };
    let device = Device {
        disk: Rc::clone(&disk),
        identity,
    };
    let (model, worker) = stage(Box::new(device), wake);
    (model, worker, disk)
}

/// A step that looks for completions only once the disk posts one.
fn stepped(worker: &DiskWorker) -> Step {
    worker.step(0, u64::MAX)
}

fn write(lba: u64, value: u8) -> Command {
    Command::Write {
        lba,
        data: vec![value; BLOCK],
        fua: false,
    }
}

#[test]
fn a_start_only_queues_and_a_result_waits_for_the_hosts_completion() {
    let (mut model, worker, disk) = staged(8);
    let tag = model.start(write(3, 7)).unwrap();
    // The model's calls never reach the host disk.
    assert_eq!(disk.borrow().calls, 0);
    assert!(worker.work() && !model.posted());
    assert_eq!(model.finish(tag), None);
    // The step gives the host the command; it has not completed.
    let step = stepped(&worker);
    assert_eq!((step.started, step.finished, step.work), (1, 0, true));
    assert_eq!(disk.borrow().running.len(), 1);
    assert!(!model.posted() && model.finish(tag).is_none());
    // Once the host completes it, a step posts the result.
    disk.borrow_mut().complete_all();
    let step = stepped(&worker);
    assert_eq!((step.started, step.finished, step.work), (0, 1, false));
    let calls = disk.borrow().calls;
    assert!(model.posted());
    assert_eq!(model.finish(tag), Some(Ok(None)));
    assert!(!model.posted() && !worker.work());
    assert_eq!(disk.borrow().calls, calls);
    assert!(!stepped(&worker).progressed());
}

#[test]
fn the_host_takes_commands_in_the_order_the_model_started_them() {
    let (mut model, worker, disk) = staged(8);
    model.start(write(1, 1)).unwrap();
    model
        .start(Command::Read {
            lba: 1,
            bytes: BLOCK,
        })
        .unwrap();
    model.start(Command::Flush).unwrap();
    model.start(Command::discard(9, 2)).unwrap();
    model.start(write(2, 2)).unwrap();
    stepped(&worker);
    assert_eq!(
        disk.borrow().taken,
        ["write", "read", "flush", "discard", "write"]
    );
}

#[test]
fn a_flush_completes_only_once_the_host_has_made_the_writes_durable() {
    let (mut model, worker, disk) = staged(8);
    let written = model.start(write(5, 9)).unwrap();
    stepped(&worker);
    disk.borrow_mut().complete_all();
    stepped(&worker);
    assert_eq!(model.finish(written), Some(Ok(None)));
    assert!(disk.borrow().durable.is_empty());
    let flush = model.start(Command::Flush).unwrap();
    stepped(&worker);
    // The host's flush runs: the model's waits.
    assert!(model.finish(flush).is_none());
    assert!(stepped(&worker).work);
    assert!(model.finish(flush).is_none());
    disk.borrow_mut().complete_all();
    stepped(&worker);
    assert_eq!(disk.borrow().durable.get(&5), Some(&9));
    assert_eq!(model.finish(flush), Some(Ok(None)));
}

#[test]
fn a_read_started_after_a_completed_write_sees_it() {
    let (mut model, worker, disk) = staged(8);
    let written = model.start(write(4, 0x5a)).unwrap();
    stepped(&worker);
    disk.borrow_mut().complete_all();
    stepped(&worker);
    assert_eq!(model.finish(written), Some(Ok(None)));
    let read = model
        .start(Command::Read {
            lba: 4,
            bytes: BLOCK,
        })
        .unwrap();
    stepped(&worker);
    disk.borrow_mut().complete_all();
    stepped(&worker);
    assert_eq!(model.finish(read), Some(Ok(Some(vec![0x5a; BLOCK]))));
}

#[test]
fn results_reach_the_model_by_tag_in_whatever_order_the_host_completes() {
    let (mut model, worker, disk) = staged(8);
    let first = model.start(write(1, 1)).unwrap();
    let second = model.start(write(2, 2)).unwrap();
    stepped(&worker);
    {
        let mut disk = disk.borrow_mut();
        let newest = disk.running.pop().unwrap();
        disk.running.insert(0, newest);
        disk.complete_oldest();
    }
    stepped(&worker);
    assert!(model.finish(first).is_none());
    assert_eq!(model.finish(second), Some(Ok(None)));
    // A result taken stays taken.
    assert!(model.finish(second).is_none());
}

#[test]
fn a_step_is_bounded_by_its_starts_and_by_the_hosts_room() {
    let (mut model, worker, disk) = staged(DEPTH);
    for lba in 0..40 {
        model.start(write(lba, 1)).unwrap();
    }
    assert_eq!(stepped(&worker).started, STARTS);
    assert_eq!(stepped(&worker).started, STARTS);
    assert_eq!(disk.borrow().running.len(), 2 * STARTS);
    // A disk with room for fewer takes no more until some complete.
    let (mut model, worker, disk) = staged(3);
    for lba in 0..5 {
        model.start(write(lba, 1)).unwrap();
    }
    assert_eq!(stepped(&worker).started, 3);
    assert_eq!(stepped(&worker).started, 0);
    disk.borrow_mut().complete_oldest();
    let step = stepped(&worker);
    assert_eq!((step.started, step.finished), (1, 1));
}

#[test]
fn a_disk_that_queues_nothing_runs_one_command_a_step() {
    let (mut model, worker, disk) = staged(0);
    let first = model.start(write(1, 3)).unwrap();
    let second = model
        .start(Command::Read {
            lba: 1,
            bytes: BLOCK,
        })
        .unwrap();
    let step = stepped(&worker);
    assert_eq!((step.started, step.work), (1, true));
    assert_eq!(model.finish(first), Some(Ok(None)));
    assert!(model.finish(second).is_none());
    stepped(&worker);
    assert_eq!(model.finish(second), Some(Ok(Some(vec![3; BLOCK]))));
    assert_eq!(disk.borrow().taken, ["write", "read"]);
}

#[test]
fn the_model_owns_at_most_its_depth_until_it_takes_results() {
    let (mut model, worker, disk) = staged(DEPTH);
    let tags: Vec<u32> = (0..DEPTH as u64)
        .map(|lba| model.start(write(lba, 1)).unwrap())
        .collect();
    assert!(!model.room());
    assert!(model.start(write(0, 1)).is_err());
    while stepped(&worker).started != 0 {}
    disk.borrow_mut().complete_all();
    stepped(&worker);
    // Completed but untaken results still count.
    assert!(!model.room());
    assert_eq!(model.finish(tags[0]), Some(Ok(None)));
    assert!(model.room());
}

#[test]
fn a_refused_start_is_its_commands_result() {
    let (mut model, worker, disk) = staged(8);
    disk.borrow_mut().refuse = true;
    let tag = model.start(write(1, 1)).unwrap();
    let step = stepped(&worker);
    assert_eq!((step.started, step.finished, step.work), (1, 0, false));
    assert_eq!(
        model.finish(tag),
        Some(Err(BlockError::Protocol("refused")))
    );
}

#[test]
fn a_synchronous_call_runs_after_every_queued_command() {
    let (mut model, _worker, disk) = staged(8);
    disk.borrow_mut().auto = true;
    let written = model.start(write(6, 4)).unwrap();
    // A shutdown's flush: the queued write reaches the host first and is
    // made durable with it.
    model.flush().unwrap();
    assert_eq!(disk.borrow().taken, ["write", "flush"]);
    assert_eq!(disk.borrow().durable.get(&6), Some(&4));
    assert_eq!(model.finish(written), Some(Ok(None)));
}

#[test]
fn a_step_while_another_processor_steps_does_nothing() {
    let (mut model, worker, disk) = staged(8);
    model.start(write(1, 1)).unwrap();
    let held = worker.staging.host.try_lock().unwrap();
    let step = stepped(&worker);
    assert!(step.held && step.work && !step.progressed());
    assert!(disk.borrow().taken.is_empty());
    // Finding it held leaves it held.
    assert!(worker.staging.host.try_lock().is_none());
    assert!(stepped(&worker).held);
    drop(held);
    assert_eq!(stepped(&worker).started, 1);
}

#[test]
fn the_block_layers_requests_complete_through_the_staged_disk() {
    let (mut model, worker, disk) = staged(8);
    // A durable write on a cache without FUA is a write, then a flush once
    // the write has completed.
    let request = Request::Write {
        lba: 8,
        data: vec![2; 2 * BLOCK],
        durable: true,
    };
    let mut submission = Submission::new(request, model.identity()).unwrap();
    assert!(submission.poll(&mut model).is_none());
    stepped(&worker);
    assert_eq!(disk.borrow().taken, ["write"]);
    disk.borrow_mut().complete_all();
    stepped(&worker);
    // The flush starts only once the write's result reached the model.
    assert!(submission.poll(&mut model).is_none());
    stepped(&worker);
    assert_eq!(disk.borrow().taken, ["write", "flush"]);
    assert!(disk.borrow().durable.is_empty());
    disk.borrow_mut().complete_all();
    stepped(&worker);
    assert_eq!(submission.poll(&mut model), Some(Ok(None)));
    assert_eq!(disk.borrow().durable.get(&9), Some(&2));
}

#[test]
fn the_model_asks_for_a_step_once_until_one_is_taken() {
    let (mut model, worker, disk) = staged(8);
    let first = model.start(write(1, 1)).unwrap();
    let second = model.start(write(2, 2)).unwrap();
    assert_eq!(wakes(), 1);
    // Looks that find nothing ask no more until a step is taken.
    assert!(model.finish(first).is_none());
    assert_eq!(wakes(), 1);
    stepped(&worker);
    assert!(model.finish(first).is_none());
    assert_eq!(wakes(), 2);
    disk.borrow_mut().complete_oldest();
    stepped(&worker);
    // A look that finds another command's result still asks for the rest.
    assert!(model.finish(second).is_none());
    assert_eq!(wakes(), 3);
    assert_eq!(model.finish(first), Some(Ok(None)));
    disk.borrow_mut().complete_all();
    stepped(&worker);
    assert_eq!(model.finish(second), Some(Ok(None)));
    // With no work left, a look asks for nothing.
    assert!(model.finish(second).is_none());
    assert_eq!(wakes(), 3);
}

/// Poll `submission` as a caller that waits for it does, at most `polls`
/// times; its result.
fn wait(submission: &mut Submission, model: &mut StagedDisk, polls: usize) -> Option<Finished> {
    (0..polls).find_map(|_| submission.poll(model))
}

#[test]
fn a_wait_with_no_stepping_processor_steps_the_host_side_itself() {
    // Before the PEs run (a machine state's load), nothing else steps.
    let (mut model, _worker, disk) = unstepped(8);
    disk.borrow_mut().auto = true;
    let request = Request::Write {
        lba: 3,
        data: vec![7; BLOCK],
        durable: true,
    };
    let mut submission = Submission::new(request, model.identity()).unwrap();
    assert_eq!(wait(&mut submission, &mut model, 100), Some(Ok(None)));
    assert_eq!(disk.borrow().taken, ["write", "flush"]);
    assert_eq!(disk.borrow().durable.get(&3), Some(&7));
    let read = Request::Read {
        lba: 3,
        bytes: BLOCK,
    };
    let mut submission = Submission::new(read, model.identity()).unwrap();
    assert_eq!(
        wait(&mut submission, &mut model, 100),
        Some(Ok(Some(vec![7; BLOCK])))
    );
    assert_eq!(wakes(), 0);
}

#[test]
fn staged_the_model_never_steps_and_unstaged_again_it_does() {
    let (mut model, worker, disk) = staged(8);
    disk.borrow_mut().auto = true;
    let request = Request::Write {
        lba: 1,
        data: vec![1; BLOCK],
        durable: false,
    };
    let mut submission = Submission::new(request, model.identity()).unwrap();
    // A stepping processor exists: the model's polls only ask for it.
    assert_eq!(wait(&mut submission, &mut model, 100), None);
    assert_eq!(disk.borrow().calls, 0);
    assert_eq!(wakes(), 1);
    // Once the PEs stop, the model's own polls finish the work.
    worker.set_staged(false);
    assert_eq!(wait(&mut submission, &mut model, 100), Some(Ok(None)));
    assert!(!worker.work());
}

#[test]
fn a_step_with_nothing_posted_reaches_no_command_until_a_signal_is_overdue() {
    let (mut model, worker, disk) = staged(8);
    model.start(write(1, 1)).unwrap();
    assert_eq!(worker.step(0, 100).started, 1);
    let calls = disk.borrow().calls;
    // In flight, nothing posted: no finish reaches the disk.
    assert!(!worker.step(50, 100).progressed());
    assert_eq!(disk.borrow().calls, calls);
    // A completion whose posting the disk does not show is found once a
    // look is overdue.
    {
        let mut disk = disk.borrow_mut();
        disk.hidden = true;
        disk.complete_oldest();
    }
    assert!(!worker.step(99, 100).progressed());
    assert_eq!(disk.borrow().calls, calls);
    let found = worker.step(100, 100);
    assert_eq!(found.finished, 1);
    // Nothing queued and nothing in flight: no work, no step.
    assert!(!worker.work());
}

#[test]
fn a_step_entered_again_during_a_step_leaves_the_first_ones_commands_alone() {
    let (mut model, worker, disk) = staged(8);
    for lba in 0..3 {
        model.start(write(lba, 1)).unwrap();
    }
    stepped(&worker);
    disk.borrow_mut().complete_all();
    // Steps the same processor enters from inside a host call: each must
    // find the host side held. (A failed attempt that released the lock let
    // the second one in to reap the first one's commands: the outer reap
    // then removed past the end of its list.)
    let inner = worker.clone();
    let found = Rc::new(RefCell::new(Vec::new()));
    let seen = Rc::clone(&found);
    disk.borrow_mut().during_finish = Some(Box::new(move || {
        for _ in 0..2 {
            seen.borrow_mut().push(inner.step(0, 0).held);
        }
    }));
    let step = stepped(&worker);
    assert_eq!((step.finished, step.work), (3, false));
    assert!(found.borrow().iter().all(|&held| held));
    assert_eq!(found.borrow().len(), 6);
    disk.borrow_mut().during_finish = None;
    for tag in 0..3 {
        assert_eq!(model.finish(tag), Some(Ok(None)));
    }
}

#[test]
fn completions_that_arrive_during_a_reap_are_taken_by_it() {
    let (mut model, worker, disk) = staged(8);
    for lba in 0..3 {
        model.start(write(lba, 1)).unwrap();
    }
    stepped(&worker);
    // The first finish completes every running command, as a driver that
    // reaps its whole completion queue at once does.
    disk.borrow_mut().auto = true;
    let step = stepped(&worker);
    assert_eq!((step.finished, step.work), (3, false));
    assert!(disk.borrow().running.is_empty());
    for tag in 0..3 {
        assert_eq!(model.finish(tag), Some(Ok(None)));
    }
}

#[test]
fn a_step_with_nothing_posted_and_nothing_to_start_reaches_no_disk() {
    let (mut model, worker, disk) = staged(8);
    model.start(write(1, 1)).unwrap();
    stepped(&worker);
    let calls = disk.borrow().calls;
    // Running, nothing completed: the step neither looks nor starts.
    let step = stepped(&worker);
    assert_eq!((step.started, step.finished), (0, 0));
    assert_eq!(disk.borrow().calls, calls);
}

#[test]
fn one_finish_reaps_every_completion_of_a_disk_that_collects_them_together() {
    let (mut model, worker, disk) = staged(8);
    disk.borrow_mut().together = true;
    let tags: Vec<u32> = (0..4)
        .map(|lba| model.start(write(lba, 1)).unwrap())
        .collect();
    stepped(&worker);
    disk.borrow_mut().complete_all();
    let step = stepped(&worker);
    assert_eq!(step.finished, 4);
    assert_eq!(disk.borrow().looks, 1);
    for tag in tags {
        assert_eq!(model.finish(tag), Some(Ok(None)));
    }
}

/// 20 us of the 24 MHz counter.
const WINDOW: u64 = 480;

#[test]
fn a_fruitless_look_excuses_the_next_ones_only_until_the_window_ends() {
    let mut gate = LookGate::default();
    assert!(!gate.skip(true, 7, 1000, WINDOW), "never looked yet");
    gate.looked(false, 7, 1000);
    assert!(gate.skip(true, 7, 1000 + WINDOW - 1, WINDOW));
    // The doorbell count moved: a signal came, so look.
    assert!(!gate.skip(true, 8, 1001, WINDOW));
    // No core took the signal, but the window passed: the completion may
    // be in memory already.
    assert!(!gate.skip(true, 7, 1000 + WINDOW, WINDOW));
    // A disk that sends no signal is always looked at.
    assert!(!gate.skip(false, 7, 1001, WINDOW));
}

#[test]
fn a_look_that_found_a_completion_or_a_new_command_excuses_nothing() {
    let mut gate = LookGate::default();
    gate.looked(true, 3, 50);
    assert!(!gate.skip(true, 3, 51, WINDOW));
    gate.looked(false, 3, 60);
    assert!(gate.skip(true, 3, 61, WINDOW));
    gate.reset();
    assert!(!gate.skip(true, 3, 62, WINDOW));
    // A counter that wrapped since the look ends the excuse too.
    gate.looked(false, 3, u64::MAX - 10);
    assert!(gate.skip(true, 3, u64::MAX, WINDOW));
    assert!(!gate.skip(true, 3, WINDOW, WINDOW));
}

#[test]
fn a_busy_core_steps_at_a_disk_signal_without_waiting_for_the_fallback() {
    const FALLBACK: u64 = 24_000;
    // A waiting core always steps.
    assert!(step_due(true, 10, 10, FALLBACK, 4, 4));
    // A busy core leaves a disk stepped moments ago to the others...
    assert!(!step_due(false, 10, 5, FALLBACK, 4, 4));
    // ...unless a completion's signal came since that step.
    assert!(step_due(false, 10, 5, FALLBACK, 5, 4));
    // ...or nobody stepped for the fallback time.
    assert!(step_due(false, 5 + FALLBACK, 5, FALLBACK, 4, 4));
}

#[test]
fn a_completion_posted_after_a_fruitless_step_is_taken_by_the_next_step() {
    let (mut model, worker, disk) = staged(4);
    let tag = model.start(write(9, 3)).unwrap();
    // The step gives the host its command; nothing is posted yet.
    assert_eq!(stepped(&worker).started, 1);
    assert_eq!(stepped(&worker).finished, 0);
    assert_eq!(model.finish(tag), None);
    disk.borrow_mut().complete_all();
    // The completion is in memory: the next step takes it at once, and the
    // model's look finds the result without reaching the disk itself.
    assert_eq!(stepped(&worker).finished, 1);
    assert!(model.posted());
    let calls = disk.borrow().calls;
    assert_eq!(model.finish(tag), Some(Ok(None)));
    assert_eq!(disk.borrow().calls, calls);
}

/// A step whose time is spent ends after the call it is in: the commands
/// and completions left keep for the next steps, which take them in order.
#[test]
fn a_step_whose_time_is_spent_leaves_the_rest_to_the_next() {
    let (mut model, worker, disk) = staged(DEPTH);
    let tags: Vec<u32> = (0..4)
        .map(|lba| model.start(write(lba, 1)).unwrap())
        .collect();
    // Out of time from the first look: one start a step.
    let mut over = || true;
    let step = worker.step_within(0, u64::MAX, &mut over);
    assert_eq!((step.started, step.work), (1, true));
    assert_eq!(disk.borrow().running.len(), 1);
    assert_eq!(worker.step_within(0, u64::MAX, &mut over).started, 1);
    // Time for two calls: two more starts.
    let mut calls = 0;
    let mut two = || {
        calls += 1;
        calls > 1
    };
    assert_eq!(worker.step_within(0, u64::MAX, &mut two).started, 2);
    assert_eq!(disk.borrow().taken.len(), 4);
    // Completions, too, are taken one a call while out of time.
    disk.borrow_mut().auto = true;
    disk.borrow_mut().complete_oldest();
    disk.borrow_mut().complete_oldest();
    let step = worker.step_within(0, 0, &mut over);
    assert_eq!((step.started, step.finished), (0, 1));
    let mut finished = 0;
    for _ in 0..8 {
        finished += worker.step_within(0, 0, &mut over).finished;
    }
    assert_eq!(finished, 3);
    assert!(!worker.work());
    for tag in tags {
        assert!(model.finish(tag).unwrap().is_ok());
    }
}

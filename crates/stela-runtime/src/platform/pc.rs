//! What the mount keeps on the host PC beside its drivers
//! (`stela_drivers`), after ExitBootServices.
//!
//! [`storage::open`] selects, among the disks the drivers found, the one
//! disk and GPT partition that `stela.cfg` names and returns a
//! [`gpt::PartitionDevice`] that cannot reach any other block; [`fat32`]
//! reads files from such a partition. [`boot_log`] keeps the mount's output
//! on the machine-state partition, and `stela` reads it back;
//! [`machine_state`] keeps that partition's layout in the tail of the Apple
//! region instead. [`acpi`] reads the firmware's ACPI tables, through which
//! [`power`] restarts and powers off the host and reads its power button.
//! [`display`] chooses the graphics mode the screen runs in while boot
//! services still allow it.
pub mod acpi;
pub mod board_flash;
pub mod boot_log;
pub mod display;
pub mod failure_record;
pub mod fat32;
pub mod fdr_store;
pub mod gpt;
pub mod machine_state;
pub mod power;
pub mod rtc_anchor;
pub mod storage;
pub mod swap;

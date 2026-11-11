//! What the boot screen says when no GPU started: which NVIDIA GPU was
//! refused and why, in words a user can act on from another computer.
use alloc::{format, string::String};
use stela_loader::host_firmware::recipes::nvidia_chip;

/// The NVIDIA generations Stela's GPU driver runs (Turing, Ampere, Ada and
/// Blackwell: every chip `recipes::nvidia_chip` names), as people know them.
pub const SUPPORTED: &str = "an NVIDIA GeForce GTX 16 or RTX 20, 30, 40 or 50 series GPU";

/// Why an NVIDIA GPU did not start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The USB holds no firmware for its chip.
    MissingFirmware,
    /// Driver policy disabled NVModern.
    PolicyDisabled,
    /// Driver policy blocked this PCI function.
    PolicyBlocked,
    /// The forced driver was disabled or did not admit this PCI function.
    PolicyForceRejected,
    /// Anything else; the boot log names it.
    Other,
}

/// The screen line for the NVIDIA GPU `device` at PCI `address` that did
/// not start for `reason`.
pub fn refused(address: &str, device: u16, reason: Reason) -> String {
    let policy = match reason {
        Reason::PolicyDisabled => Some("NVModern is disabled by nwconfig"),
        Reason::PolicyBlocked => Some("this PCI device is blocked by nwconfig"),
        Reason::PolicyForceRejected => {
            Some("the nwconfig forced driver is disabled or incompatible")
        }
        Reason::MissingFirmware | Reason::Other => None,
    };
    if let Some(policy) = policy {
        return format!(
            "the NVIDIA GPU at {address} (device {device:04x}) was left unused: {policy}"
        );
    }
    match (nvidia_chip(device), reason) {
        (None, _) => format!(
            "the NVIDIA GPU at {address} (device {device:04x}) is not supported: Stela needs \
             {SUPPORTED}"
        ),
        (Some(chip), Reason::MissingFirmware) => format!(
            "the USB has no firmware for this PC's NVIDIA {chip} GPU. On another computer, run \
             `stela setup host-firmware fetch --gpu {chip} --usb USB` (USB: this USB's STELA \
             volume)"
        ),
        (Some(chip), _) => format!(
            "the NVIDIA {chip} GPU at {address} did not start; the boot log on this USB says why"
        ),
    }
}

/// The screen line for a PC without an NVIDIA GPU. The firmware framebuffer
/// can display the boot UI, while graphical macOS requires a Metal device.
pub fn none() -> String {
    format!("this PC has no NVIDIA GPU. Stela's Metal driver needs {SUPPORTED} for graphical macOS")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_gpu_is_named_with_its_fix() {
        let missing = refused("01:00.0", 0x2484, Reason::MissingFirmware);
        assert!(missing.contains("NVIDIA ga104 GPU"), "{missing}");
        assert!(
            missing.contains("`stela setup host-firmware fetch --gpu ga104 --usb USB`"),
            "{missing}"
        );
        let other = refused("03:00.0", 0x2b85, Reason::Other);
        assert_eq!(
            other,
            "the NVIDIA gb202 GPU at 03:00.0 did not start; the boot log on this USB says why"
        );
    }

    #[test]
    fn an_older_gpu_or_none_names_what_is_supported() {
        // A GeForce GTX 1080 (Pascal, gp104).
        let pascal = refused("01:00.0", 0x1b80, Reason::MissingFirmware);
        assert_eq!(
            pascal,
            "the NVIDIA GPU at 01:00.0 (device 1b80) is not supported: Stela needs an NVIDIA \
             GeForce GTX 16 or RTX 20, 30, 40 or 50 series GPU"
        );
        assert!(none().starts_with("this PC has no NVIDIA GPU."));
        assert!(none().contains("for graphical macOS"));
    }

    #[test]
    fn an_nvidia_policy_refusal_names_the_device_and_rule() {
        for (device, reason, detail) in [
            (0x2684, Reason::PolicyDisabled, "NVModern is disabled"),
            (0x1b80, Reason::PolicyBlocked, "this PCI device is blocked"),
            (
                0x2684,
                Reason::PolicyForceRejected,
                "the nwconfig forced driver is disabled or incompatible",
            ),
        ] {
            let line = refused("01:00.0", device, reason);
            assert!(line.contains(&format!("01:00.0 (device {device:04x})")));
            assert!(line.contains(detail), "{line}");
        }
    }
}

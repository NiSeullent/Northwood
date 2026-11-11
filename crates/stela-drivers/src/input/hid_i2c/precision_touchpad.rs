// SPDX-License-Identifier: GPL-2.0-or-later
//! Descriptor-qualified mouse mode for a Precision Touchpad. The native
//! peripheral carries mouse events, so select the device's own mouse report.
//! https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/touchpad-configuration-collection

use crate::usb::hid::{Descriptor, ReportType, Translator};

const TOUCHPAD: u32 = 0x000d_0005;
const CONFIGURATION: u32 = 0x000d_000e;
const INPUT_MODE: u32 = 0x000d_0052;
const MOUSE: u32 = 0x0001_0002;
const X: u32 = 0x0001_0030;
const Y: u32 = 0x0001_0031;
pub(super) const MOUSE_MODE: u8 = 0;

fn application(descriptor: &Descriptor, mut collection: Option<usize>) -> Option<u32> {
    while let Some(index) = collection {
        let entry = descriptor.collections.get(index)?;
        if entry.kind == 1 {
            return entry.parent.is_none().then_some(entry.usage);
        }
        collection = entry.parent;
    }
    None
}

pub(super) fn mouse_mode_report(translator: &Translator) -> Option<u8> {
    let descriptor = translator.descriptor()?;
    let touchpad = descriptor
        .reports
        .iter()
        .filter(|report| report.kind == ReportType::Input)
        .flat_map(|report| &report.fields)
        .any(|field| application(descriptor, field.collection) == Some(TOUCHPAD));
    if !touchpad {
        return None;
    }

    let mouse = descriptor.layout(ReportType::Input, translator.relative_pointer_report()?)?;
    if ![X, Y].into_iter().all(|axis| {
        mouse.fields.iter().any(|field| {
            field.flags & 0x7 == 0x6
                && application(descriptor, field.collection) == Some(MOUSE)
                && field
                    .usages
                    .iter()
                    .take(field.count)
                    .any(|&usage| usage == axis)
        })
    }) {
        return None;
    }

    let mut modes = descriptor.reports.iter().filter(|report| {
        report.kind == ReportType::Feature
            && report.fields.iter().any(|field| {
                application(descriptor, field.collection) == Some(CONFIGURATION)
                    && field.usages.contains(&INPUT_MODE)
            })
    });
    let mode = modes.next()?;
    let [field] = mode.fields.as_slice() else {
        return None;
    };
    // Write only the dedicated one-byte feature, with no other fields or
    // padding to guess. The transport inserts the descriptor's report ID.
    (modes.next().is_none()
        && mode.bits == 8
        && field.offset == 0
        && field.size == 8
        && field.count == 1
        && field.flags == 0x02
        && field.minimum == 0
        && field.maximum >= 3
        && field.usages == [INPUT_MODE])
    .then_some(mode.id)
}

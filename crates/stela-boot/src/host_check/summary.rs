// SPDX-License-Identifier: GPL-2.0-or-later
//! Keep the boot-readiness checks visible after the live diagnostic console.

use super::{Machine, Verdict};
use crate::{console, ui};
use alloc::{format, string::String};

const FIRST: &[&str] = &[
    "profile",
    "memory",
    "devices",
    "install-target",
    "graphics",
    "input",
    "lpss-pci",
    "swap",
    "network",
];

pub(super) fn show(machine: &Machine, restart: &str) {
    let Some((screen, font)) = machine.screen.zip(console::font::console()) else {
        return;
    };
    let (glyph_width, glyph_height) = (font.width, font.height);
    let mut page = ui::Canvas::new(screen, font);
    let (width, height) = page.size();
    let margin = 24;
    let columns = width.saturating_sub(2 * margin) / glyph_width;
    if columns < 28 || height < 200 {
        return;
    }
    if !console::release() {
        return;
    }
    page.text(margin, 20, "Diagnostics", 2, ui::TEXT);
    let footer = height - glyph_height - 12;
    let mut y = 64;
    let checks = FIRST
        .iter()
        .filter_map(|&name| machine.checks.iter().find(|check| check.name == name))
        .chain(
            machine
                .checks
                .iter()
                .filter(|check| !FIRST.contains(&check.name)),
        );
    for check in checks {
        let colour = match check.verdict {
            Verdict::Pass => ui::TEXT,
            Verdict::Fail => ui::ORANGE,
            Verdict::Skip => ui::DIM,
        };
        let text = format!("{check}");
        let mut line = String::new();
        for word in text.split_ascii_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > columns {
                if y + glyph_height > footer - 8 {
                    break;
                }
                page.text(margin, y, &line, 1, colour);
                y += glyph_height + 2;
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if y + glyph_height > footer - 8 {
            break;
        }
        page.text(margin, y, &line, 1, colour);
        y += glyph_height + 6;
    }
    let action = format!("{restart}Hold the power button to turn off.");
    page.text(margin, footer, &action, 1, ui::TEXT);
    page.show();
}

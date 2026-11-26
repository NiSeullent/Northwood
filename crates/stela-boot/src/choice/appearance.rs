// SPDX-License-Identifier: GPL-2.0-or-later

//! The graphical startup picker uses the same rectangles for paint and input.
use super::{ChoiceKind, Choices, Item, Menu};
use crate::ui::{self, Canvas};
use alloc::format;
use stela_boot_core::screen::picker::{Layout, Rect};

impl Item {
    fn appearance(&self) -> (&'static str, u32) {
        match self {
            Self::Choice(_, ChoiceKind::Mac) => ("Startup disk", ui::BLUE),
            Self::Choice(_, ChoiceKind::Setup) => ("Installer", ui::ORANGE),
            Self::Choice(_, ChoiceKind::Update) => ("System update", ui::BLUE),
            Self::Choice(_, ChoiceKind::Other) => ("Boot option", ui::DIM),
            Self::Firmware(_) => ("Firmware entry", ui::DIM),
            Self::Diagnostics => ("Hardware report", ui::BLUE),
        }
    }

    fn action(&self) -> &'static str {
        match self {
            Self::Choice(_, ChoiceKind::Setup) => "Open installer",
            Self::Choice(_, ChoiceKind::Update) => "Start update",
            Self::Firmware(_) => "Restart to boot",
            Self::Diagnostics => "Run diagnostics",
            Self::Choice(_, _) => "Continue",
        }
    }

    fn draw_icon(&self, canvas: &mut Canvas, middle: usize, top: usize, compact: bool) {
        let (_, accent) = self.appearance();
        let (width, height) = if compact { (50, 26) } else { (62, 38) };
        let left = middle - width / 2;
        canvas.panel(left, top + 3, width, height, 7, 0x3d485c);
        canvas.panel(left, top, width, height, 7, 0x7f8ba0);
        canvas.panel(left + 2, top + 2, width - 4, height - 7, 6, 0xc1cbd9);
        canvas.fill(left + 7, top + height - 7, width - 14, 1, 0x566175);
        canvas.panel(left + width - 13, top + height - 5, 6, 2, 1, accent);

        let center = top + (height - 6) / 2;
        let shade = 0x33425b;
        match self {
            Self::Choice(_, ChoiceKind::Setup | ChoiceKind::Update) => {
                canvas.fill(middle - 1, center - 6, 2, 9, shade);
                for row in 0..4 {
                    canvas.fill(middle - (4 - row), center + row, 8 - 2 * row, 1, shade);
                }
            }
            Self::Firmware(_) => {
                canvas.panel(middle - 7, center - 6, 14, 9, 2, shade);
                canvas.fill(middle - 5, center - 4, 10, 5, 0xc1cbd9);
                canvas.fill(middle - 4, center + 4, 8, 1, shade);
            }
            Self::Diagnostics => {
                canvas.fill(middle - 8, center, 16, 1, shade);
                canvas.fill(middle - 2, center - 4, 2, 8, shade);
                canvas.fill(middle + 3, center - 2, 2, 5, shade);
            }
            _ => {
                canvas.fill(middle - 9, center - 3, 18, 2, shade);
                canvas.fill(middle - 6, center + 2, 12, 1, shade);
            }
        }
    }
}

fn caption(canvas: &mut Canvas, middle: usize, y: usize, text: &str, width: usize, colour: u32) {
    let capacity = width / canvas.text_width(" ", 1);
    if let Some((end, _)) = text.char_indices().nth(capacity) {
        if capacity > 3 {
            let end = text
                .char_indices()
                .nth(capacity - 3)
                .map_or(end, |(i, _)| i);
            canvas.text_centred(middle, y, &format!("{}...", &text[..end]), 1, colour);
        }
    } else {
        canvas.text_centred(middle, y, text, 1, colour);
    }
}

fn card_name(canvas: &mut Canvas, middle: usize, y: usize, name: &str, width: usize) {
    let capacity = width / canvas.text_width(" ", 1);
    if let Some((end, _)) = name.char_indices().nth(capacity)
        && let Some(space) = name[..end].rfind(' ')
        && space != 0
    {
        caption(canvas, middle, y, &name[..space], width, ui::TEXT);
        caption(
            canvas,
            middle,
            y + 15,
            name[space + 1..].trim_start(),
            width,
            ui::TEXT,
        );
    } else {
        caption(canvas, middle, y + 7, name, width, ui::TEXT);
    }
}

fn block(canvas: &mut Canvas, area: Rect, radius: usize, colour: u32) {
    canvas.panel(area.x, area.y, area.width, area.height, radius, colour);
}

fn page_button(canvas: &mut Canvas, button: Option<Rect>, glyph: &str) {
    if let Some(area) = button {
        block(canvas, area, 9, ui::CARD);
        canvas.text_centred(area.x + area.width / 2, area.y + 8, glyph, 1, ui::TEXT);
    }
}

impl Menu<'_> {
    pub(super) fn draw(&self, canvas: &mut Canvas, layout: Layout) {
        let (width, height) = canvas.size();
        canvas.fill(0, 0, width, height, ui::BACKGROUND);
        if layout.compact {
            canvas.text_centred(width / 2, 16, ui::NAME, 1, ui::TEXT);
        } else {
            canvas.header();
            canvas.text_centred(width / 2, 71, "Startup options", 1, ui::DIM);
        }
        for index in layout.entries() {
            let item = &self.items[index];
            let area = layout.card(index);
            let middle = area.x + area.width / 2;
            let selected = index == self.selected;
            block(canvas, area, 15, if selected { ui::BLUE } else { ui::LINE });
            let border = if selected { 2 } else { 1 };
            canvas.panel(
                area.x + border,
                area.y + border,
                area.width - border * 2,
                area.height - border * 2,
                15 - border,
                if selected { ui::RAISED } else { ui::CARD },
            );
            let (icon, name, kind) = if layout.compact {
                (10, 40, 72)
            } else {
                (18, 63, 104)
            };
            let label_width = area.width - 16;
            item.draw_icon(canvas, middle, area.y + icon, layout.compact);
            card_name(canvas, middle, area.y + name, self.name(item), label_width);
            caption(
                canvas,
                middle,
                area.y + kind,
                item.appearance().0,
                label_width,
                ui::DIM,
            );
            if let Some(key) = Choices::key(index) {
                let mut utf8 = [0; 4];
                canvas.text(
                    area.x + 9,
                    area.y + 6,
                    key.encode_utf8(&mut utf8),
                    1,
                    ui::DIM,
                );
            }
        }
        page_button(canvas, layout.previous, "<");
        page_button(canvas, layout.next, ">");

        let action = layout.action;
        block(canvas, action, 12, ui::BLUE);
        caption(
            canvas,
            width / 2,
            action.y + 10,
            self.items[self.selected].action(),
            action.width - 10,
            ui::BACKGROUND,
        );

        let countdown;
        let (status, colour) = if let Some(notice) = self.notice {
            (notice, ui::ORANGE)
        } else if let Some(seconds) = self.countdown {
            countdown = format!("Starting in {seconds}s");
            (countdown.as_str(), ui::DIM)
        } else if layout.compact {
            ("Arrows / Enter", ui::DIM)
        } else {
            ("Arrow keys to select     Enter to start", ui::DIM)
        };
        caption(canvas, width / 2, layout.footer, status, width - 24, colour);
    }
}

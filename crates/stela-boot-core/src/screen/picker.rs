// SPDX-License-Identifier: GPL-2.0-or-later

//! Shared geometry for drawing and hit-testing the firmware startup picker.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub fn contains(self, (x, y): (usize, usize)) -> bool {
        x >= self.x && y >= self.y && x - self.x < self.width && y - self.y < self.height
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    Select(usize),
    Start,
}

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub compact: bool,
    pub first: usize,
    pub count: usize,
    pub action: Rect,
    pub previous: Option<Rect>,
    pub next: Option<Rect>,
    pub footer: usize,
    first_card: Rect,
    gap: usize,
}

impl Layout {
    pub fn new((width, height): (usize, usize), items: usize, selected: usize) -> Option<Self> {
        if items == 0 || selected >= items || width < 200 || height < 220 {
            return None;
        }
        let compact = height < 390 || width < 540;
        let (tile_width, tile_height, gap) = if compact {
            (140, 94, 8)
        } else {
            (168, 128, 16)
        };
        let control = 32;
        let available = width - 2 * (control + 6);
        let card_width = tile_width.min(available);
        let count = ((available + gap) / (card_width + gap))
            .clamp(1, 6)
            .min(items);
        let first = super::window(selected, items, count);
        let strip_width = count * card_width + (count - 1) * gap;
        let left = (width - strip_width) / 2;
        let top = ((height - (tile_height + 54)) / 2).max(if compact { 48 } else { 92 });
        let arrow_y = top + (tile_height - control) / 2;
        let action_width = 176.min(available);
        let action = Rect {
            x: (width - action_width) / 2,
            y: top + tile_height + 14,
            width: action_width,
            height: 36,
        };
        Some(Self {
            compact,
            first,
            count,
            action,
            previous: (first > 0).then(|| Rect {
                x: left - control - 4,
                y: arrow_y,
                width: control,
                height: control,
            }),
            next: (first + count < items).then_some(Rect {
                x: left + strip_width + 4,
                y: arrow_y,
                width: control,
                height: control,
            }),
            footer: (action.y + action.height + 10).max(height - 34),
            first_card: Rect {
                x: left,
                y: top,
                width: card_width,
                height: tile_height,
            },
            gap,
        })
    }

    pub fn entries(self) -> core::ops::Range<usize> {
        self.first..self.first + self.count
    }

    pub fn card(self, index: usize) -> Rect {
        debug_assert!(self.entries().contains(&index));
        let mut rect = self.first_card;
        rect.x += (index - self.first) * (rect.width + self.gap);
        rect
    }

    pub fn hit(self, position: (usize, usize)) -> Option<Target> {
        if self.action.contains(position) {
            return Some(Target::Start);
        }
        if self.previous.is_some_and(|rect| rect.contains(position)) {
            return Some(Target::Select(self.first - 1));
        }
        if self.next.is_some_and(|rect| rect.contains(position)) {
            return Some(Target::Select(self.first + self.count));
        }
        self.entries()
            .find(|&index| self.card(index).contains(position))
            .map(Target::Select)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn middle(rect: Rect) -> (usize, usize) {
        (rect.x + rect.width / 2, rect.y + rect.height / 2)
    }

    #[test]
    fn controls_stay_visible_and_click_targets_match_the_drawn_tiles() {
        for (width, height) in [(200, 220), (320, 240), (640, 480), (1280, 720), (2160, 900)] {
            let layout = Layout::new((width, height), 42, 26).unwrap();
            assert!(layout.entries().contains(&26));
            assert!(layout.footer + 16 <= height);
            assert!(layout.action.y + layout.action.height <= layout.footer);
            assert_eq!(layout.hit(middle(layout.action)), Some(Target::Start));
            for index in layout.entries() {
                let rect = layout.card(index);
                assert!(rect.x + rect.width <= width && rect.y + rect.height <= layout.action.y);
                assert_eq!(layout.hit(middle(rect)), Some(Target::Select(index)));
                assert_eq!(
                    layout.hit((rect.x + rect.width, rect.y + rect.height / 2)),
                    None
                );
            }
            let previous = layout.previous.unwrap();
            let next = layout.next.unwrap();
            assert!(next.x + next.width <= width);
            assert_eq!(
                layout.hit(middle(previous)),
                Some(Target::Select(layout.first - 1))
            );
            assert_eq!(
                layout.hit(middle(next)),
                Some(Target::Select(layout.first + layout.count))
            );
        }
    }

    #[test]
    fn a_single_entry_has_no_paging_actions() {
        let layout = Layout::new((640, 480), 1, 0).unwrap();
        assert_eq!(layout.entries(), 0..1);
        assert_eq!(layout.previous, None);
        assert_eq!(layout.next, None);
        assert_eq!(layout.hit(middle(layout.card(0))), Some(Target::Select(0)));
        assert!(Layout::new((160, 120), 1, 0).is_none());
        assert!(Layout::new((640, 480), 0, 0).is_none());
    }
}

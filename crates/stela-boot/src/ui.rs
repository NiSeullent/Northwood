//! Northwood's firmware palette and drawing primitives, shared by the
//! startup picker, disk picker and boot console.
//!
//! Pages are laid out in units of [`scale`] screen pixels, so they keep
//! their proportions on any display, and drawn in memory before they are
//! shown at once.
use crate::console::{Framebuffer, font::Font};
use alloc::{vec, vec::Vec};

pub const BACKGROUND: u32 = 0x10131a;
pub const CARD: u32 = 0x1b202b;
pub const RAISED: u32 = 0x252e40;
pub const LINE: u32 = 0x3c485e;
pub const TEXT: u32 = 0xf0f4fa;
pub const DIM: u32 = 0xaebacc;
pub const ORANGE: u32 = 0xf4ad70;
pub const BLUE: u32 = 0x9cbcff;
pub const RED: u32 = 0xf77f76;

/// Page units the header takes at the top.
pub const HEADER: usize = 88;
const LOGO: usize = 32;
pub(crate) const NAME: &str = "Northwood";
pub use stela_boot_core::screen::scale;

/// A page drawn in memory and shown at once.
pub struct Canvas {
    screen: Framebuffer,
    font: Font,
    width: usize,
    height: usize,
    scale: usize,
    /// Pixels with blue in the low byte, row by row.
    pixels: Vec<[u8; 4]>,
}

impl Canvas {
    /// The whole screen.
    pub fn new(screen: Framebuffer, font: Font) -> Self {
        let height = screen.resolution().1;
        Self::rows(screen, font, height)
    }

    /// Only the header, over whatever the screen shows under it.
    pub fn header_band(screen: Framebuffer, font: Font) -> Self {
        let rows = HEADER * scale(screen.resolution().1);
        Self::rows(screen, font, rows)
    }

    fn rows(screen: Framebuffer, font: Font, height: usize) -> Self {
        let (width, rows) = screen.resolution();
        let height = height.min(rows);
        Self {
            screen,
            font,
            width,
            height,
            scale: scale(rows),
            pixels: vec![BACKGROUND.to_le_bytes(); width * height],
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.width / self.scale, self.height / self.scale)
    }

    /// Fill device pixels, cut to the canvas.
    fn device_fill(&mut self, x: usize, y: usize, width: usize, height: usize, colour: u32) {
        let right = (x + width).min(self.width);
        for row in y..(y + height).min(self.height) {
            if x < right {
                self.pixels[row * self.width + x..row * self.width + right]
                    .fill(colour.to_le_bytes());
            }
        }
    }

    pub fn fill(&mut self, x: usize, y: usize, width: usize, height: usize, colour: u32) {
        let s = self.scale;
        self.device_fill(x * s, y * s, width * s, height * s, colour);
    }

    /// A rectangle with corners of `radius` rounded.
    pub fn panel(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        radius: usize,
        colour: u32,
    ) {
        let s = self.scale;
        let (x, y, width, height) = (x * s, y * s, width * s, height * s);
        let radius = (radius * s).min(width / 2).min(height / 2) as u64;
        for row in 0..height {
            let from_edge = row.min(height - 1 - row) as u64;
            let inset = if from_edge < radius {
                let dy = radius - from_edge;
                (radius - (radius * radius - dy * dy).isqrt()) as usize
            } else {
                0
            };
            self.device_fill(
                x + inset,
                y + row,
                width.saturating_sub(2 * inset),
                1,
                colour,
            );
        }
    }

    pub fn text_width(&self, text: &str, size: usize) -> usize {
        text.chars().count() * self.font.width * size
    }

    /// `text` from (`x`, `y`) at `size` times the font.
    pub fn text(&mut self, x: usize, y: usize, text: &str, size: usize, colour: u32) {
        let s = self.scale * size;
        let (glyph_width, glyph_height) = (self.font.width, self.font.height);
        for (index, character) in text.chars().enumerate() {
            let character = if character.is_ascii_graphic() || character == ' ' {
                character as u8
            } else {
                b'?'
            };
            let left = (x + index * glyph_width * size) * self.scale;
            for gy in 0..glyph_height {
                for gx in 0..glyph_width {
                    if self.font.pixel(character, gx, gy) {
                        self.device_fill(left + gx * s, y * self.scale + gy * s, s, s, colour);
                    }
                }
            }
        }
    }

    /// `text` ending at `right`.
    pub fn text_right(&mut self, right: usize, y: usize, text: &str, size: usize, colour: u32) {
        let width = self.text_width(text, size);
        self.text(right.saturating_sub(width), y, text, size, colour);
    }

    /// `text` centred on column `middle`.
    pub fn text_centred(&mut self, middle: usize, y: usize, text: &str, size: usize, colour: u32) {
        let width = self.text_width(text, size);
        self.text(middle.saturating_sub(width / 2), y, text, size, colour);
    }

    /// Northwood's monogram from firmware-safe drawing primitives.
    fn logo(&mut self, x: usize, y: usize) {
        self.panel(x, y, LOGO, LOGO, 8, BLUE);
        self.panel(x + 1, y + 1, LOGO - 2, LOGO - 2, 7, RAISED);
        self.text_centred(x + LOGO / 2, y + 8, "Nw", 1, TEXT);
    }

    /// The first and last glyph rows any character of `text` sets.
    fn ink_rows(&self, text: &str) -> (usize, usize) {
        let set = |y: usize| {
            text.bytes()
                .any(|c| (0..self.font.width).any(|x| self.font.pixel(c, x, y)))
        };
        let rows = 0..self.font.height;
        let first = rows.clone().find(|&y| set(y)).unwrap_or(0);
        let last = rows.rev().find(|&y| set(y)).unwrap_or(self.font.height - 1);
        (first, last)
    }

    /// The logo and name, together centred at the top; the first unit row
    /// under them.
    pub fn header(&mut self) -> usize {
        self.named_header(NAME)
    }

    /// Draw the common header with a page-specific title.
    pub fn named_header(&mut self, title: &str) -> usize {
        let (width, _) = self.size();
        let gap = 12;
        let name = self.text_width(title, 2);
        let left = width.saturating_sub(LOGO + gap + name) / 2;
        let top = 28;
        self.logo(left, top);
        // Centre the name's ink, not its glyph cells, on the logo.
        let (first, last) = self.ink_rows(title);
        self.text(
            left + LOGO + gap,
            (top + LOGO / 2).saturating_sub(first + last + 1),
            title,
            2,
            TEXT,
        );
        HEADER
    }

    pub fn show(&self) {
        for (y, row) in self.pixels.chunks_exact(self.width).enumerate() {
            self.screen.show_row(y, row.as_flattened());
        }
    }

    /// Draw or erase the cursor by blitting only its affected rectangle.
    /// The page buffer remains the original artwork underneath the cursor.
    pub(crate) fn cursor(&self, (x, y): (usize, usize), visible: bool) {
        const WIDTH: usize = 11;
        const CURSOR: [&[u8; WIDTH]; 15] = [
            b"#          ",
            b"##         ",
            b"#o#        ",
            b"#oo#       ",
            b"#ooo#      ",
            b"#oooo#     ",
            b"#ooooo#    ",
            b"#oooooo#   ",
            b"#ooooooo#  ",
            b"#oooo##### ",
            b"#oo#o#     ",
            b"#o##oo#    ",
            b"##  #o#    ",
            b"#    #o#   ",
            b"      ##   ",
        ];
        let scale = self.scale;
        let (x, y) = (x * scale, y * scale);
        if x >= self.width || y >= self.height {
            return;
        }
        let columns = (WIDTH * scale).min(self.width - x);
        let rows = (CURSOR.len() * scale).min(self.height - y);
        let mut pixels = [[0u8; 4]; WIDTH * 4];
        for line in 0..rows {
            let offset = (y + line) * self.width + x;
            let row = &mut pixels[..columns];
            row.copy_from_slice(&self.pixels[offset..offset + columns]);
            if visible {
                for (column, pixel) in row.iter_mut().enumerate() {
                    match CURSOR[line / scale][column / scale] {
                        b'#' => *pixel = 0x070911u32.to_le_bytes(),
                        b'o' => *pixel = TEXT.to_le_bytes(),
                        _ => {}
                    }
                }
            }
            self.screen.show_span(x, y + line, row.as_flattened());
        }
    }
}

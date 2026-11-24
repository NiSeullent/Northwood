//! Stela's look on the firmware framebuffer, shared by the startup menu,
//! the disk picker and the boot console: dark gray pages under the logo and
//! name, centred at the top, with orange for what is chosen and blue for
//! what is new or done.
//!
//! Pages are laid out in units of [`scale`] screen pixels, so they keep
//! their proportions on any display, and drawn in memory before they are
//! shown at once.
use crate::console::{Framebuffer, font::Font};
use alloc::{vec, vec::Vec};

pub const BACKGROUND: u32 = 0x18191c;
pub const CARD: u32 = 0x222428;
pub const RAISED: u32 = 0x2c2e34;
pub const LINE: u32 = 0x3a3c43;
pub const TEXT: u32 = 0xe6e7ea;
pub const DIM: u32 = 0x8b8e96;
pub const ORANGE: u32 = 0xf0883a;
pub const BLUE: u32 = 0x3f86e8;
pub const RED: u32 = 0xe4574d;

/// Page units the header takes at the top.
pub const HEADER: usize = 88;
const LOGO: usize = 32;
const NAME: &str = "Stela";
pub use stela_boot_core::screen::scale;

/// A page drawn in memory and shown at once.
pub struct Canvas {
    screen: Framebuffer,
    font: Font,
    width: usize,
    height: usize,
    scale: usize,
    /// Pixels with blue in the low byte, row by row.
    pixels: Vec<u32>,
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
            pixels: vec![BACKGROUND; width * height],
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
                self.pixels[row * self.width + x..row * self.width + right].fill(colour);
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
        text.len() * self.font.width * size
    }

    /// `text` from (`x`, `y`) at `size` times the font.
    pub fn text(&mut self, x: usize, y: usize, text: &str, size: usize, colour: u32) {
        let s = self.scale * size;
        let (glyph_width, glyph_height) = (self.font.width, self.font.height);
        for (index, character) in text.bytes().enumerate() {
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

    /// The logo, `size` units square from (`x`, `y`), smoothed by sampling
    /// each pixel four by four.
    fn logo(&mut self, x: usize, y: usize, size: usize) {
        const SAMPLES: usize = 4;
        let side = size * self.scale;
        let (x, y) = (x * self.scale, y * self.scale);
        for py in 0..side {
            for px in 0..side {
                let mut sum = [0u32; 3];
                for sy in 0..SAMPLES {
                    for sx in 0..SAMPLES {
                        let u = (px * SAMPLES + sx) as f32 / (side * SAMPLES) as f32;
                        let v = (py * SAMPLES + sy) as f32 / (side * SAMPLES) as f32;
                        let colour =
                            stela_boot_core::screen::logo_pixel(u, v).unwrap_or(BACKGROUND);
                        for (channel, total) in sum.iter_mut().enumerate() {
                            *total += (colour >> (16 - 8 * channel)) & 0xff;
                        }
                    }
                }
                let n = (SAMPLES * SAMPLES) as u32;
                let colour = sum
                    .iter()
                    .fold(0, |pixel, total| (pixel << 8) | ((total + n / 2) / n));
                self.device_fill(x + px, y + py, 1, 1, colour);
            }
        }
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
        self.logo(left, top, LOGO);
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
        let mut row = vec![0u8; self.width * 4];
        for y in 0..self.height {
            for (out, pixel) in row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(&self.pixels[y * self.width..(y + 1) * self.width])
            {
                *out = pixel.to_le_bytes();
            }
            self.screen.show_row(y, &row);
        }
    }
}

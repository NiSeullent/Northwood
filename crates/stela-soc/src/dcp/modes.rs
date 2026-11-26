//! The one mode the modeled DCP advertises for its fixed, connected SDR
//! monitor, in the forms the 26A434 AP reads: the EDID base block, the
//! `IOAVVideoTimingData` and `IOAVVideoColorData` records `A364
//! getVideoDataForIDs` returns, and the serialized `TimingElements` and
//! `ColorElements` arrays the coprocessor publishes on the IOMFB service with
//! `D126`..`D128 setDCPAVPropStart/Chunk/End`.
//!
//! Every layout and key is the exact build's (kernel fileset offsets; VA =
//! offset + 0xfffffe0007004000), from the IOAVFamily code that serializes
//! the same elements:
//!
//! - `IOAVVideoTimingElement::initWithDetailedTiming` (0x39671ac) decodes an
//!   EDID detailed timing into the 0x50-byte timing record (flags +0, the
//!   horizontal axis +0x4 and the vertical +0x24, each `{total, active, sync
//!   width, back porch, front porch, precise sync rate 16.16, sync polarity,
//!   pixel repetition}`, the pixel clock in Hz +0x44, the aspect ratio +0x48,
//!   the scan information +0x4c); `_IOAVVideoTimingSetPixelClock` (0x3973970)
//!   and `_IOAVVideoTimingUpdateSyncRates` (0x39739b0) derive both sync rates;
//!   `_IOAVVideoActiveFormatAspectRatio` (0x396fad4, table `arRef` 0xa52edc)
//!   picks the aspect-ratio code nearest the precise ratio.
//!   `restored_external`'s own `timing_data_struct` log reads the same
//!   offsets (active +0x28 / +0x8, interlaced +0 bit 0, clock +0x44, ratio
//!   +0x48, scan +0x4c).
//! - `IOAVVideoColorElement::withCapabilities` (0x3965c60) packs the 0x20-byte
//!   color record: depth (bits per component) +0, pixel encoding +4, dynamic
//!   range +8, colorimetry +0xc, EOTF +0x10, the supports-DSC byte +0x14.
//! - `IOAVElement::copyProperties` (0x39a31e4: `Score` 32-bit when not
//!   virtual, `ID` 64-bit, `IsVirtual`), `IOAVVideoTimingElement::
//!   copyProperties` (0x39acfec) with `IOAVVideoTimingAxisCopy` (0x3970928),
//!   and `IOAVVideoColorElement::copyProperties` (0x3966258) give the keys and
//!   their widths. The enumerations are the IOAV string tables
//!   (`_IOAVVideoPixelEncodingString` 0x396ed34 → 0 "RGB 4:4:4",
//!   `_IOAVVideoColorEOTFString` → 0 "Traditional gamma (SDR)",
//!   `_IOAVVideoColorDynamicRangeString` → 0 "Full", `_IOAVVideoColorimetryString`
//!   → 10 "sRGB", `_IOAVVideoTimingTypeString` → 3 "Detailed",
//!   `_IOAVStandardTypeString` → 1 "VESA", `_IOAVElementTypeString` → 0
//!   "Video Timing", 1 "Video Color").
//! - The scores follow `IOAVVideoTimingElement::setScoreOffsetMask` (0x3967004)
//!   with no offset bits and `__scoreColorData` (0x3965cc0, stored
//!   `* 1000 + 999` by `initWithData` 0x3965b2c). The firmware's own offset
//!   mask is not pinned; with one advertised timing the score only has to be
//!   positive, which `restored_external` requires (it keeps the timing whose
//!   `Score` exceeds the best so far, starting from 0).
//!
//! The timing's standard type (VESA, an EDID base-block timing without a CEA
//! extension), its being preferred and the color's sRGB colorimetry (the EDID
//! declares sRGB its default color space) are Inferred from the EDID the
//! model serves.
use super::super::scanout::{DEFAULT_REFRESH_MILLIHERTZ, Mode};
use crate::Result;
use alloc::string::String;
use core::fmt::Write;
use stela_contracts::refuse;

/// The timing element's ID. `IOMobileFramebufferShim::set_digital_out_mode`
/// (0x2a66aac) treats timing IDs in 500..=10027 as display-wall modes, so
/// the single-display ID lies outside them.
pub const TIMING_ID: u32 = 1;
/// The color element's ID (a separate namespace: `getVideoDataForIDs` takes
/// both).
pub const COLOR_ID: u32 = 2;
/// `sizeof(IOAVVideoTimingData)`: the `ElementData` restored copies and
/// `A364`'s first argument.
pub const TIMING_BYTES: usize = 0x50;
/// `sizeof(IOAVVideoColorData)`: `A364`'s second argument.
pub const COLOR_BYTES: usize = 0x20;

/// `IOAVVideoPixelEncoding` RGB 4:4:4.
const RGB_444: u32 = 0;
/// `IOAVVideoColorDynamicRange` full.
const FULL_RANGE: u32 = 0;
/// `IOAVVideoColorEOTF` traditional gamma (SDR).
const SDR: u32 = 0;
/// `IOAVVideoColorimetry` sRGB.
const SRGB: u32 = 10;
/// `IOAVStandardType` VESA.
const VESA: u32 = 1;
/// `IOAVVideoTimingType` detailed.
const DETAILED: u32 = 3;
/// `IOAVElementType` video timing and video color.
const TIMING_ELEMENT: u32 = 0;
const COLOR_ELEMENT: u32 = 1;
/// 8 bits per component.
const DEPTH: u32 = 8;

/// `IOAVVideoActiveFormatAspectRatio::arRef` (0xa52edc): each code's
/// width/height as 16.16, ascending.
const ASPECT_RATIOS: [(u32, u32); 18] = [
    (0x6c00, 1),
    (0x8700, 2),
    (0x9000, 3),
    (0x999a, 4),
    (0xa000, 5),
    (0xa492, 6),
    (0xc000, 7),
    (0xcccd, 8),
    (0x10000, 9),
    (0x14000, 10),
    (0x15555, 11),
    (0x18e39, 12),
    (0x1999a, 13),
    (0x1aaab, 14),
    (0x1c71c, 15),
    (0x1e574, 16),
    (0x25ed1, 17),
    (0x3097b, 0),
];

/// Nominal CVT-style blanking of the advertised timing.
const H_FRONT: u32 = 48;
const H_SYNC: u32 = 32;
const H_BLANK: u32 = 320;
const V_FRONT: u32 = 3;
const V_SYNC: u32 = 10;
const V_BLANK: u32 = 45;
/// CVT reduced blanking v2's horizontal blanking (VESA CVT 2.0 section
/// 5.2: front porch 8, sync 32, back porch 40), which a mode takes when its
/// nominal blanking needs more pixel clock than a detailed timing holds.
const RB_H_FRONT: u32 = 8;
const RB_H_BLANK: u32 = 80;
/// CVT's minimum vertical blanking time, in nanoseconds.
const RB_V_BLANK_NANOS: u64 = 460_000;
/// A detailed timing's pixel clock, in its 10 kHz units.
const MAX_CLOCK: u64 = 0xffff;

/// A detailed timing's blanking: `(h_front, h_blank, v_blank)`, with the
/// syncs and the vertical front porch fixed.
type Blanking = (u32, u32, u32);

/// The pixel clock, in 10 kHz units, of `width` by `height` with `blanking`
/// at `refresh` millihertz, rounded to the nearest unit.
fn clock(width: u32, height: u32, (_, h_blank, v_blank): Blanking, refresh: u32) -> u64 {
    let total = u64::from(width + h_blank) * u64::from(height + v_blank);
    (total * u64::from(refresh) + 5_000_000) / 10_000_000
}

/// The blanking of `width` by `height` at `refresh` millihertz: the nominal
/// one, or reduced blanking (vertical blanking of at least CVT's 460 us)
/// when the nominal clock does not fit a detailed timing.
fn blanking(width: u32, height: u32, refresh: u32) -> Blanking {
    let nominal = (H_FRONT, H_BLANK, V_BLANK);
    if clock(width, height, nominal, refresh) <= MAX_CLOCK {
        return nominal;
    }
    // A fraction k of each frame is blank: v_blank / (height + v_blank) >= k,
    // with k = 460 us * refresh.
    let k = RB_V_BLANK_NANOS * u64::from(refresh) / 1000;
    let lines = (k * u64::from(height)).div_ceil(1_000_000_000u64.saturating_sub(k).max(1));
    let v_blank = (lines as u32).max(V_FRONT + V_SYNC + 1);
    (RB_H_FRONT, RB_H_BLANK, v_blank)
}

/// The image size in millimetres to report for a `width` by `height` mode
/// whose monitor reported none: a desktop monitor's usual diagonal for the
/// resolution (27" from 2560 pixels across, 24" from 1920, else 21.5"), so
/// that macOS's default scale follows a sane pixel density.
pub fn assumed_size_mm(width: u32, height: u32) -> (u32, u32) {
    let tenths_of_inch: u64 = match width {
        2560.. => 270,
        1920.. => 240,
        _ => 215,
    };
    let (w, h) = (u64::from(width.max(1)), u64::from(height.max(1)));
    let diagonal_pixels = (w * w + h * h).isqrt().max(1);
    let diagonal_mm = tenths_of_inch * 254 / 100;
    let mm = |pixels: u64| (diagonal_mm * pixels / diagonal_pixels) as u32;
    (mm(w), mm(h))
}

/// An 18-byte EDID detailed timing descriptor for `mode` at `refresh`
/// millihertz, digital separate sync, both polarities positive. A mode
/// whose fields do not fit the descriptor refuses rather than wraps.
pub fn detailed_timing(mode: Mode, refresh: u32) -> Result<[u8; 18]> {
    detailed_timing_sized(mode, refresh, None)
}

/// [`detailed_timing`] with the monitor's image size in millimetres, else
/// [`assumed_size_mm`].
pub fn detailed_timing_sized(
    mode: Mode,
    refresh: u32,
    size_mm: Option<(u32, u32)>,
) -> Result<[u8; 18]> {
    let (w, h) = (mode.width, mode.height);
    if w == 0 || h == 0 || w > 0xfff || h > 0xfff {
        return Err(refuse!("DCP mode does not fit an EDID detailed timing"));
    }
    let (h_front, h_blank, v_blank) = blanking(w, h, refresh);
    let clock = u16::try_from(clock(w, h, (h_front, h_blank, v_blank), refresh))
        .ok()
        .filter(|&clock| clock != 0)
        .ok_or(refuse!(
            "DCP mode's pixel clock does not fit an EDID detailed timing"
        ))?;
    if v_blank > 0xfff {
        return Err(refuse!(
            "DCP mode's vertical blanking does not fit an EDID detailed timing"
        ));
    }
    let mut d = [0u8; 18];
    d[0..2].copy_from_slice(&clock.to_le_bytes());
    d[2] = w as u8;
    d[3] = h_blank as u8;
    d[4] = (((w >> 8) << 4) | (h_blank >> 8)) as u8;
    d[5] = h as u8;
    d[6] = v_blank as u8;
    d[7] = (((h >> 8) << 4) | (v_blank >> 8)) as u8;
    d[8] = h_front as u8;
    d[9] = H_SYNC as u8;
    d[10] = ((V_FRONT << 4) | V_SYNC) as u8;
    d[11] = 0;
    // Image size in mm, no borders.
    let (width_mm, height_mm) = size_mm.unwrap_or_else(|| assumed_size_mm(w, h));
    let (width_mm, height_mm) = (width_mm.min(0xfff), height_mm.min(0xfff));
    d[12] = width_mm as u8;
    d[13] = height_mm as u8;
    d[14] = (((width_mm >> 8) << 4) | (height_mm >> 8)) as u8;
    // Digital separate sync (bits 4:3), vsync and hsync positive.
    d[17] = 0x1e;
    Ok(d)
}

/// The range limits descriptor's five rate bytes (E-EDID 1.4 section
/// 3.10.3.3) for detailed timing `d`: vertical rates in Hz and horizontal
/// in kHz, each range holding the timing's and at least 50..=75 Hz and
/// 30..=80 kHz, and the maximum pixel clock in 10 MHz units.
fn range_limits(d: &[u8; 18]) -> Result<[u8; 5]> {
    let clock = u64::from(u16::from_le_bytes([d[0], d[1]])) * 10_000;
    let h_total = u64::from(u32::from(d[2]) | u32::from(d[4] >> 4) << 8)
        + u64::from(u32::from(d[3]) | u32::from(d[4] & 0xf) << 8);
    let v_total = u64::from(u32::from(d[5]) | u32::from(d[7] >> 4) << 8)
        + u64::from(u32::from(d[6]) | u32::from(d[7] & 0xf) << 8);
    let line = clock / h_total;
    let frame = line / v_total;
    let byte = |value: u64| {
        u8::try_from(value).map_err(|_| refuse!("DCP mode's rates exceed EDID range limits"))
    };
    Ok([
        byte(frame.min(50))?,
        byte(line.div_ceil(v_total).max(75))?,
        byte((line / 1000).min(30))?,
        byte(line.div_ceil(1000).max(80))?,
        byte(clock.div_ceil(10_000_000).max(17))?,
    ])
}

/// A truthful EDID 1.4 base block whose one detailed timing is `mode`'s at
/// `refresh` millihertz, of [`assumed_size_mm`].
pub fn edid(mode: Mode, refresh: u32) -> Result<[u8; 128]> {
    edid_sized(mode, refresh, None)
}

/// [`edid`] for a monitor whose image is `size_mm` millimetres, as its own
/// EDID reported, when known.
pub fn edid_sized(mode: Mode, refresh: u32, size_mm: Option<(u32, u32)>) -> Result<[u8; 128]> {
    let size_mm = size_mm.unwrap_or_else(|| assumed_size_mm(mode.width, mode.height));
    let mut e = [0u8; 128];
    e[0..8].copy_from_slice(&[0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00]);
    // Manufacturer "STE" (Stela), product 1, serial 1, week 1, year 2026.
    let man = ((b'S' as u16 - 64) << 10) | ((b'T' as u16 - 64) << 5) | (b'E' as u16 - 64);
    e[8..10].copy_from_slice(&man.to_be_bytes());
    e[10..12].copy_from_slice(&1u16.to_le_bytes());
    e[12..16].copy_from_slice(&1u32.to_le_bytes());
    e[16] = 1;
    e[17] = 2026u16.wrapping_sub(1990) as u8;
    e[18] = 1; // EDID 1.4
    e[19] = 4;
    // Digital input, 8 bpc, DisplayPort.
    e[20] = 0xa5;
    // The image size in whole centimetres.
    e[21] = (size_mm.0.div_ceil(10)).clamp(1, 255) as u8;
    e[22] = (size_mm.1.div_ceil(10)).clamp(1, 255) as u8;
    e[23] = 0x78; // gamma 2.2
    e[24] = 0x06; // RGB, sRGB default color space, preferred timing native
    e[25..35].copy_from_slice(&[0xee, 0x91, 0xa3, 0x54, 0x4c, 0x99, 0x26, 0x0f, 0x50, 0x54]);
    let timing = detailed_timing_sized(mode, refresh, Some(size_mm))?;
    e[54..72].copy_from_slice(&timing);
    e[75] = 0xfc; // display name
    e[77..77 + 6].copy_from_slice(b"Stela\n");
    e[93] = 0xfd; // range limits
    e[95..95 + 5].copy_from_slice(&range_limits(&timing)?);
    e[101] = 0x0a;
    e[111] = 0x10; // dummy descriptor
    let sum = e[0..127].iter().fold(0u8, |a, b| a.wrapping_add(*b));
    e[127] = 0u8.wrapping_sub(sum);
    Ok(e)
}

fn put(data: &mut [u8], at: usize, value: u32) {
    data[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn get(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().expect("4 bytes"))
}

/// The `IOAVVideoTimingData` of an EDID detailed timing, decoded as
/// `IOAVVideoTimingElement::initWithDetailedTiming` does.
pub fn timing_data(d: &[u8; 18]) -> Result<[u8; TIMING_BYTES]> {
    let clock = u32::from(u16::from_le_bytes([d[0], d[1]]));
    let interlaced = u32::from(d[17] >> 7);
    let h_active = u32::from(d[2]) | u32::from(d[4] >> 4) << 8;
    let h_blank = u32::from(d[3]) | u32::from(d[4] & 0xf) << 8;
    let h_front = u32::from(d[8]) | u32::from(d[11] >> 6 & 3) << 8;
    let h_sync = u32::from(d[9]) | u32::from(d[11] >> 4 & 3) << 8;
    let h_back = h_blank.checked_sub(h_front + h_sync).ok_or(refuse!(
        "DCP detailed timing's horizontal porches exceed its blanking"
    ))?;
    let field = |lines: u32| {
        if interlaced != 0 {
            lines * 2 + 1
        } else {
            lines
        }
    };
    let v_active = (u32::from(d[5]) | u32::from(d[7] >> 4) << 8) << interlaced;
    let v_front = field(u32::from(d[10] >> 4) | u32::from(d[11] >> 2 & 3) << 4);
    let v_sync = (u32::from(d[10] & 0xf) | u32::from(d[11] & 3) << 4) << interlaced;
    let v_blank = field(u32::from(d[6]) | u32::from(d[7] & 0xf) << 8);
    let v_back = v_blank.checked_sub(v_front + v_sync).ok_or(refuse!(
        "DCP detailed timing's vertical porches exceed its blanking"
    ))?;
    let mut t = [0u8; TIMING_BYTES];
    put(&mut t, 0x00, interlaced);
    for (axis, [total, active, sync, back, front]) in [
        (
            0x04,
            [h_active + h_blank, h_active, h_sync, h_back, h_front],
        ),
        (
            0x24,
            [v_active + v_blank, v_active, v_sync, v_back, v_front],
        ),
    ] {
        put(&mut t, axis, total);
        put(&mut t, axis + 0x4, active);
        put(&mut t, axis + 0x8, sync);
        put(&mut t, axis + 0xc, back);
        put(&mut t, axis + 0x10, front);
    }
    // Digital sync: bit 3 separate (vsync polarity bit 2) or composite.
    if d[17] & 0x10 != 0 {
        let h = u32::from(d[17] >> 1 & 1);
        let v = if d[17] & 0x8 == 0 {
            h
        } else {
            u32::from(d[17] >> 2 & 1)
        };
        put(&mut t, 0x1c, h);
        put(&mut t, 0x3c, v);
    }
    // _IOAVVideoTimingSetPixelClock, then _IOAVVideoTimingUpdateSyncRates.
    let pixel_clock = clock * 10_000;
    put(&mut t, 0x44, pixel_clock);
    let line = u64::from(get(&t, 0x04)) * u64::from(get(&t, 0x20) + 1);
    let frame = u64::from(get(&t, 0x24)) * line;
    if line == 0 || frame == 0 {
        return Err(refuse!("DCP detailed timing has no blanking totals"));
    }
    let q16 = u64::from(pixel_clock) << 16;
    let h_rate = (q16 + line * 500) / (line * 1000);
    let mut v_rate = (q16 + frame / 2) / frame;
    if interlaced != 0 {
        v_rate *= 2;
    }
    put(&mut t, 0x18, h_rate as u32);
    put(&mut t, 0x38, v_rate as u32);
    if v_rate == 0 || h_active == 0 || v_active == 0 {
        return Err(refuse!("DCP detailed timing has no active area or refresh"));
    }
    let ratio = aspect_ratio(precise_aspect_ratio(&t));
    put(&mut t, 0x48, ratio);
    Ok(t)
}

/// `getPreciseAspectRatio`: horizontal over vertical active, 16.16.
fn precise_aspect_ratio(t: &[u8; TIMING_BYTES]) -> u32 {
    ((u64::from(get(t, 0x08)) << 16) / u64::from(get(t, 0x28))) as u32
}

/// `_IOAVVideoActiveFormatAspectRatio`: the code of the nearest ratio.
fn aspect_ratio(precise: u32) -> u32 {
    let distance = |ratio: u32| ratio.abs_diff(precise);
    let mut best = ASPECT_RATIOS.len() - 1;
    for (index, &(ratio, _)) in ASPECT_RATIOS.iter().enumerate() {
        if distance(ratio) < distance(ASPECT_RATIOS[best].0) {
            best = index;
        }
    }
    ASPECT_RATIOS[best].1
}

/// The `IOAVVideoColorData` of the one color mode: RGB 4:4:4, 8 bits per
/// component, full range, SDR, sRGB, no DSC.
pub fn color_data() -> [u8; COLOR_BYTES] {
    let mut c = [0u8; COLOR_BYTES];
    put(&mut c, 0x00, DEPTH);
    put(&mut c, 0x04, RGB_444);
    put(&mut c, 0x08, FULL_RANGE);
    put(&mut c, 0x0c, SRGB);
    put(&mut c, 0x10, SDR);
    c
}

/// `setScoreOffsetMask(0)`: the active pixel clock in MHz, rounded, plus
/// the vertical sync rate, rounded.
fn timing_score(t: &[u8; TIMING_BYTES]) -> u32 {
    let active =
        (u64::from(get(t, 0x08)) * u64::from(get(t, 0x28)) * u64::from(get(t, 0x38))) >> 16;
    let active = (active / u64::from(get(t, 0x20) + 1)) >> (get(t, 0) & 1);
    let megahertz = (active + 500_000) / 1_000_000;
    let refresh = (get(t, 0x38) + 0x8000) >> 16;
    megahertz as u32 + refresh
}

/// `__scoreColorData` for 8-bit RGB 4:4:4 sRGB (depth 200, encoding 40,
/// colorimetry 2), stored as `score * 1000 + 999`.
const COLOR_SCORE: u32 = (200 + 40 + 2) * 1000 + 999;

/// The manufacturer the EDID encodes.
const MANUFACTURER: &str = "STE";
/// `IOMFBUUID`: one fixed monitor, so its settings persist across boots.
const UUID: &str = "53544c41-0001-4000-8000-000000000001";

/// An OSUnserializeXML document under construction.
struct Xml(String);

impl Xml {
    fn key(&mut self, key: &str) -> &mut Self {
        let _ = write!(self.0, "<key>{key}</key>");
        self
    }
    fn int(&mut self, key: &str, value: u64, bits: u32) -> &mut Self {
        self.key(key);
        let _ = write!(self.0, "<integer size=\"{bits}\">{value}</integer>");
        self
    }
    fn bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.key(key);
        self.0.push_str(if value { "<true/>" } else { "<false/>" });
        self
    }
    fn string(&mut self, key: &str, value: &str) -> &mut Self {
        self.key(key);
        let _ = write!(self.0, "<string>{value}</string>");
        self
    }
    fn data(&mut self, key: &str, bytes: &[u8]) -> &mut Self {
        self.key(key);
        self.0.push_str("<data>");
        base64(&mut self.0, bytes);
        self.0.push_str("</data>");
        self
    }
    fn raw(&mut self, text: &str) -> &mut Self {
        self.0.push_str(text);
        self
    }
}

fn base64(out: &mut String, bytes: &[u8]) {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for chunk in bytes.chunks(3) {
        let word = u32::from(chunk[0]) << 16
            | u32::from(*chunk.get(1).unwrap_or(&0)) << 8
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(ALPHABET[(word >> (18 - 6 * index) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
}

/// A chromaticity coordinate pair.
fn xy(xml: &mut Xml, key: &str, x: u64, y: u64) {
    xml.key(key)
        .raw("<dict>")
        .int("X", x, 32)
        .int("Y", y, 32)
        .raw("</dict>");
}

/// The one color element's properties (`IOAVVideoColorElement::
/// copyProperties`).
fn color_dict(xml: &mut Xml) {
    xml.raw("<dict>")
        .int("Score", u64::from(COLOR_SCORE), 32)
        .int("ID", u64::from(COLOR_ID), 64)
        .bool("IsVirtual", false)
        .int("ElementType", u64::from(COLOR_ELEMENT), 32)
        .data("ElementData", &color_data())
        .int("Depth", u64::from(DEPTH), 32)
        .int("PixelEncoding", u64::from(RGB_444), 32)
        .int("DynamicRange", u64::from(FULL_RANGE), 32)
        .int("EOTF", u64::from(SDR), 32)
        .int("Colorimetry", u64::from(SRGB), 32)
        .bool("SupportsDSC", false)
        .int("StandardType", u64::from(VESA), 32)
        .raw("</dict>");
}

/// One axis's attributes (`IOAVVideoTimingAxisCopy`).
fn axis_dict(xml: &mut Xml, t: &[u8; TIMING_BYTES], axis: usize) {
    let precise = get(t, axis + 0x14);
    xml.raw("<dict>")
        .int("Total", u64::from(get(t, axis)), 32)
        .int("FrontPorch", u64::from(get(t, axis + 0x10)), 32)
        .int("SyncWidth", u64::from(get(t, axis + 0x8)), 32)
        .int("BackPorch", u64::from(get(t, axis + 0xc)), 32)
        .int("Active", u64::from(get(t, axis + 0x4)), 32)
        .int("SyncRate", u64::from((precise + 0x8000) & 0xffff_0000), 32)
        .int("PreciseSyncRate", u64::from(precise), 32)
        .int("SyncPolarity", u64::from(get(t, axis + 0x18)), 32)
        .int("PixelRepetition", u64::from(get(t, axis + 0x1c)), 32)
        .raw("</dict>");
}

/// The advertised mode: its EDID, records and published properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advertised {
    pub edid: [u8; 128],
    pub timing: [u8; TIMING_BYTES],
    pub color: [u8; COLOR_BYTES],
}

impl Advertised {
    /// The one mode for the scanout's `mode` at 60 Hz, or why it cannot be
    /// advertised.
    pub fn new(mode: Mode) -> Result<Self> {
        Self::at(mode, DEFAULT_REFRESH_MILLIHERTZ)
    }

    /// The one mode for the scanout's `mode` at `refresh` millihertz, or why
    /// it cannot be advertised.
    pub fn at(mode: Mode, refresh: u32) -> Result<Self> {
        Self::sized(mode, refresh, None)
    }

    /// [`Self::at`] for a monitor whose image is `size_mm` millimetres,
    /// when its own EDID said.
    pub fn sized(mode: Mode, refresh: u32, size_mm: Option<(u32, u32)>) -> Result<Self> {
        let size_mm = size_mm.map(|(width_mm, height_mm)| {
            // SkyLight may default a dense 1080p panel to a 960-point desktop.
            // Publish narrow panels at no more than 140 PPI to fit Setup.
            let minimum_width_mm = (u64::from(mode.width) * 254).div_ceil(1400);
            if mode.width >= 2560
                || width_mm == 0
                || height_mm == 0
                || u64::from(width_mm) >= minimum_width_mm
            {
                return (width_mm, height_mm);
            }
            let height = (u64::from(height_mm) * minimum_width_mm).div_ceil(u64::from(width_mm));
            (
                minimum_width_mm as u32,
                height.min(u64::from(u32::MAX)) as u32,
            )
        });
        let edid = edid_sized(mode, refresh, size_mm)?;
        let dtd: [u8; 18] = edid[54..72].try_into().expect("18 bytes");
        Ok(Self {
            edid,
            timing: timing_data(&dtd)?,
            color: color_data(),
        })
    }

    /// The video records for `(timing, color)` IDs, when both are the
    /// advertised ones.
    pub fn video_data(&self, timing: u32, color: u32) -> Option<(&[u8], &[u8])> {
        (timing == TIMING_ID && color == COLOR_ID).then_some((&self.timing[..], &self.color[..]))
    }

    /// The monitor's `DisplayAttributes`, the EDID's identity as WindowServer
    /// reads it. Chromaticities are sRGB's, in 16.16 fixed point.
    pub fn display_attributes(&self) -> String {
        let e = &self.edid;
        let name = e[77..90].split(|&byte| byte == b'\n').next().unwrap_or(&[]);
        let (width, height) = (get(&self.timing, 0x08), get(&self.timing, 0x28));
        let mut xml = Xml(String::new());
        xml.raw("<dict>")
            .key("ProductAttributes")
            .raw("<dict>")
            .string("ManufacturerID", MANUFACTURER)
            .int(
                "LegacyManufacturerID",
                u64::from(u16::from_be_bytes([e[8], e[9]])),
                32,
            )
            .int(
                "ProductID",
                u64::from(u16::from_le_bytes([e[10], e[11]])),
                32,
            )
            .int(
                "SerialNumber",
                u64::from(u32::from_le_bytes([e[12], e[13], e[14], e[15]])),
                32,
            )
            .int("WeekOfManufacture", u64::from(e[16]), 32)
            .int("YearOfManufacture", 1990 + u64::from(e[17]), 32)
            .string("ProductName", core::str::from_utf8(name).unwrap_or(""))
            .raw("</dict>")
            .int("NativeFormatHorizontalPixels", u64::from(width), 32)
            .int("NativeFormatVerticalPixels", u64::from(height), 32)
            // The base block's image size in whole centimetres, as Apple's
            // EDID parser publishes it. QuartzCore's IOMFBDisplay reads these
            // two keys alone for the panel's physical size; without them
            // WindowServer assumes 72 dpi and CGDisplayScreenSize is zero.
            .int("MaxHorizontalImageSize", u64::from(e[21]), 32)
            .int("MaxVerticalImageSize", u64::from(e[22]), 32)
            .bool("ContinuousFrequencySupport", false)
            .bool("SupportsActiveOff", false)
            .bool("SupportsSuspend", false)
            .key("Chromaticity")
            .raw("<dict>");
        for (key, x, y) in [
            ("Red", 41943, 21627),
            ("Green", 19661, 39322),
            ("Blue", 9830, 3932),
        ] {
            xy(&mut xml, key, x, y);
        }
        xml.raw("</dict>");
        xy(&mut xml, "DefaultWhite", 20493, 21561);
        xml.raw("</dict>");
        xml.0
    }

    /// How the monitor connects: DCP's DisplayPort through the Mac's HDMI port.
    pub fn transport() -> String {
        let mut xml = Xml(String::new());
        xml.raw("<dict>")
            .string("Upstream", "DP")
            .string("Downstream", "HDMI")
            .raw("</dict>");
        xml.0
    }

    /// The display's fixed identity across boots, as a plist string.
    pub fn uuid() -> String {
        let mut xml = Xml(String::new());
        let _ = write!(xml.0, "<string>{UUID}</string>");
        xml.0
    }

    /// The `TimingElements` array (`IOAVVideoTimingElement::copyProperties`
    /// of the one timing, its `ColorModes` the one color element).
    pub fn timing_elements(&self) -> String {
        let t = &self.timing;
        let mut xml = Xml(String::new());
        xml.raw("<array><dict>")
            .int("Score", u64::from(timing_score(t)), 32)
            .int("ID", u64::from(TIMING_ID), 64)
            .bool("IsVirtual", false)
            .int("ElementType", u64::from(TIMING_ELEMENT), 32)
            .data("ElementData", t)
            .key("ColorModes")
            .raw("<array>");
        color_dict(&mut xml);
        xml.raw("</array>")
            .bool("IsInterlaced", get(t, 0) & 1 != 0)
            .bool("IsSplit", get(t, 0) & 2 != 0)
            .bool("IsOverscanned", get(t, 0x4c) == 1)
            .bool("IsPreferred", true)
            .bool("IsPromoted", false)
            .int("ScanInformation", u64::from(get(t, 0x4c)), 32)
            .int("ValidPixelEncodings", 1 << RGB_444, 32)
            .int("StandardType", u64::from(VESA), 32)
            .int("TimingType", u64::from(DETAILED), 32)
            .int("AspectRatio", u64::from(get(t, 0x48)), 32)
            .int("PreciseAspectRatio", u64::from(precise_aspect_ratio(t)), 32)
            .key("HorizontalAttributes");
        axis_dict(&mut xml, t, 0x04);
        xml.key("VerticalAttributes");
        axis_dict(&mut xml, t, 0x24);
        xml.raw("</dict></array>");
        xml.0
    }

    /// The `ColorElements` array: the one color element.
    pub fn color_elements(&self) -> String {
        let mut xml = Xml(String::from("<array>"));
        color_dict(&mut xml);
        xml.raw("</array>");
        xml.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::scanout::PixelFormat;
    use super::*;
    use alloc::format;

    const FULL_HD: Mode = Mode {
        width: 1920,
        height: 1080,
        format: PixelFormat::Bgra8888,
    };

    #[test]
    fn the_edid_is_well_formed_for_the_mode() {
        let e = edid(FULL_HD, 60_000).unwrap();
        assert_eq!(&e[0..8], &[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
        assert_eq!(e.iter().fold(0u8, |a, b| a.wrapping_add(*b)), 0, "checksum");
        let dtd = &e[54..72];
        assert_eq!(u32::from(dtd[2]) | (u32::from(dtd[4] >> 4) << 8), 1920);
        // A mode the descriptor cannot hold refuses rather than wraps.
        let huge = Mode {
            width: 5120,
            ..FULL_HD
        };
        assert!(edid(huge, 60_000).is_err());
        // The range limits are the old fixed 50..=75 Hz, 30..=80 kHz, 170 MHz
        // for a mode within them.
        assert_eq!(&e[95..101], &[0x32, 0x4b, 0x1e, 0x50, 0x11, 0x00]);
    }

    fn dtd_size_mm(e: &[u8; 128]) -> (u32, u32) {
        let d = &e[54..72];
        (
            u32::from(d[12]) | u32::from(d[14] >> 4) << 8,
            u32::from(d[13]) | u32::from(d[14] & 0xf) << 8,
        )
    }

    #[test]
    fn a_monitors_reported_size_reaches_both_size_fields() {
        let uhd = Mode {
            width: 3840,
            height: 2160,
            ..FULL_HD
        };
        // A 27" 4K panel: 597 x 336 mm.
        let e = edid_sized(uhd, 60_000, Some((597, 336))).unwrap();
        assert_eq!(dtd_size_mm(&e), (597, 336));
        assert_eq!((e[21], e[22]), (60, 34));
        assert_eq!(e.iter().fold(0u8, |a, b| a.wrapping_add(*b)), 0, "checksum");
        let a = Advertised::sized(uhd, 60_000, Some((597, 336))).unwrap();
        assert_eq!(a.edid, e);
        // WindowServer learns the size only from DisplayAttributes.
        let attributes = a.display_attributes();
        for key in [
            "<key>MaxHorizontalImageSize</key><integer size=\"32\">60</integer>",
            "<key>MaxVerticalImageSize</key><integer size=\"32\">34</integer>",
        ] {
            assert!(attributes.contains(key), "{key} in {attributes}");
        }
    }

    #[test]
    fn an_unreported_size_follows_a_usual_diagonal_for_the_resolution() {
        // 24" 1080p, 27" 1440p and 4K, 21.5" below 1920 pixels across.
        assert_eq!(assumed_size_mm(1920, 1080), (531, 298));
        assert_eq!(assumed_size_mm(2560, 1440), (597, 335));
        assert_eq!(assumed_size_mm(3840, 2160), (597, 335));
        assert_eq!(assumed_size_mm(1280, 720), (476, 267));
        let e = edid(FULL_HD, 60_000).unwrap();
        assert_eq!(dtd_size_mm(&e), (531, 298));
        assert_eq!((e[21], e[22]), (54, 30));
    }

    #[test]
    fn the_timing_record_is_the_ioav_decode_of_the_edid_timing() {
        let a = Advertised::new(FULL_HD).unwrap();
        let t = &a.timing;
        // 1920 + 320 by 1080 + 45 at 151.2 MHz: exactly 60 Hz, 67.5 kHz.
        assert_eq!(
            [
                get(t, 0x04),
                get(t, 0x08),
                get(t, 0x0c),
                get(t, 0x10),
                get(t, 0x14)
            ],
            [2240, 1920, 32, 240, 48]
        );
        assert_eq!(
            [
                get(t, 0x24),
                get(t, 0x28),
                get(t, 0x2c),
                get(t, 0x30),
                get(t, 0x34)
            ],
            [1125, 1080, 10, 32, 3]
        );
        assert_eq!(get(t, 0x44), 151_200_000);
        assert_eq!(get(t, 0x38), 60 << 16);
        assert_eq!(get(t, 0x18), 0x0043_8000);
        assert_eq!((get(t, 0x1c), get(t, 0x3c)), (1, 1));
        // 16:9 is code 15 (IOAV string table index 14 plus one).
        assert_eq!(get(t, 0x48), 15);
        assert_eq!(get(t, 0x4c), 0);
        // 124 MHz of active pixels at 60 Hz.
        assert_eq!(timing_score(t), 184);
    }

    /// The timing record's vertical rate in Hz, rounded.
    fn hertz(a: &Advertised) -> u32 {
        (get(&a.timing, 0x38) + 0x8000) >> 16
    }

    #[test]
    fn a_fast_monitor_is_advertised_at_its_refresh() {
        let qhd = Mode {
            width: 2560,
            height: 1440,
            ..FULL_HD
        };
        // 2880 by 1485 at 144 Hz fits a detailed timing with the nominal
        // blanking: 615.86 MHz.
        let a = Advertised::at(qhd, 144_000).unwrap();
        assert_eq!(hertz(&a), 144);
        assert_eq!((get(&a.timing, 0x04), get(&a.timing, 0x24)), (2880, 1485));
        assert_eq!(get(&a.timing, 0x44), 615_860_000);
        // The range limits hold the timing's 144 Hz and 214 kHz, 620 MHz.
        assert_eq!(&a.edid[95..100], &[50, 144, 30, 214, 62]);
        // 2560x1600 at 144 Hz needs reduced blanking: 80 pixels, and 114
        // lines for 460 us.
        let wide = Mode {
            height: 1600,
            ..qhd
        };
        let a = Advertised::at(wide, 144_000).unwrap();
        assert_eq!(hertz(&a), 144);
        assert_eq!((get(&a.timing, 0x04), get(&a.timing, 0x24)), (2640, 1714));
        assert_eq!([get(&a.timing, 0x14), get(&a.timing, 0x10)], [8, 40]);
        // Without it the clock would not fit, and 4K at 120 Hz does not even
        // with it.
        let uhd = Mode {
            width: 3840,
            height: 2160,
            ..FULL_HD
        };
        assert_eq!(hertz(&Advertised::at(uhd, 60_000).unwrap()), 60);
        assert!(Advertised::at(uhd, 120_000).is_err());
    }

    #[test]
    fn the_color_record_packs_like_with_capabilities() {
        let c = color_data();
        assert_eq!(
            [
                get(&c, 0),
                get(&c, 4),
                get(&c, 8),
                get(&c, 0xc),
                get(&c, 0x10)
            ],
            [8, 0, 0, 10, 0]
        );
        assert!(c[0x14..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn video_data_answers_only_the_advertised_ids() {
        let a = Advertised::new(FULL_HD).unwrap();
        let (timing, color) = a.video_data(TIMING_ID, COLOR_ID).unwrap();
        assert_eq!((timing.len(), color.len()), (TIMING_BYTES, COLOR_BYTES));
        assert!(a.video_data(COLOR_ID, TIMING_ID).is_none());
        assert!(!(500..=10027).contains(&TIMING_ID));
    }

    #[test]
    fn the_properties_carry_the_keys_restore_reads() {
        let a = Advertised::new(FULL_HD).unwrap();
        let timings = a.timing_elements();
        for key in [
            "<key>ID</key><integer size=\"64\">1</integer>",
            "<key>Score</key><integer size=\"32\">184</integer>",
            "<key>ElementData</key><data>",
            "<key>ColorModes</key><array><dict>",
            "<key>Active</key><integer size=\"32\">1920</integer>",
            "<key>Active</key><integer size=\"32\">1080</integer>",
            "<key>SyncRate</key><integer size=\"32\">3932160</integer>",
            "<key>PixelEncoding</key><integer size=\"32\">0</integer>",
            "<key>Depth</key><integer size=\"32\">8</integer>",
            "<key>EOTF</key><integer size=\"32\">0</integer>",
            "<key>DynamicRange</key><integer size=\"32\">0</integer>",
        ] {
            assert!(timings.contains(key), "{key}");
        }
        assert!(timings.starts_with("<array><dict>") && timings.ends_with("</dict></array>"));
        let colors = a.color_elements();
        assert!(colors.contains("<key>ID</key><integer size=\"64\">2</integer>"));
        assert!(colors.contains("<key>Score</key><integer size=\"32\">242999</integer>"));
    }

    /// The value of integer `key` in a serialized property.
    fn integer(xml: &str, key: &str) -> u64 {
        let open = format!("<key>{key}</key><integer size=\"");
        let start = xml.find(&open).unwrap_or_else(|| panic!("{key}")) + open.len();
        let value = &xml[start..];
        let value = &value[value.find('>').unwrap() + 1..];
        value[..value.find('<').unwrap()].parse().unwrap()
    }

    /// The UI scale SkyLight 27's density rule picks as the default for a
    /// display publishing `attributes` (`GenerateModeListForDisplay`, guest
    /// dyld cache 0x1828f9cec..0x1828fad38, no override file for a vendor
    /// that is not Apple's): physical width in inches from
    /// `MaxHorizontalImageSize` cm x 0.3937 (QuartzCore 0x18e0bdfe0); half
    /// the native density from 3840 x 2160 up, else whichever of the native
    /// density and its half is nearer 110 ppi.
    fn density_default_scale(attributes: &str) -> u32 {
        let width = integer(attributes, "NativeFormatHorizontalPixels") as f64;
        let height = integer(attributes, "NativeFormatVerticalPixels") as f64;
        let inches = integer(attributes, "MaxHorizontalImageSize") as f64 * 0.3937;
        assert!(inches > 6.5, "a physical width SkyLight uses");
        let ppi = width / inches;
        let half = ppi / 2.0;
        let target =
            if width >= 3840.0 && height >= 2160.0 || (half - 110.0).abs() < (ppi - 110.0).abs() {
                half
            } else {
                ppi
            };
        if target == ppi { 1 } else { 2 }
    }

    #[test]
    fn the_first_published_attributes_already_give_the_settled_default_mode() {
        // The monitor of the measured runs (EDID 650 x 369 mm) and the sizes
        // assumed when a monitor reports none. WindowServer's settled mode in
        // those runs, 2560 x 1440 at scale 1, carries the default flag from
        // the first mode list on (run 20261110-221312, `FMD` lines).
        let qhd = Mode {
            width: 2560,
            height: 1440,
            format: PixelFormat::Bgra8888,
        };
        let uhd = Mode {
            width: 3840,
            height: 2160,
            format: PixelFormat::Bgra8888,
        };
        for (mode, size, scale) in [
            (qhd, Some((650, 369)), 1),
            (qhd, None, 1),
            (FULL_HD, None, 1),
            (FULL_HD, Some((309, 174)), 1),
            (
                Mode {
                    width: 2560,
                    height: 1600,
                    format: PixelFormat::Bgra8888,
                },
                Some((286, 179)),
                2,
            ),
            (uhd, None, 2),
        ] {
            let a = Advertised::sized(mode, 60_000, size).unwrap();
            let attributes = a.display_attributes();
            assert_eq!(
                integer(&attributes, "NativeFormatHorizontalPixels"),
                u64::from(mode.width)
            );
            assert_eq!(
                integer(&attributes, "NativeFormatVerticalPixels"),
                u64::from(mode.height)
            );
            // The image size in the attributes is the EDID's, which the
            // detailed timing gives in millimetres.
            let (width_mm, height_mm) = dtd_size_mm(&a.edid);
            assert_eq!(
                integer(&attributes, "MaxHorizontalImageSize"),
                u64::from(width_mm.div_ceil(10))
            );
            assert_eq!(
                integer(&attributes, "MaxVerticalImageSize"),
                u64::from(height_mm.div_ceil(10))
            );
            // The one timing is the native size and preferred, so the
            // native size SkyLight reads matches the attributes.
            let timings = a.timing_elements();
            assert_eq!(timings.matches("<key>IsPreferred</key><true/>").count(), 1);
            assert_eq!(timings.matches("<key>TimingType</key>").count(), 1);
            assert!(timings.contains(&format!(
                "<key>Active</key><integer size=\"32\">{}</integer>",
                mode.width
            )));
            assert!(timings.contains(&format!(
                "<key>Active</key><integer size=\"32\">{}</integer>",
                mode.height
            )));
            assert_eq!(
                density_default_scale(&attributes),
                scale,
                "{mode:?} {size:?}"
            );
            // Building the answers again gives the same bytes: nothing in
            // them depends on when they are asked.
            assert_eq!(Advertised::sized(mode, 60_000, size).unwrap(), a);
        }
    }

    #[test]
    fn base64_pads_like_the_xml_data_element_expects() {
        let mut out = String::new();
        base64(&mut out, b"Man");
        base64(&mut out, b"Ma");
        base64(&mut out, b"M");
        assert_eq!(out, "TWFuTWE=TQ==");
    }
}

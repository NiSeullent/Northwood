//! corpus/host.metal: the host's side of the device. Drawables of a layer
//! in each pixel format, IOSurface-backed textures over one and two
//! planes, a texture and an event another process opens, binary archives
//! written and loaded, the errors creation calls report, command buffer
//! status and enqueue order, purgeable state, and work at scale: grids at
//! their dimension limits, long command streams, threads and queues
//! encoding at once, resources dropped while in flight and a buffer past
//! 4 GiB.
use super::plan::{Buffer, Case, Output, Plan, Texture, Tolerance};
use super::values::{Dtype, Random, f32s, u32s};
use serde_json::json;

const SOURCE: &str = "host.metal";

fn props(count: usize) -> Buffer {
    Buffer::zeroed("props", 0, count * 4).unbound()
}

/// One texel of colour `c`, whose components are 0 or 1, in `format`.
fn texel(format: &str, c: [u32; 4]) -> Vec<u8> {
    let [r, g, b, a] = c;
    match format {
        "bgra8unorm" | "bgra8unorm_srgb" => [b, g, r, a].map(|v| (v * 255) as u8).to_vec(),
        "rgba16float" => c
            .iter()
            .flat_map(|&v| (if v == 1 { 0x3c00u16 } else { 0 }).to_le_bytes())
            .collect(),
        "bgr10a2unorm" => ((b * 1023) | (g * 1023) << 10 | (r * 1023) << 20 | (a * 3) << 30)
            .to_le_bytes()
            .to_vec(),
        _ => unreachable!("{format}"),
    }
}

/// Three drawables of an 8x4 layer, each cleared and drawn on its left
/// half, then read back (unless framebuffer only) and presented.
fn drawables(plan: &mut Plan, library: &str) {
    let (w, h) = (8usize, 4usize);
    let clears = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]];
    let color = [1, 1, 0, 1];
    let values = [
        ("bgra8unorm", 80),
        ("bgra8unorm_srgb", 81),
        ("rgba16float", 115),
        ("bgr10a2unorm", 94),
    ];
    for (format, value) in values {
        let bytes = texel(format, color).len();
        let image: Vec<u8> = clears
            .iter()
            .flat_map(|&clear| {
                (0..w * h)
                    .flat_map(move |i| texel(format, if i % w < w / 2 { color } else { clear }))
            })
            .collect();
        let expected: Vec<u32> = (0..3)
            .flat_map(|_| [w as u32, h as u32, value, 1, 1])
            .collect();
        plan.add(
            Case::host(
                &format!("host.drawables.{format}"),
                library,
                json!({"kind": "drawables", "format": format, "width": w, "height": h,
                       "count": 3, "clears": clears, "color": color}),
            )
            .buffer(Buffer::zeroed("out", 0, 3 * w * h * bytes).unbound())
            .buffer(props(15))
            .output(Output::exact("out", Dtype::U8, image))
            .output(Output::exact("props", Dtype::U32, u32s(&expected))),
        );
    }
    let expected: Vec<u32> = (0..3)
        .flat_map(|_| [w as u32, h as u32, 80, 1, 0])
        .collect();
    plan.add(
        Case::host(
            "host.drawables.framebuffer_only",
            library,
            json!({"kind": "drawables", "format": "bgra8unorm", "width": w, "height": h,
                   "count": 3, "clears": clears, "color": color, "framebuffer_only": true}),
        )
        .buffer(props(15))
        .output(Output::exact("props", Dtype::U32, u32s(&expected))),
    );
    // Six drawables, each presented a different way, every presentation
    // reported to its handler.
    let present = [
        "command_buffer",
        "command_buffer_at",
        "command_buffer_after",
        "drawable",
        "drawable_at",
        "drawable_after",
    ];
    let six: Vec<u32> = (0..6)
        .flat_map(|_| [w as u32, h as u32, 80, 1, 0])
        .collect();
    let six_clears = [clears, clears].concat();
    plan.add(
        Case::host(
            "host.drawables.presentation",
            library,
            json!({"kind": "drawables", "format": "bgra8unorm", "width": w, "height": h,
                   "count": 6, "clears": six_clears, "color": color,
                   "framebuffer_only": true, "present": present}),
        )
        .buffer(props(30))
        .buffer(Buffer::zeroed("presented", 0, 32).unbound())
        .output(Output::exact("props", Dtype::U32, u32s(&six)))
        .output(Output::exact(
            "presented",
            Dtype::U32,
            u32s(&[1, 1, 1, 1, 1, 1, 0, 0]),
        )),
    );
}

/// A BGRA surface the CPU fills and a kernel inverts into another the CPU
/// reads, and a two-plane 4:2:0 surface a kernel reads both planes of.
fn iosurface(plan: &mut Plan, library: &str, rng: &mut Random) {
    let (w, h) = (16usize, 8usize);
    let bgra: Vec<u8> = (0..w * h * 4).map(|_| rng.u32() as u8).collect();
    let inverted: Vec<u8> = bgra
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 4 == 3 { v } else { 255 - v })
        .collect();
    let luma: Vec<u8> = (0..w * h).map(|_| rng.u32() as u8).collect();
    let chroma: Vec<u8> = (0..w * h / 2).map(|_| rng.u32() as u8).collect();
    let planes: Vec<u32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let c = (y / 2 * (w / 2) + x / 2) * 2;
            u32::from(luma[i]) | u32::from(chroma[c]) << 8 | u32::from(chroma[c + 1]) << 16
        })
        .collect();
    plan.add(
        Case::host(
            "host.iosurface",
            library,
            json!({"kind": "iosurface", "width": w, "height": h}),
        )
        .buffer(Buffer::data("in", 0, bgra).unbound())
        .buffer(Buffer::data("luma", 0, luma).unbound())
        .buffer(Buffer::data("chroma", 0, chroma).unbound())
        .buffer(Buffer::zeroed("out", 0, w * h * 4).unbound())
        .buffer(Buffer::zeroed("planes", 0, w * h * 4).unbound())
        .buffer(props(4))
        .output(Output::new("out", Dtype::U8, Tolerance::units(1.0)).expect_same(inverted))
        .output(Output::exact("planes", Dtype::U32, u32s(&planes)))
        .output(Output::exact(
            "props",
            Dtype::U32,
            u32s(&[1, 0, w as u32, h as u32]),
        )),
    );
}

/// A buffer over an IOSurface's memory: the CPU fills the surface, a kernel
/// adds to each word in place, and the CPU reads the sums through the
/// surface. The buffer names its surface and is shared with the CPU.
fn iosurface_buffer(plan: &mut Plan, library: &str, rng: &mut Random) {
    let words = 256usize;
    let add = 0x0101_0101u32;
    let input: Vec<u32> = (0..words).map(|_| rng.u32()).collect();
    let sums: Vec<u32> = input.iter().map(|v| v.wrapping_add(add)).collect();
    plan.add(
        Case::host(
            "host.iosurface_buffer",
            library,
            json!({"kind": "iosurface_buffer", "words": words, "add": add}),
        )
        .buffer(Buffer::data("in", 0, u32s(&input)).unbound())
        .buffer(Buffer::zeroed("out", 0, words * 4).unbound())
        .buffer(props(3))
        .output(Output::exact("out", Dtype::U32, u32s(&sums)))
        .output(Output::exact("props", Dtype::U32, u32s(&[1, 1, 0]))),
    );
}

/// Surfaces of Apple's private two-plane RGB formats (540, 541 and 542 for
/// 4:2:0, 4:2:2 and 4:4:4): green bytes in the first plane, blue and red
/// pairs in the second. A kernel reads each texel and samples each at its
/// centre; both give the second plane filtered linearly at its own
/// resolution there, so a read of a subsampled plane blends its texels.
fn two_plane(plan: &mut Plan, library: &str, rng: &mut Random) {
    let (w, h) = (16usize, 8usize);
    let formats = [
        ("420", 540, 2, 2),
        ("422", 541, 2, 1),
        ("444", 542, 1, 1),
        ("ycbcr_420", 500, 2, 2),
        ("ycbcr_422", 502, 2, 1),
        ("ycbcr_444", 503, 1, 1),
    ];
    // YCbCr cases draw from their own stream, leaving later cases' inputs.
    let mut own = Random::new(500);
    for (name, format, across, down) in formats {
        let ycbcr = format < 540;
        let source = if ycbcr { &mut own } else { &mut *rng };
        let (cw, ch) = (w / across, h / down);
        let green: Vec<u8> = (0..w * h).map(|_| source.u32() as u8).collect();
        let pairs: Vec<u8> = (0..cw * ch * 2).map(|_| source.u32() as u8).collect();
        let pack = |[r, g, b]: [f64; 3]| {
            let unorm = |v: f64| (v.round() as u32).min(255);
            unorm(r) | unorm(g) << 8 | unorm(b) << 16 | 255 << 24
        };
        // The first plane's value and the second's two, in whole steps, as
        // RGB: a YCbCr texel through full-range BT.601, its chroma scaled
        // by 255 / 256 as Apple GPUs scale it.
        let rgb = |first: f64, second: [f64; 2]| {
            if !ycbcr {
                return [second[1], first, second[0]];
            }
            let [cb, cr] = second.map(|c| (c - 128.0) * 255.0 / 256.0);
            [
                first + 1.402 * cr,
                first - 0.344136 * cb - 0.714136 * cr,
                first + 1.772 * cb,
            ]
            .map(|v| v.clamp(0.0, 255.0))
        };
        let pair = |x: usize, y: usize, c: usize| f64::from(pairs[(y * cw + x) * 2 + c]);
        // A clamped linear tap's texels and the second's weight along one axis.
        let taps = |p: usize, n: usize, size: usize| {
            let u = (p as f64 + 0.5) * size as f64 / n as f64 - 0.5;
            let first = u.floor();
            let clamp = |i: f64| i.clamp(0.0, size as f64 - 1.0) as usize;
            (clamp(first), clamp(first + 1.0), u - first)
        };
        let mut read = Vec::new();
        let mut sampled = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let g = f64::from(green[y * w + x]);
                let (x0, x1, fx) = taps(x, w, cw);
                let (y0, y1, fy) = taps(y, h, ch);
                let filter = |c| {
                    let row = |y| pair(x0, y, c) * (1.0 - fx) + pair(x1, y, c) * fx;
                    row(y0) * (1.0 - fy) + row(y1) * fy
                };
                // Apple GPUs upsample a subsampled plane to whole values,
                // rounding halves up; sampling keeps the fraction.
                let filtered = [filter(0), filter(1)];
                read.push(pack(rgb(g, filtered.map(f64::round))));
                sampled.push(pack(rgb(g, filtered)));
            }
        }
        // Luma and chroma convert in hardware to within half a step.
        let read = if ycbcr {
            Output::new("read", Dtype::U8, Tolerance::units(1.0)).expect_same(u32s(&read))
        } else {
            Output::exact("read", Dtype::U32, u32s(&read))
        };
        plan.add(
            Case::host(
                &format!("host.iosurface.two_plane_{name}"),
                library,
                json!({"kind": "two_plane", "format": format, "width": w, "height": h,
                       "across": across, "down": down}),
            )
            .buffer(Buffer::data("green", 0, green).unbound())
            .buffer(Buffer::data("blue_red", 0, pairs).unbound())
            .buffer(Buffer::zeroed("read", 0, w * h * 4).unbound())
            .buffer(Buffer::zeroed("sampled", 0, w * h * 4).unbound())
            .output(read)
            .output(
                Output::new("sampled", Dtype::U8, Tolerance::units(1.0))
                    .expect_same(u32s(&sampled)),
            ),
        );
    }
}

/// Surfaces of Apple's 10-bit two-plane formats (505-507 YCbCr and 543-545
/// RGB for 4:2:0, 4:2:2 and 4:4:4), each component the high 10 bits of a
/// 16-bit word, whose low 6 bits shaders ignore, read as floats. As
/// measured on an M1 Pro: a read gives the second plane filtered linearly
/// at the texel centre and rounded to whole steps, halves up; RGB reads
/// are those steps over 1023. YCbCr converts through BT.601 in the range
/// the surface's pixel format names, not the Metal format: full for
/// '444f' (luma over 1023, chroma (C - 512) / 1024, as the 8-bit formats'
/// (C - 128) / 256) and video for 'x420' and '420v' (luma (Y - 64) / 876,
/// chroma (C - 512) / 896, in 8-bit steps a quarter of those), rounded to
/// 1/2048 and clamped. The 8-bit 500 over a '420v' surface is here too,
/// since its range also follows the surface. A linear sample at the
/// centre agrees with the read to within a step of each component, as
/// the conversion scales it: filtering keeps the fraction, and Stela's
/// samples also take the share of a step the words' low bits hold.
fn two_plane_ten_bit(plan: &mut Plan, library: &str, rng: &mut Random) {
    let (w, h) = (16usize, 8usize);
    // Name, format, subsampling, bits and the surface's pixel format.
    let formats = [
        ("rgb10_420", 543, 2, 2, 10, "444f"),
        ("rgb10_422", 544, 2, 1, 10, "444f"),
        ("rgb10_444", 545, 1, 1, 10, "444f"),
        ("ycbcr10_420", 505, 2, 2, 10, "444f"),
        ("ycbcr10_422", 506, 2, 1, 10, "444f"),
        ("ycbcr10_444", 507, 1, 1, 10, "444f"),
        ("ycbcr10_420_video", 505, 2, 2, 10, "x420"),
        ("ycbcr_420_video", 500, 2, 2, 8, "420v"),
    ];
    for (name, format, across, down, bits, fourcc) in formats {
        let ycbcr = format < 540;
        let video = fourcc != "444f";
        let wide = bits == 10;
        let (cw, ch) = (w / across, h / down);
        let max = f64::from((1u32 << bits) - 1);
        // A component's step: the high 10 bits of a word whose low ones
        // hold noise, or a byte.
        let mut component = || {
            let random = rng.u32();
            if wide {
                (random & 0x3ff) << 6 | random >> 26
            } else {
                random & 0xff
            }
        };
        let green: Vec<u32> = (0..w * h).map(|_| component()).collect();
        let pairs: Vec<u32> = (0..cw * ch * 2).map(|_| component()).collect();
        let step = |value: u32| f64::from(if wide { value >> 6 } else { value });
        let pair = |x: usize, y: usize, c: usize| step(pairs[(y * cw + x) * 2 + c]);
        let taps = |p: usize, n: usize, size: usize| {
            let u = (p as f64 + 0.5) * size as f64 / n as f64 - 0.5;
            let first = u.floor();
            let clamp = |i: f64| i.clamp(0.0, size as f64 - 1.0) as usize;
            (clamp(first), clamp(first + 1.0), u - first)
        };
        let scale = (max + 1.0) / 256.0;
        let rgba = |first: f64, [b, r]: [f64; 2]| {
            if !ycbcr {
                return [r / max, first / max, b / max, 1.0];
            }
            let half = (max + 1.0) / 2.0;
            let (y, [cb, cr]) = if video {
                (
                    (first - 16.0 * scale) / (219.0 * scale),
                    [b, r].map(|c| (c - half) / (224.0 * scale)),
                )
            } else {
                (first / max, [b, r].map(|c| (c - half) / (max + 1.0)))
            };
            let quantized = |v: f64| (v.clamp(0.0, 1.0) * 2048.0).round() / 2048.0;
            [
                quantized(y + 1.402 * cr),
                quantized(y - 0.344136 * cb - 0.714136 * cr),
                quantized(y + 1.772 * cb),
                1.0,
            ]
        };
        let mut read = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let (x0, x1, fx) = taps(x, w, cw);
                let (y0, y1, fy) = taps(y, h, ch);
                let filter = |c| {
                    let row = |y| pair(x0, y, c) * (1.0 - fx) + pair(x1, y, c) * fx;
                    (row(y0) * (1.0 - fy) + row(y1) * fy + 0.5).floor()
                };
                read.extend(rgba(step(green[y * w + x]), [filter(0), filter(1)]));
            }
        }
        // Half a step, and for YCbCr two of the conversion's 1/2048 more.
        let step_tolerance = 0.5 / max;
        let tolerance = if ycbcr {
            step_tolerance + 2.0 / 2048.0
        } else {
            step_tolerance
        };
        // A step of each component in the result: a luma step with a
        // chroma step through blue's 1.772, the largest coefficient.
        let filtered_step = if !ycbcr {
            1.0 / max
        } else if video {
            1.0 / (219.0 * scale) + 1.772 / (224.0 * scale)
        } else {
            1.0 / max + 1.772 / (max + 1.0)
        };
        let elements = |values: &[u32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|&value| {
                    if wide {
                        (value as u16).to_le_bytes().to_vec()
                    } else {
                        vec![value as u8]
                    }
                })
                .collect()
        };
        plan.add(
            Case::host(
                &format!("host.iosurface.two_plane_{name}"),
                library,
                json!({"kind": "two_plane", "format": format, "width": w, "height": h,
                       "across": across, "down": down, "bytes": if wide { 2 } else { 1 },
                       "wide": 1, "fourcc": fourcc}),
            )
            .buffer(Buffer::data("green", 0, elements(&green)).unbound())
            .buffer(Buffer::data("blue_red", 0, elements(&pairs)).unbound())
            .buffer(Buffer::zeroed("read", 0, w * h * 16).unbound())
            .buffer(Buffer::zeroed("sampled", 0, w * h * 16).unbound())
            .output(
                Output::new("read", Dtype::F32, Tolerance::abs(tolerance)).expect_same(f32s(&read)),
            )
            .output(
                Output::new(
                    "sampled",
                    Dtype::F32,
                    Tolerance::abs(tolerance + filtered_step),
                )
                .expect_same(f32s(&read)),
            ),
        );
    }
}

/// A 4:4:4 two-plane format's planes of an RGB color, in whole steps:
/// green and blue-red, or full-range BT.601 luma and chroma whose chroma is
/// scaled by 256 / 255 around 128, as Apple GPUs store them.
fn planes_of(ycbcr: bool, [r, g, b]: [f64; 3]) -> [f64; 3] {
    if !ycbcr {
        return [g, b, r].map(|v| v * 255.0);
    }
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let chroma = |difference: f64, scale: f64| 128.0 + difference * 255.0 / scale * 256.0 / 255.0;
    [y * 255.0, chroma(b - y, 1.772), chroma(r - y, 1.402)].map(|v| v.clamp(0.0, 255.0))
}

/// The RGB color, in [0, 1], of a 4:4:4 two-plane texel's bytes.
fn rgb_of(ycbcr: bool, [first, a, b]: [u8; 3]) -> [f64; 3] {
    let [first, a, b] = [first, a, b].map(f64::from);
    if !ycbcr {
        return [b, first, a].map(|v| v / 255.0);
    }
    let [cb, cr] = [a, b].map(|c| (c - 128.0) * 255.0 / 256.0);
    [
        first + 1.402 * cr,
        first - 0.344136 * cb - 0.714136 * cr,
        first + 1.772 * cb,
    ]
    .map(|v| (v / 255.0).clamp(0.0, 1.0))
}

/// Rendering into 4:4:4 two-plane surfaces: blending over loaded planes,
/// and clearing then drawing half (`host_two_plane_render`). The surfaces
/// start with colors inside the RGB cube; luma and chroma convert to
/// within a step.
fn two_plane_render(plan: &mut Plan, library: &str, rng: &mut Random) {
    let (w, h) = (8usize, 4usize);
    for (name, format) in [("rgb", 542), ("ycbcr", 503)] {
        let ycbcr = format == 503;
        let mut unit = || rng.uniform(0.0, 1.0);
        let initial: Vec<[f64; 3]> = (0..w * h).map(|_| [unit(), unit(), unit()]).collect();
        let colors: Vec<[f64; 4]> = (0..w * h)
            .map(|_| [unit(), unit(), unit(), unit()])
            .collect();
        let clear = [unit(), unit(), unit(), 1.0];
        let bytes = |rgb: [f64; 3]| planes_of(ycbcr, rgb).map(|v| v.round() as u8);
        let texels: Vec<[u8; 3]> = initial.iter().map(|&rgb| bytes(rgb)).collect();
        let planes = |texels: &[[u8; 3]]| -> Vec<u8> {
            let mut out: Vec<u8> = texels.iter().map(|t| t[0]).collect();
            out.extend(texels.iter().flat_map(|t| [t[1], t[2]]));
            out
        };
        let loaded: Vec<[u8; 3]> = texels
            .iter()
            .zip(&colors)
            .map(|(&texel, color)| {
                let destination = rgb_of(ycbcr, texel);
                let a = color[3];
                bytes([0, 1, 2].map(|i| color[i] * a + destination[i] * (1.0 - a)))
            })
            .collect();
        let cleared: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let color = if i % w < w / 2 { colors[i] } else { clear };
                bytes([color[0], color[1], color[2]])
            })
            .collect();
        let flat: Vec<f64> = colors.iter().flatten().copied().collect();
        let first: Vec<u8> = texels.iter().map(|t| t[0]).collect();
        let second: Vec<u8> = texels.iter().flat_map(|t| [t[1], t[2]]).collect();
        let output = |name: &str, texels: &[[u8; 3]]| {
            Output::new(name, Dtype::U8, Tolerance::units(1.0)).expect_same(planes(texels))
        };
        plan.add(
            Case::host(
                &format!("host.iosurface.two_plane_render_{name}"),
                library,
                json!({"kind": "two_plane_render", "format": format, "width": w, "height": h,
                       "clear": clear}),
            )
            .buffer(Buffer::data("green", 0, first).unbound())
            .buffer(Buffer::data("blue_red", 0, second).unbound())
            .buffer(Buffer::data("colors", 0, f32s(&flat)).unbound())
            .buffer(Buffer::zeroed("loaded", 0, w * h * 3).unbound())
            .buffer(Buffer::zeroed("cleared", 0, w * h * 3).unbound())
            .output(output("loaded", &loaded))
            .output(output("cleared", &cleared)),
        );
    }
}

/// A shared texture this process clears and the share service writes the
/// top rows of, ordered by a shared event both processes wait on.
fn share(plan: &mut Plan, library: &str, rng: &mut Random) {
    let (w, h, rows) = (8usize, 8usize, 4usize);
    let top: Vec<u8> = (0..w * rows * 4).map(|_| rng.u32() as u8).collect();
    let mut image = top.clone();
    image.extend((0..w * (h - rows)).flat_map(|_| [0, 255, 0, 255]));
    plan.add(
        Case::host(
            "host.share.texture_event",
            library,
            json!({"kind": "share", "width": w, "height": h, "clear": [0, 1, 0, 1]}),
        )
        .buffer(Buffer::data("rows", 0, top).unbound())
        .buffer(Buffer::zeroed("out", 0, w * h * 4).unbound())
        .buffer(props(1))
        .output(Output::exact("out", Dtype::U8, image))
        .output(Output::exact("props", Dtype::U32, u32s(&[2])).set("length", 4)),
    );
}

fn archive(plan: &mut Plan, library: &str, rng: &mut Random) {
    let data: Vec<u32> = (0..64).map(|_| rng.u32() >> 4).collect();
    let expected: Vec<u32> = data
        .iter()
        .map(|v| v.wrapping_mul(3).wrapping_add(1))
        .collect();
    plan.add(
        Case::host("host.binary_archive", library, json!({"kind": "archive"}))
            .buffer(Buffer::data("data", 0, u32s(&data)).unbound())
            .buffer(props(12))
            .output(Output::exact("data", Dtype::U32, u32s(&expected)))
            .output(Output::exact(
                "props",
                Dtype::U32,
                u32s(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
            )),
    );
}

/// A library that does not compile reports MTLLibraryErrorCompileFailure
/// (3), an absent function is nil and an unlinked varying fails the
/// pipeline; a command buffer goes from not enqueued (0) through enqueued
/// (1) to completed (4) without error; and the buffer enqueued first runs
/// first though committed second (`v * 10 + digit`: 2, then 21).
fn errors(plan: &mut Plan, library: &str) {
    plan.add(
        Case::host("host.errors_and_status", library, json!({"kind": "errors"}))
            .buffer(Buffer::zeroed("order", 0, 4).unbound())
            .buffer(props(9))
            .output(Output::exact("order", Dtype::U32, u32s(&[21])).set("length", 4))
            .output(Output::exact(
                "props",
                Dtype::U32,
                u32s(&[1, 1, 3, 1, 1, 0, 1, 4, 1]),
            )),
    );
}

/// setPurgeableState: returns the state before it (non-volatile 2,
/// volatile 3, empty 4): keep, empty, keep, non-volatile, keep, volatile,
/// for a buffer, a private texture and a heap.
fn purgeable(plan: &mut Plan, library: &str) {
    let states: Vec<u32> = (0..3).flat_map(|_| [2, 2, 4, 4, 2, 2]).collect();
    plan.add(
        Case::host("host.purgeable", library, json!({"kind": "purgeable"}))
            .buffer(props(18))
            .output(Output::exact("props", Dtype::U32, u32s(&states))),
    );
}

/// 10000 dispatches in one encoder, 512 command buffers in one queue, 8
/// threads committing 64 each into one queue, and 8 queues adding 64
/// each at once.
fn streams(plan: &mut Plan, library: &str) {
    let threads = 8u32;
    let mut counts = vec![10000, 512];
    counts.extend((0..threads).map(|_| 64));
    counts.push(64 * threads);
    plan.add(
        Case::host(
            "host.streams",
            library,
            json!({"kind": "streams", "dispatches": 10000, "command_buffers": 512,
                   "threads": threads, "per_thread": 64}),
        )
        .set("timeout", 60.0)
        .buffer(Buffer::zeroed("counts", 0, counts.len() * 4).unbound())
        .output(Output::exact("counts", Dtype::U32, u32s(&counts))),
    );
}

/// 256 rounds of a 1 MiB buffer and a private texture made, used by one
/// command buffer and dropped before it runs.
fn churn(plan: &mut Plan, library: &str) {
    let rounds = 256u32;
    let words: Vec<u32> = (0..rounds).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let texels: Vec<u32> = (0..rounds).collect();
    plan.add(
        Case::host(
            "host.churn",
            library,
            json!({"kind": "churn", "rounds": rounds, "size": 1 << 20}),
        )
        .set("timeout", 60.0)
        .buffer(Buffer::zeroed("words", 0, rounds as usize * 4).unbound())
        .buffer(Buffer::zeroed("texels", 0, rounds as usize * 4).unbound())
        .output(Output::exact("words", Dtype::U32, u32s(&words)))
        .output(Output::exact("texels", Dtype::U32, u32s(&texels))),
    );
}

/// Rounds that create buffers and textures of each storage mode, read them
/// back, fill them with the round's byte and free them, so later rounds
/// take memory and recycled textures earlier ones held. Metal documents a
/// new buffer as zero-filled (`newBufferWithLength:options:`): each counts
/// its nonzero bytes. A new texture's contents are undefined, but never
/// bytes of another resource: each counts bytes that are neither zero nor a
/// byte this case wrote. Both counts are zero.
fn fresh_resources(plan: &mut Plan, library: &str) {
    let rounds = 6usize;
    // Pooled small shared buffers, buffers placed in blocks, and buffers
    // with memory of their own.
    let buffer_sizes = [4096u32, 1 << 20, 48 << 20];
    let texture_sizes = 2;
    let per_round = buffer_sizes.len() * 3 + texture_sizes * 3;
    plan.add(
        Case::host(
            "host.fresh_resources",
            library,
            json!({"kind": "fresh_resources", "rounds": rounds, "buffer_sizes": buffer_sizes,
                   "first_fill": 0xa1}),
        )
        .set("timeout", 60.0)
        .buffer(props(rounds * per_round))
        .output(Output::exact(
            "props",
            Dtype::U32,
            u32s(&vec![0; rounds * per_round]),
        )),
    );
}

/// A private buffer of 4 GiB and 32 MiB, stamped below 2 GiB, across 2
/// and 4 GiB and at its end, and filled past 4 GiB.
fn large_buffer(plan: &mut Plan, library: &str) {
    let length: u64 = (1 << 32) + (1 << 25);
    let count = 256u32;
    let windows = [
        0u32,
        (1 << 29) - count / 2,
        (1 << 30) - count / 2,
        (length / 4) as u32 - count,
    ];
    let fill = ((1u64 << 32) + (1 << 24), 128usize, 0xa5u8);
    let mut expected: Vec<u8> = windows
        .iter()
        .flat_map(|&first| u32s(&(first..first + count).collect::<Vec<_>>()))
        .collect();
    expected.extend(std::iter::repeat_n(fill.2, fill.1));
    plan.add(
        Case::host(
            "host.large_buffer",
            library,
            json!({"kind": "large_buffer", "length": length,
                   "windows": windows.map(|w| [w, count]), "fill": [fill.0, fill.1, fill.2]}),
        )
        .set("timeout", 60.0)
        .buffer(Buffer::zeroed("out", 0, expected.len()).unbound())
        .output(Output::exact("out", Dtype::U8, expected)),
    );
}

/// The sum, xor, count and largest `y + z` of every thread's position
/// hash, as grid_hash reduces them.
fn grid_hash(grid: [u32; 3]) -> [u32; 4] {
    let mut out = [0u32; 4];
    for z in 0..grid[2] {
        for y in 0..grid[1] {
            let yz = y.wrapping_mul(0x85eb_ca77) ^ z.wrapping_mul(0xc2b2_ae3d);
            for x in 0..grid[0] {
                let h = x.wrapping_mul(0x9e37_79b1) ^ yz;
                out[0] = out[0].wrapping_add(h);
                out[1] ^= h;
            }
            out[3] = out[3].max(y + z);
        }
    }
    out[2] = grid[0].wrapping_mul(grid[1]).wrapping_mul(grid[2]);
    out
}

/// A non-uniform grid of 67.7 million threads, and 65535 threadgroups
/// along y and along z.
fn grids(plan: &mut Plan, library: &str) {
    let cases: [(&str, [u32; 3], [u32; 3], bool); 3] = [
        ("non_uniform", [100_003, 677, 1], [256, 1, 1], true),
        ("y_limit", [1, 65535, 1], [32, 1, 1], false),
        ("z_limit", [1, 1, 65535], [32, 1, 1], false),
    ];
    for (label, grid, group, threads) in cases {
        let total = if threads {
            grid
        } else {
            [grid[0] * group[0], grid[1] * group[1], grid[2] * group[2]]
        };
        let mut case = Case::compute(
            &format!("host.grid.{label}"),
            library,
            "grid_hash",
            grid,
            group,
        )
        .set("threadgroup_memory", json!([{"index": 0, "length": 16}]))
        .set("timeout", 60.0)
        .buffer(Buffer::zeroed("sums", 0, 16))
        .output(Output::exact("sums", Dtype::U32, u32s(&grid_hash(total))));
        if threads {
            case = case.dispatch_threads();
        }
        plan.add(case);
    }
}

/// An 8x4 RGBA8 image, its left half `left` and right half `right`.
fn halves(left: [u8; 4], right: [u8; 4]) -> Vec<u8> {
    (0..32)
        .flat_map(|i| if i % 8 < 4 { left } else { right })
        .collect()
}

/// Two rectangles (x0, y0, x1, y1, z) or colours, 256 bytes apart.
fn spaced(first: &[f64], second: &[f64]) -> Vec<u8> {
    let mut bytes = f32s(first);
    bytes.resize(256, 0);
    bytes.extend(f32s(second));
    bytes.resize(512, 0);
    bytes
}

/// Offsets moved on bound buffers, buffers and textures bound as ranges
/// and samplers bound with their own level clamps (levels hold 51 * k).
fn rebinding(plan: &mut Plan, library: &str, rng: &mut Random) {
    let words: Vec<u32> = (0..4).map(|_| rng.u32()).collect();
    let level2 = 102.0 / 255.0;
    plan.add(
        Case::host("host.rebind", library, json!({"kind": "rebind"}))
            .buffer(Buffer::data("words", 0, u32s(&words)).unbound())
            .buffer(Buffer::zeroed("out", 0, 12).unbound())
            .buffer(Buffer::zeroed("levels", 0, 8).unbound())
            .buffer(
                Buffer::data(
                    "rects",
                    0,
                    spaced(&[-1.0, -1.0, 0.0, 1.0, 0.0], &[0.0, -1.0, 1.0, 1.0, 0.0]),
                )
                .unbound(),
            )
            .buffer(
                Buffer::data(
                    "colors",
                    0,
                    spaced(&[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 1.0, 1.0]),
                )
                .unbound(),
            )
            .buffer(Buffer::zeroed("image", 0, 128).unbound())
            .output(Output::exact("out", Dtype::U32, u32s(&words[..3])).set("length", 12))
            .output(
                Output::new("levels", Dtype::F32, Tolerance::ulp(1.0))
                    .expect_same(f32s(&[level2, level2])),
            )
            .output(Output::exact(
                "image",
                Dtype::U8,
                halves([255, 0, 0, 51], [0, 0, 255, 153]),
            )),
    );
    let mut corners = f32s(&[-1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 1.0]);
    corners.resize(64, 0);
    for [x, y] in [[0.0, -1.0], [1.0, -1.0], [0.0, 1.0], [1.0, 1.0]] {
        corners.extend(f32s(&[x, y, 0.0, 0.0]));
    }
    plan.add(
        Case::host(
            "host.dynamic_stride",
            library,
            json!({"kind": "dynamic_stride", "second_offset": 64, "colors": [[1, 0, 0, 1], [0, 1, 0, 1]]}),
        )
        .buffer(Buffer::data("corners", 0, corners).unbound())
        .buffer(Buffer::zeroed("image", 0, 128).unbound())
        .output(Output::exact("image", Dtype::U8, halves([255, 0, 0, 255], [0, 255, 0, 255]))),
    );
}

/// Why render barriers after the fragment stage, or over render targets,
/// hold no device to them: the feature set tables' note 3 leaves them
/// out of Apple3 to Apple11, and the driver claims Apple3.
const FRAGMENT_BARRIERS: &str =
    "Apple3 to Apple11 have no render barriers after the fragment stage or over render targets";

/// A resource barrier between concurrent dispatches, and stage and
/// resource barriers between draws whose fragments feed the next draw's
/// vertex function (41, then 42, 43, 44).
fn barriers(plan: &mut Plan, library: &str) {
    let n = 64u32;
    let y: Vec<u32> = (0..n).map(|i| (i * 3 + 1) * 2).collect();
    plan.add(
        Case::host("host.barriers", library, json!({"kind": "barriers"}))
            .buffer(Buffer::zeroed("x", 0, n as usize * 4).unbound())
            .buffer(Buffer::zeroed("y", 0, n as usize * 4).unbound())
            .output(Output::exact("y", Dtype::U32, u32s(&y))),
    );
    plan.add(
        Case::host(
            "host.barriers.render",
            library,
            json!({"kind": "barriers", "render": true}),
        )
        .buffer(Buffer::data("chain", 0, u32s(&[0, 0, 0, 41])).unbound())
        .output(Output::exact("chain", Dtype::U32, u32s(&[42, 43, 44, 41])))
        .informational()
        .set("note", FRAGMENT_BARRIERS),
    );
}

/// One image of a managed array texture synchronized alone; two viewports
/// each with its own scissor; store actions chosen after the draw.
fn passes(plan: &mut Plan, library: &str) {
    // A target read by its own pass after a render-target barrier, or
    // after the textureBarrier it replaced: the left half inverted, the
    // right as the first draw filled it.
    let (filled, inverted) = ([51u8, 102, 153, 255], [204u8, 153, 102, 255]);
    let image: Vec<u8> = (0..16)
        .flat_map(|i| if i % 4 < 2 { inverted } else { filled })
        .collect();
    for (name, barrier) in [
        ("host.render_target_barrier", "memory"),
        ("host.render_target_barrier.texture", "texture"),
    ] {
        plan.add(
            Case::host(
                name,
                library,
                json!({"kind": "render_target_barrier", "barrier": barrier}),
            )
            .buffer(Buffer::zeroed("image", 0, 64).unbound())
            .output(Output::exact("image", Dtype::U8, image.clone()))
            .informational()
            .set("note", FRAGMENT_BARRIERS),
        );
    }
    // RenderBox's path fill: a fan's facings summed into a float count by
    // blending, read back by framebuffer fetch on the same pass. A 16 x 16
    // ring between squares of half-sides 6 and 2 pixels fills; the hole and
    // the outside keep the clear, and the count ends cleared.
    let ring = |x: i32, y: i32| {
        let inside =
            |half: i32| (8 - half..8 + half).contains(&x) && (8 - half..8 + half).contains(&y);
        inside(6) && !inside(2)
    };
    let image: Vec<u8> = (0..256)
        .flat_map(|i| {
            if ring(i % 16, i / 16) {
                [255u8, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        })
        .collect();
    plan.add(
        Case::host(
            "host.winding_coverage",
            library,
            json!({"kind": "winding_coverage"}),
        )
        .requires(&["apple3"], &[])
        .buffer(Buffer::zeroed("color", 0, 1024).unbound())
        .buffer(Buffer::zeroed("count", 0, 1024).unbound())
        .output(Output::exact("color", Dtype::U8, image))
        .output(Output::exact("count", Dtype::U8, vec![0; 1024])),
    );
    plan.add(
        Case::host(
            "host.synchronize_texture",
            library,
            json!({"kind": "sync_texture"}),
        )
        .buffer(Buffer::zeroed("out", 0, 16).unbound())
        .output(Output::exact(
            "out",
            Dtype::U8,
            [51, 153, 255, 255].repeat(4),
        )),
    );
    plan.add(
        Case::host("host.scissors", library, json!({"kind": "scissors"}))
            .buffer(
                Buffer::data("colors", 0, f32s(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]))
                    .unbound(),
            )
            .buffer(Buffer::zeroed("image", 0, 128).unbound())
            .output(Output::exact(
                "image",
                Dtype::U8,
                halves([255, 0, 0, 255], [0, 0, 255, 255]),
            )),
    );
    let depth: Vec<f64> = (0..32)
        .map(|i| if i % 8 < 4 { 0.25 } else { 1.0 })
        .collect();
    let stencil: Vec<u8> = (0..32).map(|i| if i % 8 < 4 { 7 } else { 0 }).collect();
    plan.add(
        Case::host(
            "host.deferred_store",
            library,
            json!({"kind": "deferred_store"}),
        )
        .buffer(Buffer::data("rects", 0, f32s(&[-1.0, -1.0, 0.0, 1.0, 0.25])).unbound())
        .buffer(Buffer::data("colors", 0, f32s(&[0.0, 1.0, 0.0, 1.0])).unbound())
        .buffer(Buffer::zeroed("image", 0, 128).unbound())
        .buffer(Buffer::zeroed("depth", 0, 128).unbound())
        .buffer(Buffer::zeroed("stencil", 0, 32).unbound())
        .output(Output::exact(
            "image",
            Dtype::U8,
            halves([0, 255, 0, 255], [0, 0, 0, 255]),
        ))
        .output(Output::exact("depth", Dtype::F32, f32s(&depth)))
        .output(Output::exact("stencil", Dtype::U8, stencil)),
    );
}

/// The entry points a source declares, sorted.
fn entry_points(source: &str) -> Vec<String> {
    let mut names: Vec<String> = source
        .lines()
        .filter_map(|line| {
            let rest = [
                "kernel void ",
                "vertex ",
                "fragment ",
                "[[visible]] ",
                "[[stitchable]] ",
                "[[mesh]] ",
                "[[object]] ",
            ]
            .iter()
            .find_map(|w| line.strip_prefix(w))?;
            let name = rest[..rest.find('(')?].split_whitespace().last()?;
            Some(name.to_string())
        })
        .collect();
    names.sort();
    names
}

/// What the device and its objects report, each used where it can be: a
/// library's function names, a library from a metallib's bytes, texture
/// alignments over buffers, heap sizes and use, timestamps and their
/// frequency, event waits with timeouts, pipeline thread limits and a
/// dynamic library's install name.
fn queries(plan: &mut Plan, library: &str, rng: &mut Random) {
    let source =
        std::fs::read_to_string(super::assets().join("corpus").join(SOURCE)).unwrap_or_default();
    let mut names = entry_points(&source).join("\n").into_bytes();
    names.resize(2048, 0);
    let data: Vec<u32> = (0..64).map(|_| rng.u32() >> 4).collect();
    let tripled: Vec<u32> = data.iter().map(|v| v * 3 + 1).collect();
    let linear: Vec<u8> = (0..4u8)
        .flat_map(|y| (0..16u8).flat_map(move |x| [x, y, 255, 255]))
        .collect();
    let texels: Vec<u8> = (0..16u8).flat_map(|x| [0, x, 255, 255]).collect();
    let mut placed = vec![0x5a; 4];
    placed.extend([255, 0, 255, 255]);
    plan.add(
        Case::host("host.queries", library, json!({"kind": "queries"}))
            .buffer(Buffer::zeroed("names", 0, names.len()).unbound())
            .buffer(Buffer::data("data", 0, u32s(&data)).unbound())
            .buffer(Buffer::zeroed("linear", 0, linear.len()).unbound())
            .buffer(Buffer::zeroed("texels", 0, texels.len()).unbound())
            .buffer(Buffer::zeroed("placed", 0, 8).unbound())
            .buffer(Buffer::zeroed("cpu_times", 0, 16).unbound())
            .buffer(Buffer::zeroed("gpu_times", 0, 16).unbound())
            .buffer(props(20))
            .output(Output::exact("names", Dtype::U8, names))
            .output(Output::exact("data", Dtype::U32, u32s(&tripled)))
            .output(Output::exact("linear", Dtype::U8, linear))
            .output(Output::exact("texels", Dtype::U8, texels))
            .output(Output::exact("placed", Dtype::U8, placed))
            .output(Output::exact("cpu_times", Dtype::U64, u32s(&[1, 1])).set("derive", "ordered"))
            .output(Output::exact("gpu_times", Dtype::U64, u32s(&[1, 1])).set("derive", "ordered"))
            .output(Output::exact("props", Dtype::U32, u32s(&[1; 20]))),
    );
    let install = "@executable_path/stela_suite_dynamic.metallib";
    // Its install name, then the install name of it serialized and loaded.
    let mut name = format!("{install}\n{install}").into_bytes();
    name.resize(128, 0);
    plan.add(
        Case::host(
            "host.dynamic_library_install_name",
            library,
            json!({"kind": "dynamic_library", "install_name": install}),
        )
        .requires(&[], &["dynamic_libraries"])
        .buffer(Buffer::zeroed("name", 0, 128).unbound())
        .output(Output::exact("name", Dtype::U8, name)),
    );
}

/// Timestamps at a pass's boundaries resolved by a blit, and between
/// dispatches; an argument buffer laid out from descriptors; and a heap
/// that holds a second buffer only once the first is aliasable.
fn resources(plan: &mut Plan, library: &str, rng: &mut Random) {
    for (label, between, feature) in [
        ("pass_boundaries", false, "counters_stage_boundary"),
        ("between_dispatches", true, "counters_dispatch_boundary"),
    ] {
        plan.add(
            Case::host(
                &format!("host.counters.{label}"),
                library,
                json!({"kind": "resolve_counters", "between_dispatches": between}),
            )
            .requires(&[], &["counter_set_timestamp", feature])
            .buffer(Buffer::zeroed("x", 0, 256).unbound())
            .buffer(Buffer::zeroed("stamps", 0, 16).unbound())
            .output(Output::exact("stamps", Dtype::U64, u32s(&[1, 1])).set("derive", "ordered")),
        );
    }
    let data: Vec<u32> = (0..64).map(|_| rng.u32() >> 4).collect();
    let k = 17u32;
    let through: Vec<u32> = data.iter().map(|v| v * 2 + k).collect();
    plan.add(
        Case::host(
            "host.argument_descriptors",
            library,
            json!({"kind": "argument_descriptors"}),
        )
        .buffer(Buffer::data("data", 0, u32s(&data)).unbound())
        .buffer(Buffer::data("k", 0, u32s(&[k])).unbound())
        .output(Output::exact("data", Dtype::U32, u32s(&through))),
    );
    plan.add(
        Case::host("host.heap_aliasable", library, json!({"kind": "aliasable"}))
            .buffer(Buffer::zeroed("out", 0, 4).unbound())
            .buffer(props(5))
            .output(Output::exact("out", Dtype::U8, vec![0x3c; 4]).set("length", 4))
            .output(Output::exact("props", Dtype::U32, u32s(&[1; 5]))),
    );
    // The heap answers and fits a Mac gives (M1 Pro, macOS 27): sizes and
    // alignments of R8 8x8, RG8 16x16, BGRA8 32x32 (read), R16Float
    // 188x77, RGBA16Float 68x67 and 122x67, BGR10A2 64x64 and BGRA8
    // 2560x1440 (read and render); 1,024 8x8 R8 glyphs filling 128 KiB with
    // its usedSize and maxAvailableSizeWithAlignment: 128, usedSize after
    // makeAliasable and one more glyph; a 100-byte heap's free size and
    // currentAllocatedSize, a page; two 188x77 R16Float masks in 128 KiB
    // (one fits) and usedSize; a 68x67 RGBA16Float layer in 64 KiB (nil);
    // and 64x64 BGR10A2 textures in a heap of three answers (three fit);
    // the 100-byte heap's size, a page; and Core Animation's masks and
    // layers: a heap's currentAllocatedSize, then per texture whether it
    // fit, usedSize and maxAvailableSizeWithAlignment: 128 and 16384 (each
    // heap holds its size and no more, so the last texture is nil).
    let facts = [
        128, 128, 640, 128, 4224, 128, 66_560, 128, 98_816, 128, 98_816, 128, 16_512, 128,
        15_335_424, 128, 1024, 131_072, 0, 130_944, 1, 16_384, 16_384, 1, 0, 66_560, 0, 3, 49_536,
        16_384, // the 100-byte heap's size
        131_072, 1, 66_560, 64_512, 49_152, 0, 66_560, 64_512, 49_152, // R16Float 72x151
        131_072, 1, 66_560, 64_512, 49_152, 0, 66_560, 64_512, 49_152, // R16Float 113x153
        196_608, 1, 66_560, 130_048, 114_688, 1, 133_120, 63_488, 49_152, 0, 133_120, 63_488,
        49_152, // R16Float 248x81
        65_536, 1, 32_896, 32_640, 16_384, 0, 32_896, 32_640, 16_384, // RGBA16Float 36x35
    ];
    plan.add(
        Case::host("host.heap_fit", library, json!({"kind": "heap_fit"}))
            .buffer(Buffer::zeroed("facts", 0, facts.len() * 4).unbound())
            .output(Output::exact("facts", Dtype::U32, u32s(&facts))),
    );
    // Textures over CPU memory the CPU clears and redraws, as Core
    // Animation's backing stores are (Mac: M1 Pro, macOS 27): a no-copy
    // buffer over vm_allocate memory, a shared buffer, an IOSurface,
    // replaceRegion from vm_allocate memory and a blit from a shared buffer,
    // each cleared by bzero of all of it and from an unaligned offset,
    // memset_pattern4, MADV_FREE_REUSABLE then MADV_FREE_REUSE,
    // vm_deallocate then vm_allocate at the same address, and on the GPU by
    // a buffer fill or a render pass clear. Per source and clear: the texels
    // a kernel reads that differ from the memory after the first pattern,
    // the nonzero words a GPU clear leaves in the memory, and the texels
    // that differ after the second pattern. None differ, except that a
    // no-copy buffer's GPU pages stay those it was made over (all 32,768).
    let cleared = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 32_768, 0, 0, 0, // no-copy buffer
        0, 0, 0, 0, 0, 0, 0, 0, 0, // shared buffer
        0, 0, 0, 0, 0, 0, 0, 0, 0, // IOSurface
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // replaceRegion
        0, 0, 0, 0, 0, 0, // blit
    ];
    plan.add(
        Case::host("host.cpu_clear", library, json!({"kind": "cpu_clear"}))
            .buffer(Buffer::zeroed("facts", 0, cleared.len() * 4).unbound())
            .output(Output::exact("facts", Dtype::U32, u32s(&cleared))),
    );
    // A sub-rectangle cleared with Core Animation's own path_blit_vert_lph
    // and attachment_clear_frag_lph, unblended, on opaque red, then 5%
    // white blended over it, three times, on a private BGRA8 texture and
    // one over an IOSurface: the rectangle's centre holds one layer of white
    // over transparent (13 in each byte), outside stays red (Mac: M1 Pro,
    // macOS 27).
    let region: Vec<u8> = [0x0d0d_0d0d_u32, 0xffff_0000, 0x0d0d_0d0d, 0xffff_0000]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    plan.add(
        Case::host(
            "host.region_clear",
            library,
            json!({"kind": "region_clear"}),
        )
        .buffer(Buffer::zeroed("facts", 0, region.len()).unbound())
        .output(Output::new("facts", Dtype::U8, Tolerance::units(1.0)).expect_same(region)),
    );
    // The same on a private RGB10A8_2P_XR10 (550) target, whose first plane
    // (extended-range BGR10) a blit copies out: 0.05 inside, 1, 0, 0
    // outside.
    let xr10: Vec<u8> = [0x1996_6599_u32, 0x37e6_0180]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    plan.add(
        Case::host(
            "host.region_clear_550",
            library,
            json!({"kind": "region_clear", "format": 550}),
        )
        .buffer(Buffer::zeroed("facts", 0, xr10.len()).unbound())
        .output(Output::new("facts", Dtype::U8, Tolerance::units(1.0)).expect_same(xr10)),
    );
}

/// Fast resource loading: random bytes followed by a compressible run,
/// loaded plain and through every compression method; the texture takes
/// 1 KiB from 4 KiB in.
fn io(plan: &mut Plan, library: &str, rng: &mut Random) {
    let size = 96 * 1024;
    let mut source: Vec<u8> = (0..size / 3).map(|_| rng.u32() as u8).collect();
    source.extend((0..size - size / 3).map(|i| (i % 251) as u8 ^ ((i / 4096) as u8)));
    let texture_offset = 4096;
    let mut case = Case::host(
        "host.io",
        library,
        json!({"kind": "io", "size": size, "texture_offset": texture_offset}),
    )
    .buffer(Buffer::data("source", 0, source.clone()).unbound())
    .buffer(Buffer::zeroed("texture", 0, 1024).unbound())
    .buffer(Buffer::zeroed("status", 0, 16).unbound())
    .output(Output::exact(
        "texture",
        Dtype::U8,
        source[texture_offset..texture_offset + 1024].to_vec(),
    ))
    // The second buffer's status as it copied it (complete, 3), and the
    // one whose cancellation was requested while it waited: cancelled
    // (1), or complete (3), since tryCancel is only a request.
    .output(
        Output::exact("status", Dtype::U32, u32s(&[3, 0, 1, 0]))
            .expect_alternative(u32s(&[3, 0, 3, 0])),
    );
    for name in [
        "plain",
        "memory",
        "zlib",
        "lzfse",
        "lz4",
        "lzma",
        "lzbitmap",
        "legacy",
        "legacy_zlib",
    ] {
        case = case
            .buffer(Buffer::zeroed(name, 0, size).unbound())
            .output(Output::exact(name, Dtype::U8, source.clone()));
    }
    plan.add(case);
}

/// What textures and resources report of how they were made (in order):
/// a cube from its class method (type 5, 8x8, four levels, one slice,
/// shader read); a view of levels 1-2 and slices 1-2 of an 8x8 array
/// (its parent, level 1, slice 1, 4 wide, two levels, two slices) and
/// its root resource; a texture over a buffer at 256 with 64-byte rows;
/// a descriptor's BGR1 swizzle (blue 4, green 3, red 2, one 1), with
/// GPU-optimized contents off and lossless compression (0); a shareable
/// texture and a plain one; a buffer and a texture placed at their heap
/// offsets; allocated sizes covering lengths; ownership set to this
/// process; a descriptor view of level 1 of slice 2 (4 wide, one slice,
/// slice 2) and one with minimum LOD 1.5 (doubled, 3); and a view by
/// pixel format alone (its format, all three levels).
const TEXTURE_PROPERTIES: [u32; 34] = [
    5, 8, 8, 4, 1, 1, 1, 1, 1, 4, 2, 2, 1, 1, 256, 64, 4, 3, 2, 1, 0, 0, 1, 0, 1, 1, 1, 1, 4, 1, 2,
    3, 1, 3,
];

/// Host-side properties and object lifetimes: textures, residency sets,
/// queues, function reflection, library loading and debug markers.
fn properties(plan: &mut Plan, library: &str, rng: &mut Random) {
    plan.add(
        Case::host(
            "host.texture_properties",
            library,
            json!({"kind": "texture_properties"}),
        )
        .buffer(props(36))
        .output(Output::exact(
            "props",
            Dtype::U32,
            u32s(&[&TEXTURE_PROPERTIES[..], &[0, 0]].concat()),
        )),
    );
    // A view's minimum LOD of 1 over levels holding 1, 2 and 3.
    let levels: Vec<f64> = [(16, 1.0), (4, 2.0), (1, 3.0)]
        .iter()
        .flat_map(|&(n, v)| std::iter::repeat_n([v, 0.0, 0.0, 1.0], n).flatten())
        .collect();
    plan.add(
        Case::compute(
            "host.texture_view.min_lod",
            library,
            "sample_min_lod",
            [1, 1, 1],
            [1, 1, 1],
        )
        .texture(
            Texture::d2("source", "rgba32float", 4, 4)
                .mipmaps(3)
                .data(f32s(&levels)),
        )
        .texture(
            Texture::d2("view", "rgba32float", 4, 4)
                .set("view_of", "source")
                .set("min_lod", 1.0)
                .index(0),
        )
        .sampler(
            json!({"name": "s", "index": 0, "min": "nearest", "mag": "nearest", "mip": "nearest"}),
        )
        .buffer(Buffer::zeroed("out", 0, 16))
        .output(Output::exact(
            "out",
            Dtype::F32,
            f32s(&[2.0, 2.0, 3.0, 0.0]),
        ))
        .set(
            "reference_divergence",
            "the M1 Pro samples below a view's minLOD",
        ),
    );
    // Three allocations added, one removed and added and removed again;
    // then the set alone makes an argument buffer's targets resident,
    // from the queue and then from the command buffer: x * 2 + k twice;
    // and none are left once all are removed.
    let n = 64usize;
    let data: Vec<u32> = (0..n).map(|_| rng.u32() >> 4).collect();
    let k = 9u32;
    let twice: Vec<u32> = data
        .iter()
        .map(|&x| {
            (x.wrapping_mul(2).wrapping_add(k))
                .wrapping_mul(2)
                .wrapping_add(k)
        })
        .collect();
    plan.add(
        Case::host("host.residency", library, json!({"kind": "residency"}))
            .buffer(Buffer::data("data", 0, u32s(&data)).unbound())
            .buffer(Buffer::data("k", 0, u32s(&[k, 0, 0, 0])).unbound())
            .buffer(Buffer::zeroed("spare", 0, 16).unbound())
            .buffer(Buffer::zeroed("arguments", 0, 16).unbound())
            .buffer(props(8))
            .output(Output::exact("data", Dtype::U32, u32s(&twice)))
            .output(Output::exact(
                "props",
                Dtype::U32,
                u32s(&[3, 1, 0, 2, 1, 0, 0, 0]),
            )),
    );
    // Two queues of at most two command buffers each run four in order:
    // the digits 1234, then 5 more on the first; a command buffer's
    // queue, its retained references and an unretained one's, each
    // queue's last command buffer scheduled once it is waited for, and
    // CPU scheduling times in order.
    plan.add(
        Case::host("host.queues", library, json!({"kind": "queues"}))
            .buffer(Buffer::zeroed("order_count", 0, 16).unbound())
            .buffer(Buffer::zeroed("order_descriptor", 0, 16).unbound())
            .buffer(Buffer::zeroed("times", 0, 16).unbound())
            .buffer(props(8))
            .output(Output::exact(
                "order_count",
                Dtype::U32,
                u32s(&[12345, 0, 0, 0]),
            ))
            .output(Output::exact(
                "order_descriptor",
                Dtype::U32,
                u32s(&[1234, 0, 0, 0]),
            ))
            .output(Output::exact(
                "props",
                Dtype::U32,
                u32s(&[2, 1, 1, 0, 0, 1, 1, 0]),
            ))
            .output(
                Output::new("times", Dtype::U64, Tolerance::exact())
                    .expect_same(u32s(&[1, 1]))
                    .set("derive", "ordered"),
            ),
    );
    for (name, library, functions, lines) in [
        (
            "arguments",
            "arguments",
            &["constants", "twice", "constant_range"][..],
            &ARGUMENTS_REFLECTION[..],
        ),
        (
            "stages",
            "stages",
            &["tess_quad", "tess_triangle"][..],
            &STAGES_REFLECTION[..],
        ),
        (
            "extras",
            "extras",
            &["stage_in_compute"][..],
            &EXTRAS_REFLECTION[..],
        ),
        (
            "host",
            library,
            &["vert_attribute", "frag_constant", "append_digit"][..],
            &HOST_REFLECTION[..],
        ),
    ] {
        let mut text = lines.join("\n").into_bytes();
        text.push(b'\n');
        text.resize(2048, 0);
        plan.add(
            Case::host(
                &format!("host.function_reflection.{name}"),
                library,
                json!({"kind": "function_reflection", "functions": functions}),
            )
            .buffer(Buffer::zeroed("text", 0, 2048).unbound())
            .output(Output::exact("text", Dtype::U8, text)),
        );
    }
    // The depth bounds test discards the second draw where the first
    // stored depth 0.25: red on the left, blue on the right. The feature
    // set tables give the test to Apple10.
    let bounded: Vec<u8> = (0..8 * 4)
        .flat_map(|i| {
            if i % 8 < 4 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        })
        .collect();
    plan.add(
        Case::host(
            "host.depth_bounds",
            library,
            json!({"kind": "depth_bounds"}),
        )
        .requires(&["apple10"], &[])
        .buffer(Buffer::zeroed("image", 0, 8 * 4 * 4).unbound())
        .output(Output::exact("image", Dtype::U8, bounded)),
    );
    // A full-quality map is the identity at Apple's granularity and
    // parameter size; a lower-quality map shrinks within its screen. A pass
    // with a map draws as one without.
    let facts: Vec<u32> = [
        0, 1, 1, 0, 1, 640, 480, 2, 32, 32, 1, 24624, 4, 640, 480, 640, 480,
    ]
    .into_iter()
    .chain([100.5f32, 7.25, 100.5, 7.25].map(f32::to_bits))
    .chain([1, 1, 1, 1])
    .collect();
    let mapped: Vec<f64> = (0..8)
        .flat_map(|i| {
            let point = [f64::from(i) * 10.5, f64::from(i) * 3.25];
            [point, point]
        })
        .flatten()
        .collect();
    let left: Vec<u8> = (0..8 * 4)
        .flat_map(|i| if i % 8 < 4 { [255, 0, 0, 255] } else { [0; 4] })
        .collect();
    plan.add(
        Case::host("host.rate_maps", library, json!({"kind": "rate_maps"}))
            .buffer(Buffer::zeroed("facts", 0, 25 * 4).unbound())
            .buffer(Buffer::zeroed("mapped", 1, 16 * 8).unbound())
            .buffer(Buffer::zeroed("image", 2, 8 * 4 * 4).unbound())
            .output(Output::exact("facts", Dtype::U32, u32s(&facts)))
            .output(Output::exact("mapped", Dtype::F32, f32s(&mapped)))
            .output(Output::exact("image", Dtype::U8, left)),
    );
    // A pipeline from an MTLRenderPipelineDescriptor maps each logical
    // output to its own attachment, whatever map the encoder sets.
    let gray = [128, 128, 128, 255];
    let images: Vec<u8> = (0..3)
        .flat_map(|attachment| {
            (0..8 * 4).flat_map(move |i| {
                let clear = [
                    if attachment == 0 { 255 } else { 0 },
                    if attachment == 1 { 255 } else { 0 },
                    if attachment == 2 { 255 } else { 0 },
                    255,
                ];
                match (attachment, i % 8 < 6) {
                    (0, true) => gray,
                    (0, false) => [255; 4],
                    (1, false) => [255, 255, 0, 255],
                    _ => clear,
                }
            })
        })
        .collect();
    plan.add(
        Case::host(
            "host.color_attachment_map",
            library,
            json!({"kind": "color_attachment_map"}),
        )
        .buffer(Buffer::zeroed("facts", 0, 14 * 4).unbound())
        .buffer(Buffer::zeroed("images", 1, 3 * 8 * 4 * 4).unbound())
        .output(Output::exact(
            "facts",
            Dtype::U32,
            u32s(&COLOR_ATTACHMENT_MAP_FACTS),
        ))
        // Gray's 0.5 lies halfway between two unorm8 values, which
        // conversion may round either way; a misplaced output differs
        // by 255.
        .output(Output::new("images", Dtype::U8, Tolerance::units(1.0)).expect_same(images)),
    );
    // A pool's views read as their textures do, through the IDs its
    // setters return, which follow its base ID and stay with their entries;
    // a copy into another pool reads the same.
    let texel = [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0];
    let texels: Vec<f64> = [
        texel,
        [texel[2], texel[1], texel[0], 1.0],
        [1.0, 0.0, 0.0, 1.0],
        texel,
    ]
    .into_iter()
    .flatten()
    .collect();
    plan.add(
        Case::host(
            "host.texture_view_pools",
            library,
            json!({"kind": "texture_view_pools"}),
        )
        .buffer(Buffer::zeroed("facts", 0, 7 * 4).unbound())
        .buffer(Buffer::zeroed("texels", 1, 4 * 16).unbound())
        .output(Output::exact(
            "facts",
            Dtype::U32,
            u32s(&[4, 1, 1, 1, 1, 1, 1]),
        ))
        .output(Output::exact("texels", Dtype::F32, f32s(&texels))),
    );
    // Mapped tiles hold what a pass stores and read as resident; unmapped
    // ones drop stores and read zero, and a moved tile's memory reads in its
    // new texture.
    let filled = [128.0 / 255.0, 64.0 / 255.0, 192.0 / 255.0, 1.0];
    let empty = [0.0; 4];
    let probes: Vec<f64> = [
        [filled, filled, empty, empty],
        [filled, empty, empty, empty],
        [empty; 4],
        [filled, empty, empty, empty],
    ]
    .into_iter()
    .flatten()
    .flatten()
    .collect();
    // Mapped tiles 0 and 1 read the clear and unmapped ones zero, each as
    // resident or not; the depth texture is not sparse, and its compares of
    // 0.5 less than 0.25 fail.
    let red = 128.0 / 255.0;
    let functions: Vec<f64> = (0..4)
        .flat_map(|tile| {
            let (value, resident) = if tile < 2 { (red, 1.0) } else { (0.0, 0.0) };
            let colour = [value, resident];
            [
                colour,
                colour,
                colour,
                [0.0, 1.0],
                [0.0, 1.0],
                colour,
                colour,
            ]
        })
        .flatten()
        .collect();
    plan.add(
        Case::host(
            "host.sparse_textures",
            library,
            json!({"kind": "sparse_textures"}),
        )
        .buffer(Buffer::zeroed("facts", 0, 18 * 4).unbound())
        .buffer(Buffer::zeroed("probes", 1, 4 * 4 * 16).unbound())
        .output(Output::exact("facts", Dtype::U32, u32s(&SPARSE_FACTS)))
        .buffer(Buffer::zeroed("functions", 2, 4 * 7 * 8).unbound())
        .output(Output::exact("probes", Dtype::F32, f32s(&probes)))
        .output(Output::exact("functions", Dtype::F32, f32s(&functions))),
    );
    // Stencil resolves by sample 0 and by the sample whose depth resolves,
    // each sample's stencil set by a draw that covers it alone.
    let stencils: Vec<u8> = [1u8, 1, 4].iter().flat_map(|&s| [s; 64]).collect();
    let depths: Vec<f64> = [0.1f32, 0.1, 0.4]
        .iter()
        .flat_map(|&depth| [f64::from(depth); 64])
        .collect();
    plan.add(
        Case::host(
            "host.stencil_resolve",
            library,
            json!({"kind": "stencil_resolve"}),
        )
        .buffer(Buffer::zeroed("stencils", 0, 3 * 64).unbound())
        .buffer(Buffer::zeroed("depths", 1, 3 * 256).unbound())
        .output(Output::exact("stencils", Dtype::U8, stencils))
        .output(Output::exact("depths", Dtype::F32, f32s(&depths))),
    );
    plan.add(
        Case::host(
            "host.library_loading",
            library,
            json!({"kind": "library_loading"}),
        )
        .buffer(props(4))
        .output(Output::exact("props", Dtype::U32, u32s(&[1, 1, 1, 0]))),
    );
    plan.add(
        Case::host(
            "host.extern_functions",
            library,
            json!({"kind": "extern_functions"}),
        )
        .buffer(props(4))
        .output(Output::exact("props", Dtype::U32, u32s(&[1, 1, 0, 0]))),
    );
    let values: Vec<u32> = (0..n as u32).map(|i| i * 3 + 1).collect();
    let doubled: Vec<u32> = values.iter().map(|&x| x * 2).collect();
    plan.add(
        Case::host("host.debug_markers", library, json!({"kind": "debug"}))
            .buffer(Buffer::zeroed("values", 0, n * 4).unbound())
            .buffer(Buffer::zeroed("doubled", 0, n * 4).unbound())
            .buffer(Buffer::zeroed("copied", 0, n * 4).unbound())
            .output(Output::exact("values", Dtype::U32, u32s(&values)))
            .output(Output::exact("copied", Dtype::U32, u32s(&doubled))),
    );
}

/// Whether tile sizes in texels agree with 64 KiB pages and the default
/// page size; a pixel region's tiles aligned outward and inward and a tile
/// region's pixels in tiles, as origin and size; and the textures'
/// sparseness, tail, and whether their sparse texture tier is 1 or more
/// while a buffer's is none.
const SPARSE_FACTS: [u32; 18] = [1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1];

/// A fresh map's physical index of logical indices 0 to 7; logical 3's
/// once set to 5, in a copy, after a reset, and in the copy after it; and
/// the render pass and indirect command buffer descriptors' default
/// `supportColorAttachmentMapping`.
const COLOR_ATTACHMENT_MAP_FACTS: [u32; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 5, 5, 3, 5, 0, 0];

/// Each function's type, patch type and control points; its vertex and
/// stage-in attributes (index:data type, active, patch data, control
/// point data); its constants by name (index:data type, required, an
/// `is_function_constant_defined` one not); and the library's count of
/// its bindings.
const ARGUMENTS_REFLECTION: [&str; 18] = [
    "constants type=3 patch=0 points=-1",
    " constant c_bool#2:53 required=1",
    " constant c_extra#5:53 required=1",
    " constant c_float#1:3 required=1",
    " constant c_half#4:16 required=1",
    " constant c_int#0:29 required=1",
    " constant c_named#8:29 required=0",
    " constant c_short#6:37 required=1",
    " constant c_uint#3:33 required=1",
    " constant c_vector#7:6 required=1",
    " library bindings=2 annotation=-",
    "twice type=5 patch=0 points=-1",
    " library bindings=0 annotation=-",
    "constant_range type=3 patch=0 points=-1",
    " constant c_range0#40:33 required=1",
    " constant c_range1#41:33 required=1",
    " constant c_range2#42:33 required=1",
    " library bindings=1 annotation=-",
];

/// Post-tessellation vertex functions: quad (2) and triangle (1) patches
/// of 4 and 3 control points, whose stage-in attributes are control
/// point data.
const STAGES_REFLECTION: [&str; 12] = [
    "tess_quad type=1 patch=2 points=4",
    " vertex position#0:4 active=1 patch=0 control=1",
    " vertex value#1:3 active=1 patch=0 control=1",
    " input position#0:4 active=1 patch=0 control=1",
    " input value#1:3 active=1 patch=0 control=1",
    " library bindings=0 annotation=-",
    "tess_triangle type=1 patch=1 points=3",
    " vertex position#0:4 active=1 patch=0 control=1",
    " vertex value#1:3 active=1 patch=0 control=1",
    " input position#0:4 active=1 patch=0 control=1",
    " input value#1:3 active=1 patch=0 control=1",
    " library bindings=0 annotation=-",
];

const EXTRAS_REFLECTION: [&str; 4] = [
    "stage_in_compute type=3 patch=0 points=-1",
    " input a#0:4 active=1 patch=0 control=0",
    " input b#1:52 active=1 patch=0 control=0",
    " library bindings=1 annotation=-",
];

const HOST_REFLECTION: [&str; 8] = [
    "vert_attribute type=1 patch=0 points=-1",
    " vertex position#0:4 active=1 patch=0 control=0",
    " input position#0:4 active=1 patch=0 control=0",
    " library bindings=0 annotation=-",
    "frag_constant type=2 patch=0 points=-1",
    " library bindings=1 annotation=-",
    "append_digit type=3 patch=0 points=-1",
    " library bindings=2 annotation=-",
];

pub fn build(plan: &mut Plan) {
    let library = plan.library("host", SOURCE, &[], super::plan::LATEST);
    plan.default_library(&library);
    let mut rng = Random::new(1301);
    drawables(plan, &library);
    properties(plan, &library, &mut Random::new(1302));
    iosurface(plan, &library, &mut rng);
    two_plane(plan, &library, &mut Random::new(1303));
    iosurface_buffer(plan, &library, &mut Random::new(1320));
    two_plane_render(plan, &library, &mut Random::new(1304));
    two_plane_ten_bit(plan, &library, &mut Random::new(1305));
    share(plan, &library, &mut rng);
    archive(plan, &library, &mut rng);
    errors(plan, &library);
    purgeable(plan, &library);
    streams(plan, &library);
    churn(plan, &library);
    fresh_resources(plan, &library);
    large_buffer(plan, &library);
    grids(plan, &library);
    rebinding(plan, &library, &mut rng);
    barriers(plan, &library);
    passes(plan, &library);
    queries(plan, &library, &mut rng);
    resources(plan, &library, &mut rng);
    io(plan, &library, &mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_hash_reduces_every_thread() {
        let [sum, xor, count, top] = grid_hash([2, 2, 1]);
        let h = |x: u32, y: u32| x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77);
        let all = [h(0, 0), h(1, 0), h(0, 1), h(1, 1)];
        assert_eq!(sum, all.iter().fold(0u32, |a, &v| a.wrapping_add(v)));
        assert_eq!(xor, all.iter().fold(0, |a, &v| a ^ v));
        assert_eq!((count, top), (4, 1));
    }

    #[test]
    fn entry_points_are_named_and_sorted() {
        let names = entry_points(
            "kernel void b(int)\nvertex V a(uint v)\nfragment float4 c(\nint d();\n[[visible]] uint e(uint x)\n",
        );
        assert_eq!(names, ["a", "b", "c", "e"]);
    }

    #[test]
    fn packed_texels_put_blue_lowest() {
        assert_eq!(texel("bgr10a2unorm", [0, 0, 1, 0]), 1023u32.to_le_bytes());
        assert_eq!(texel("bgra8unorm", [1, 0, 0, 1]), [0, 0, 255, 255]);
    }
}

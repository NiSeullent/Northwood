// SPDX-License-Identifier: GPL-2.0-or-later
// The shaders host cases run: drawing into drawables, reading and writing
// IOSurface planes, archived pipelines, ordered and concurrent counting,
// and stamping a buffer past 4 GiB.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

// The left half of the target, as a strip.
vertex Position vert_left_half(uint v [[vertex_id]])
{
    return {float4(v & 1 ? 0.0f : -1.0f, v & 2 ? 1.0f : -1.0f, 0.0f, 1.0f)};
}

fragment float4 frag_constant(constant float4 &color [[buffer(0)]])
{
    return color;
}

struct Tinted {
    float4 position [[position]];
    float4 tint [[user(tint)]];
};

// Reads a varying the vertex function never writes: linking fails.
fragment float4 frag_unwritten_input(Tinted in [[stage_in]])
{
    return in.tint;
}

kernel void surface_invert(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           uint2 p [[thread_position_in_grid]])
{
    float4 v = src.read(p);
    dst.write(float4(1.0f - v.rgb, v.a), p);
}

// Adds `add` to each word of a buffer over an IOSurface, in place.
kernel void surface_buffer_add(device uint *words [[buffer(0)]],
                               constant uint &add [[buffer(1)]],
                               uint i [[thread_position_in_grid]])
{
    words[i] += add;
}

// Luma with the chroma pair of each 2x2 block, as bytes of one word.
kernel void surface_planes(texture2d<float, access::read> luma [[texture(0)]],
                           texture2d<float, access::read> chroma [[texture(1)]],
                           device uint *out [[buffer(0)]],
                           uint2 p [[thread_position_in_grid]], uint2 n [[threads_per_grid]])
{
    float y = luma.read(p).r;
    float2 uv = chroma.read(p / 2).rg;
    out[p.y * n.x + p.x] = uint(rint(y * 255.0f)) | uint(rint(uv.x * 255.0f)) << 8 | uint(rint(uv.y * 255.0f)) << 16;
}

/* Apple's two-plane RGB formats: green in the first plane, blue and red in
 * the second, each read at its own resolution and filtered on its own. */
kernel void surface_two_plane(texture2d<float, access::sample> t [[texture(0)]],
                              device uint *read [[buffer(0)]], device uint *sampled [[buffer(1)]],
                              uint2 p [[thread_position_in_grid]], uint2 n [[threads_per_grid]])
{
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    uint i = p.y * n.x + p.x;
    read[i] = pack_float_to_unorm4x8(t.read(p));
    sampled[i] = pack_float_to_unorm4x8(t.sample(linear, (float2(p) + 0.5f) / float2(n)));
}

/* The 10-bit two-plane formats: the read and the sample as floats, which
 * hold every 10-bit step. */
kernel void surface_two_plane_wide(texture2d<float, access::sample> t [[texture(0)]],
                                   device float4 *read [[buffer(0)]], device float4 *sampled [[buffer(1)]],
                                   uint2 p [[thread_position_in_grid]], uint2 n [[threads_per_grid]])
{
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    uint i = p.y * n.x + p.x;
    read[i] = t.read(p);
    sampled[i] = t.sample(linear, (float2(p) + 0.5f) / float2(n));
}

kernel void archived(device uint *data [[buffer(0)]], constant uint &unused [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    data[i] = data[i] * 3u + 1u;
}

kernel void not_archived(device uint *data [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    data[i] = 0u;
}

kernel void append_digit(device uint *value [[buffer(0)]], constant uint &digit [[buffer(1)]])
{
    value[0] = value[0] * 10u + digit;
}

kernel void bump(device uint *counts [[buffer(0)]], constant uint &slot [[buffer(1)]])
{
    counts[slot] += 1u;
}

kernel void bump_atomic(device atomic_uint *counts [[buffer(0)]], constant uint &slot [[buffer(1)]])
{
    atomic_fetch_add_explicit(&counts[slot], 1u, memory_order_relaxed);
}

kernel void stamp(device uint *big [[buffer(0)]], constant uint &first [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    big[first + i] = first + i;
}

// Each thread's hash of its position, reduced in threadgroup memory and
// then once per group into a sum, an xor, a count and the largest y.
kernel void grid_hash(device atomic_uint *sums [[buffer(0)]], threadgroup atomic_uint *local [[threadgroup(0)]],
                      uint3 p [[thread_position_in_grid]], uint l [[thread_index_in_threadgroup]])
{
    if (l < 4)
        atomic_store_explicit(&local[l], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint h = (p.x * 0x9E3779B1u) ^ (p.y * 0x85EBCA77u) ^ (p.z * 0xC2B2AE3Du);
    atomic_fetch_add_explicit(&local[0], h, memory_order_relaxed);
    atomic_fetch_xor_explicit(&local[1], h, memory_order_relaxed);
    atomic_fetch_add_explicit(&local[2], 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(&local[3], p.y + p.z, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0)
        atomic_fetch_add_explicit(&sums[0], atomic_load_explicit(&local[0], memory_order_relaxed), memory_order_relaxed);
    if (l == 1)
        atomic_fetch_xor_explicit(&sums[1], atomic_load_explicit(&local[1], memory_order_relaxed), memory_order_relaxed);
    if (l == 2)
        atomic_fetch_add_explicit(&sums[2], atomic_load_explicit(&local[2], memory_order_relaxed), memory_order_relaxed);
    if (l == 3)
        atomic_fetch_max_explicit(&sums[3], atomic_load_explicit(&local[3], memory_order_relaxed), memory_order_relaxed);
}

// A strip over a rectangle at a depth: x0, y0, x1, y1, z.
vertex Position vert_offset_rect(constant float *rect [[buffer(0)]], uint v [[vertex_id]])
{
    return {float4(v & 1 ? rect[2] : rect[0], v & 2 ? rect[3] : rect[1], rect[4], 1.0f)};
}

// The colour's RGB with, as alpha, a sample at level 0 that the bound
// sampler's clamps move to another level.
fragment float4 frag_rebind(constant float4 &color [[buffer(0)]], texture2d<float> t [[texture(1)]],
                            sampler s [[sampler(0)]])
{
    return float4(color.rgb, t.sample(s, float2(0.5f), level(0)).r);
}

fragment float4 frag_rebind_color(constant float4 &color [[buffer(0)]])
{
    return color;
}

kernel void copy_word(device uint *dst [[buffer(0)]], device const uint *src [[buffer(1)]], constant uint &k [[buffer(2)]])
{
    dst[k] = src[0];
}

kernel void sample_levels(texture2d<float> a [[texture(0)]], texture2d<float> b [[texture(1)]], sampler s [[sampler(0)]],
                          device float *out [[buffer(0)]])
{
    out[0] = a.sample(s, float2(0.5f), level(0)).r;
    out[1] = b.sample(s, float2(0.5f), level(3)).r;
}

struct Corner {
    float2 position [[attribute(0)]];
};

vertex Position vert_attribute(Corner in [[stage_in]])
{
    return {float4(in.position, 0.0f, 1.0f)};
}

kernel void write_values(device uint *x [[buffer(0)]], constant uint &k [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    x[i] = i * k + 1u;
}

kernel void double_values(device const uint *x [[buffer(0)]], device uint *y [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    y[i] = x[i] * 2u;
}

struct Link {
    float4 position [[position]];
    float size [[point_size]];
    uint value [[flat]];
};

// One point over the single pixel, carrying the chain's word `from`.
vertex Link vert_chain(device const uint *chain [[buffer(0)]], constant uint &from [[buffer(1)]])
{
    return {float4(0.0f, 0.0f, 0.0f, 1.0f), 1.0f, chain[from]};
}

fragment void frag_chain(Link in [[stage_in]], device uint *chain [[buffer(0)]], constant uint &to [[buffer(1)]])
{
    chain[to] = in.value + 1u;
}

struct Viewported {
    float4 position [[position]];
    uint viewport [[viewport_array_index]];
    uint instance [[flat]];
};

vertex Viewported vert_viewport(uint v [[vertex_id]], uint i [[instance_id]])
{
    return {float4(v & 1 ? 1.0f : -1.0f, v & 2 ? 1.0f : -1.0f, 0.0f, 1.0f), i, i};
}

fragment float4 frag_instance(Viewported in [[stage_in]], constant float4 *colors [[buffer(0)]])
{
    return colors[in.instance];
}

// Each pixel's own color, from a row-major table 8 pixels wide.
fragment float4 frag_pixel(float4 p [[position]], constant float4 *colors [[buffer(0)]])
{
    uint2 q = uint2(p.xy);
    return colors[q.y * 8 + q.x];
}

kernel void fill_linear(texture2d<float, access::write> t [[texture(0)]],
                        texture_buffer<float, access::write> texels [[texture(1)]],
                        uint2 p [[thread_position_in_grid]])
{
    float4 v = float4(float(p.x) / 255.0f, float(p.y) / 255.0f, 1.0f, 1.0f);
    t.write(v, p);
    if (p.y == 0)
        texels.write(v.yxzw, p.x);
}

// Each texel's word of a texture over CPU memory, row by row.
kernel void texel_words(texture2d<uint, access::read> t [[texture(0)]], device uint *out [[buffer(0)]],
                        uint2 p [[thread_position_in_grid]])
{
    if (p.x < t.get_width() && p.y < t.get_height())
        out[p.y * t.get_width() + p.x] = t.read(p).x;
}

[[max_total_threads_per_threadgroup(128)]]
kernel void limited(device uint *data [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    data[i] = i;
}

// A threadgroup sum, run as a command of an indirect command buffer.
kernel void group_sum(device const uint *src [[buffer(0)]], device uint *out [[buffer(1)]],
                      threadgroup uint *partial [[threadgroup(0)]], uint l [[thread_index_in_threadgroup]],
                      uint n [[threads_per_threadgroup]])
{
    partial[l] = src[l];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) {
        uint sum = 0;
        for (uint k = 0; k < n; k++)
            sum += partial[k];
        out[0] = sum;
    }
}

// Two pointers an argument buffer laid out from descriptors holds.
struct Pointers {
    device uint *data [[id(0)]];
    constant uint *k [[id(1)]];
};

kernel void through_arguments(device Pointers &p [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    p.data[i] = p.data[i] * 2u + p.k[0];
}

// One minus what the target holds at this pixel, read from the target
// itself after a render-target barrier.
fragment float4 frag_feedback(Position in [[stage_in]], texture2d<float> earlier [[texture(0)]])
{
    float4 v = earlier.read(uint2(in.position.xy));
    return float4(1.0f - v.rgb, v.a);
}

// A view whose minimum LOD is 1: a sample at level 0 reaches level 1,
// and reads of levels 1 and 2 are within it.
kernel void sample_min_lod(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], device float *out [[buffer(0)]])
{
    out[0] = t.sample(s, float2(0.5f), level(0)).r;
    out[1] = t.read(uint2(0), 1).r;
    out[2] = t.read(uint2(0), 2).r;
}

// Functions an archive holds beyond pipelines: a visible function, a
// stitchable one for a stitched library, and a mesh pipeline's mesh.
[[visible]] uint archived_visible(uint x)
{
    return x * 5u;
}

[[stitchable]] uint archived_stitch(uint x)
{
    return x + 3u;
}

[[mesh]] void archived_mesh(mesh<Position, void, 3, 1, topology::triangle> m, uint l [[thread_index_in_threadgroup]])
{
    if (l == 0) {
        m.set_primitive_count(1);
        for (uint v = 0; v < 3; v++) {
            m.set_vertex(v, {float4(v == 1 ? 1.0f : -1.0f, v == 2 ? 1.0f : -1.0f, 0.0f, 1.0f)});
            m.set_index(v, v);
        }
    }
}

// A rasterization rate map decoder over the map's parameter data: odd
// threads map at layer 1.
kernel void rate_map_decode(constant rasterization_rate_map_data &data [[buffer(0)]], device const float2 *in [[buffer(1)]],
                            device float2 *out [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    rasterization_rate_map_decoder map(data);
    out[i * 2] = map.map_screen_to_physical_coordinates(in[i], i & 1);
    out[i * 2 + 1] = map.map_physical_to_screen_coordinates(in[i], i & 1);
}

struct TwoColors {
    float4 first [[color(0)]];
    float4 second [[color(1)]];
};

// Two logical outputs, which a color attachment map sends to any physical
// attachments.
fragment TwoColors frag_two_colors()
{
    return {float4(1.0f), float4(1.0f, 1.0f, 0.0f, 1.0f)};
}

struct PoolViews {
    texture2d<float> views[4];
};

// Reads each view a texture view pool holds, through their resource IDs.
kernel void pool_views(constant PoolViews &pool [[buffer(0)]], device float4 *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    out[i] = pool.views[i].read(uint2(0));
}

// The centre texel of each of a 2 x 2 tile texture's tiles: its RGB, and
// whether it is resident as alpha.
kernel void sparse_probe(texture2d<float, access::read> t [[texture(0)]], constant uint2 &tile [[buffer(0)]],
                         device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    sparse_color<float4> c = t.sparse_read(uint2(i & 1, i >> 1) * tile + tile / 2);
    out[i] = float4(c.value().rgb, c.resident() ? 1.0f : 0.0f);
}

// Each sparse texture function at each tile's centre, as its value's first
// channel and whether it is resident: a sample, a gather, a read through a
// sampler, depth compares of a texture that is not sparse, and a block
// across the tiles of the centre's row.
kernel void sparse_functions(texture2d<float> t [[texture(0)]], depth2d<float> z [[texture(1)]],
                             sampler s [[sampler(0)]], constant uint2 &tile [[buffer(0)]],
                             device float2 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint2 texel = uint2(i & 1, i >> 1) * tile + tile / 2;
    float2 uv = (float2(texel) + 0.5f) / float2(tile * 2);
    constexpr sampler compare(filter::nearest, compare_func::less);
    sparse_color<float4> sampled = t.sparse_sample(s, uv);
    sparse_color<float4> gathered = t.sparse_gather(s, uv);
    sparse_color<float4> read = t.sparse_read(s, int2(texel));
    sparse_color<float> compared = z.sparse_sample_compare(compare, uv, 0.5f);
    sparse_color<float4> gathered_compares = z.sparse_gather_compare(compare, uv, 0.5f);
    auto block = t.sparse_block_read<2, 1>(s, int2(tile.x - 1, texel.y));
    sparse_color<float4> left = get<0>(block), right = get<1>(block);
    device float2 *o = out + i * 7;
    o[0] = float2(sampled.value().x, sampled.resident());
    o[1] = float2(gathered.value().x, gathered.resident());
    o[2] = float2(read.value().x, read.resident());
    o[3] = float2(compared.value(), compared.resident());
    o[4] = float2(gathered_compares.value().x, gathered_compares.resident());
    o[5] = float2(left.value().x, left.resident());
    o[6] = float2(right.value().x, right.resident());
}

struct OneSample {
    float depth [[depth(any)]];
    uint mask [[sample_mask]];
};

// Covers sample k alone, at depth 0.1 * (k + 1); the draw's stencil
// reference marks it.
fragment OneSample frag_one_sample(constant uint &k [[buffer(0)]])
{
    return {0.1f * float(k + 1), 1u << k};
}

// A path filled as RenderBox fills one: each triangle of a fan from a
// pivot to each contour edge adds its facing to a winding count in a float
// target, with additive blending.
vertex Position vert_points(constant float2 *p [[buffer(0)]], uint v [[vertex_id]])
{
    return {float4(p[v], 0.0f, 1.0f)};
}

struct Winding {
    float2 coverage [[color(1)]];
};

[[early_fragment_tests]] fragment Winding frag_winding(bool front [[front_facing]])
{
    return {float2(0.0f, front ? 1.0f : -1.0f)};
}

constant bool reads_targets [[function_constant(0)]];

struct Resolved {
    half4 color [[color(0)]];
    float2 coverage [[color(1)]];
};

// Reads the count and the destination back as RenderBox's accumulator
// does, through inputs a constant enables, fills where the count is
// nonzero and clears the count for the next path.
[[early_fragment_tests]] fragment Resolved
frag_resolve_winding(constant half4 &fill [[buffer(0)]],
                     float2 coverage [[color(1), function_constant(reads_targets)]],
                     half4 dest [[color(0), function_constant(reads_targets)]])
{
    float2 c = reads_targets ? coverage : float2(0.0f);
    half4 d = reads_targets ? dest : half4(0.0h);
    return {mix(d, fill, half(saturate(abs(c.y)))), float2(0.0f)};
}

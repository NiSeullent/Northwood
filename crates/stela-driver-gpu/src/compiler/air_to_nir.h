/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef STELA_AIR_TO_NIR_H
#define STELA_AIR_TO_NIR_H

#include "compiler/nir/nir_builder.h"
#include "air_metadata.h"
#include "air_soft_half.h"
#include <stddef.h>

/* User-space compiler boundary. LLVM and Mesa must be built for this process's
 * architecture and linked with their normal runtime services. No GPU access,
 * IOKit call, Metal device registration, or command submission happens here.
 */
/* Resolve a validated Metal buffer binding to a 64-bit device virtual address.
 * The caller owns resource lifetime, size/access validation, actual allocation,
 * residency and runtime argument ABI. Returning NULL refuses compilation. This
 * callback must build straight-line NIR in the entry prelude; it must not
 * introduce control flow, submit commands or mutate resources.
 * Test callbacks may use fixed addresses; production bindings must not do so.
 */
typedef nir_def *(*stela_air_bind_buffer)(nir_builder *b,
                                        const struct stela_air_buffer *buffer,
                                        void *data);

/* Alternative for a backend with indexed buffer surfaces rather than global
 * device addresses. The callback maps a Metal argument to a fixed SSBO slot.
 * Its surface must describe exactly that argument's buffer view, including its
 * offset and byte length; the shader's logical offset starts at zero. Distinct
 * arguments require distinct slots, even when their runtime views alias. A slot
 * is not an encoded or truncated GPU VA. */
typedef bool (*stela_air_bind_ssbo)(const struct stela_air_buffer *buffer,
                                   unsigned *slot, void *data);

/* Resolve a texture argument to the deref of a descriptor of type, which its
 * accesses need: an image type (nir_var_image) for write and read_write
 * access, otherwise a texture type (nir_var_uniform). An image holds one mip
 * level: level is the 32-bit level an access names, which the callback must
 * keep within the texture, or NULL for the first. It is NULL for texture
 * types. faces asks for a cube texture's faces as the sampled 2D array type
 * gives, layer face + 6 * cube, which reads of its texels fetch. Called at
 * each use, where it must emit straight-line NIR at the cursor. Returning
 * NULL refuses compilation. */
typedef nir_deref_instr *(*stela_air_bind_texture)(nir_builder *b,
                                                   const struct stela_air_texture *texture,
                                                   const struct glsl_type *type, bool faces,
                                                   nir_def *level, void *data);

/* Resolve render target `location` for a fragment function's access of it:
 * a 2D texture of type for a read, or, for an attachment its function
 * accesses as storage (stela_air_attachment_access), a 2D or 2D multisample
 * array image of type of the attached level, its layers from the attached
 * slice, whose texels are the format's raw bits. Returning NULL refuses
 * compilation. */
typedef nir_deref_instr *(*stela_air_bind_attachment)(nir_builder *b, unsigned location,
                                                      const struct glsl_type *type, void *data);

/* Resolve the pass's depth or stencil copy for a fragment function that
 * tests and updates them itself (stela_air_stage_options::stencil_export):
 * a 2D or 2D multisample array image of type, its layers from the attached
 * slice, whose texels are the attachment's raw depth (16 or 32 bits) or
 * stencil (8) bits. Returning NULL refuses compilation. */
typedef nir_deref_instr *(*stela_air_bind_depth_stencil)(nir_builder *b, bool stencil,
                                                         const struct glsl_type *type, void *data);

/* Resolve a sampler argument for a use with a texture whose texels are
 * integers (integer is the border color class), as bind_texture does, to a
 * deref of glsl_bare_sampler_type(). */
typedef nir_deref_instr *(*stela_air_bind_sampler)(nir_builder *b,
                                                   const struct stela_air_sampler *sampler,
                                                   bool integer, void *data);

/* Resolve a texture or sampler whose 64-bit MTLResourceID an argument buffer
 * holds, as bind_texture and bind_sampler do: id is that loaded identifier.
 * The identifier is the application's; an invalid one is undefined, as in
 * Metal. */
typedef nir_deref_instr *(*stela_air_bind_texture_id)(nir_builder *b, nir_def *id,
                                                      const struct glsl_type *type, bool faces,
                                                      nir_def *level, void *data);
typedef nir_deref_instr *(*stela_air_bind_sampler_id)(nir_builder *b, nir_def *id, bool integer,
                                                      void *data);

/* Whether a texture ID an argument buffer holds names the word holding the
 * MTLResourceID bind_texture_id resolves for it, with that word's device
 * address in *address; id is the loaded identifier. The translator reads the
 * word only where the result is true, since any other ID names no memory. */
typedef nir_def *(*stela_air_texture_view_word)(nir_builder *b, nir_def *id, nir_def **address,
                                                void *data);

/* The 64-bit MTLResourceID of an element of a texture array argument, or of
 * a sampler array argument when texture is NULL, which bind_texture_id or
 * bind_sampler_id then resolve. element is the 64-bit index an access names;
 * Metal leaves one past the array undefined, but the callback must not read
 * past the array for it. Emits straight-line NIR at the cursor; returning
 * NULL refuses compilation. */
typedef nir_def *(*stela_air_resource_id)(nir_builder *b, const struct stela_air_texture *texture,
                                          const struct stela_air_sampler *sampler,
                                          nir_def *element, void *data);

/* How a texture's storage images hold extended-range codes
 * (stela_air_options' texture_extended_storage). */
enum stela_air_extended_storage {
   STELA_AIR_EXTENDED_STORAGE_NONE,
   /* Half floats of RGB10A8_2P_XR10's codes and 8-bit alpha. */
   STELA_AIR_EXTENDED_STORAGE_CODES,
   /* The same, the codes through the sRGB curve mirrored below zero. */
   STELA_AIR_EXTENDED_STORAGE_SRGB,
};

/* What a texture argument's binding says, or with id what the identifier an
 * argument buffer holds does (texture is then NULL). */
typedef nir_def *(*stela_air_texture_query)(nir_builder *b,
                                            const struct stela_air_texture *texture,
                                            nir_def *id, void *data);

/* What a sampler argument's binding says, or with id what the identifier an
 * argument buffer holds does (sampler is then NULL). */
typedef nir_def *(*stela_air_sampler_query)(nir_builder *b,
                                            const struct stela_air_sampler *sampler,
                                            nir_def *id, void *data);

/* How a float texture's texels filter bicubically: as they are, clamped to
 * a normalized format's range, or (32-bit float channels) not at all, the
 * texel a nearest filter takes in place of the bicubic one. */
enum stela_air_filter_class {
   STELA_AIR_FILTER_FLOAT,
   STELA_AIR_FILTER_UNORM,
   STELA_AIR_FILTER_SNORM,
   STELA_AIR_FILTER_NEAREST,
};

/* sampler_filter's bits: Metal's bicubic filter, which Vulkan's samplers
 * lack and Apple GPUs apply to 2D textures only, a mip filter, and pixel
 * coordinates, which shaders divide by the base level's size for the
 * normalized Vulkan sampler. */
enum stela_air_sampler_filter {
   STELA_AIR_SAMPLER_CUBIC = 1,
   STELA_AIR_SAMPLER_MIPPED = 2,
   STELA_AIR_SAMPLER_PIXELS = 4,
};

/* texture_planes' values: a two-plane texture's second plane holds one
 * texel per 1 x 1, 2 x 1 or 2 x 2 of the first, in the low two bits. Its
 * first plane holds green, its second blue and red; with
 * STELA_AIR_PLANES_YCBCR, luma and Cb and Cr, which with
 * STELA_AIR_PLANES_VIDEO are video range (the surface's pixel format's,
 * not the Metal format's), else full range. With STELA_AIR_PLANES_TEN_BIT
 * each component is the high 10 bits of a 16-bit word, else a byte. */
enum stela_air_planes {
   STELA_AIR_PLANES_NONE = 0,
   STELA_AIR_PLANES_FULL = 1,
   STELA_AIR_PLANES_HORIZONTAL = 2,
   STELA_AIR_PLANES_BOTH = 3,
   STELA_AIR_PLANES_SUBSAMPLING = 3,
   STELA_AIR_PLANES_YCBCR = 4,
   STELA_AIR_PLANES_TEN_BIT = 8,
   STELA_AIR_PLANES_VIDEO = 16,
   /* Apple's RGB10A8_2P_XR10: the first plane holds blue, green and red
    * extended-range codes from bit 0 of a 32-bit word, read through an
    * A2B10G10R10 UNORM view, each reading (code - 384) / 510, through the
    * sRGB curve mirrored below zero with STELA_AIR_PLANES_XR10_SRGB; the
    * second plane holds 8-bit alpha. */
   STELA_AIR_PLANES_XR10 = 32,
   STELA_AIR_PLANES_XR10_SRGB = 64,
};

/* AIR's indirect command functions (air_commands.c), as Metal's
 * render_command and compute_command methods. */
enum stela_air_command_op {
   STELA_AIR_COMMAND_RESET,
   STELA_AIR_COMMAND_COPY,
   STELA_AIR_COMMAND_SET_BARRIER,
   STELA_AIR_COMMAND_CLEAR_BARRIER,
   STELA_AIR_COMMAND_PIPELINE,
   STELA_AIR_COMMAND_BUFFER,
   STELA_AIR_COMMAND_THREADGROUP_LENGTH,
   STELA_AIR_COMMAND_STAGE_IN_REGION,
   STELA_AIR_COMMAND_DRAW,
   STELA_AIR_COMMAND_DRAW_INDEXED,
   STELA_AIR_COMMAND_DRAW_PATCHES,
   STELA_AIR_COMMAND_DRAW_INDEXED_PATCHES,
   STELA_AIR_COMMAND_DISPATCH_THREADGROUPS,
   STELA_AIR_COMMAND_DISPATCH_THREADS,
};
/* One call on command `index` (a uint) of the indirect command buffer
 * whose 64-bit gpuResourceID is `buffer`. Operands are uints unless noted. */
struct stela_air_command {
   enum stela_air_command_op op;
   bool render; /* A render_command's, otherwise a compute_command's. */
   nir_def *buffer, *index;
   union {
      /* COPY: the command it copies. */
      struct {
         nir_def *buffer, *index;
      } source;
      /* PIPELINE: the pipeline state's 64-bit gpuResourceID. */
      nir_def *pipeline;
      /* BUFFER: a 64-bit address bound at slot of a vertex or fragment
       * function, or of a kernel when stage is zero, with a 64-bit vertex
       * stride that is ~0 when the call names none. */
      struct {
         enum stela_air_stage stage;
         nir_def *slot, *address, *stride;
      } binding;
      /* THREADGROUP_LENGTH. */
      struct {
         nir_def *slot, *length;
      } threadgroup;
      /* STAGE_IN_REGION: uvec3s. */
      struct {
         nir_def *origin, *size;
      } region;
      /* DRAW and DRAW_INDEXED: an MTLPrimitiveType, the vertex or index
       * count and the instance count; a draw's first vertex, an indexed
       * draw's indices' 64-bit address, their index_bytes and its signed
       * base vertex; and the base instance. */
      struct {
         nir_def *primitive, *count, *instances, *first_vertex, *indices, *base_vertex,
            *base_instance;
         unsigned index_bytes;
      } draw;
      /* DRAW_PATCHES and DRAW_INDEXED_PATCHES: the control points per
       * patch, patch start and count, the patch indices' 64-bit address
       * (zero for none), an indexed draw's control point indices' 64-bit
       * address, the instance count and base instance, the factors' 64-bit
       * address and instance stride, and the factor scale, a float. */
      struct {
         nir_def *control_points, *start, *count, *patch_indices, *control_point_indices,
            *instances, *base_instance, *factors, *instance_stride, *scale;
      } patches;
      /* DISPATCH_*: the threadgroups or threads of the grid, and the
       * threads per threadgroup, as uvec3s. */
      struct {
         nir_def *grid, *threads;
      } dispatch;
   };
};
/* command_buffer::size(): the command count, as a uint, of the indirect
 * command buffer whose gpuResourceID is buffer. Emits straight-line NIR at
 * the cursor; returning NULL refuses compilation. */
typedef nir_def *(*stela_air_command_count)(nir_builder *b, nir_def *buffer, void *data);
/* The writes that encode a command, which the translator makes: each store
 * of value at its 64-bit address when its 1-bit valid holds, in order, and
 * then, when bytes is set and copy_valid holds, a copy of bytes (a uint
 * multiple of 16) to destination from source, or of zeros without a
 * source. Addresses are aligned to their values' components, and copies
 * to 16 bytes. */
#define STELA_AIR_COMMAND_STORES 6u
struct stela_air_command_store {
   nir_def *address, *value, *valid;
};
struct stela_air_command_encoding {
   unsigned store_count;
   struct stela_air_command_store stores[STELA_AIR_COMMAND_STORES];
   nir_def *destination, *source, *bytes, *copy_valid;
};
/* Encodes command as the writes in *encoding, building straight-line NIR at
 * the cursor. An invalid buffer or pipeline state is undefined, as in
 * Metal, but no write may leave the buffer for an index or slot past it.
 * Returning false refuses compilation. */
typedef bool (*stela_air_encode_command)(nir_builder *b, const struct stela_air_command *command,
                                         struct stela_air_command_encoding *encoding,
                                         void *data);

struct stela_air_threadgroup_view {
   uint64_t size; /* Exact nonempty per-dispatch byte extent, not type_size. */
   unsigned alignment;
   enum stela_air_buffer_access available_access;
};

/* AS3 uses a separate Metal index namespace. Return the actual dispatch's
 * allocation length/rights/alignment; NIR owns the shared-memory layout, never
 * a CPU/GPU VA. The exact sizes/alignment and workgroup specialization must be
 * retained in receipts and the pipeline cache key. Changing them requires a
 * matching compiled specialization and dispatch shared-memory reservation. */
typedef bool (*stela_air_bind_threadgroup)(const struct stela_air_buffer *buffer,
                                          struct stela_air_threadgroup_view *view,
                                          void *data);

/* A threadgroup buffer's byte offset, a uint multiple of 16, within the
 * threadgroup memory that dispatches place buffers in. Emits straight-line
 * NIR in the entry prelude; returning NULL refuses compilation. */
typedef nir_def *(*stela_air_threadgroup_offset)(nir_builder *b,
                                                 const struct stela_air_buffer *buffer,
                                                 void *data);

struct stela_air_ssbo_view {
   unsigned slot;
   uint64_t buffer_size;
   uint64_t offset;
   uint64_t size;
   enum stela_air_buffer_access access; /* Required by this shader argument. */
   enum stela_air_buffer_access available_access; /* Actual resource rights. */
   unsigned alignment; /* Required power-of-two byte alignment of this view. */
};

/* Command binding precondition for the 32-bit get_ssbo_size ABI: nonempty exact
 * byte views, <=UINT32_MAX bytes, wholly within the real allocation, and a slot
 * within the pipeline's explicit table extent (1..UINT8_MAX). The native driver
 * must additionally validate its GPU base VA/canonical form and base alignment,
 * lifetime and residency when constructing the original hardware surface state.
 * No physical VA is constructed by the SSBO compiler path. */
bool stela_air_ssbo_view_valid(const struct stela_air_ssbo_view *view,
                               unsigned ssbo_count);

enum stela_air_fp_profile {
   /* Ordinary LLVM environment; private AIR denorm options are not inferred. */
   STELA_AIR_FP_IEEE_DEFAULT = 0,
   /* Metal's environment: FP32 rounds to nearest even and flushes denormal
    * inputs and results when the module's compile options disable them
    * (air.compile.denorms_disable, Metal's default), preserving them
    * otherwise. */
   STELA_AIR_FP_METAL = 1,
};

/* Values the device supplies rather than the stage. */
enum stela_air_device_value {
   /* A dispatchThreads grid's threads as a uvec3; zero for threadgroup
    * dispatches. */
   STELA_AIR_DISPATCH_GRID,
   /* The amplification ID of an amplified draw, as a uint. */
   STELA_AIR_AMPLIFICATION,
   /* The amplified draw's views as a uint. */
   STELA_AIR_AMPLIFICATIONS,
   /* That view's mapping as a uint: the viewport offset in the low half, the
    * render target array index offset in the high half. */
   STELA_AIR_VIEW_MAPPING,
   /* A post-tessellation vertex function's patch ID, instance ID and base
    * instance as a uvec3. */
   STELA_AIR_PATCH_IDS,
   /* A dynamic kernel's threadgroup shape as a uvec3. */
   STELA_AIR_THREADGROUP_SHAPE,
   /* The address of the buffer os_log appends to, zero for none. */
   STELA_AIR_LOG_BUFFER,
   /* A fragment function's pass's samples per pixel, as a uint. */
   STELA_AIR_SAMPLE_COUNT,
   /* The render encoder's blend color as a vec4. */
   STELA_AIR_BLEND_COLOR,
   /* Its pass's sample positions as a uvec4 of a byte per sample for up to
    * 16, first sample lowest: x in 1/16 pixel in the low nibble, y in the
    * high. */
   STELA_AIR_SAMPLE_POSITIONS,
   /* A stage-in dispatch's region's origin and size, each a uvec3. */
   STELA_AIR_STAGE_IN_ORIGIN,
   STELA_AIR_STAGE_IN_SIZE,
   /* A tile function's dispatch, as a uvec4: the pass's tile width in the
    * low half of x and its height in the high half, then the bytes of each
    * tile's threadgroup memory that persist across the pass's dispatches,
    * and the 64-bit address, low word first, of the memory that holds them
    * between dispatches, tile after tile in rows. */
   STELA_AIR_TILE,
   /* A render pass's explicit imageblock data, as a uvec4: the bytes each
    * sample takes, the pixels in each row, and the 64-bit address, low
    * word first, of the data, each pixel's samples in turn, row after row;
    * the rows and their pixels are whole tiles'. */
   STELA_AIR_IMAGEBLOCK,
   /* The depth and stencil state a fragment function that tests them itself
    * applies, as a uvec4: the depth word, the front and back faces' stencil
    * words, laid out below, and the encoder's stencil reference values, the
    * front face's in bits 0 to 7 and the back face's in bits 8 to 15, which
    * a function without a stencil output tests against. */
   STELA_AIR_DEPTH_STENCIL,
   /* The viewport's near and far depths as a uvec4 of float bits, then two
    * zeros. */
   STELA_AIR_DEPTH_RANGE,
   /* A fragment function's pass's dirty tiles, as a uvec4: the stamp each
    * fragment stores at its tile of a color attachment's dirty layer, zero
    * for none, and that attachment, then two zeros
    * (STELA_METAL_TABLE_DIRTY). */
   STELA_AIR_DIRTY_TILES,
};

/* The log2 of the side of STELA_AIR_DIRTY_TILES's square tiles, and the
 * layer of the attachment's raw image that holds them, a texel each. */
#define STELA_AIR_DIRTY_TILE_SHIFT 6u
#define STELA_AIR_DIRTY_LAYER 2u

/* STELA_AIR_DEPTH_STENCIL's depth word: the MTLCompareFunction of the depth
 * test in bits 0 to 2, which Vulkan's VkCompareOp numbers alike, depth
 * writes in bit 3, the depth attachment's encoding in bits 4 and 5, a
 * stencil attachment in bit 6, and depth clamping to the viewport's range in
 * bit 7. A face's stencil word: its compare function in bits 0 to 2, its
 * MTLStencilOperations (VkStencilOp alike) on stencil failure, depth failure
 * and pass in bits 3 to 5, 6 to 8 and 9 to 11, its read mask in bits 16 to
 * 23 and its write mask in bits 24 to 31. A pass without a depth attachment
 * passes every depth test; one without a stencil attachment, every stencil
 * test. */
#define STELA_AIR_DEPTH_WRITE 8u
#define STELA_AIR_DEPTH_FORMAT_SHIFT 4u
#define STELA_AIR_DEPTH_STENCIL_ATTACHED 64u
#define STELA_AIR_DEPTH_CLAMP 128u
enum stela_air_depth_format {
   STELA_AIR_DEPTH_NONE,
   STELA_AIR_DEPTH_UNORM16,
   STELA_AIR_DEPTH_UNORM24,
   STELA_AIR_DEPTH_FLOAT32,
};
typedef nir_def *(*stela_air_load_device_value)(nir_builder *, enum stela_air_device_value,
                                                void *binding_data);

/* A stage-in attribute at location, as four 32-bit lanes. For a
 * post-tessellation vertex function, `at` is its STELA_AIR_PATCH_IDS, and
 * the attribute is that of the patch's control point `point`, or of the
 * patch when point is NULL; for a kernel, `at` is the thread's position in
 * the grid and point is NULL. integer says whether the function reads it as
 * integers. NULL when the stage-in layout does not provide it so. */
typedef nir_def *(*stela_air_fetch_attribute)(nir_builder *, unsigned location, nir_def *at,
                                              nir_def *point, bool integer, void *binding_data);

/* A tile function's render pass: the texel layouts of its color attachments,
 * by location, PIPE_FORMAT_NONE where it has none, and its samples per pixel.
 * The function, a dynamic kernel, runs a threadgroup per tile, and its
 * implicit imageblock is the attachments' texels under the tile, which it
 * accesses as storage. Its threadgroup memory, which its dispatches place
 * buffers in, persists across the pass's dispatches as STELA_AIR_TILE gives
 * it. */
struct stela_air_tile {
   enum pipe_format formats[8];
   unsigned samples;
};

struct stela_air_options {
   unsigned workgroup_size[3]; /* Exact nonzero pipeline specialization. */
   /* A kernel's explicit imageblock's width and height, as its encoder's
    * setImageblockWidth:height: gives them; zero for the threadgroup's. */
   unsigned imageblock_size[2];
   /* A tile function's pass; NULL for a kernel that is not one. */
   const struct stela_air_tile *tile;
   unsigned max_instructions; /* Nonzero caller-supplied compilation budget. */
   stela_air_bind_buffer bind_buffer;
   void *binding_data;
   /* Select one buffer callback, never both. Zero-initialized additions retain
    * the original global64 path. SSBO mode uses the existing NIR SSBO intrinsics
    * and size query; the backend must lower these through its actual resource
    * ABI. Full 64-bit offsets are checked before narrowing; out-of-view loads
    * return zero and stores are discarded, never wrapped into another offset.
    * Every runtime view must pass stela_air_ssbo_view_valid before dispatch. */
   stela_air_bind_ssbo bind_ssbo;
   unsigned ssbo_count;
   enum stela_air_fp_profile fp_profile;
   stela_air_bind_threadgroup bind_threadgroup;
   /* Explicit target/backend representability limit. Zero refuses shared
    * resources/barriers; NAK callers must additionally stay within UINT16_MAX
    * before its original smem_size conversion. No target limit is inferred. */
   unsigned max_threadgroup_bytes;
   /* A dynamic kernel, whose dispatches shape its threadgroups: its
    * workgroup_size is (n, 1, 1) for the most threads a threadgroup holds,
    * STELA_AIR_THREADGROUP_SHAPE supplies the shape, and threads past it
    * skip the function. Its dispatches may also place its threadgroup
    * buffers, within dynamic_threadgroup_bytes of threadgroup memory at
    * the offsets threadgroup_offset gives, instead of bind_threadgroup's
    * views. */
   bool dynamic_shape;
   unsigned dynamic_threadgroup_bytes;
   stela_air_threadgroup_offset threadgroup_offset;
   /* Explicit target limits on a thread's private memory (allocas, lowered
    * to NIR scratch) and on the shader's constant data (initialized constant
    * globals, lowered to load_constant over nir_shader::constant_data). Zero
    * refuses that storage. */
   unsigned max_private_bytes, max_constant_bytes;
   /* Texture and sampler arguments; zero refuses them. */
   stela_air_bind_texture bind_texture;
   stela_air_bind_sampler bind_sampler;
   /* [[color(n)]] inputs; zero refuses them. */
   stela_air_bind_attachment bind_attachment;
   /* The depth and stencil copies of a function that tests them itself;
    * zero refuses stencil outputs without stencil export. */
   stela_air_bind_depth_stencil bind_depth_stencil;
   /* Textures and samplers in argument buffers; zero refuses them. Argument
    * buffers hold device addresses, so they require bind_buffer. */
   stela_air_bind_texture_id bind_texture_id;
   stela_air_bind_sampler_id bind_sampler_id;
   /* The words an argument buffer's texture IDs name; NULL takes each ID
    * as bind_texture_id takes it. */
   stela_air_texture_view_word texture_view_word;
   /* Texture and sampler array arguments' elements, which also need
    * bind_texture_id or bind_sampler_id; zero refuses the arrays. */
   stela_air_resource_id resource_id;
   /* The samplers of the constexpr sampler states of the input and its
    * linked modules, which bind_sampler_id resolves; a state with none is
    * refused. */
   const struct stela_air_constexpr_sampler *constexpr_samplers;
   unsigned constexpr_sampler_count;
   /* The count of the images bind_texture selects among for a write or
    * read_write texture; NULL refuses get_num_mip_levels of them. */
   stela_air_texture_query texture_levels;
   /* Whether a texture is unbound; NULL refuses is_null_texture. */
   stela_air_texture_query texture_null;
   /* Whether a write or read_write float texture's storage images hold an
    * sRGB format's texels, whose color channels the shader encodes as it
    * writes and decodes as it reads; NULL when none do. */
   stela_air_texture_query texture_srgb_storage;
   /* A write or read_write float texture's enum stela_air_extended_storage:
    * whether its storage images hold a two-plane extended-range format's
    * half floats, which the shader holds to the format's codes and 8-bit
    * alpha as it writes; NULL when none do. */
   stela_air_texture_query texture_extended_storage;
   /* Whether a sampler compares with Metal's Never, through which a
    * comparison is zero though Vulkan's sampler for it does not compare;
    * NULL leaves comparisons to the sampler bind_sampler resolves. */
   stela_air_sampler_query sampler_never;
   /* A sampler's stela_air_sampler_filter bits and a float texture's
    * stela_air_filter_class, for the bicubic filter shaders apply in
    * Vulkan's place; NULL when no sampler filters bicubically. */
   stela_air_sampler_query sampler_filter;
   stela_air_texture_query texture_filter_class;
   /* A float 2D texture's second plane (enum stela_air_planes), zero for
    * textures of one plane, whose samples and reads it then combines with
    * the first's; consulted where dynamic_planes says textures may have
    * two. */
   stela_air_texture_query texture_planes;
   bool dynamic_planes;
   /* Samplers of argument values may filter bicubically or take pixel
    * coordinates: their samples read the sampler's filter bits, which a
    * constexpr sampler's fold to a constant. */
   bool dynamic_bicubic;
   /* The samplers bound at these indices filter bicubically or take pixel
    * coordinates: their samples read the sampler's filter bits, and those
    * of the others do not. */
   uint32_t bicubic_samplers;
   /* Indirect command buffers and pipeline states in argument buffers;
    * NULL refuses AIR's command functions. */
   stela_air_command_count command_count;
   stela_air_encode_command encode_command;
   /* Device values; NULL refuses the builtins that read them. With it,
    * kernel threads outside a dispatchThreads grid return before the
    * function runs, as Metal's smaller edge threadgroups never run them. */
   stela_air_load_device_value load_device_value;
   /* Stage-in attributes of kernels and post-tessellation vertex
    * functions; NULL refuses them. */
   stela_air_fetch_attribute fetch_attribute;
   /* Function constant values by index (see air_constants.c). A required
    * constant without one is refused; values for constants the module does
    * not declare are ignored. */
   const struct stela_air_constant_value *constants;
   unsigned constant_count;
   /* Modules linked with the input's, compiled as it was and each
    * specialized with its own constant values, which define its external
    * functions.
    * A visible function table's handle i + 1 calls linked_functions[i],
    * which the input or a module defines; any other handle calls none, and
    * a call through it returns zero. */
   const struct stela_air_module *modules;
   unsigned module_count;
   const struct stela_air_linked_function *linked_functions;
   unsigned linked_function_count;
   /* The pipeline stage's maxCallStackDepth: how deeply calls through
    * tables nest when they recurse. */
   unsigned max_call_stack_depth;
   /* Narrowing floating-point conversions round toward zero rather than to
    * nearest even (MTLCompileOptions.floatingPointConversionRoundingMode). */
   bool round_toward_zero;
};

/* Additive VS/FS frontend. No runtime/Metal pipeline is constructed. resources
 * supplies the same actual AS2-readonly view/bounds and FP contract as compute;
 * workgroup_size/max_threadgroup_bytes/bind_threadgroup must be zero/NULL for
 * vertex and fragment functions. Object and mesh functions are compute-like
 * and take them as kernels do: they become task and mesh shaders, which
 * keep a few bytes of threadgroup memory for their grid or primitive count.
 * AS2 and AS1 are not interchangeable permissions/address spaces. All declared
 * varying identifiers/types must match the explicit shared VS/FS map. Target
 * IO limits are real caller facts, not inferred Metal family advertisement.
 * Returned NIR owns variables/names; no descriptor/map/input pointer survives.
 * Original NIR stage IO lowering belongs to the actual vendor compiler owner.
 * Packed flat FP32 scalar/vector aggregates assembled through insertvalue may
 * cross PHIs, including loop backedges; every reachable incoming leaf must be
 * completely defined. Nested/unpacked aggregates, other leaf types and direct
 * aggregate constants remain unsupported. The source path awaits qualification.
 * As with the compute entrypoint, nir_options must outlive the returned shader;
 * the owned vendor compiler wrapper must retain a stable qualified options copy. */
/* MTLBlendFactor and MTLBlendOperation. */
enum stela_air_blend_factor {
   STELA_AIR_BLEND_ZERO,
   STELA_AIR_BLEND_ONE,
   STELA_AIR_BLEND_SOURCE_COLOR,
   STELA_AIR_BLEND_ONE_MINUS_SOURCE_COLOR,
   STELA_AIR_BLEND_SOURCE_ALPHA,
   STELA_AIR_BLEND_ONE_MINUS_SOURCE_ALPHA,
   STELA_AIR_BLEND_DESTINATION_COLOR,
   STELA_AIR_BLEND_ONE_MINUS_DESTINATION_COLOR,
   STELA_AIR_BLEND_DESTINATION_ALPHA,
   STELA_AIR_BLEND_ONE_MINUS_DESTINATION_ALPHA,
   STELA_AIR_BLEND_SOURCE_ALPHA_SATURATED,
   STELA_AIR_BLEND_COLOR_FACTOR,
   STELA_AIR_BLEND_ONE_MINUS_COLOR_FACTOR,
   STELA_AIR_BLEND_ALPHA_FACTOR,
   STELA_AIR_BLEND_ONE_MINUS_ALPHA_FACTOR,
   /* Dual-source blending's second source, color(0) index(1), which only
    * attachment 0 blends with. */
   STELA_AIR_BLEND_SOURCE1_COLOR,
   STELA_AIR_BLEND_ONE_MINUS_SOURCE1_COLOR,
   STELA_AIR_BLEND_SOURCE1_ALPHA,
   STELA_AIR_BLEND_ONE_MINUS_SOURCE1_ALPHA,
};
enum stela_air_blend_op {
   STELA_AIR_BLEND_ADD,
   STELA_AIR_BLEND_SUBTRACT,
   STELA_AIR_BLEND_REVERSE_SUBTRACT,
   STELA_AIR_BLEND_MIN,
   STELA_AIR_BLEND_MAX,
};

/* A color attachment a fragment function reads, which it accesses as
 * storage instead of through the blend unit: the function loads the texel
 * where it first uses the read and, if it writes the attachment, blends,
 * masks and stores its output when it ends, all in one fragment shader
 * interlock section, so overlapping fragments read and write each texel in
 * primitive order. The pipeline writes no color to the attachment. */
struct stela_air_attachment_access {
   /* The texel layout; PIPE_FORMAT_NONE leaves the attachment to the blend
    * unit. */
   enum pipe_format format;
   bool blend;
   /* Bit i writes channel i: red, green, blue, alpha. */
   unsigned write_mask;
   enum stela_air_blend_op rgb_op, alpha_op;
   enum stela_air_blend_factor source_rgb, destination_rgb, source_alpha, destination_alpha;
   /* A two-plane format's planes (enum stela_air_planes), zero for one:
    * `format` is its first plane's, and its second plane's raw texels, two
    * bytes, follow in the next storage descriptor. Only a 4:4:4 second
    * plane renders. */
   unsigned planes;
};

/* How an extended-range color attachment encodes its color channels. Its
 * alpha, where it has one, holds [0, 1]. */
enum stela_air_extended_range {
   STELA_AIR_EXTENDED_RANGE_NONE,
   /* [-384, 639] / 510. */
   STELA_AIR_EXTENDED_RANGE_LINEAR,
   /* The same codes through the sRGB curve, mirrored below zero; the
    * attachment holds the linear values. */
   STELA_AIR_EXTENDED_RANGE_SRGB,
   /* RGB10A8_2P_XR10's codes, linear or through the curve, and 8-bit
    * alpha, of an attachment the pipeline blends: Apple GPUs blend the
    * output as the texture holds it. The device renders these formats as
    * half floats and encodes them where the pass ends. */
   STELA_AIR_EXTENDED_RANGE_PLANES_HELD,
   STELA_AIR_EXTENDED_RANGE_PLANES_HELD_SRGB,
   /* RGB10A8_2P_XR10 unblended: the output as Apple GPUs' tile memory
    * holds it, each channel its nearest half, ties to even, which tile
    * functions read unclamped; the pass's end clamps it to the codes. */
   STELA_AIR_EXTENDED_RANGE_PLANES,
};
#define STELA_AIR_EXTENDED_RANGE_BITS 3u

struct stela_air_stage_options {
   enum stela_air_stage stage;
   struct stela_air_options resources;
   const struct stela_air_varying_slot *varyings;
   unsigned varying_count, max_varying_slots;
   /* A vertex function of a pipeline that amplifies vertices: it adds its
    * view's mapping to the render target array and viewport indices it
    * writes, or writes the offsets alone. */
   bool amplified;
   /* An object function, or a mesh function of a pipeline without one: the
    * stage drawMeshThreads launches. As a kernel's do, its threads outside
    * the dispatch grid, which load_device_value supplies, skip the function. */
   bool launched_by_grid;
   /* Target limits on an object payload's bytes and a mesh function's
    * vertices and primitives; zero refuses them. */
   unsigned max_payload_bytes, max_mesh_vertices, max_mesh_primitives;
   /* Whether a fragment function may write the stencil reference. Without
    * it, a function writing one runs the pass's depth and stencil tests and
    * updates itself, in its interlock section where it ends, on copies of
    * the attachments (bind_depth_stencil): it takes its stencil output as
    * each fragment's reference, as Metal does, and covers only the samples
    * that pass. Its pipeline tests no depth or stencil itself. That needs
    * raster_order_groups, and runs for each sample the fragment covers, of
    * `samples` per pixel. */
   bool stencil_export, fragment_tests;
   /* Whether a fragment function runs the depth and stencil tests itself,
    * as fragment_tests describes, whatever it writes: a function reading or
    * writing storage attachments (attachments) whose pipeline cannot test
    * fragments before it runs, because it writes depth, stencil or coverage,
    * discards under a depth or stencil test, or its pipeline turns alpha
    * into coverage. It then writes its storage attachments for the samples
    * that pass, in the same interlock section, and its pipeline tests no
    * depth or stencil. alpha_to_coverage and alpha_to_one are its pipeline's,
    * which it applies itself: color(0)'s alpha choosing the samples kept,
    * as many as it is of the pixel's, the lowest first, before the tests,
    * then 1 as every color output's alpha. */
   bool self_tests, alpha_to_coverage, alpha_to_one;
   /* Whether a fragment function may order its accesses after those of
    * overlapping fragments: the device's fragment shader pixel interlock. */
   bool raster_order_groups;
   /* A fragment function's color attachments, indexed by location, of a
    * pipeline of samples per pixel; NULL accesses none as storage. Storage
    * access needs raster_order_groups, and runs per sample for more than
    * one sample. */
   const struct stela_air_attachment_access *attachments;
   unsigned samples;
   /* A fragment function's color attachments of Apple's extended-range
    * formats, three bits each by location (enum stela_air_extended_range),
    * which the device stores as half floats: the function clamps its
    * outputs to the range the formats encode, as their stores do. */
   unsigned extended_range;
   /* A vertex function's inputs, by location, that its pipeline fetches as
    * this many 32-bit words, the vertex input lacking their formats: each
    * reads as decode_vertex_input makes four 32-bit lanes of its words, with
    * the resources' binding data. NULL, or zero words, for an input fetched
    * in its own format. */
   const uint8_t *vertex_input_words;
   nir_def *(*decode_vertex_input)(nir_builder *, unsigned location, nir_def *words,
                                   void *binding_data);
};
nir_shader *stela_air_stage_to_nir(const void *, size_t, const char *entry,
                                  const nir_shader_compiler_options *,
                                  const struct stela_air_stage_options *,
                                  struct stela_air_error *);

/* Metal's stage callbacks write only to the temporary NIR, so translation
 * can retry the ordinary lowering order when early specialization fails. */
nir_shader *stela_air_metal_stage_to_nir(const void *, size_t, const char *entry,
                                        const nir_shader_compiler_options *,
                                        const struct stela_air_stage_options *,
                                        struct stela_air_error *);

/* Ordinary vertex/fragment stages with software binary16 ALU lowering. The
 * backend profile controls which FP32 arithmetic the shader may request. */
nir_shader *stela_air_metal_software_stage_to_nir(const void *, size_t, const char *entry,
                                                 const nir_shader_compiler_options *,
                                                 const struct stela_air_stage_options *,
                                                 enum stela_air_backend_profile,
                                                 struct stela_air_error *);

/* Input is the hash-checked raw AIR bitcode selected by the archive reader.
 * Success transfers ownership of a real NIR shader to the caller (ralloc_free).
 * Failure returns NULL, destroys partial NIR, and reports an explicit error.
 * The caller retains input bytes/digest/build ID and records the exact selected
 * target, workgroup specialization and argument mapping alongside vendor output.
 * Reachable integer/FP32 scalar/vector CFGs use the original LLVM switch
 * legalization and Mesa goto-to-if structurization, including numeric PHIs,
 * early returns, nested/multiple-latch loops and irreducible entries. Original
 * and legalized inputs share the caller's instruction budget; total switch
 * cases also obey that budget before legalization. Production has no iteration
 * cap. Pointer PHIs require a pre-emission, cycle-aware proof of one validated
 * resource root and homogeneous AS2 root/derived reflection state. The numeric
 * PHI carries only the offset/VA; its root's exact rights/resource/view metadata
 * remain static and every incoming edge must agree. Pointer eq/ne require the
 * same resource authority, ignoring only AS2 root-reflection typing state;
 * relational and cross-resource comparisons remain explicit refusals.
 * Integer division/remainder admit i8/i16/i32/i64 scalar or fixed2-4 lanes
 * through original NIR udiv/idiv/umod/irem. Only LLVM-defined inputs constrain
 * their values: zero divisors and signed MIN/-1 division/remainder are UB;
 * inexact exact-div results are poison. No result/trap behavior is promised
 * there. Direct undef/poison and freeze remain refused. Target owners must
 * run their actual Mesa division/int64 lowering; this source unit is UNRUN.
 * Cross-resource joins,
 * rootless/undef/poison/integer-derived pointers, unsupported pointer producers,
 * compute aggregate PHIs and reachable unsupported terminators are refused
 * explicitly. Unreachable blocks consume the budget but emit no
 * effects. These source paths await semantic/vendor qualification; no family
 * capability follows from accepting a control-flow shape.
 * Qualified AS3 storage is returned as explicit shared32 NIR with the original
 * aligned layout in info.shared_size. Dynamic view sizes/alignment and target
 * byte limits are caller specializations, never inferred GPU capabilities.
 * Only the observed ordinary-shared collective barrier form is supported.
 * LLVM is a native compiler library, not a hostile-input sandbox.
 */
nir_shader *stela_air_to_nir(const void *bitcode, size_t size, const char *entry,
                            const nir_shader_compiler_options *nir_options,
                            const struct stela_air_options *options,
                            struct stela_air_error *error);

/* Compute frontend for a software target with 16-bit integer SSA and no
 * native FP16. Supports register-only half add, subtract, multiply, fused
 * multiply-add and round-to-nearest-even FP32 conversions. Other half
 * arithmetic and 16-bit memory/texture interfaces refuse explicitly. An
 * input that requires preserving FP32 subnormals cannot use arbitrary FP32
 * arithmetic on this target. */
nir_shader *stela_air_software_to_nir(const void *bitcode, size_t size, const char *entry,
                                     const nir_shader_compiler_options *nir_options,
                                     const struct stela_air_options *options,
                                     enum stela_air_backend_profile profile,
                                     struct stela_air_error *error);
#endif

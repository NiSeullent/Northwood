/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef STELA_METAL_COMPILER_H
#define STELA_METAL_COMPILER_H
/* The Metal device's compiler service: shader::Compiler's reflection and
 * compilation of one AIR function, over the binding ABI of metal_abi.h. The
 * platform links it with LLVM and Mesa and hands the device a table of these
 * functions; metal/compiler.rs mirrors every layout here, and the
 * STELA_METAL_*_BYTES sizes are checked on both sides. */
#include "metal_abi.h"

/* MTLFunctionType values of the stages. */
enum stela_metal_function_stage {
   STELA_METAL_VERTEX = 1,
   STELA_METAL_FRAGMENT = 2,
   STELA_METAL_KERNEL = 3,
   STELA_METAL_VISIBLE = 5,
   STELA_METAL_MESH = 7,
   STELA_METAL_OBJECT = 8,
};

#define STELA_METAL_ATTRIBUTES 31u
#define STELA_METAL_VARYINGS 32u
/* Eight render targets and the second source of dual-source blending. */
#define STELA_METAL_COLOR_OUTPUTS 9u
#define STELA_METAL_FUNCTION_CONSTANTS 256u
#define STELA_METAL_NAME_BYTES 8192u

/* Access flags, as shader.rs Access. */
#define STELA_METAL_ACCESS_READ 1u
#define STELA_METAL_ACCESS_WRITE 2u
#define STELA_METAL_ACCESS_SAMPLE 4u

/* A buffer slot; argument_buffer marks an argument buffer. */
struct stela_metal_buffer_use {
   uint32_t index, access, argument_buffer;
};
/* A texture slot: type is an enum stela_air_texture_type and primitive the
 * enum stela_air_primitive of its components. */
struct stela_metal_texture_use {
   uint32_t index, access, type, primitive;
};
/* A threadgroup buffer slot and the bytes its type declares. */
struct stela_metal_threadgroup_use {
   uint32_t index, bytes;
};
/* A stage-in attribute the function declares; of a post-tessellation
 * vertex function, a control point's or else its patch's. active marks one
 * the function reads, which alone needs a layout; its name is in names. */
struct stela_metal_attribute_use {
   uint32_t index, primitive, components, patch_control_point, active, name, name_bytes;
};
/* A varying: its identifier is names[identifier, identifier + bytes). */
struct stela_metal_varying {
   uint32_t identifier, identifier_bytes, primitive, components, per_primitive;
};
/* A render target output; dual_source marks [[index(1)]]. */
struct stela_metal_color_output {
   uint32_t index, primitive, dual_source;
};

/* A function constant: its name and MSL type spelling are in names. */
struct stela_metal_function_constant {
   uint32_t index, name, name_bytes, type, type_bytes, required;
};

/* One function's reflection; a visible function's holds only its stage and
 * the module's facts. names holds the attribute names, the varying
 * identifiers, the function constants' names and types and the module's
 * llvm.ident, whose bytes are zero when it records none. */
struct stela_metal_reflection {
   uint32_t stage, max_total_threads;
   uint64_t threadgroup_bytes;
   uint32_t buffer_count, texture_count, sampler_count, threadgroup_count;
   uint32_t attribute_count, varying_count, color_output_count;
   uint32_t writes_depth, writes_stencil, writes_sample_mask, discards, amplification;
   uint32_t ident, ident_bytes, names_bytes, function_constant_count;
   /* A post-tessellation vertex function's enum stela_air_patch and the
    * control points it declares, zero when it declares none. */
   uint32_t patch, patch_control_points;
   /* Whether a fragment function reads its barycentric coordinates, and
    * interpolates an interpolant, and whether a function takes a visible
    * function table. */
   uint32_t barycentrics, interpolants, function_tables;
   /* Whether the function's module encodes indirect commands, and may set
    * or copy their barriers. */
   uint32_t commands, command_barriers;
   /* The color attachments a fragment function reads, one bit each. */
   uint32_t attachment_reads;
   /* The bytes of each element of a kernel's explicit imageblock, or zero
    * for a function without one. */
   uint32_t imageblock_bytes;
   struct stela_metal_buffer_use buffers[STELA_METAL_BUFFERS];
   struct stela_metal_texture_use textures[STELA_METAL_TEXTURES];
   uint32_t samplers[STELA_METAL_SAMPLERS];
   struct stela_metal_threadgroup_use threadgroup[STELA_METAL_BUFFERS];
   struct stela_metal_attribute_use attributes[STELA_METAL_ATTRIBUTES];
   struct stela_metal_varying varyings[STELA_METAL_VARYINGS];
   struct stela_metal_color_output color_outputs[STELA_METAL_COLOR_OUTPUTS];
   struct stela_metal_function_constant function_constants[STELA_METAL_FUNCTION_CONSTANTS];
   char names[STELA_METAL_NAME_BYTES];
};
#define STELA_METAL_REFLECTION_BYTES 18800u

/* The target's facts, fixed for a device. nir is the Vulkan driver's
 * compiler options, which must outlive every shader compiled with them. */
struct stela_metal_target {
   const nir_shader_compiler_options *nir;
   uint32_t max_instructions, max_threadgroup_bytes, max_private_bytes, max_constant_bytes;
   uint32_t max_varying_slots, max_payload_bytes, max_mesh_vertices, max_mesh_primitives;
   /* Nonzero when fragment shaders may write the stencil reference. */
   uint32_t stencil_export;
   /* Nonzero when fragment shaders may use raster order groups. */
   uint32_t raster_order_groups;
   /* Nonzero when a fragment function writing the stencil reference without
    * stencil export runs the fragment tests itself. */
   uint32_t fragment_tests;
};
#define STELA_METAL_TARGET_BYTES 56u

/* A threadgroup buffer's setThreadgroupMemoryLength:atIndex:. */
struct stela_metal_threadgroup_length {
   uint32_t index, bytes;
};
/* A linked varying, at the location of its position. */
struct stela_metal_linked_varying {
   const char *identifier;
   size_t identifier_bytes;
   uint32_t per_primitive;
};
#define STELA_METAL_LINKED_VARYING_BYTES 24u

#define STELA_METAL_CONSTANT_VALUE_BYTES 24u

/* A color attachment a fragment function reads and writes as storage
 * (stela_air_attachment_access): the VkFormat realizing its pixel format,
 * zero for one the blend unit writes, and its texel bytes; then its
 * MTLRenderPipelineColorAttachmentDescriptor's blending: whether it blends,
 * its MTLColorWriteMask, its MTLBlendOperations of color and alpha, and its
 * MTLBlendFactors of source and destination color and alpha. */
struct stela_metal_attachment {
   uint32_t format, bytes;
   uint8_t blend, write_mask, rgb_op, alpha_op;
   uint8_t source_rgb, destination_rgb, source_alpha, destination_alpha;
   /* A two-plane format's planes (enum stela_air_planes), zero for one. */
   uint8_t planes;
};

/* shader.rs Specialization, with a post-tessellation vertex function's
 * pipeline tessellation, and the stage-in layout a kernel or
 * post-tessellation vertex function fetches by; either may be NULL, and
 * every other function has neither. The modules are those of the
 * pipeline's linked functions and preloaded dynamic libraries, which the
 * function's module does not hold, and handle i + 1 of a visible function
 * table calls linked_functions[i] (air_to_nir.h). */
struct stela_metal_specialization {
   uint32_t workgroup_size[3];
   uint32_t threadgroup_length_count;
   const struct stela_metal_threadgroup_length *threadgroup_lengths;
   const struct stela_metal_linked_varying *varyings;
   uint32_t varying_count, amplified, launched_by_grid;
   /* A kernel of indirect commands: its dispatches' DispatchArguments shape
    * its threadgroups, of at most workgroup_size[0] threads, and place its
    * threadgroup buffers. */
   uint32_t dynamic_threadgroups;
   uint32_t constant_count;
   const struct stela_air_constant_value *constants;
   const struct stela_metal_tessellation *tessellation;
   const struct stela_metal_stage_in *stage_in;
   const struct stela_air_module *modules;
   const struct stela_air_linked_function *linked_functions;
   uint32_t module_count, linked_function_count;
   /* The stage's maxCallStackDepth, to which recursion through visible
    * function tables unrolls. */
   uint32_t max_call_stack_depth;
   /* The sampler of each constexpr sampler state of the function's and the
    * linked modules (stela_metal_sampler_states). */
   uint32_t sampler_count;
   const struct stela_air_constexpr_sampler *samplers;
   /* A fragment function's color attachments, attachment_count of them,
    * and its pipeline's samples per pixel. */
   const struct stela_metal_attachment *attachments;
   uint32_t attachment_count, samples;
   /* Narrowing floating-point conversions round toward zero, as the
    * library's MTLCompileOptions ask, not to nearest even. Then a fragment
    * function's extended-range color attachments, three bits each by location
    * (air_to_nir.h enum stela_air_extended_range). */
   uint32_t round_toward_zero, extended_range;
   /* A vertex function's MTLVertexFormat of each input, by location, that
    * its pipeline fetches as 32-bit words, the device's vertex input
    * lacking the format; zero for the rest. */
   uint8_t vertex_formats[32];
   /* A kernel's explicit imageblock's width and height, its encoder's
    * setImageblockWidth:height:, or zero for its threadgroup's. */
   uint32_t imageblock_size[2];
   /* A tile function of dynamic threadgroups: its pass's color attachments
    * are attachments' formats, which its implicit imageblock accesses, and
    * samples its samples per pixel. */
   uint32_t tile;
   /* A fragment function's fragment tests: STELA_METAL_SELF_TESTS when it
    * runs them itself, accessing storage attachments its pipeline cannot
    * test fragments for first, and its pipeline's alpha to coverage and
    * alpha to one (stela_air_stage_options::self_tests). */
   uint32_t fragment_tests;
   /* Metal's bicubic filter and pixel coordinates, which shaders apply:
    * bit n when the sampler bound at index n has either at the draws or
    * dispatches this compilation serves, and STELA_METAL_BICUBIC_IDS when
    * the function's process has made such samplers, which its samplers of
    * argument values may then be. */
   uint32_t bicubic;
};
#define STELA_METAL_BICUBIC_IDS 0x80000000u
#define STELA_METAL_SPECIALIZATION_BYTES 200u
#define STELA_METAL_SELF_TESTS 1u
#define STELA_METAL_ALPHA_TO_COVERAGE 2u
#define STELA_METAL_ALPHA_TO_ONE 4u
#define STELA_METAL_ATTACHMENT_BYTES 20u
#define STELA_METAL_CONSTEXPR_SAMPLER_BYTES 16u
#define STELA_METAL_MODULE_BYTES 48u
#define STELA_METAL_LINKED_FUNCTION_BYTES 16u
/* The metal_abi.h layouts it points to. */
#define STELA_METAL_PIPELINE_TESSELLATION_BYTES 28u
#define STELA_METAL_STAGE_IN_BYTES 752u

/* Reflects entry of the AIR module specialized with the function constant
 * values, without the arguments a false [[function_constant]] makes absent;
 * with NULL values, unspecialized, with every argument it declares. False
 * after reporting why in error. */
bool stela_metal_reflect(const void *air, size_t size, const char *entry,
                         const struct stela_air_constant_value *constants,
                         uint32_t constant_count, struct stela_metal_reflection *reflection,
                         struct stela_air_error *error);

/* MTLBindingType's kinds of binding a function argument is. */
enum stela_metal_binding_kind {
   STELA_METAL_BINDING_BUFFER,
   STELA_METAL_BINDING_THREADGROUP,
   STELA_METAL_BINDING_TEXTURE,
   STELA_METAL_BINDING_SAMPLER,
   STELA_METAL_BINDING_FUNCTION_TABLE,
   STELA_METAL_BINDING_OBJECT_PAYLOAD,
   STELA_METAL_BINDING_IMAGEBLOCK,
};

#define STELA_METAL_BINDINGS 128u
#define STELA_METAL_MEMBERS 1024u
#define STELA_METAL_BINDING_NAME_BYTES 32768u

/* A binding: its kind, index, count of array elements, STELA_METAL_ACCESS
 * flags and whether the function uses it; a buffer's argument_buffer, and
 * the size, alignment and MSL type spelling of what a buffer, threadgroup
 * buffer or payload points to, with its structure's members; a texture's
 * enum stela_air_texture_type, component enum stela_air_primitive and
 * depth. Its name and type are in names. */
struct stela_metal_binding {
   uint32_t kind, index, count, access, used, argument_buffer;
   uint32_t name, name_bytes, type, type_bytes;
   uint32_t size, alignment;
   uint32_t texture_type, primitive, depth;
   uint32_t members, member_count;
};
/* A structure member: its offset, the bytes of one element, its element
 * count (zero for one), its name and MSL type spelling, and the members of
 * the structure it is or, in an argument buffer, points to. An argument
 * buffer's member is an argument holding count resources of an enum
 * stela_air_field_kind from [[id]] id: as a binding of that kind, its
 * access, pointee size and alignment, and texture type, component and
 * depth. */
struct stela_metal_member {
   uint32_t offset, size, elements, count;
   uint32_t name, name_bytes, type, type_bytes;
   uint32_t members, member_count;
   uint32_t argument, kind, id, access, size_pointed, alignment_pointed;
   uint32_t texture_type, primitive, depth;
};
/* A function's bindings, in argument order, and their structures'
 * members, each structure's contiguous. */
struct stela_metal_bindings {
   uint32_t binding_count, member_count, names_bytes;
   struct stela_metal_binding bindings[STELA_METAL_BINDINGS];
   struct stela_metal_member members[STELA_METAL_MEMBERS];
   char names[STELA_METAL_BINDING_NAME_BYTES];
};
#define STELA_METAL_BINDINGS_BYTES 119308u

/* The bindings of entry, specialized as stela_metal_reflect specializes
 * it: MTLPipelineOptionBindingInfo's reflection and the layouts argument
 * encoders write. False after reporting why in error. */
bool stela_metal_reflect_bindings(const void *air, size_t size, const char *entry,
                                  const struct stela_air_constant_value *constants,
                                  uint32_t constant_count, struct stela_metal_bindings *bindings,
                                  struct stela_air_error *error);

/* Calls visit with the name and stage of each function the AIR module lists,
 * for a library of source, whose container has no list. This reads only module
 * metadata. False after reporting why, or when visit refuses one. */
typedef bool (*stela_metal_visit_function)(void *data, const char *name, size_t name_bytes,
                                           enum stela_metal_function_stage stage);
bool stela_metal_functions(const void *air, size_t size, stela_metal_visit_function visit,
                           void *data, struct stela_air_error *error);
/* Calls visit with each function a dynamic library of the module exports,
 * its stage STELA_METAL_KERNEL, which nothing reads. False after
 * reporting why, or when visit refuses one. */
bool stela_metal_exports(const void *air, size_t size, stela_metal_visit_function visit,
                         void *data, struct stela_air_error *error);

/* Calls visit with the AIR of a module of stitched functions, whose calls
 * stay declarations a pipeline links. False after reporting why. */
bool stela_metal_stitch(const struct stela_air_stitch_source *sources, uint32_t source_count,
                        const struct stela_air_stitch_graph *graphs, uint32_t graph_count,
                        stela_air_visit_bytes visit, void *data, struct stela_air_error *error);

#define STELA_METAL_SAMPLER_STATES 256u

/* The constexpr sampler states the AIR module declares, each Apple's
 * encoding, which metal/sampler.rs decodes. This reads only module metadata.
 * False after reporting why. */
bool stela_metal_sampler_states(const void *air, size_t size,
                                uint64_t states[STELA_METAL_SAMPLER_STATES], uint32_t *count,
                                struct stela_air_error *error);

/* Compiles entry against the specialization: NIR whose entry point is
 * main, which stela_metal_release frees, or NULL after reporting why. A
 * visible function compiles only into the functions that link it. */
nir_shader *stela_metal_compile(const void *air, size_t size, const char *entry,
                                const struct stela_metal_target *target,
                                const struct stela_metal_specialization *specialization,
                                struct stela_air_error *error);

/* As stela_metal_compile, with the compiler's own successful reflection of
 * this exact AIR, entry and constant values. This reads only the stage, patch,
 * threadgroup bytes and varying descriptors with their identifiers. The
 * caller owns the reflection through the call. NULL asks it to reflect. */
nir_shader *stela_metal_compile_reflected(const void *air, size_t size, const char *entry,
                                         const struct stela_metal_target *target,
                                         const struct stela_metal_specialization *specialization,
                                         const struct stela_metal_reflection *reflection,
                                         struct stela_air_error *error);

/* Software binary16 for kernels and ordinary vertex/fragment stages. Render
 * stages require explicit FP32 flush-to-zero, 32-bit inputs, and 32-bit
 * outputs or a primary half color target. The caller supplies a backend with
 * 16-bit integer SSA and a separate shader cache key. */
nir_shader *stela_metal_compile_software_reflected(const void *air, size_t size, const char *entry,
                                                  const struct stela_metal_target *target,
                                                  const struct stela_metal_specialization *specialization,
                                                  const struct stela_metal_reflection *reflection,
                                                  struct stela_air_error *error);

/* The software profile with FP32 denorm preservation for bounded shader
 * arithmetic. The caller must select the patched lavapipe that implements
 * the private NIR contract and use a separate shader cache key. */
nir_shader *stela_metal_compile_lavapipe_reflected(const void *air, size_t size, const char *entry,
                                                 const struct stela_metal_target *target,
                                                 const struct stela_metal_specialization *specialization,
                                                 const struct stela_metal_reflection *reflection,
                                                 struct stela_air_error *error);

/* The vertex and control stages a tessellated pipeline runs ahead of its
 * post-tessellation vertex function (metal_abi.h), each with entry point
 * main, which stela_metal_release frees. */
void stela_metal_tessellation_stages(const struct stela_metal_target *target,
                                     const struct stela_metal_tessellation *tessellation,
                                     nir_shader *stages[2]);

void stela_metal_release(nir_shader *shader);

/* The buffer slots a compiled function accesses through its stage's table,
 * one bit each: the arguments it references, which Metal requires bound. */
uint32_t stela_metal_buffer_reads(const nir_shader *shader);
#endif

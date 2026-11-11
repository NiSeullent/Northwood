/* SPDX-License-Identifier: GPL-2.0-or-later */
#define STELA_AIR_METADATA_INTERNAL
#include "air_to_nir.h"
#include "air_calls.h"
#include "air_commands.h"
#include "air_log.h"
#include "air_constants.h"
#include "air_math.h"
#include "air_matrix.h"
#include "air_precise.h"
#include "air_simd.h"
#include "air_texture.h"
#include "compiler/nir/nir_format_convert.h"
#include "llvm-c/Analysis.h"
#include "llvm-c/BitReader.h"
#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm-c/Error.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/set.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

struct air_block {
   LLVMBasicBlockRef llvm;
   nir_block *start;
   nir_block *end; /* Actual predecessor after any guarded-access splits. */
   bool reachable;
   LLVMValueRef incoming_phi, incoming_value; /* Deduplicate one PHI in linear time. */
};

/* The most roots one pointer chooses among: each buffer a function binds,
 * of Metal's 31, and null. */
#define POINTER_CHOICES 32u

/* The roots a pointer may be at run time, as a generic pointer cast from
 * different address spaces or a join of different resources is: arguments,
 * variables, allocations and null. */
struct pointer_choices {
   unsigned count;
   LLVMValueRef roots[POINTER_CHOICES];
};

struct value {
   nir_def *def;
   unsigned access; /* Buffer provenance; zero for integer values. */
   LLVMValueRef pointer_root; /* Proven resource identity, never a numeric address. */
   /* A pointer chosen among roots, whose accesses branch on choice, the
    * index of the root it is; def is the address or offset that root takes.
    * NULL for a pointer of one root. */
   const struct pointer_choices *choices;
   nir_def *choice;
   bool ssbo;
   unsigned ssbo_slot;
   /* The alignment of the buffer argument's declared type, the most a
    * pointer into it is known to have whatever its accesses claim: Apple
    * GPUs accept vector accesses at their element alignment, which LLVM may
    * claim more of. Zero when the type is unknown. */
   unsigned declared_alignment;
   /* The size of a bound buffer argument's declared type, the bytes from its
    * address a fragment function may read wherever it runs: a draw binds
    * every buffer its functions access. Zero for others. */
   unsigned declared_size;
   /* Threadgroup or thread memory's variable, or, without one, constant data
    * at constant_base in the shader's data. Either is storage_size bytes. */
   nir_variable *storage;
   bool constant_data;
   unsigned storage_size, constant_base;
   /* A null pointer or one off it: a buffer a function constant left out,
    * whose accesses a shader guards at run time. They reach no memory. */
   bool null;
   /* A pointer into an argument buffer: the buffer's fields, the constant
    * part of the pointer's offset and the greatest common divisor of the
    * steps its dynamic indices add (zero without one). PHIs lose the
    * offset. */
   struct stela_air_fields fields;
   bool offset_known;
   uint64_t offset, steps;
   /* Partial values have no vector def until every lane is defined. Missing
    * lanes are tracked symbolically and read undefined (whole_value). */
   nir_def *lanes[4];
   unsigned components, bits, aggregate_count;
   struct value **elements;
   /* Texture and sampler arguments, used only as texture call operands, and
    * the MTLResourceID of one an argument buffer or array holds. A texture or
    * sampler array argument, and each pointer into it, is instead an array:
    * def is the pointer's byte offset among the array's handles. */
   const struct stela_air_texture *texture;
   const struct stela_air_sampler *sampler;
   nir_def *id;
   bool array;
   /* A texture chosen at run time between this one and a null texture: true
    * where it is this one. NULL for a texture always present. */
   nir_def *present;
   /* The argument buffer field of an indirect command buffer or pipeline
    * state whose gpuResourceID is id, which only command calls use. */
   const struct stela_air_field *object;
   /* A two-member result: a texture call's {texel, residency} or cmpxchg's
    * {old value, exchanged}. A member is NULL when nothing reads it. */
   nir_def *members[2];
};

bool
stela_air_ssbo_view_valid(const struct stela_air_ssbo_view *view,
                          unsigned ssbo_count)
{
   return view && ssbo_count && ssbo_count <= UINT8_MAX &&
          view->slot < ssbo_count && view->size && view->size <= UINT32_MAX &&
          view->access >= STELA_AIR_READ && view->access <= STELA_AIR_READ_WRITE &&
          view->available_access >= STELA_AIR_READ && view->available_access <= STELA_AIR_READ_WRITE &&
          !(view->access & ~view->available_access) &&
          view->alignment && !(view->alignment & (view->alignment - 1)) &&
          !(view->offset & (view->alignment - 1)) &&
          view->offset <= view->buffer_size &&
          view->size <= view->buffer_size - view->offset;
}

struct pointer_node {
   LLVMValueRef llvm, root;
   const struct pointer_choices *choices; /* Instead of root, for several. */
   unsigned index, low, next, component;
   bool active;
};

struct translator {
   LLVMContextRef context;
   LLVMModuleRef module;
   LLVMTargetDataRef layout; /* Borrowed from module. */
   struct stela_air_metadata metadata;
   nir_builder b;
   struct hash_table *values;
   const struct stela_air_options *options;
   const struct stela_air_stage_options *stage_options;
   struct stela_air_render render;
   const struct stela_air_field *fields; /* The entry's argument buffer fields. */
   nir_variable *outputs[AIR_MAX_ARGUMENTS];
   /* An object or mesh function's mesh grid properties or mesh, its payload,
    * a mesh's primitive indices, the grid or primitive count any thread may
    * set, and the block every thread leaves the function through. */
   LLVMValueRef mesh, mesh_grid;
   nir_variable *payload, *indices, *mesh_state;
   nir_block *exit;
   /* A kernel's explicit imageblock: threadgroup memory of width by height
    * elements of element bytes, row by row. */
   nir_variable *imageblock;
   unsigned imageblock_elements, imageblock_element;
   /* The explicit imageblock's dimensions, as the dispatch's encoder sets
    * them, within its imageblock_elements elements. */
   nir_def *imageblock_extent;
   /* A tile function's STELA_AIR_TILE, and whether its threadgroup memory
    * persists, which its exit block copies out. */
   nir_def *tile;
   bool tile_memory;
   /* STELA_AIR_IMAGEBLOCK of a tile function with an explicit imageblock
    * or a fragment function with imageblock data; the fragment's pixel and
    * the samples it covers, and whether it accesses the data. */
   nir_def *imageblock_pass, *imageblock_pixel, *imageblock_mask;
   bool tile_explicit, tile_implicit, fragment_imageblock;
   /* The first air.imageblock_data call, which stands for the origin of
    * every pointer into the imageblock. */
   LLVMValueRef imageblock_root;
   /* An amplified vertex function's layer and viewport outputs. */
   nir_variable *layer, *viewport;
   /* A post-tessellation vertex function's STELA_AIR_PATCH_IDS. */
   nir_def *patch_ids;
   /* A fragment function's storage attachments (stela_air_attachment_access):
    * their mask, each one's heap image and the descriptor index the prelude
    * resolved, its output, the blend color, and the reads of each: a load of
    * a variable nothing stores, which stands for the texel until the
    * interlock section loads it. */
   unsigned storage_attachments;
   nir_variable *attachment_heaps[AIR_MAX_RENDER_TARGETS];
   nir_def *attachment_indices[AIR_MAX_RENDER_TARGETS];
   nir_variable *attachment_outputs[AIR_MAX_RENDER_TARGETS];
   /* Storage attachment 0's second blend source, color(0) index(1). */
   nir_variable *attachment_source1;
   nir_def *blend_color;
   /* STELA_AIR_DIRTY_TILES, and the two-plane extended-range color
    * attachments whose working image may take it. */
   nir_def *dirty;
   unsigned dirty_attachments;
   struct attachment_read {
      const struct stela_air_stage_io *io;
      nir_def *placeholder;
   } attachment_reads[AIR_MAX_ARGUMENTS];
   unsigned attachment_read_count;
   /* A fragment function that runs the depth and stencil tests itself
    * (stela_air_stage_options::stencil_export): whether it does. */
   bool tests_fragments;
   struct stela_air_error *error;
   unsigned air_minor;
   unsigned language_major, language_minor;
   unsigned max_device_buffers;
   unsigned max_threadgroup_buffers;
   unsigned shared_resources;
   /* A dynamic kernel's threadgroup memory its dispatches place buffers in. */
   nir_variable *dynamic_threadgroup;
   unsigned shared_barriers;
   uint64_t shared_layout_bound;
   unsigned private_allocations;
   uint64_t private_layout_bound;
   /* Thread memory variables that hold only SIMD-group matrices, compacted
    * by matrix_storage. */
   struct set *matrix_storage;
   bool denorms_disable;
   bool fast_math;
   bool fp_checked;
   LLVMValueRef entry;
   void *cfg_scratch;
   struct hash_table *block_map, *pointer_map;
   struct pointer_node *pointers;
   unsigned pointer_count;
   struct air_block *blocks;
   struct air_block **work;
   unsigned block_count, reachable_count;
};

static bool
fail(struct translator *t, enum stela_air_error_kind kind, const char *fmt, ...)
{
   if (t->error->kind == STELA_AIR_OK) {
      t->error->kind = kind;
      va_list args;
      va_start(args, fmt);
      vsnprintf(t->error->detail, sizeof(t->error->detail), fmt, args);
      va_end(args);
   }
   return false;
}

static void
diagnostic(LLVMDiagnosticInfoRef info, void *data)
{
   struct translator *t = data;
   if (LLVMGetDiagInfoSeverity(info) != LLVMDSError)
      return;
   char *message = LLVMGetDiagInfoDescription(info);
   fail(t, STELA_AIR_BITCODE, "%s", message ? message : "LLVM bitcode diagnostic");
   LLVMDisposeMessage(message);
}

#define md_string stela_air_md_string
#define md_uint stela_air_md_uint
static bool
md_node(struct translator *t, LLVMValueRef node, LLVMValueRef *operands, unsigned *count)
{
   return stela_air_md_node(&t->metadata, node, operands, count);
}
static bool
named_md(struct translator *t, const char *name, LLVMValueRef *operands, unsigned *count)
{
   return stela_air_named_md(&t->metadata, name, operands, count);
}

static bool
shape(LLVMTypeRef type, unsigned *components, unsigned *bits)
{
   *components = 1;
   if (LLVMGetTypeKind(type) == LLVMVectorTypeKind) {
      *components = LLVMGetVectorSize(type);
      if (*components < 2 || *components > 4)
         return false;
      type = LLVMGetElementType(type);
   }
   if (LLVMGetTypeKind(type) != LLVMIntegerTypeKind)
      return false;
   *bits = LLVMGetIntTypeWidth(type);
   return *bits == 1 || *bits == 8 || *bits == 16 || *bits == 32 || *bits == 64;
}

static bool
float_shape(LLVMTypeRef type, unsigned *components, unsigned *bits)
{
   *components = 1;
   if (stela_air_matrix_type(type, bits)) {
      /* A SIMD-group matrix's arithmetic is its thread elements'. */
      *components = 2;
      type = LLVMGetElementType(type);
   } else if (LLVMGetTypeKind(type) == LLVMVectorTypeKind) {
      *components = LLVMGetVectorSize(type);
      if (*components < 2 || *components > 4)
         return false;
      type = LLVMGetElementType(type);
   }
   *bits = LLVMGetTypeKind(type) == LLVMHalfTypeKind ? 16 : 32;
   return LLVMGetTypeKind(type) == LLVMFloatTypeKind || LLVMGetTypeKind(type) == LLVMHalfTypeKind;
}

/* A bfloat's 16 bits, which only conversions and memory read as a number:
 * its arithmetic is refused rather than taken for a half's. */
static bool
bfloat_shape(LLVMTypeRef type, unsigned *components, unsigned *bits)
{
   *components = LLVMGetTypeKind(type) == LLVMVectorTypeKind ? LLVMGetVectorSize(type) : 1;
   *bits = 16;
   return stela_air_call_bfloat(type);
}

static bool
value_shape(LLVMTypeRef type, unsigned *components, unsigned *bits)
{
   return shape(type, components, bits) || float_shape(type, components, bits) ||
          bfloat_shape(type, components, bits);
}

/* An aggregate leaf: an integer, float or half scalar or vector. */
static bool
leaf_shape(LLVMTypeRef type)
{
   unsigned components, bits;
   return value_shape(type, &components, &bits);
}

/* A narrowing conversion of `value` to half, rounded as the compile options
 * select. */
static nir_def *
narrow_to_half(struct translator *t, nir_def *value)
{
   return t->options->round_toward_zero ? nir_f2f16_rtz(&t->b, value) : nir_f2f16_rtne(&t->b, value);
}

static bool
fp_environment(struct translator *t)
{
   if (t->fp_checked)
      return true;
   if (t->denorms_disable && t->options->fp_profile != STELA_AIR_FP_METAL)
      return fail(t, STELA_AIR_METADATA, "AIR denorms_disable arithmetic requires the Metal FP profile");
   bool ftz = t->denorms_disable;
   const char *special[] = {"strictfp", "denormal_fpenv"};
   for (unsigned i = 0; i < ARRAY_SIZE(special); i++) {
      unsigned kind = LLVMGetEnumAttributeKindForName(special[i], strlen(special[i]));
      if (kind && LLVMGetEnumAttributeAtIndex(t->entry, LLVMAttributeFunctionIndex, kind))
         return fail(t, STELA_AIR_TYPE, "unsupported FP environment attribute: %s", special[i]);
   }
   const char *strings[] = {"denormal-fp-math", "denormal-fp-math-f32", "rounding-math"};
   for (unsigned i = 0; i < ARRAY_SIZE(strings); i++) {
      if (LLVMGetStringAttributeAtIndex(t->entry, LLVMAttributeFunctionIndex,
                                       strings[i], strlen(strings[i])))
         return fail(t, STELA_AIR_TYPE, "unsupported FP environment attribute: %s", strings[i]);
   }
   t->b.shader->info.float_controls_execution_mode =
      (ftz ? FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32 : FLOAT_CONTROLS_DENORM_PRESERVE_FP32) |
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   t->fp_checked = true;
   return true;
}

/* Whether any ALU operation reads or writes a half. */
static bool
half_arithmetic(nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            const nir_op_info *info = &nir_op_infos[alu->op];
            if (nir_alu_type_get_base_type(info->output_type) == nir_type_float &&
                alu->def.bit_size == 16)
               return true;
            for (unsigned i = 0; i < info->num_inputs; i++) {
               if (nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float &&
                   nir_src_bit_size(alu->src[i].src) == 16)
                  return true;
            }
         }
      }
   }
   return false;
}

static unsigned
fp_math_control(LLVMValueRef inst)
{
   unsigned control = nir_fp_no_fast_math;
   LLVMFastMathFlags flags = LLVMCanValueUseFastMathFlags(inst) ? LLVMGetFastMathFlags(inst) : 0;
   if (flags & LLVMFastMathNoNaNs)
      control &= ~nir_fp_preserve_nan;
   if (flags & LLVMFastMathNoInfs)
      control &= ~nir_fp_preserve_inf;
   /* LLVM 23 nsz permits changing zero input signs; NIR's corresponding bit
    * also relaxes output signs. Keeping it set is a conservative refinement. */
   if (flags & LLVMFastMathAllowContract)
      control &= ~nir_fp_no_contract;
   if (flags & LLVMFastMathAllowReassoc)
      control &= ~nir_fp_no_reassoc;
   /* Keep NIR's broader real-number transform restriction. LLVM arcp/afn are
    * permissions, not required approximations; division/functions are refused. */
   return control;
}

static bool
put_value(struct translator *t, LLVMValueRef llvm, nir_def *def, unsigned access)
{
   if (!def)
      return fail(t, STELA_AIR_ALLOCATION, "AIR value definition unavailable");
   struct value *value = ralloc(t->values, struct value);
   if (!value)
      return fail(t, STELA_AIR_ALLOCATION, "AIR value allocation failed");
   *value = (struct value){.def = def, .access = access,
      .components = def->num_components, .bits = def->bit_size};
   if (!_mesa_hash_table_insert(t->values, llvm, value))
      return fail(t, STELA_AIR_ALLOCATION, "AIR value map allocation failed");
   return true;
}

static bool
put_pair(struct translator *t, LLVMValueRef llvm, nir_def *first, nir_def *second)
{
   struct value *value = rzalloc(t->values, struct value);
   if (!value || !_mesa_hash_table_insert(t->values, llvm, value))
      return fail(t, STELA_AIR_ALLOCATION, "AIR pair allocation failed");
   value->members[0] = first;
   value->members[1] = second;
   return true;
}

static struct value *
find_value(struct translator *t, LLVMValueRef llvm)
{
   struct hash_entry *entry = _mesa_hash_table_search(t->values, llvm);
   return entry ? entry->data : NULL;
}

static nir_def *module_global(struct translator *t, LLVMValueRef global);
static nir_variable *shared_resource(struct translator *t, uint64_t size, unsigned alignment);
static bool tile_prelude(struct translator *t);
static nir_def *tile_pixel(struct translator *t, nir_def *at);
static void workgroup_barrier(struct translator *t);
static nir_variable *dynamic_threadgroup(struct translator *t, const struct stela_air_buffer *buffer);
static nir_def *stage_narrow(nir_builder *b, nir_def *def, enum stela_air_primitive p);

/* Static pointer authority is separate from the offset/VA in def. */
static void
pointer_authority(struct value *target, const struct value *root)
{
   target->pointer_root = root->pointer_root;
   target->choices = root->choices;
   target->choice = root->choice;
   target->access = root->access;
   target->ssbo = root->ssbo;
   target->ssbo_slot = root->ssbo_slot;
   target->declared_alignment = root->declared_alignment;
   target->declared_size = root->declared_size;
   target->storage = root->storage;
   target->constant_data = root->constant_data;
   target->storage_size = root->storage_size;
   target->constant_base = root->constant_base;
   target->null = root->null;
   target->fields = root->fields;
   target->array = root->array;
   target->texture = root->texture;
   target->sampler = root->sampler;
}

/* Whether v is a texture argument, or one chosen with null textures, which
 * a PHI or select may choose with null textures. */
static bool
texture_choice_root(const struct value *v)
{
   return v && v->texture && !v->array && !v->id && v->pointer_root;
}

/* Whether a choice the texture choice `target` makes takes `incoming`: the
 * same texture, a choice of it, or a null texture. *present is where it is
 * the texture; NULL after reporting why not. */
static nir_def *texture_choice_present(struct translator *t, const struct value *target,
                                       LLVMValueRef incoming);

static uint64_t
gcd(uint64_t a, uint64_t b)
{
   while (b) {
      uint64_t rest = a % b;
      a = b;
      b = rest;
   }
   return a;
}

/* A derived pointer's offset within its argument buffer: origin's, plus
 * offset and dynamic indices of steps. */
static void
pointer_offset(struct value *target, const struct value *origin, uint64_t offset, uint64_t steps)
{
   target->offset_known = origin->offset_known;
   target->offset = origin->offset + offset;
   target->steps = gcd(origin->steps, steps);
}

/* Thread, device, constant, threadgroup, imageblock, object data or generic
 * memory; roots validate the rest. */
static bool
pointer_space(LLVMTypeRef type)
{
   return LLVMGetTypeKind(type) == LLVMPointerTypeKind &&
          (LLVMGetPointerAddressSpace(type) <= 4 || LLVMGetPointerAddressSpace(type) == 6 ||
           LLVMGetPointerAddressSpace(type) == 8);
}

/* A kernel, object or mesh function. */
static bool
compute_like(const struct translator *t)
{
   return !t->stage_options || t->stage_options->stage == STELA_AIR_OBJECT ||
          t->stage_options->stage == STELA_AIR_MESH;
}

/* A value as a vector, or NULL for an aggregate. LLVM lets a lane poison or
 * undef hold anything, so each use reads a partial value's as undefined;
 * building the vector per use keeps its lanes dominating it. */
static nir_def *
whole_value(struct translator *t, const struct value *v)
{
   if (v->def || !v->components)
      return v->def;
   nir_def *lanes[4];
   for (unsigned i = 0; i < v->components; i++)
      lanes[i] = v->lanes[i] ? v->lanes[i] : nir_undef(&t->b, 1, v->bits);
   return nir_vec(&t->b, lanes, v->components);
}

static nir_def *
get_value(struct translator *t, LLVMValueRef llvm)
{
   struct value *value = find_value(t, llvm);
   if (value) {
      nir_def *def = whole_value(t, value);
      if (!def) fail(t, STELA_AIR_INSTRUCTION, "aggregate reaches scalar-vector use");
      return def;
   }
   if (LLVMIsAGlobalVariable(llvm))
      return module_global(t, llvm);
   if (LLVMIsAConstantPointerNull(llvm))
      return nir_imm_int64(&t->b, 0);
   unsigned components, bits;
   if (stela_air_matrix_type(LLVMTypeOf(llvm), &bits)) {
      if (LLVMIsNull(llvm))
         return nir_imm_zero(&t->b, 2, bits);
      if (LLVMIsUndef(llvm))
         return nir_undef(&t->b, 2, bits);
      /* A constant fills both of a thread's elements when it fills every
       * element alike, or names them as its elements 0 and 1, the thread's
       * thread_elements(), leaving the rest undefined, as Apple's compiler
       * builds a thread's elements. */
      LLVMValueRef first = LLVMGetAggregateElement(llvm, 0);
      LLVMValueRef second = LLVMGetAggregateElement(llvm, 1);
      bool filled = first != NULL, elements = first && second;
      for (unsigned i = 1; (filled || elements) && i < 64; i++) {
         LLVMValueRef element = LLVMGetAggregateElement(llvm, i);
         filled &= element == first;
         elements &= i == 1 || (element && LLVMIsUndef(element));
      }
      if (!filled && !elements) {
         fail(t, STELA_AIR_TYPE, "SIMD-group matrix constants vary by element");
         return NULL;
      }
      nir_def *lanes[2];
      for (unsigned i = 0; i < 2; i++) {
         LLVMValueRef element = filled || !i ? first : second;
         lanes[i] = LLVMIsUndef(element) ? nir_undef(&t->b, 1, bits) : get_value(t, element);
         if (!lanes[i])
            return NULL;
      }
      return nir_vec2(&t->b, lanes[0], lanes[1]);
   }
   if (!value_shape(LLVMTypeOf(llvm), &components, &bits)) {
      fail(t, STELA_AIR_TYPE, "unsupported AIR operand type");
      return NULL;
   }
   /* Folding a function constant's select leaves undef in ordinary uses,
    * after LLVM's own optimizers have run. */
   if (LLVMIsUndef(llvm))
      return nir_undef(&t->b, components, bits);
   if (LLVMIsAConstantAggregateZero(llvm))
      return nir_imm_zero(&t->b, components, bits);
   if (components == 1 && LLVMIsAConstantInt(llvm))
      return nir_imm_intN_t(&t->b, LLVMConstIntGetZExtValue(llvm), bits);
   if (components == 1 && LLVMIsAConstantFP(llvm)) {
      /* Let LLVM serialize its APFloat bits; a host float/double conversion
       * could change a NaN's payload or signed zero before NIR sees it. */
      LLVMValueRef encoded = LLVMConstBitCast(llvm, LLVMIntTypeInContext(t->context, bits));
      if (LLVMIsAConstantInt(encoded))
         return nir_imm_intN_t(&t->b, LLVMConstIntGetZExtValue(encoded), bits);
   }
   if (components > 1 && (LLVMIsAConstantVector(llvm) ||
                         LLVMIsAConstantDataVector(llvm) ||
                         LLVMIsAConstantFP(llvm) || LLVMIsAConstantInt(llvm))) {
      nir_def *elements[4];
      for (unsigned i = 0; i < components; i++) {
         LLVMValueRef element = LLVMGetAggregateElement(llvm, i);
         elements[i] = get_value(t, element);
         if (!elements[i])
            return NULL;
      }
      return nir_vec(&t->b, elements, components);
   }
   /* Unoptimized code keeps the casts of constants. */
   if (LLVMIsAConstantExpr(llvm) && LLVMGetConstOpcode(llvm) == LLVMBitCast) {
      LLVMValueRef from = LLVMGetOperand(llvm, 0);
      bool lone = LLVMGetTypeKind(LLVMTypeOf(from)) == LLVMVectorTypeKind &&
                  LLVMGetVectorSize(LLVMTypeOf(from)) == 1;
      nir_def *source = get_value(t, lone ? LLVMGetAggregateElement(from, 0) : from);
      return source ? nir_bitcast_vector(&t->b, source, bits) : NULL;
   }
   fail(t, STELA_AIR_INSTRUCTION, "forward/constant-expression operand");
   return NULL;
}

/* A pointer operand's value, materializing a module global on first use. */
static struct value *
pointer_value(struct translator *t, LLVMValueRef pointer)
{
   if (LLVMIsAConstantPointerNull(pointer) && pointer_space(LLVMTypeOf(pointer))) {
      /* Made at each use: a constant's def dominates no later block. */
      struct value *null = rzalloc(t->values, struct value);
      if (!null) {
         fail(t, STELA_AIR_ALLOCATION, "AIR null pointer allocation failed");
         return NULL;
      }
      *null = (struct value){.def = nir_imm_int64(&t->b, 0), .access = STELA_AIR_READ_WRITE,
         .pointer_root = pointer, .null = true, .offset_known = true, .components = 1, .bits = 64};
      return null;
   }
   if (LLVMIsAGlobalVariable(pointer) && !get_value(t, pointer))
      return NULL;
   return find_value(t, pointer);
}

/* A texture or sampler argument's value, which only texture calls use, or an
 * array argument's, which only its element loads do. */
static bool
put_resource(struct translator *t, LLVMValueRef param, const struct stela_air_texture *texture,
             const struct stela_air_sampler *sampler)
{
   unsigned length = texture ? texture->length : sampler->length;
   if (texture ? texture->index >= AIR_MAX_TEXTURES : sampler->index >= AIR_MAX_SAMPLERS)
      return fail(t, STELA_AIR_ARGUMENT, "absent texture or sampler %u is read",
                  texture ? texture->argument : sampler->argument);
   bool bound = texture ? t->options->bind_texture_id != NULL : t->options->bind_sampler_id != NULL;
   if (length && (!t->options->resource_id || !bound))
      return fail(t, STELA_AIR_ARGUMENT, "unbound texture or sampler array %u",
                  texture ? texture->argument : sampler->argument);
   struct value *value = rzalloc(t->values, struct value);
   if (!value)
      return fail(t, STELA_AIR_ALLOCATION, "AIR resource value allocation failed");
   if (length) {
      *value = (struct value){.def = nir_imm_int64(&t->b, 0), .access = STELA_AIR_READ,
         .pointer_root = param, .offset_known = true, .array = true, .components = 1, .bits = 64};
   }
   if (texture) {
      struct stela_air_texture *copy = ralloc(t->values, struct stela_air_texture);
      if (!copy)
         return fail(t, STELA_AIR_ALLOCATION, "AIR texture allocation failed");
      *copy = *texture;
      value->texture = copy;
      /* The root of the choices a PHI or select makes with null textures. */
      if (!length)
         value->pointer_root = param;
   } else {
      struct stela_air_sampler *copy = ralloc(t->values, struct stela_air_sampler);
      if (!copy)
         return fail(t, STELA_AIR_ALLOCATION, "AIR sampler allocation failed");
      *copy = *sampler;
      value->sampler = copy;
   }
   if (!_mesa_hash_table_insert(t->values, param, value))
      return fail(t, STELA_AIR_ALLOCATION, "AIR value map allocation failed");
   return true;
}

static bool
binding_prelude(struct translator *t)
{
   if (exec_list_length(&t->b.impl->body) != 1 ||
       nir_block_ends_in_jump(nir_start_block(t->b.impl)))
      return fail(t, STELA_AIR_CONTROL_FLOW, "resource callback must emit a straight-line prelude");
   return true;
}

static nir_def *
device_value(struct translator *t, enum stela_air_device_value value)
{
   if (!t->options->load_device_value) {
      fail(t, STELA_AIR_ARGUMENT, "builtin needs a device value the target does not supply");
      return NULL;
   }
   nir_def *def = t->options->load_device_value(&t->b, value, t->options->binding_data);
   unsigned components = value == STELA_AIR_DISPATCH_GRID || value == STELA_AIR_PATCH_IDS ||
                         value == STELA_AIR_THREADGROUP_SHAPE || value == STELA_AIR_STAGE_IN_ORIGIN ||
                         value == STELA_AIR_STAGE_IN_SIZE ? 3 :
                         value == STELA_AIR_SAMPLE_POSITIONS || value == STELA_AIR_TILE ||
                         value == STELA_AIR_BLEND_COLOR || value == STELA_AIR_IMAGEBLOCK ? 4 : 1;
   if (!def || def->num_components != components || def->bit_size != 32 || !binding_prelude(t)) {
      fail(t, STELA_AIR_ARGUMENT, "invalid device value");
      return NULL;
   }
   return def;
}

static nir_def *
workgroup_size(struct translator *t)
{
   const unsigned *size = t->options->workgroup_size;
   return nir_imm_ivec3(&t->b, (int)size[0], (int)size[1], (int)size[2]);
}

/* The threadgroup's shape: the workgroup's, or a dynamic kernel's
 * dispatch's. */
static nir_def *
threadgroup_shape(struct translator *t)
{
   return t->options->dynamic_shape ? device_value(t, STELA_AIR_THREADGROUP_SHAPE) :
                                      workgroup_size(t);
}

/* Whether the function's threads past a dispatchThreads grid skip it. */
static bool
grid_bounded(const struct translator *t)
{
   return !t->stage_options ? t->options->load_device_value != NULL :
                              compute_like(t) && t->stage_options->launched_by_grid;
}

/* The threads of this threadgroup that a dispatchThreads grid holds. */
static nir_def *
dispatch_size(struct translator *t)
{
   nir_def *grid = device_value(t, STELA_AIR_DISPATCH_GRID);
   nir_def *size = threadgroup_shape(t);
   if (!grid || !size)
      return NULL;
   nir_def *first = nir_imul(&t->b, nir_load_workgroup_id(&t->b), size);
   nir_def *edge = nir_umin(&t->b, size, nir_isub(&t->b, grid, first));
   return nir_bcsel(&t->b, nir_ieq_imm(&t->b, nir_channel(&t->b, grid, 0), 0), size, edge);
}

/* The threadgroup's shape as it runs: Metal's threads_per_threadgroup, cut
 * short at a dispatchThreads grid's edge. */
static nir_def *
running_shape(struct translator *t)
{
   return grid_bounded(t) ? dispatch_size(t) : threadgroup_shape(t);
}

static nir_def *
volume(nir_builder *b, nir_def *size)
{
   return nir_imul(b, nir_imul(b, nir_channel(b, size, 0), nir_channel(b, size, 1)),
                   nir_channel(b, size, 2));
}

/* A tile function's threadgroup, of shape threads, padded to even sides:
 * Apple GPUs run a tile's threads as 2 by 2 quads. */
static nir_def *
tile_padded(nir_builder *b, nir_def *shape)
{
   return nir_iand_imm(b, nir_iadd_imm(b, nir_trim_vector(b, shape, 2), 1), ~1u);
}

/* A tile function's thread at this invocation, as Apple GPUs place them:
 * the threadgroup's sides padded to even, then cut into blocks 8 threads
 * wide and 4 high, shorter at its edges, which fill SIMD-groups in rows;
 * within a block, 2 by 2 quads in rows two threads high. Consecutive
 * invocations fill a NIR subgroup as Apple's lanes fill a SIMD-group, so
 * quads and SIMD-groups hold the threads Apple's do. A position past the
 * shape is padding, which skips the function. Its index in the threadgroup
 * runs the same quads across the whole padded width. */
static void
tile_thread(struct translator *t, nir_def *shape, nir_def **position, nir_def **index)
{
   nir_builder *b = &t->b;
   nir_def *padded = tile_padded(b, shape);
   nir_def *width = nir_channel(b, padded, 0), *height = nir_channel(b, padded, 1);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *band = nir_ishl_imm(b, width, 2);
   nir_def *by = nir_udiv(b, lane, band);
   nir_def *in_band = nir_isub(b, lane, nir_imul(b, by, band));
   nir_def *rows = nir_umin(b, nir_imm_int(b, 4), nir_isub(b, height, nir_ishl_imm(b, by, 2)));
   nir_def *block = nir_ishl_imm(b, rows, 3);
   nir_def *bx = nir_udiv(b, in_band, block);
   nir_def *inner = nir_isub(b, in_band, nir_imul(b, bx, block));
   nir_def *columns = nir_umin(b, nir_imm_int(b, 8), nir_isub(b, width, nir_ishl_imm(b, bx, 3)));
   nir_def *quads = nir_ushr_imm(b, columns, 1);
   nir_def *quad = nir_ushr_imm(b, inner, 2);
   nir_def *qy = nir_udiv(b, quad, quads);
   nir_def *qx = nir_isub(b, quad, nir_imul(b, qy, quads));
   nir_def *x = nir_iadd(b, nir_iadd(b, nir_ishl_imm(b, bx, 3), nir_ishl_imm(b, qx, 1)),
                         nir_iand_imm(b, inner, 1));
   nir_def *y = nir_iadd(b, nir_iadd(b, nir_ishl_imm(b, by, 2), nir_ishl_imm(b, qy, 1)),
                         nir_iand_imm(b, nir_ushr_imm(b, inner, 1), 1));
   *position = nir_vec3(b, x, y, nir_imm_int(b, 0));
   nir_def *row_quads = nir_ushr_imm(b, width, 1);
   nir_def *at = nir_iadd(b, nir_imul(b, nir_ushr_imm(b, y, 1), row_quads), nir_ushr_imm(b, x, 1));
   *index = nir_ior(b, nir_ishl_imm(b, at, 2),
                    nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, y, 1), 1), nir_iand_imm(b, x, 1)));
}

/* The thread's position in its threadgroup: its linear index's in the shape
 * the threadgroup runs. A threadgroup at a dispatchThreads grid's edge is
 * smaller than the workgroup, and Metal numbers its threads, and fills its
 * SIMD-groups, in its own shape, so the workgroup's leading invocations take
 * its positions; a full threadgroup's are the invocations' own. */
static nir_def *
local_position(struct translator *t)
{
   nir_builder *b = &t->b;
   const unsigned *size = t->options->workgroup_size;
   if (t->options->tile) {
      nir_def *shape = threadgroup_shape(t), *position, *index;
      if (!shape)
         return NULL;
      tile_thread(t, shape, &position, &index);
      return position;
   }
   if (!t->options->dynamic_shape && !grid_bounded(t))
      return nir_load_local_invocation_id(b);
   if (!t->options->dynamic_shape && size[1] == 1 && size[2] == 1)
      return nir_vec3(b, nir_load_local_invocation_index(b), nir_imm_int(b, 0), nir_imm_int(b, 0));
   nir_def *shape = running_shape(t);
   if (!shape)
      return NULL;
   /* Threads of an empty shape skip the function; avoid dividing by zero. */
   nir_def *width = nir_umax_imm(b, nir_channel(b, shape, 0), 1);
   nir_def *height = nir_umax_imm(b, nir_channel(b, shape, 1), 1);
   nir_def *index = nir_load_local_invocation_index(b);
   nir_def *row = nir_udiv(b, index, width);
   return nir_vec3(b, nir_umod(b, index, width), nir_umod(b, row, height), nir_udiv(b, row, height));
}

/* Metal's grid starts at zero, with no dispatch-base parameter. Form its ID
 * directly from NIR's workgroup/local builtins; a generic global-ID lowering
 * may add a Vulkan base input. */
static nir_def *
grid_position(struct translator *t)
{
   nir_def *shape = threadgroup_shape(t), *local = local_position(t);
   if (!shape || !local)
      return NULL;
   return nir_iadd(&t->b, nir_imul(&t->b, nir_load_workgroup_id(&t->b), shape), local);
}

/* Whether `name` is `stem.` followed by more. */
static bool
named_after(const char *name, size_t name_length, const char *stem)
{
   size_t length = strlen(stem);
   return name_length > length + 1 && !memcmp(name, stem, length) && name[length] == '.';
}

/* A stitched kernel's read of its builtin, as air_stitch.c declares it: a
 * call of `name.TYPE` that returns the builtin as an integer scalar or vector
 * of up to three lanes. */
static bool
stitched_builtin_call(struct translator *t, LLVMValueRef call, LLVMValueRef callee, const char *name,
                      size_t name_length, bool *lowered)
{
   bool grid = named_after(name, name_length, STELA_AIR_STITCH_THREAD_POSITION_IN_GRID_NAME);
   *lowered = grid ||
              named_after(name, name_length, STELA_AIR_STITCH_THREAD_POSITION_IN_THREADGROUP_NAME);
   if (!*lowered)
      return true;
   unsigned components, bits;
   if (!LLVMIsDeclaration(callee) || LLVMGetNumArgOperands(call) ||
       !value_shape(LLVMTypeOf(call), &components, &bits) || components > 3 ||
       (bits != 16 && bits != 32))
      return fail(t, STELA_AIR_TYPE, "a stitched thread position of an unsupported type");
   if (!compute_like(t))
      return fail(t, STELA_AIR_INPUT, "a stitched thread position outside a kernel, object or mesh function");
   nir_def *position = grid ? grid_position(t) : local_position(t);
   if (!position)
      return false;
   position = nir_u2uN(&t->b, nir_channels(&t->b, position, nir_component_mask(components)), bits);
   return put_value(t, call, position, 0);
}

/* Threadgroup buffer desc: in a dynamic kernel, where its dispatches place
 * it in the kernel's threadgroup memory, else its own memory of the size its
 * binding gives, as the offset def into *shared of *shared_size bytes. */
static bool
threadgroup_memory(struct translator *t, const struct stela_air_buffer *desc, nir_def **def,
                   nir_variable **shared, unsigned *shared_size)
{
   if (t->options->dynamic_threadgroup_bytes) {
      *shared = dynamic_threadgroup(t, desc);
      *def = *shared ? t->options->threadgroup_offset(&t->b, desc, t->options->binding_data) : NULL;
      if (!binding_prelude(t))
         return false;
      if (!*def || (*def)->num_components != 1 || (*def)->bit_size != 32)
         return fail(t, STELA_AIR_ARGUMENT, "threadgroup offset callback refused buffer %u",
                     desc->argument);
      *def = nir_u2u64(&t->b, *def);
      *shared_size = t->options->dynamic_threadgroup_bytes;
      return true;
   }
   struct stela_air_threadgroup_view view = {0};
   if (!t->options->bind_threadgroup ||
       !t->options->bind_threadgroup(desc, &view, t->options->binding_data) ||
       view.size < desc->type_size || view.alignment < desc->type_alignment ||
       view.available_access < STELA_AIR_READ || view.available_access > STELA_AIR_READ_WRITE ||
       (desc->access & ~view.available_access))
      return fail(t, STELA_AIR_ARGUMENT, "invalid/unbound threadgroup buffer %u", desc->argument);
   *shared = shared_resource(t, view.size, view.alignment);
   if (!*shared)
      return false;
   *shared_size = (unsigned)view.size;
   *def = nir_imm_int64(&t->b, 0);
   return true;
}

/* Whether `name` is one of air_stitch.c's reads of a bound resource. */
static bool
stitched_resource_name(const char *name, size_t name_length)
{
   return named_after(name, name_length, STELA_AIR_STITCH_BUFFER_ADDRESS_NAME) ||
          (name_length == strlen(STELA_AIR_STITCH_TEXTURE_NAME) &&
           !memcmp(name, STELA_AIR_STITCH_TEXTURE_NAME, name_length)) ||
          (name_length == strlen(STELA_AIR_STITCH_SAMPLER_NAME) &&
           !memcmp(name, STELA_AIR_STITCH_SAMPLER_NAME, name_length)) ||
          (name_length == strlen(STELA_AIR_STITCH_THREADGROUP_NAME) &&
           !memcmp(name, STELA_AIR_STITCH_THREADGROUP_NAME, name_length));
}

/* Whether inst calls one of air_stitch.c's reads of a bound resource. */
static bool
stitched_resource_call(LLVMValueRef inst)
{
   LLVMValueRef callee = LLVMIsACallInst(inst) ? LLVMGetCalledValue(inst) : NULL;
   size_t name_length = 0;
   const char *name = callee && LLVMIsAFunction(callee) ? LLVMGetValueName2(callee, &name_length) : "";
   return stitched_resource_name(name, name_length);
}

/* Whether inst is air.imageblock_data, whose pointer is a root. */
static bool
imageblock_data_call(LLVMValueRef inst)
{
   LLVMValueRef callee = LLVMIsACallInst(inst) ? LLVMGetCalledValue(inst) : NULL;
   size_t name_length = 0;
   const char *name = callee && LLVMIsAFunction(callee) ? LLVMGetValueName2(callee, &name_length) : "";
   return stela_air_call_named(name, name_length, "air.imageblock_data");
}

/* What extractvalue `extract` reads, when an insertvalue chain names it:
 * extracts of extracts compose their index paths, an insert at the path
 * gives its value, one above it is entered and one beside it skipped, as
 * LLVM's instcombine folds them. NULL when an insert lies within the path's
 * aggregate or the chain starts elsewhere. */
static LLVMValueRef
extracted_value(LLVMValueRef extract)
{
   unsigned path[16], depth = LLVMGetNumIndices(extract);
   if (depth > ARRAY_SIZE(path))
      return NULL;
   memcpy(path, LLVMGetIndices(extract), depth * sizeof(*path));
   LLVMValueRef aggregate = LLVMGetOperand(extract, 0);
   for (unsigned steps = 0; steps < 4096; steps++) {
      if (LLVMIsAExtractValueInst(aggregate)) {
         unsigned outer = LLVMGetNumIndices(aggregate);
         if (outer + depth > ARRAY_SIZE(path))
            return NULL;
         memmove(path + outer, path, depth * sizeof(*path));
         memcpy(path, LLVMGetIndices(aggregate), outer * sizeof(*path));
         depth += outer;
         aggregate = LLVMGetOperand(aggregate, 0);
         continue;
      }
      if (!LLVMIsAInsertValueInst(aggregate))
         return NULL;
      unsigned inserted = LLVMGetNumIndices(aggregate);
      const unsigned *at = LLVMGetIndices(aggregate);
      unsigned common = MIN2(inserted, depth);
      if (memcmp(at, path, common * sizeof(*path))) {
         aggregate = LLVMGetOperand(aggregate, 0);
         continue;
      }
      if (inserted > depth)
         return NULL;
      memmove(path, path + inserted, (depth - inserted) * sizeof(*path));
      depth -= inserted;
      aggregate = LLVMGetOperand(aggregate, 1);
      if (!depth)
         return aggregate;
   }
   return NULL;
}

/* Forwards fn's extractvalues to the values insertvalue chains put there,
 * so a texture or sampler packed into Metal's one-pointer wrapper structs,
 * or into a larger struct such as Core Image's coreimage::Sampler, reaches
 * its texture calls as the read itself (Core Image's stitched filters pass
 * _ci_sampler_builder's result to their kernels by value). */
static void
forward_extracts(LLVMValueRef fn)
{
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
        block = LLVMGetNextBasicBlock(block)) {
      LLVMValueRef inst = LLVMGetFirstInstruction(block);
      while (inst) {
         LLVMValueRef next = LLVMGetNextInstruction(inst);
         LLVMValueRef value = LLVMIsAExtractValueInst(inst) ? extracted_value(inst) : NULL;
         if (value && LLVMTypeOf(value) == LLVMTypeOf(inst)) {
            LLVMReplaceAllUsesWith(inst, value);
            LLVMInstructionEraseFromParent(inst);
         }
         inst = next;
      }
   }
   /* The chains those extracts read may now feed nothing. */
   for (bool erased = true; erased;) {
      erased = false;
      for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
           block = LLVMGetNextBasicBlock(block)) {
         LLVMValueRef inst = LLVMGetFirstInstruction(block);
         while (inst) {
            LLVMValueRef next = LLVMGetNextInstruction(inst);
            if ((LLVMIsAInsertValueInst(inst) || LLVMIsAExtractValueInst(inst)) &&
                !LLVMGetFirstUse(inst)) {
               LLVMInstructionEraseFromParent(inst);
               erased = true;
            }
            inst = next;
         }
      }
   }
}

/* The texture or sampler a stitched function's call reads from the binding
 * at `index`. A texture's type, texel type and access are what the texture
 * calls that take it need, since a stitched parameter's metadata names only
 * its kind; one no call takes is never read. */
static bool
stitched_texture(struct translator *t, LLVMValueRef call, unsigned index, bool texture)
{
   if (!texture)
      return put_resource(t, call, NULL, &(struct stela_air_sampler){.argument = index, .index = index});
   struct stela_air_texture described = {.argument = index, .index = index};
   for (LLVMUseRef use = LLVMGetFirstUse(call); use; use = LLVMGetNextUse(use)) {
      LLVMValueRef user = LLVMGetUser(use);
      if (!LLVMIsACallInst(user) || LLVMGetOperand(user, 0) != call ||
          !stela_air_texture_use(user, &described))
         return fail(t, STELA_AIR_INPUT, "stitched texture %u is used other than as texture calls agree", index);
   }
   if (!described.type)
      return true;
   if (described.primitive == STELA_AIR_PRIMITIVE_UNKNOWN)
      described.primitive = STELA_AIR_FLOAT;
   return put_resource(t, call, &described, NULL);
}

/* Whether `name` is one of air_stitch.c's reads of a builtin. */
static bool
stitched_builtin_name(const char *name, size_t name_length)
{
   return named_after(name, name_length, STELA_AIR_STITCH_THREAD_POSITION_IN_GRID_NAME) ||
          named_after(name, name_length, STELA_AIR_STITCH_THREAD_POSITION_IN_THREADGROUP_NAME);
}

/* The builtins and resources stitched kernel fn reads, in the prelude as a
 * kernel's arguments are: a thread position may need the dispatch's device
 * values, and the kernel's early returns come before calls that read it. */
static bool
stitched_resources(struct translator *t, LLVMValueRef fn)
{
   forward_extracts(fn);
   for (LLVMValueRef read = LLVMGetFirstFunction(LLVMGetGlobalParent(fn)); read;
        read = LLVMGetNextFunction(read)) {
      size_t name_length;
      const char *name = LLVMGetValueName2(read, &name_length);
      if (stitched_builtin_name(name, name_length)) {
         for (LLVMUseRef use = LLVMGetFirstUse(read); use; use = LLVMGetNextUse(use)) {
            LLVMValueRef call = LLVMGetUser(use);
            bool lowered;
            if (!LLVMIsACallInst(call) || LLVMGetCalledValue(call) != read)
               return fail(t, STELA_AIR_INPUT, "a stitched builtin that is not called");
            if (LLVMGetBasicBlockParent(LLVMGetInstructionParent(call)) == fn &&
                !stitched_builtin_call(t, call, read, name, name_length, &lowered))
               return false;
         }
         continue;
      }
      if (!stitched_resource_name(name, name_length))
         continue;
      bool buffer = named_after(name, name_length, STELA_AIR_STITCH_BUFFER_ADDRESS_NAME);
      for (LLVMUseRef use = LLVMGetFirstUse(read); use; use = LLVMGetNextUse(use)) {
         LLVMValueRef call = LLVMGetUser(use);
         if (!LLVMIsACallInst(call) || LLVMGetCalledValue(call) != read)
            return fail(t, STELA_AIR_INPUT, "a stitched buffer address that is not called");
         if (LLVMGetBasicBlockParent(LLVMGetInstructionParent(call)) != fn)
            continue;
         LLVMValueRef index = LLVMGetNumArgOperands(call) == 1 ? LLVMGetOperand(call, 0) : NULL;
         unsigned space = LLVMGetPointerAddressSpace(LLVMTypeOf(call));
         if (!index || !LLVMIsAConstantInt(index) || space < 1 || space > 3 || (buffer && space == 3))
            return fail(t, STELA_AIR_INPUT, "a stitched resource of an unknown form");
         if (space == 3) {
            /* Threadgroup memory a dispatch sizes, read and written as a
             * kernel's threadgroup buffer is. */
            unsigned bound = (unsigned)LLVMConstIntGetZExtValue(index);
            struct stela_air_buffer memory = {.argument = bound, .index = bound, .address_space = 3,
                                              .access = STELA_AIR_READ_WRITE, .type_alignment = 1};
            nir_def *def;
            nir_variable *shared;
            unsigned shared_size;
            if (!threadgroup_memory(t, &memory, &def, &shared, &shared_size) ||
                !put_value(t, call, def, memory.access))
               return false;
            struct value *value = find_value(t, call);
            value->pointer_root = call;
            value->storage = shared;
            value->storage_size = shared_size;
            continue;
         }
         if (!buffer) {
            if (!stitched_texture(t, call, (unsigned)LLVMConstIntGetZExtValue(index), space == 1))
               return false;
            continue;
         }
         if (t->options->bind_ssbo || !t->options->bind_buffer)
            return fail(t, STELA_AIR_ARGUMENT, "a stitched buffer address requires device addresses");
         struct stela_air_buffer buffer = {
            .index = (unsigned)LLVMConstIntGetZExtValue(index), .address_space = space,
            .access = space == 2 ? STELA_AIR_READ : STELA_AIR_READ_WRITE};
         nir_def *def = t->options->bind_buffer(&t->b, &buffer, t->options->binding_data);
         if (!binding_prelude(t))
            return false;
         if (!def || def->num_components != 1 || def->bit_size != 64)
            return fail(t, STELA_AIR_ARGUMENT, "binding callback refused stitched buffer %u",
                        buffer.index);
         if (!put_value(t, call, def, buffer.access))
            return false;
         find_value(t, call)->pointer_root = call;
      }
   }
   return true;
}


/* A fragment input the rasterizer supplies as a flat uint varying. */
static nir_def *
flat_input(struct translator *t, gl_varying_slot slot, const char *name)
{
   nir_variable *var = nir_variable_create(t->b.shader, nir_var_shader_in, glsl_uint_type(), name);
   if (!var) {
      fail(t, STELA_AIR_ALLOCATION, "builtin variable allocation failed");
      return NULL;
   }
   var->data.location = slot;
   var->data.interpolation = INTERP_MODE_FLAT;
   return nir_load_var(&t->b, var);
}

/* A builtin input's 32-bit value; positions and sizes are uvec3. */
static nir_def *
builtin_value(struct translator *t, enum stela_air_builtin builtin)
{
   nir_builder *b = &t->b;
   const unsigned *size = t->options->workgroup_size;
   nir_def *value;
   switch (builtin) {
   case STELA_AIR_THREAD_POSITION_IN_GRID:
      return grid_position(t);
   case STELA_AIR_THREAD_POSITION_IN_THREADGROUP:
      return local_position(t);
   case STELA_AIR_THREADGROUP_POSITION_IN_GRID:
      return nir_load_workgroup_id(b);
   case STELA_AIR_THREADS_PER_THREADGROUP:
      return running_shape(t);
   case STELA_AIR_DISPATCH_THREADS_PER_THREADGROUP:
      return threadgroup_shape(t);
   case STELA_AIR_THREADGROUPS_PER_GRID:
      return nir_load_num_workgroups(b);
   case STELA_AIR_THREADS_PER_GRID: {
      nir_def *grid = device_value(t, STELA_AIR_DISPATCH_GRID);
      nir_def *shape = threadgroup_shape(t);
      if (!grid || !shape)
         return NULL;
      nir_def *uniform = nir_imul(b, nir_load_num_workgroups(b), shape);
      return nir_bcsel(b, nir_ieq_imm(b, nir_channel(b, grid, 0), 0), uniform, grid);
   }
   case STELA_AIR_STAGE_IN_GRID_ORIGIN:
   case STELA_AIR_STAGE_IN_GRID_SIZE:
      /* Only a dispatch of a pipeline with stage-in attributes has a region. */
      if (!t->options->fetch_attribute) {
         fail(t, STELA_AIR_ARGUMENT, "a stage-in region needs the pipeline's stage-in attributes");
         return NULL;
      }
      return device_value(t, builtin == STELA_AIR_STAGE_IN_GRID_ORIGIN ? STELA_AIR_STAGE_IN_ORIGIN :
                                                                          STELA_AIR_STAGE_IN_SIZE);
   case STELA_AIR_THREAD_INDEX_IN_THREADGROUP:
   case STELA_AIR_QUADGROUP_INDEX_IN_THREADGROUP:
      if (t->options->tile) {
         nir_def *shape = threadgroup_shape(t), *position, *index;
         if (!shape)
            return NULL;
         tile_thread(t, shape, &position, &index);
         return builtin == STELA_AIR_THREAD_INDEX_IN_THREADGROUP ? index :
                                                                  nir_ushr_imm(b, index, 2);
      }
      if (builtin == STELA_AIR_QUADGROUP_INDEX_IN_THREADGROUP)
         return nir_ushr_imm(b, nir_load_local_invocation_index(b), 2);
      return nir_load_local_invocation_index(b);
   case STELA_AIR_SIMDGROUP_INDEX_IN_THREADGROUP:
      return nir_load_subgroup_id(b);
   case STELA_AIR_SIMDGROUPS_PER_THREADGROUP:
   case STELA_AIR_DISPATCH_SIMDGROUPS_PER_THREADGROUP: {
      bool running = builtin == STELA_AIR_SIMDGROUPS_PER_THREADGROUP;
      /* A tile's workgroup is its padded threads, as Apple's SIMD-groups. */
      if (t->options->tile || (!t->options->dynamic_shape && !(running && grid_bounded(t))))
         return nir_load_num_subgroups(b);
      value = running ? running_shape(t) : threadgroup_shape(t);
      if (!value)
         return NULL;
      return nir_udiv(b, nir_iadd(b, volume(b, value), nir_iadd_imm(b, nir_load_subgroup_size(b), -1)),
                      nir_load_subgroup_size(b));
   }
   case STELA_AIR_QUADGROUPS_PER_THREADGROUP:
   case STELA_AIR_DISPATCH_QUADGROUPS_PER_THREADGROUP: {
      bool running = builtin == STELA_AIR_QUADGROUPS_PER_THREADGROUP;
      if (t->options->tile) {
         nir_def *shape = threadgroup_shape(t);
         if (!shape)
            return NULL;
         nir_def *padded = tile_padded(b, shape);
         return nir_ushr_imm(b, nir_imul(b, nir_channel(b, padded, 0), nir_channel(b, padded, 1)), 2);
      }
      if (!t->options->dynamic_shape && !(running && grid_bounded(t)))
         return nir_imm_int(b, (int)DIV_ROUND_UP(size[0] * size[1] * size[2], 4));
      value = running ? running_shape(t) : threadgroup_shape(t);
      if (!value)
         return NULL;
      return nir_ushr_imm(b, nir_iadd_imm(b, volume(b, value), 3), 2);
   }
   case STELA_AIR_THREAD_INDEX_IN_SIMDGROUP:
      return nir_load_subgroup_invocation(b);
   case STELA_AIR_THREADS_PER_SIMDGROUP:
   case STELA_AIR_THREAD_EXECUTION_WIDTH:
      return nir_load_subgroup_size(b);
   case STELA_AIR_THREAD_INDEX_IN_QUADGROUP:
      return nir_iand_imm(b, nir_load_subgroup_invocation(b), 3);
   case STELA_AIR_VERTEX_ID:
      /* Metal's vertex_id includes the base vertex, as NIR's vertex_id
       * does, unlike vertex_id_zero_base. */
      return nir_load_vertex_id(b);
   case STELA_AIR_INSTANCE_ID:
      /* Metal's instance_id includes the base instance; NIR's does not. A
       * post-tessellation function's comes with its patch. */
      if (t->patch_ids)
         return nir_channel(b, t->patch_ids, 1);
      return nir_iadd(b, nir_load_instance_id(b), nir_load_base_instance(b));
   case STELA_AIR_BASE_VERTEX:
      return nir_load_base_vertex(b);
   case STELA_AIR_BASE_INSTANCE:
      return t->patch_ids ? nir_channel(b, t->patch_ids, 2) : nir_load_base_instance(b);
   case STELA_AIR_PATCH_ID:
      return nir_channel(b, t->patch_ids, 0);
   case STELA_AIR_AMPLIFICATION_ID:
      return device_value(t, STELA_AIR_AMPLIFICATION);
   case STELA_AIR_AMPLIFICATION_COUNT:
      return device_value(t, STELA_AIR_AMPLIFICATIONS);
   case STELA_AIR_FRONT_FACING:
      return nir_load_front_face(b, 1);
   case STELA_AIR_SAMPLE_ID:
      return nir_load_sample_id(b);
   case STELA_AIR_SAMPLE_MASK_IN:
      return nir_load_sample_mask_in(b);
   case STELA_AIR_PRIMITIVE_ID:
      return flat_input(t, VARYING_SLOT_PRIMITIVE_ID, "air.primitive_id");
   case STELA_AIR_RENDER_TARGET_ARRAY_INDEX_IN:
      return flat_input(t, VARYING_SLOT_LAYER, "air.render_target_array_index");
   case STELA_AIR_VIEWPORT_ARRAY_INDEX_IN:
      return flat_input(t, VARYING_SLOT_VIEWPORT, "air.viewport_array_index");
   case STELA_AIR_POINT_COORD:
      /* Both Metal's and Vulkan's have an upper-left origin. */
      return nir_load_point_coord(b);
   case STELA_AIR_BUILTIN_NONE:
   case STELA_AIR_BUILTIN_COUNT:
      break;
   }
   UNREACHABLE("builtins are validated with their stage");
}

/* A builtin input in its argument's type: the leading lanes, narrowed to
 * 16-bit lanes for ushort forms. */
static nir_def *
builtin_argument(struct translator *t, enum stela_air_builtin builtin, LLVMValueRef param)
{
   nir_def *value = builtin_value(t, builtin);
   if (!value)
      return NULL;
   LLVMTypeRef type = LLVMTypeOf(param);
   unsigned components = 1;
   if (LLVMGetTypeKind(type) == LLVMVectorTypeKind) {
      components = LLVMGetVectorSize(type);
      type = LLVMGetElementType(type);
   }
   value = nir_trim_vector(&t->b, value, components);
   return LLVMGetTypeKind(type) == LLVMIntegerTypeKind && LLVMGetIntTypeWidth(type) == 16 ?
      nir_u2u16(&t->b, value) : value;
}

/* A stage-in attribute at location in its type, as fetch_attribute reads
 * it at `at`. */
static nir_def *
stage_in_attribute(struct translator *t, unsigned location, nir_def *at, nir_def *point,
                   enum stela_air_primitive p, unsigned components)
{
   bool integer = p != STELA_AIR_FLOAT && p != STELA_AIR_HALF;
   const struct stela_air_options *options = t->options;
   nir_def *lanes = options->fetch_attribute ?
      options->fetch_attribute(&t->b, location, at, point, integer, options->binding_data) : NULL;
   if (!lanes || lanes->num_components != 4 || lanes->bit_size != 32) {
      fail(t, STELA_AIR_ARGUMENT, "stage-in attribute %u is not provided", location);
      return NULL;
   }
   return stage_narrow(&t->b, nir_trim_vector(&t->b, lanes, components), p);
}

/* A buffer argument's value as the root of its pointers. Argument buffers
 * hold device addresses, which SSBO bindings have no counterpart of. */
static bool
argument_root(struct translator *t, struct value *value, LLVMValueRef param,
              const struct stela_air_argument *argument)
{
   value->pointer_root = param;
   value->declared_alignment = argument->buffer.type_alignment;
   if (!argument->is_argument_buffer) {
      value->declared_size = argument->buffer.type_size;
      return true;
   }
   if (t->options->bind_ssbo)
      return fail(t, STELA_AIR_ARGUMENT, "argument buffer %u requires device addresses",
                  argument->buffer.argument);
   value->fields = argument->fields;
   value->offset_known = true;
   return true;
}

/* The object payload: task payload memory, which both stages address from
 * zero. */
static bool
payload_argument(struct translator *t, LLVMValueRef param, const struct stela_air_buffer *desc)
{
   if (desc->type_size > t->stage_options->max_payload_bytes)
      return fail(t, STELA_AIR_LIMIT, "object payload exceeds the target limit");
   nir_variable *var = nir_variable_create(t->b.shader, nir_var_mem_task_payload,
      glsl_array_type(glsl_uint8_t_type(), desc->type_size, 1), "air.payload");
   if (!var || !put_value(t, param, nir_imm_int64(&t->b, 0), desc->access))
      return fail(t, STELA_AIR_ALLOCATION, "object payload allocation failed");
   var->data.alignment = desc->type_alignment;
   struct value *value = find_value(t, param);
   value->pointer_root = param;
   value->storage = var;
   value->storage_size = desc->type_size;
   t->payload = var;
   return true;
}

/* A kernel's explicit imageblock of element-byte elements, which Metal
 * keeps in tile memory: threadgroup memory here, of the dimensions the
 * dispatch's encoder sets. Without them it is 0 x 0, as on Apple GPUs, and
 * its memory holds an element for each thread of a threadgroup's first row,
 * where a 0-wide imageblock's elements lie. */
static bool
imageblock_memory(struct translator *t, unsigned element)
{
   const unsigned *given = t->options->imageblock_size;
   unsigned width = given[0], height = given[0] ? given[1] : 0;
   unsigned elements = given[0] ? width * height : t->options->workgroup_size[0];
   if (width > UINT16_MAX || height > UINT16_MAX || (given[0] && !height) || !elements)
      return fail(t, STELA_AIR_ARGUMENT, "an imageblock of %ux%u in a kernel of %u threads", width,
                  height, t->options->workgroup_size[0]);
   t->imageblock_extent = nir_imm_ivec2(&t->b, (int)width, (int)height);
   t->imageblock = shared_resource(t, (uint64_t)elements * element, 16);
   t->imageblock_elements = elements;
   t->imageblock_element = element;
   return t->imageblock != NULL;
}

/* Whether an argument is absent, its [[function_constant]] false: it binds
 * nothing. A buffer's accesses that folding leaves go through null. */
static bool
absent_argument(struct translator *t, const struct stela_air_argument *argument, bool *absent)
{
   if (argument->absent && argument->is_buffer)
      LLVMReplaceAllUsesWith(argument->value, LLVMConstPointerNull(LLVMTypeOf(argument->value)));
   *absent = argument->absent && !LLVMGetFirstUse(argument->value);
   return true;
}

/* The arguments of a kernel, object or mesh function. */
static struct value *storage_root(struct translator *t, LLVMValueRef llvm, unsigned access);

/* A buffer array, metal::array_ref<device T *>: thread memory holding the
 * bound address of each of its buffers, whose loads read the one field
 * those addresses are. */
static bool
buffer_array(struct translator *t, const struct stela_air_argument *argument)
{
   const struct stela_air_buffer *desc = &argument->buffer;
   unsigned length = desc->length;
   if (length && (t->options->bind_ssbo || !t->options->bind_buffer))
      return fail(t, STELA_AIR_ARGUMENT, "buffer array %u requires device addresses",
                  desc->argument);
   uint64_t size = 8ull * length;
   if (t->private_layout_bound + size + 7 > t->options->max_private_bytes)
      return fail(t, STELA_AIR_LIMIT, "thread memory size/alignment exceeds the target limit");
   nir_variable *var = length ? nir_local_variable_create(
      t->b.impl, glsl_array_type(glsl_uint8_t_type(), (unsigned)size, 1), "air.buffer_array") : NULL;
   if (length && !var)
      return fail(t, STELA_AIR_ALLOCATION, "thread memory variable allocation failed");
   if (var) {
      var->data.alignment = 8;
      t->private_allocations++;
      t->private_layout_bound += size + 7;
   }
   for (unsigned i = 0; i < length; i++) {
      struct stela_air_buffer element = *desc;
      element.index = desc->index + i;
      element.array = false;
      element.length = 0;
      nir_def *address = t->options->bind_buffer(&t->b, &element, t->options->binding_data);
      if (!binding_prelude(t))
         return false;
      if (!address || address->num_components != 1 || address->bit_size != 64)
         return fail(t, STELA_AIR_ARGUMENT, "binding callback refused/invalid buffer %u",
                     element.index);
      nir_deref_instr *byte = nir_build_deref_array_imm(&t->b, nir_build_deref_var(&t->b, var), 8 * i);
      nir_deref_instr *slot = nir_build_deref_cast_with_alignment(
         &t->b, &byte->def, nir_var_function_temp, glsl_uint64_t_type(), 8, 8, 0);
      nir_store_deref(&t->b, slot, address, 1);
   }
   struct value *value = storage_root(t, argument->value, STELA_AIR_READ);
   if (!value)
      return false;
   value->storage = var;
   value->storage_size = (unsigned)size;
   value->fields = argument->fields;
   value->offset_known = true;
   /* No array is empty memory: its loads are refused as unread fields. */
   value->null = !length;
   return true;
}

static bool
compute_arguments(struct translator *t, const struct stela_air_kernel *kernel)
{
   t->fields = kernel->fields;
   if (t->options->tile && !tile_prelude(t))
      return false;
   bool ssbo_slots[UINT8_MAX + 1] = {false};
   for (unsigned a = 0; a < kernel->count; a++) {
      const struct stela_air_argument *argument = &kernel->arguments[a];
      LLVMValueRef param = argument->value;
      bool absent;
      if (!absent_argument(t, argument, &absent))
         return false;
      if (absent)
         continue;
      if (argument->is_imageblock) {
         if (LLVMGetFirstUse(param))
            return fail(t, STELA_AIR_ARGUMENT, "an imageblock read other than through its functions");
         if (argument->implicit_imageblock && !t->tile)
            return fail(t, STELA_AIR_ARGUMENT, "an implicit imageblock outside a tile function");
         if (argument->implicit_imageblock) {
            t->tile_implicit = true;
            continue;
         }
         /* A tile function's is its pass's imageblock data, which the
          * pass's fragment functions share. */
         if (t->tile) {
            t->tile_explicit = true;
            if (!(t->imageblock_pass = device_value(t, STELA_AIR_IMAGEBLOCK)))
               return false;
            continue;
         }
         if (!imageblock_memory(t, kernel->imageblock_size))
            return false;
         continue;
      }
      if (argument->is_mesh || argument->is_mesh_grid) {
         /* Only the mesh setters take these. */
         *(argument->is_mesh ? &t->mesh : &t->mesh_grid) = param;
         continue;
      }
      if (argument->is_payload) {
         if (!payload_argument(t, param, &argument->buffer))
            return false;
         continue;
      }
      if (argument->is_texture || argument->is_sampler) {
         if (!put_resource(t, param, argument->is_texture ? &argument->texture : NULL,
                           &argument->sampler))
            return false;
         continue;
      }
      if (argument->is_stage_in) {
         /* Attributes the kernel never reads need no layout. */
         if (!LLVMGetFirstUse(param))
            continue;
         nir_def *def = stage_in_attribute(t, argument->location, grid_position(t), NULL,
                                           argument->primitive, argument->type_components);
         if (!def || !put_value(t, param, def, 0))
            return false;
         continue;
      }
      if (argument->is_buffer && argument->buffer.array) {
         if (!buffer_array(t, argument))
            return false;
         continue;
      }
      struct stela_air_buffer desc = argument->buffer;
      unsigned index = desc.argument;
      bool buffer = argument->is_buffer;
      nir_def *def;
      unsigned ssbo_slot = 0;
      nir_variable *shared = NULL;
      unsigned shared_size = 0;
      if (buffer) {
         if (desc.address_space == 3) {
            if (!threadgroup_memory(t, &desc, &def, &shared, &shared_size))
               return false;
         } else if (t->options->bind_ssbo) {
            if (!t->options->bind_ssbo(&desc, &ssbo_slot, t->options->binding_data) ||
                ssbo_slot >= t->options->ssbo_count || ssbo_slots[ssbo_slot])
               return fail(t, STELA_AIR_ARGUMENT, "SSBO callback refused/invalid slot for buffer %u", index);
            ssbo_slots[ssbo_slot] = true;
            /* The exact Metal view is the surface, not an inferred raw VA. */
            def = nir_imm_int64(&t->b, 0);
         } else {
            if (!t->options->bind_buffer)
               return fail(t, STELA_AIR_ARGUMENT, "unbound device buffer %u", index);
            def = t->options->bind_buffer(&t->b, &desc, t->options->binding_data);
            if (!binding_prelude(t))
               return false;
            if (!def || def->num_components != 1 || def->bit_size != 64)
               return fail(t, STELA_AIR_ARGUMENT, "binding callback refused/invalid buffer %u", index);
         }
      } else {
         def = builtin_argument(t, argument->builtin, param);
         if (!def)
            return false;
      }
      if (!put_value(t, param, def, desc.access))
         return false;
      if (buffer && !argument_root(t, find_value(t, param), param, argument))
         return false;
      if (shared) {
         struct value *value = find_value(t, param);
         value->storage = shared;
         value->storage_size = shared_size;
      } else if (buffer && t->options->bind_ssbo) {
         struct value *value = find_value(t, param);
         value->ssbo = true;
         value->ssbo_slot = ssbo_slot;
      }
   }
   return true;
}

/* Whether the workgroup is within a function's air.max_work_group_size. */
static bool
workgroup_within(struct translator *t, unsigned max_threads)
{
   const unsigned *size = t->options->workgroup_size;
   return !max_threads || (uint64_t)size[0] * size[1] * size[2] <= max_threads ||
      fail(t, STELA_AIR_LIMIT, "workgroup exceeds the function's maximum threads");
}

static bool
kernel_arguments(struct translator *t, LLVMValueRef fn)
{
   /* The fields outlive this function in the translation's values. */
   struct stela_air_kernel *kernel = ralloc(t->values, struct stela_air_kernel);
   if (!kernel)
      return fail(t, STELA_AIR_ALLOCATION, "AIR kernel reflection allocation failed");
   return stela_air_metadata_kernel(&t->metadata, fn, kernel) &&
          workgroup_within(t, kernel->max_threads) && compute_arguments(t, kernel);
}

static bool
fixed_storage_type(struct translator *t, LLVMTypeRef type, unsigned depth,
                   unsigned *budget)
{
   if (depth >= 64 || !*budget)
      return fail(t, STELA_AIR_LIMIT, "GEP type-layout budget exceeded");
   (*budget)--;
   switch (LLVMGetTypeKind(type)) {
   case LLVMArrayTypeKind:
   case LLVMVectorTypeKind:
      return fixed_storage_type(t, LLVMGetElementType(type), depth + 1, budget);
   case LLVMStructTypeKind:
      if (LLVMIsOpaqueStruct(type))
         return fail(t, STELA_AIR_TYPE, "opaque GEP storage type");
      for (unsigned i = 0; i < LLVMCountStructElementTypes(type); i++) {
         if (!fixed_storage_type(t, LLVMStructGetTypeAtIndex(type, i), depth + 1, budget))
            return false;
      }
      return true;
   case LLVMIntegerTypeKind:
   case LLVMPointerTypeKind:
   case LLVMHalfTypeKind:
   case LLVMFloatTypeKind:
   case LLVMDoubleTypeKind:
   case LLVMX86_FP80TypeKind:
   case LLVMFP128TypeKind:
   case LLVMPPC_FP128TypeKind:
   case LLVMBFloatTypeKind:
      return LLVMTypeIsSized(type) || fail(t, STELA_AIR_TYPE, "unsized GEP storage type");
   default:
      /* In particular, LLVM's fixed-size C DataLayout queries must not be
       * called on scalable vectors or aggregates containing them. */
      return fail(t, STELA_AIR_TYPE, "unsupported/scalable GEP storage type");
   }
}

static bool
shared_profile(struct translator *t)
{
   return stela_air_metadata_shared_profile(&t->metadata);
}

static nir_variable *
shared_resource(struct translator *t, uint64_t size, unsigned alignment)
{
   if (!size || size > t->options->max_threadgroup_bytes ||
       !alignment || (alignment & (alignment - 1)) ||
       t->shared_resources == AIR_MAX_ARGUMENTS ||
       t->shared_layout_bound + size + alignment - 1 > UINT32_MAX) {
      fail(t, STELA_AIR_LIMIT, "shared-memory size/alignment/layout budget exceeded");
      return NULL;
   }
   if (!shared_profile(t))
      return NULL;
   /* Original NIR owns byte-storage placement, alignment and final lowering.
    * The conservative bound above only prevents its unsigned layout overflow. */
   nir_variable *var = nir_variable_create(t->b.shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint8_t_type(), (unsigned)size, 1), "air.shared");
   if (!var) {
      fail(t, STELA_AIR_ALLOCATION, "shared-memory variable allocation failed");
      return NULL;
   }
   var->data.alignment = alignment;
   t->shared_resources++;
   t->shared_layout_bound += size + alignment - 1;
   return var;
}

/* A dynamic kernel's threadgroup memory, in which its dispatches place
 * threadgroup buffers at offsets aligned to 16 bytes. */
static nir_variable *
dynamic_threadgroup(struct translator *t, const struct stela_air_buffer *buffer)
{
   if (buffer->type_alignment > 16) {
      fail(t, STELA_AIR_ARGUMENT, "threadgroup buffer %u is aligned past its placement",
           buffer->argument);
      return NULL;
   }
   if (!t->dynamic_threadgroup)
      t->dynamic_threadgroup = shared_resource(t, t->options->dynamic_threadgroup_bytes, 16);
   return t->dynamic_threadgroup;
}

/* A storage root at offset zero, defined at entry: it dominates every use, even
 * when first referenced in conditional or loop code. Memory effects keep their
 * source cursor. */
static struct value *
storage_root(struct translator *t, LLVMValueRef llvm, unsigned access)
{
   nir_builder entry = nir_builder_at(nir_before_impl(t->b.impl));
   if (!put_value(t, llvm, nir_imm_int64(&entry, 0), access))
      return NULL;
   struct value *value = find_value(t, llvm);
   value->pointer_root = llvm;
   return value;
}

/* The bytes of a constant initializer of `type`, which the data starts
 * zeroed: undefined bytes may hold any value. */
static bool
constant_bytes(struct translator *t, LLVMValueRef value, LLVMTypeRef type, uint8_t *bytes,
               unsigned depth)
{
   if (depth == 32)
      return fail(t, STELA_AIR_LIMIT, "constant data nesting exceeded");
   if (LLVMIsAConstantAggregateZero(value) || LLVMIsUndef(value) ||
       !LLVMABISizeOfType(t->layout, type))
      return true;
   LLVMTypeKind kind = LLVMGetTypeKind(type);
   if (kind == LLVMIntegerTypeKind || kind == LLVMHalfTypeKind || kind == LLVMBFloatTypeKind ||
       kind == LLVMFloatTypeKind) {
      unsigned bits = kind == LLVMIntegerTypeKind ? LLVMGetIntTypeWidth(type) :
                      kind == LLVMFloatTypeKind ? 32 : 16;
      /* LLVM serializes its APFloat bits, as for immediate operands. */
      LLVMValueRef encoded = kind == LLVMIntegerTypeKind ? value :
         LLVMConstBitCast(value, LLVMIntTypeInContext(t->context, bits));
      if (!LLVMIsAConstantInt(encoded))
         return fail(t, STELA_AIR_TYPE, "unsupported constant data initializer");
      uint64_t encoding = LLVMConstIntGetZExtValue(encoded);
      for (unsigned i = 0; i < bits / 8; i++)
         bytes[i] = encoding >> (8 * i);
      return true;
   }
   uint64_t count = kind == LLVMStructTypeKind ? LLVMCountStructElementTypes(type) :
                    kind == LLVMArrayTypeKind ? LLVMGetArrayLength2(type) : LLVMGetVectorSize(type);
   for (uint64_t i = 0; i < count; i++) {
      LLVMValueRef element = LLVMGetAggregateElement(value, (unsigned)i);
      if (!element)
         return fail(t, STELA_AIR_TYPE, "unsupported constant data initializer");
      LLVMTypeRef element_type = kind == LLVMStructTypeKind ?
         LLVMStructGetTypeAtIndex(type, (unsigned)i) : LLVMGetElementType(type);
      /* Vectors bit-pack their lanes; arrays step by allocation size. */
      uint64_t offset = kind == LLVMStructTypeKind ? LLVMOffsetOfElement(t->layout, type, (unsigned)i) :
         i * (kind == LLVMVectorTypeKind ? LLVMStoreSizeOfType(t->layout, element_type) :
                                           LLVMABISizeOfType(t->layout, element_type));
      if (!constant_bytes(t, element, element_type, bytes + offset, depth + 1))
         return false;
   }
   return true;
}

/* Lays a constant global out in the shader's constant data at `base`. */
static bool
constant_data(struct translator *t, LLVMValueRef global, const struct stela_air_global *info,
              unsigned *base)
{
   nir_shader *shader = t->b.shader;
   uint64_t start = ALIGN_POT((uint64_t)shader->constant_data_size, info->alignment);
   if (start + info->size > t->options->max_constant_bytes)
      return fail(t, STELA_AIR_LIMIT, "constant data exceeds the target limit");
   if (LLVMByteOrder(t->layout) != LLVMLittleEndian)
      return fail(t, STELA_AIR_TARGET, "constant data requires a little-endian layout");
   uint8_t *data = rerzalloc_size(shader, shader->constant_data, shader->constant_data_size,
                                  start + info->size);
   if (!data)
      return fail(t, STELA_AIR_ALLOCATION, "constant data allocation failed");
   shader->constant_data = data;
   shader->constant_data_size = start + info->size;
   *base = start;
   return constant_bytes(t, LLVMGetInitializer(global), LLVMGlobalGetValueType(global),
                         data + start, 0);
}

static nir_def *
module_global(struct translator *t, LLVMValueRef global)
{
   struct stela_air_global info;
   if (!stela_air_metadata_global(&t->metadata, global, &info))
      return NULL;
   nir_variable *var = NULL;
   unsigned base = 0;
   if (info.constant ? !constant_data(t, global, &info, &base) :
                       !(var = shared_resource(t, info.size, info.alignment)))
      return NULL;
   struct value *value = storage_root(t, global, info.constant ? STELA_AIR_READ : STELA_AIR_READ_WRITE);
   if (!value)
      return NULL;
   value->storage = var;
   value->constant_data = info.constant;
   value->storage_size = (unsigned)info.size;
   value->constant_base = base;
   return value->def;
}

/* A SIMD-group matrix's LLVM type holds all 64 elements, but a thread uses
 * only its thread elements 0 and 1: a 32nd of the type's size. */
#define MATRIX_STORAGE_SHIFT 5

/* `type`, or the member of a struct of one member, down to a SIMD-group
 * matrix; NULL for anything else. */
static LLVMTypeRef
matrix_leaf(LLVMTypeRef type)
{
   unsigned bits;
   while (LLVMGetTypeKind(type) == LLVMStructTypeKind && !LLVMIsOpaqueStruct(type) &&
          LLVMCountStructElementTypes(type) == 1)
      type = LLVMStructGetTypeAtIndex(type, 0);
   return stela_air_matrix_type(type, &bits) ? type : NULL;
}

/* Folds a GEP's offsets into *steps, the greatest common divisor of every
 * constant offset and dynamic stride it adds. False for a GEP whose offset
 * this cannot bound. */
static bool
matrix_gep_steps(struct translator *t, LLVMValueRef gep, uint64_t *steps)
{
   LLVMTypeRef type = LLVMGetGEPSourceElementType(gep);
   unsigned operands = LLVMGetNumOperands(gep);
   for (unsigned i = 1; i < operands; i++) {
      LLVMValueRef index = LLVMGetOperand(gep, i);
      if (LLVMGetTypeKind(LLVMTypeOf(index)) != LLVMIntegerTypeKind || !LLVMTypeIsSized(type))
         return false;
      uint64_t stride;
      if (i == 1) {
         stride = LLVMABISizeOfType(t->layout, type);
      } else if (LLVMGetTypeKind(type) == LLVMStructTypeKind) {
         if (!LLVMIsAConstantInt(index) || LLVMIsOpaqueStruct(type) ||
             LLVMConstIntGetZExtValue(index) >= LLVMCountStructElementTypes(type))
            return false;
         unsigned field = (unsigned)LLVMConstIntGetZExtValue(index);
         *steps = gcd(*steps, LLVMOffsetOfElement(t->layout, type, field));
         type = LLVMStructGetTypeAtIndex(type, field);
         continue;
      } else if (LLVMGetTypeKind(type) == LLVMArrayTypeKind) {
         type = LLVMGetElementType(type);
         stride = LLVMABISizeOfType(t->layout, type);
      } else if (LLVMGetTypeKind(type) == LLVMVectorTypeKind) {
         type = LLVMGetElementType(type);
         stride = LLVMStoreSizeOfType(t->layout, type);
      } else {
         return false;
      }
      if (LLVMIsAConstantInt(index)) {
         int64_t offset = LLVMConstIntGetSExtValue(index) * (int64_t)stride;
         *steps = gcd(*steps, (uint64_t)(offset < 0 ? -offset : offset));
      } else {
         *steps = gcd(*steps, stride);
      }
   }
   return true;
}

/* Whether every use of `pointer`, through GEPs and bitcasts, is a lifetime
 * marker or a plain load or store of a whole *matrix (one matrix type
 * throughout), folding GEP offsets into *steps. */
static bool
matrix_uses(struct translator *t, LLVMValueRef pointer, LLVMTypeRef *matrix, uint64_t *steps,
            unsigned depth)
{
   if (depth > 16)
      return false;
   for (LLVMUseRef use = LLVMGetFirstUse(pointer); use; use = LLVMGetNextUse(use)) {
      LLVMValueRef user = LLVMGetUser(use);
      if (!LLVMIsAInstruction(user))
         return false;
      LLVMTypeRef leaf = NULL;
      switch (LLVMGetInstructionOpcode(user)) {
      case LLVMLoad:
         leaf = matrix_leaf(LLVMTypeOf(user));
         break;
      case LLVMStore:
         if (LLVMGetOperand(user, 1) != pointer || LLVMGetOperand(user, 0) == pointer)
            return false;
         leaf = matrix_leaf(LLVMTypeOf(LLVMGetOperand(user, 0)));
         break;
      case LLVMBitCast:
         if (!matrix_uses(t, user, matrix, steps, depth + 1))
            return false;
         continue;
      case LLVMGetElementPtr:
         if (LLVMGetOperand(user, 0) != pointer || !matrix_gep_steps(t, user, steps) ||
             !matrix_uses(t, user, matrix, steps, depth + 1))
            return false;
         continue;
      case LLVMCall: {
         size_t length = 0;
         LLVMValueRef callee = LLVMGetCalledValue(user);
         const char *name = LLVMIsAFunction(callee) ? LLVMGetValueName2(callee, &length) : "";
         if (strncmp(name, "llvm.lifetime.", strlen("llvm.lifetime.")))
            return false;
         continue;
      }
      default:
         return false;
      }
      if (!leaf || LLVMGetVolatile(user) || LLVMGetOrdering(user) != LLVMAtomicOrderingNotAtomic ||
          (*matrix && *matrix != leaf))
         return false;
      *matrix = leaf;
   }
   return true;
}

/* Whether `alloca` holds only SIMD-group matrices of one type, each loaded
 * and stored whole at a multiple of the matrix's size. Such storage keeps
 * only each matrix's thread elements, at a 32nd of every offset, as Apple's
 * GPUs keep a matrix in its SIMD-group's registers. */
static bool
matrix_storage(struct translator *t, LLVMValueRef alloca)
{
   LLVMTypeRef matrix = NULL;
   uint64_t steps = 0;
   if (!matrix_uses(t, alloca, &matrix, &steps, 0) || !matrix)
      return false;
   unsigned bits;
   stela_air_matrix_type(matrix, &bits);
   uint64_t bytes = LLVMABISizeOfType(t->layout, matrix);
   /* Elements 0 and 1 fill exactly the compacted slot of each matrix. */
   return bytes == (2u * bits / 8) << MATRIX_STORAGE_SHIFT && steps % bytes == 0;
}

/* An entry-block alloca of a fixed size: thread memory. */
static struct value *
private_allocation(struct translator *t, LLVMValueRef alloca)
{
   struct value *value = find_value(t, alloca);
   if (value)
      return value;
   LLVMTypeRef type = LLVMGetAllocatedType(alloca);
   LLVMValueRef count = LLVMGetOperand(alloca, 0);
   if (LLVMGetInstructionParent(alloca) != LLVMGetEntryBasicBlock(t->entry) ||
       !LLVMIsAConstantInt(count) || LLVMGetPointerAddressSpace(LLVMTypeOf(alloca)) != 0) {
      fail(t, STELA_AIR_MEMORY, "thread memory requires a fixed entry-block allocation");
      return NULL;
   }
   if (LLVMPointerSizeForAS(t->layout, 0) != 8) {
      fail(t, STELA_AIR_TARGET, "thread memory pointers require 64-bit layout");
      return NULL;
   }
   unsigned budget = 4096;
   if (!fixed_storage_type(t, type, 0, &budget))
      return NULL;
   uint64_t element = LLVMABISizeOfType(t->layout, type), length = LLVMConstIntGetZExtValue(count);
   uint64_t limit = t->options->max_private_bytes;
   unsigned alignment = LLVMGetAlignment(alloca);
   /* The view's size is 32-bit; the layout is held to the limit below. */
   if (!element || !length || length > UINT32_MAX / element ||
       !alignment || (alignment & (alignment - 1))) {
      fail(t, STELA_AIR_LIMIT, "thread memory size/alignment exceeds the target limit");
      return NULL;
   }
   uint64_t size = element * length;
   /* Matrix storage keeps a 32nd of its bytes: each matrix's thread elements. */
   bool matrices = matrix_storage(t, alloca);
   uint64_t stored = matrices ? DIV_ROUND_UP(size, 1u << MATRIX_STORAGE_SHIFT) : size;
   unsigned stored_alignment = matrices ? MIN2(alignment, 8u) : alignment;
   if (t->private_layout_bound + stored + stored_alignment - 1 > limit) {
      fail(t, STELA_AIR_LIMIT, "thread memory size/alignment exceeds the target limit");
      return NULL;
   }
   nir_variable *var = nir_local_variable_create(t->b.impl,
      glsl_array_type(glsl_uint8_t_type(), (unsigned)stored, 1), "air.private");
   if (!var || (matrices && !t->matrix_storage &&
                !(t->matrix_storage = _mesa_pointer_set_create(t->cfg_scratch))) ||
       (matrices && !_mesa_set_add(t->matrix_storage, var))) {
      fail(t, STELA_AIR_ALLOCATION, "thread memory variable allocation failed");
      return NULL;
   }
   var->data.alignment = stored_alignment;
   t->private_allocations++;
   t->private_layout_bound += stored + stored_alignment - 1;
   value = storage_root(t, alloca, STELA_AIR_READ_WRITE);
   if (value) {
      value->storage = var;
      value->storage_size = (unsigned)size;
   }
   return value;
}

static bool
thread_memory(const struct value *value)
{
   return value->storage && value->storage->data.mode == nir_var_function_temp;
}

/* A GEP's address, with the constant part of its offset and the greatest
 * common divisor of its dynamic indices' steps. */
static nir_def *
gep(struct translator *t, LLVMValueRef inst, const struct value **origin, uint64_t *constant,
    uint64_t *steps)
{
   struct value *base = pointer_value(t, LLVMGetOperand(inst, 0));
   LLVMTypeRef result_type = LLVMTypeOf(inst);
   if (!base || !base->access || LLVMGetTypeKind(result_type) != LLVMPointerTypeKind) {
      fail(t, STELA_AIR_MEMORY, "GEP requires a validated buffer/shared pointer");
      return NULL;
   }
   LLVMTypeRef type = LLVMGetGEPSourceElementType(inst);
   unsigned type_budget = 4096;
   if (!fixed_storage_type(t, type, 0, &type_budget))
      return NULL;
   nir_def *address = base->def;
   *constant = *steps = 0;
   unsigned operands = LLVMGetNumOperands(inst);
   for (unsigned i = 1; i < operands; i++) {
      LLVMValueRef index = LLVMGetOperand(inst, i);
      LLVMTypeRef index_type = LLVMTypeOf(index);
      if (LLVMGetTypeKind(index_type) != LLVMIntegerTypeKind ||
          LLVMGetIntTypeWidth(index_type) > 64) {
         fail(t, STELA_AIR_TYPE, "unsupported GEP index type");
         return NULL;
      }
      if (i > 1 && LLVMGetTypeKind(type) == LLVMStructTypeKind) {
         if (!LLVMIsAConstantInt(index) || LLVMIsOpaqueStruct(type)) {
            fail(t, STELA_AIR_TYPE, "GEP requires a constant sized-struct index");
            return NULL;
         }
         uint64_t field = LLVMConstIntGetZExtValue(index);
         if (field >= LLVMCountStructElementTypes(type)) {
            fail(t, STELA_AIR_TYPE, "GEP struct index out of range");
            return NULL;
         }
         uint64_t offset = LLVMOffsetOfElement(t->layout, type, field);
         address = nir_iadd_imm(&t->b, address, offset);
         *constant += offset;
         type = LLVMStructGetTypeAtIndex(type, field);
      } else {
         bool vector_element = false;
         if (i > 1) {
            LLVMTypeKind kind = LLVMGetTypeKind(type);
            if (kind != LLVMArrayTypeKind && kind != LLVMVectorTypeKind) {
               fail(t, STELA_AIR_TYPE, "unsupported GEP aggregate");
               return NULL;
            }
            vector_element = kind == LLVMVectorTypeKind;
            type = LLVMGetElementType(type);
         }
         if (!LLVMTypeIsSized(type)) {
            fail(t, STELA_AIR_TYPE, "unsized GEP element");
            return NULL;
         }
         nir_def *offset = get_value(t, index);
         if (!offset)
            return NULL;
         offset = offset->bit_size == 1 ?
            nir_ineg(&t->b, nir_b2i64(&t->b, offset)) : nir_i2i64(&t->b, offset);
         uint64_t stride;
         if (vector_element) {
            /* LLVM's getSequentialElementStride bit-packs vectors, even when
             * the scalar element has a larger ABI allocation alignment.
             * Arrays and the root pointer step retain allocation-size stride. */
            stride = LLVMStoreSizeOfType(t->layout, type);
            if (LLVMSizeOfTypeInBits(t->layout, type) != stride * 8) {
               fail(t, STELA_AIR_TYPE, "non-byte-addressable GEP vector element");
               return NULL;
            }
         } else {
            stride = LLVMABISizeOfType(t->layout, type);
         }
         if (LLVMIsAConstantInt(index))
            *constant += (uint64_t)LLVMConstIntGetSExtValue(index) * stride;
         else
            *steps = gcd(*steps, stride);
         offset = nir_imul_imm(&t->b, offset, stride);
         address = nir_iadd(&t->b, address, offset);
      }
   }
   *origin = base;
   return address;
}

/* All frontend branches, including memory guards introduced within an LLVM
 * block, use the same flat CFG. Mesa alone performs structurization. */
static nir_block *
new_block(struct translator *t)
{
   nir_block *block = nir_block_create(t->b.impl);
   if (!block) {
      fail(t, STELA_AIR_ALLOCATION, "AIR NIR block allocation failed");
      return NULL;
   }
   exec_list_push_tail(&t->b.impl->body, &block->cf_node.node);
   block->cf_node.parent = &t->b.impl->cf_node;
   return block;
}

struct memory_guard {
   nir_block *then_block, *else_block, *merge;
};

static bool
memory_guard_begin(struct translator *t, nir_def *condition, struct memory_guard *guard)
{
   *guard = (struct memory_guard){new_block(t), new_block(t), new_block(t)};
   if (!guard->then_block || !guard->else_block || !guard->merge)
      return false;
   nir_goto_if(&t->b, guard->then_block, condition, guard->else_block);
   t->b.cursor = nir_after_block(guard->then_block);
   return true;
}

static nir_def *
memory_guard_end(struct translator *t, const struct memory_guard *guard, nir_def *loaded)
{
   nir_block *then_end = nir_cursor_current_block(t->b.cursor);
   nir_goto(&t->b, guard->merge);
   t->b.cursor = nir_after_block(guard->else_block);
   nir_def *zero = loaded ? nir_imm_zero(&t->b, loaded->num_components, loaded->bit_size) : NULL;
   nir_goto(&t->b, guard->merge);
   t->b.cursor = nir_after_block(guard->merge);
   if (!loaded)
      return NULL;
   nir_phi_instr *phi = nir_phi_instr_create(t->b.shader);
   if (!phi) {
      fail(t, STELA_AIR_ALLOCATION, "AIR memory-guard PHI allocation failed");
      return NULL;
   }
   nir_def_init(&phi->instr, &phi->def, loaded->num_components, loaded->bit_size);
   nir_phi_instr_add_src(phi, then_end, loaded);
   nir_phi_instr_add_src(phi, guard->else_block, zero);
   nir_builder_instr_insert(&t->b, &phi->instr);
   return &phi->def;
}

/* A relaxed atomic read-modify-write of `data`, compared with `compare` for a
 * compare-exchange. */
struct atomic {
   nir_atomic_op op;
   nir_def *data, *compare;
};

static nir_def *
ssbo_access(struct translator *t, const struct value *address, nir_def *stored,
            const struct atomic *atomic, enum gl_access_qualifier access,
            unsigned components, unsigned bits, unsigned alignment)
{
   nir_builder *b = &t->b;
   nir_def *slot = nir_imm_int(b, address->ssbo_slot);
   nir_def *size = nir_u2u64(b, nir_get_ssbo_size(b, 32, slot));
   unsigned bytes = components * (bits / 8);
   /* Compare at LLVM pointer width before narrowing to the native SSBO offset.
    * Subtract the small access size only with a separate size>=bytes guard;
    * offset+bytes could overflow, and offset truncation would alias low bytes.
    * The surface byte length ABI is validated at command binding, not guessed
    * from the argument's scalar type size or a synthetic backing allocation. */
   nir_def *valid = nir_iand(b, nir_uge_imm(b, size, bytes),
      nir_uge(b, nir_iadd_imm(b, size, -(int64_t)bytes), address->def));
   nir_def *loaded = NULL;
   struct memory_guard guard;
   if (!memory_guard_begin(t, valid, &guard))
      return NULL;
   nir_def *offset = nir_u2u32(b, address->def);
   if (atomic && atomic->compare)
      loaded = nir_ssbo_atomic_swap(b, bits, slot, offset, atomic->compare, atomic->data,
                                    .atomic_op = atomic->op);
   else if (atomic)
      loaded = nir_ssbo_atomic(b, bits, slot, offset, atomic->data, .atomic_op = atomic->op);
   else if (stored)
      nir_store_ssbo(b, stored, slot, offset, .align_mul = alignment, .access = access);
   else
      loaded = nir_load_ssbo(b, components, bits, slot, offset, .align_mul = alignment, .access = access);
   return memory_guard_end(t, &guard, loaded);
}

/* Begins a guard for an access of `bytes` within a storage_size view. LLVM64
 * offsets are compared before narrowing, so an out-of-view access never
 * truncates into another allocation or a valid low offset. */
static bool
storage_guard_begin(struct translator *t, const struct value *address, unsigned bytes,
                    struct memory_guard *guard)
{
   nir_builder *b = &t->b;
   nir_def *valid = address->storage_size >= bytes ?
      nir_uge(b, nir_imm_int64(b, address->storage_size - bytes), address->def) : nir_imm_false(b);
   return memory_guard_begin(t, valid, guard);
}

/* An access to threadgroup or thread memory, through its byte variable. */
static nir_def *
variable_access(struct translator *t, const struct value *address, nir_def *stored,
                const struct atomic *atomic, enum gl_access_qualifier access,
                unsigned components, unsigned bits, unsigned alignment)
{
   nir_builder *b = &t->b;
   unsigned bytes = components * (bits / 8);
   struct memory_guard guard;
   if (!storage_guard_begin(t, address, bytes, &guard))
      return NULL;
   nir_def *offset = nir_u2u32(b, address->def);
   if (t->matrix_storage && _mesa_set_search(t->matrix_storage, address->storage)) {
      /* A matrix's thread elements, at a 32nd of its offset: a multiple of
       * their size (matrix_storage). */
      offset = nir_ushr_imm(b, offset, MATRIX_STORAGE_SHIFT);
      alignment = MIN2(bytes, address->storage->data.alignment);
   }
   nir_deref_instr *byte = nir_build_deref_array(b, nir_build_deref_var(b, address->storage),
                                                offset);
   const struct glsl_type *type = glsl_vector_type(glsl_get_base_type(glsl_uintN_t_type(bits)), components);
   nir_deref_instr *typed = nir_build_deref_cast_with_alignment(b, &byte->def,
      address->storage->data.mode, type, bytes, alignment, 0);
   nir_def *loaded = NULL;
   if (atomic && atomic->compare)
      loaded = nir_deref_atomic_swap(b, bits, &typed->def, atomic->compare, atomic->data,
                                     .atomic_op = atomic->op);
   else if (atomic)
      loaded = nir_deref_atomic(b, bits, &typed->def, atomic->data, .atomic_op = atomic->op);
   else if (stored)
      nir_store_deref_with_access(b, typed, stored, BITFIELD_MASK(components), access);
   else
      loaded = nir_load_deref_with_access(b, typed, access);
   return memory_guard_end(t, &guard, loaded);
}

/* A load of constant data. */
static nir_def *
constant_access(struct translator *t, const struct value *address, unsigned components,
                unsigned bits, unsigned alignment)
{
   struct memory_guard guard;
   if (!storage_guard_begin(t, address, components * (bits / 8), &guard))
      return NULL;
   nir_def *loaded = nir_load_constant(&t->b, components, bits, nir_u2u32(&t->b, address->def),
                                       .base = address->constant_base,
                                       .range = address->storage_size, .align_mul = alignment);
   return memory_guard_end(t, &guard, loaded);
}

static nir_def *
global_access(struct translator *t, const struct value *address, nir_def *stored,
              const struct atomic *atomic, enum gl_access_qualifier access,
              unsigned components, unsigned bits, unsigned alignment)
{
   nir_builder *b = &t->b;
   if (atomic && atomic->compare)
      return nir_global_atomic_swap(b, bits, address->def, atomic->compare, atomic->data,
                                    .atomic_op = atomic->op);
   if (atomic)
      return nir_global_atomic(b, bits, address->def, atomic->data, .atomic_op = atomic->op);
   if (stored) {
      nir_store_global(b, stored, address->def, .align_mul = alignment, .access = access);
      return NULL;
   }
   return nir_load_global(b, components, bits, address->def, .align_mul = alignment, .access = access);
}

/* Whether `address` points into a buffer argument nothing writes while the
 * function runs: one it only reads (`const device` or `constant`, which
 * translation refuses to store to) that Apple's compiler marks
 * "air-buffer-no-alias", so that no pointer it writes through reaches it.
 * Its loads may then use a read-only cache and move. */
static bool
read_only_buffer(struct translator *t, const struct value *address)
{
   LLVMValueRef root = address->pointer_root;
   if (address->access != STELA_AIR_READ || address->choices || address->storage ||
       address->constant_data || !root || !LLVMIsAArgument(root) || LLVMGetParamParent(root) != t->entry)
      return false;
   static const char key[] = "air-buffer-no-alias";
   unsigned count = LLVMCountParams(t->entry);
   for (unsigned i = 0; i < count; i++) {
      if (LLVMGetParam(t->entry, i) == root)
         return LLVMGetStringAttributeAtIndex(t->entry, i + 1, key, sizeof(key) - 1) != NULL;
   }
   return false;
}

/* Whether a fragment function's read of `bytes` at `address` lies within its
 * bound buffer argument's declared type at a fixed offset, so that it reads
 * mapped memory whichever pixel runs it: a draw binds every buffer its
 * functions access. Speculating it lets a derivative of the value it reads
 * leave a branch (hoist_derivatives). */
static bool
fixed_field(struct translator *t, const struct value *address, unsigned bytes)
{
   if (!t->stage_options || t->stage_options->stage != STELA_AIR_FRAGMENT ||
       !address->declared_size || address->choices || !address->pointer_root)
      return false;
   const struct value *root = find_value(t, address->pointer_root);
   if (!root || !root->def)
      return false;
   uint64_t offset = 0;
   nir_def *def = address->def;
   for (unsigned steps = 0; def != root->def; steps++) {
      nir_instr *instr = nir_def_instr(def);
      if (steps == 16 || instr->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_iadd)
         return false;
      unsigned constant = nir_src_is_const(alu->src[1].src) ? 1 : 0;
      if (!nir_src_is_const(alu->src[constant].src))
         return false;
      uint64_t added = nir_src_comp_as_uint(alu->src[constant].src, alu->src[constant].swizzle[0]);
      if (added >= address->declared_size)
         return false;
      offset += added;
      def = alu->src[!constant].src.ssa;
   }
   return offset + bytes <= address->declared_size;
}

/* One access at `address`: a load, a store of `stored` or an atomic, which
 * return the loaded or old value. */
static nir_def *
memory_access(struct translator *t, const struct value *address, nir_def *stored,
              const struct atomic *atomic, enum gl_access_qualifier access,
              unsigned components, unsigned bits, unsigned alignment);

/* A choice's root `i`: its value, or a null pointer's. */
static struct value
choice_root(struct translator *t, const struct pointer_choices *choices, unsigned i)
{
   if (LLVMIsAConstantPointerNull(choices->roots[i]))
      return (struct value){.access = STELA_AIR_READ_WRITE, .pointer_root = choices->roots[i],
                            .null = true, .components = 1, .bits = 64};
   const struct value *root = find_value(t, choices->roots[i]);
   return root ? *root : (struct value){0};
}

/* An access through a pointer chosen among roots: the chosen root's at the
 * pointer's address or offset, each guarded by whether it is chosen. */
static nir_def *
choice_access(struct translator *t, const struct value *address, nir_def *stored,
              const struct atomic *atomic, enum gl_access_qualifier access,
              unsigned components, unsigned bits, unsigned alignment)
{
   nir_builder *b = &t->b;
   nir_def *result = NULL;
   for (unsigned i = 0; i < address->choices->count; i++) {
      struct value chosen = choice_root(t, address->choices, i);
      if (!chosen.access) {
         fail(t, STELA_AIR_MEMORY, "a pointer chooses a root without provenance");
         return NULL;
      }
      chosen.def = address->def;
      chosen.choices = NULL;
      chosen.choice = NULL;
      nir_def *taken = nir_ieq_imm(b, address->choice, i);
      struct memory_guard guard;
      if (!memory_guard_begin(t, taken, &guard))
         return NULL;
      nir_def *loaded = memory_access(t, &chosen, stored, atomic, access, components, bits,
                                      alignment);
      if (t->error->kind != STELA_AIR_OK)
         return NULL;
      loaded = memory_guard_end(t, &guard, loaded);
      if (loaded)
         result = result ? nir_bcsel(b, taken, loaded, result) : loaded;
   }
   return result;
}

static nir_def *
memory_access(struct translator *t, const struct value *address, nir_def *stored,
              const struct atomic *atomic, enum gl_access_qualifier access,
              unsigned components, unsigned bits, unsigned alignment)
{
   if (address->choices)
      return choice_access(t, address, stored, atomic, access, components, bits, alignment);
   if (address->array) {
      fail(t, STELA_AIR_MEMORY, "texture and sampler arrays are only read by element");
      return NULL;
   }
   if (address->null)
      return stored ? NULL : nir_imm_zero(&t->b, components, bits);
   /* Buffer memory is aligned only as its declared type is; a pointer of
    * unknown type, only as its elements. Thread memory and constant data
    * are the translator's own, aligned as LLVM claims. */
   if (!address->constant_data && (!address->storage || address->declared_alignment)) {
      unsigned element = MAX2(bits / 8, 1);
      alignment = MIN2(alignment, MAX2(address->declared_alignment, element));
   }
   if (address->storage)
      return variable_access(t, address, stored, atomic, access, components, bits, alignment);
   if (address->constant_data)
      return constant_access(t, address, components, bits, alignment);
   if (address->ssbo)
      return ssbo_access(t, address, stored, atomic, access, components, bits, alignment);
   if (!stored && !atomic && fixed_field(t, address, components * (bits / 8)))
      access |= ACCESS_CAN_SPECULATE;
   if (!stored && !atomic && !(access & (ACCESS_ATOMIC | ACCESS_COHERENT)) &&
       read_only_buffer(t, address))
      access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   return global_access(t, address, stored, atomic, access, components, bits, alignment);
}

/* The value an atomic `op` of `data` at `pointer` replaces, negated first
 * when `negate`; `expected` makes it a compare-exchange. Atomics act on 32-
 * and 64-bit integers, and on floats exchange, by their bits, and add. NIR's
 * atomics are device-scope, which refines every Metal scope. */
/* Whether a pointer chosen among roots may be thread memory. */
static bool
chooses_thread_memory(struct translator *t, const struct value *address)
{
   for (unsigned i = 0; address->choices && i < address->choices->count; i++) {
      struct value root = choice_root(t, address->choices, i);
      if (thread_memory(&root))
         return true;
   }
   return false;
}

static nir_def *
atomic_old(struct translator *t, LLVMValueRef pointer, nir_atomic_op op, bool negate,
           LLVMTypeRef type, nir_def *data, nir_def *expected, unsigned alignment)
{
   unsigned components, bits;
   bool floating = LLVMGetTypeKind(type) == LLVMFloatTypeKind;
   if (!value_shape(type, &components, &bits) || components != 1 ||
       (floating ? op != nir_atomic_op_xchg && op != nir_atomic_op_fadd &&
                      op != nir_atomic_op_cmpxchg :
                   (bits != 32 && bits != 64) || op == nir_atomic_op_fadd)) {
      fail(t, STELA_AIR_TYPE, "unsupported atomic operand type");
      return NULL;
   }
   if (alignment < bits / 8) {
      fail(t, STELA_AIR_MEMORY, "atomic access is not naturally aligned");
      return NULL;
   }
   const struct value *address = pointer_value(t, pointer);
   if (!address || (address->access & STELA_AIR_READ_WRITE) != STELA_AIR_READ_WRITE ||
       thread_memory(address) || chooses_thread_memory(t, address)) {
      fail(t, STELA_AIR_MEMORY, "atomic access lacks a device or threadgroup read-write pointer");
      return NULL;
   }
   if (negate)
      data = floating ? nir_ixor(&t->b, data, nir_imm_int(&t->b, INT32_MIN)) : nir_ineg(&t->b, data);
   const struct atomic atomic = {op, data, expected};
   return memory_access(t, address, NULL, &atomic, 0, 1, bits, alignment);
}

/* An atomic `op` of `operand` at `pointer`, as atomic_old; `compare` makes
 * it a compare-exchange whose result is {old value, exchanged}. */
static bool
atomic_access(struct translator *t, LLVMValueRef inst, LLVMValueRef pointer, nir_atomic_op op,
              bool negate, LLVMValueRef operand, LLVMValueRef compare, unsigned alignment)
{
   nir_def *data = get_value(t, operand);
   nir_def *expected = compare ? get_value(t, compare) : NULL;
   if (!data || (compare && !expected))
      return false;
   nir_def *old = atomic_old(t, pointer, op, negate, LLVMTypeOf(operand), data, expected, alignment);
   if (!old)
      return false;
   return compare ? put_pair(t, inst, old, nir_ieq(&t->b, old, expected)) : put_value(t, inst, old, 0);
}

/* LLVM's atomicrmw and cmpxchg. Metal's atomics are relaxed, so stronger
 * orderings are refused rather than approximated. */
static bool
atomic_instruction(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   bool exchange = op == LLVMAtomicCmpXchg;
   bool relaxed = exchange ?
      LLVMGetCmpXchgSuccessOrdering(inst) == LLVMAtomicOrderingMonotonic &&
      LLVMGetCmpXchgFailureOrdering(inst) == LLVMAtomicOrderingMonotonic :
      LLVMGetOrdering(inst) == LLVMAtomicOrderingMonotonic;
   if (!relaxed)
      return fail(t, STELA_AIR_MEMORY, "only relaxed (monotonic) atomics are supported");
   nir_atomic_op atomic = nir_atomic_op_cmpxchg;
   bool negate = false;
   if (!exchange) {
      switch (LLVMGetAtomicRMWBinOp(inst)) {
      case LLVMAtomicRMWBinOpXchg: atomic = nir_atomic_op_xchg; break;
      case LLVMAtomicRMWBinOpSub: negate = true; FALLTHROUGH;
      case LLVMAtomicRMWBinOpAdd: atomic = nir_atomic_op_iadd; break;
      case LLVMAtomicRMWBinOpAnd: atomic = nir_atomic_op_iand; break;
      case LLVMAtomicRMWBinOpOr: atomic = nir_atomic_op_ior; break;
      case LLVMAtomicRMWBinOpXor: atomic = nir_atomic_op_ixor; break;
      case LLVMAtomicRMWBinOpMax: atomic = nir_atomic_op_imax; break;
      case LLVMAtomicRMWBinOpMin: atomic = nir_atomic_op_imin; break;
      case LLVMAtomicRMWBinOpUMax: atomic = nir_atomic_op_umax; break;
      case LLVMAtomicRMWBinOpUMin: atomic = nir_atomic_op_umin; break;
      case LLVMAtomicRMWBinOpFSub: negate = true; FALLTHROUGH;
      case LLVMAtomicRMWBinOpFAdd: atomic = nir_atomic_op_fadd; break;
      default: return fail(t, STELA_AIR_INSTRUCTION, "unsupported atomicrmw operation");
      }
   }
   return atomic_access(t, inst, LLVMGetOperand(inst, 0), atomic, negate,
                        LLVMGetOperand(inst, exchange ? 2 : 1),
                        exchange ? LLVMGetOperand(inst, 1) : NULL, LLVMGetAlignment(inst));
}

static nir_def *
compare(struct translator *t, LLVMValueRef inst, nir_def *a, nir_def *b)
{
   switch (LLVMGetICmpPredicate(inst)) {
   case LLVMIntEQ: return nir_ieq(&t->b, a, b);
   case LLVMIntNE: return nir_ine(&t->b, a, b);
   case LLVMIntUGT: return nir_ult(&t->b, b, a);
   case LLVMIntUGE: return nir_uge(&t->b, a, b);
   case LLVMIntULT: return nir_ult(&t->b, a, b);
   case LLVMIntULE: return nir_uge(&t->b, b, a);
   case LLVMIntSGT: return nir_ilt(&t->b, b, a);
   case LLVMIntSGE: return nir_ige(&t->b, a, b);
   case LLVMIntSLT: return nir_ilt(&t->b, a, b);
   case LLVMIntSLE: return nir_ige(&t->b, b, a);
   default: fail(t, STELA_AIR_INSTRUCTION, "unsupported integer predicate"); return NULL;
   }
}

static nir_def *
value_lane(struct translator *t, struct value *v, unsigned lane)
{
   if (!v || lane >= v->components) return NULL;
   return v->def ? nir_channel(&t->b, v->def, lane) : v->lanes[lane];
}

static struct value *
partial_value(struct translator *t, LLVMValueRef llvm)
{
   struct value *v = find_value(t, llvm);
   if (v) return v;
   unsigned n, bits;
   if (!value_shape(LLVMTypeOf(llvm), &n, &bits)) {
      fail(t, STELA_AIR_TYPE, "unsupported partial value type"); return NULL;
   }
   if (LLVMIsUndef(llvm) || LLVMIsPoison(llvm)) {
      v = rzalloc(t->values, struct value);
      if (!v) {
         fail(t, STELA_AIR_ALLOCATION, "partial value allocation failed"); return NULL;
      }
      v->components = n; v->bits = bits;
      return v;
   }
   if (n > 1 && (LLVMIsAConstantVector(llvm) || LLVMIsAConstantDataVector(llvm) ||
                 LLVMIsAConstantFP(llvm) || LLVMIsAConstantInt(llvm))) {
      v = rzalloc(t->values, struct value);
      if (!v) { fail(t, STELA_AIR_ALLOCATION, "partial vector allocation failed"); return NULL; }
      v->components = n; v->bits = bits;
      bool complete = true;
      for (unsigned i = 0; i < n; i++) {
         LLVMValueRef element = LLVMGetAggregateElement(llvm, i);
         if (LLVMIsUndef(element) || LLVMIsPoison(element)) complete = false;
         else if (!(v->lanes[i] = get_value(t, element))) return NULL;
      }
      if (complete) v->def = nir_vec(&t->b, v->lanes, n);
      /* Constants are rematerialized at the actual use. Caching a vector's
       * branch-local channels under the LLVM constant would violate dominance
       * at a later use in another region. Instruction defs remain in the map. */
      return v;
   }
   nir_def *def = get_value(t, llvm);
   if (!def) return NULL;
   v = rzalloc(t->values, struct value);
   if (!v) { fail(t, STELA_AIR_ALLOCATION, "constant value wrapper allocation failed"); return NULL; }
   v->def = def; v->components = n; v->bits = bits;
   return v;
}

static bool
put_lanes(struct translator *t, LLVMValueRef llvm, nir_def **lanes, unsigned n, unsigned bits)
{
   struct value *v = rzalloc(t->values, struct value);
   if (!v) return fail(t, STELA_AIR_ALLOCATION, "vector lane allocation failed");
   v->components = n; v->bits = bits;
   bool complete = true;
   for (unsigned i = 0; i < n; i++) { v->lanes[i] = lanes[i]; complete &= lanes[i] != NULL; }
   if (complete) v->def = nir_vec(&t->b, lanes, n);
   return _mesa_hash_table_insert(t->values, llvm, v) != NULL ||
      fail(t, STELA_AIR_ALLOCATION, "vector lane map allocation failed");
}

static LLVMTypeRef
aggregate_element_type(LLVMTypeRef type, unsigned index)
{
   return LLVMGetTypeKind(type) == LLVMStructTypeKind ? LLVMStructGetTypeAtIndex(type, index) :
                                                        LLVMGetElementType(type);
}

/* An aggregate's element count: a struct's or array's of leaves and smaller
 * aggregates. Zero for any other type. */
static unsigned
aggregate_count(LLVMTypeRef type)
{
   bool members = LLVMGetTypeKind(type) == LLVMStructTypeKind;
   if ((!members && LLVMGetTypeKind(type) != LLVMArrayTypeKind) ||
       (members && LLVMIsOpaqueStruct(type)))
      return 0;
   uint64_t count = members ? LLVMCountStructElementTypes(type) : LLVMGetArrayLength2(type);
   if (!count || count > AIR_MAX_ARGUMENTS)
      return 0;
   for (unsigned i = 0; i < (members ? count : 1); i++) {
      LLVMTypeRef element = aggregate_element_type(type, i);
      if (!leaf_shape(element) && !aggregate_count(element))
         return 0;
   }
   return count;
}

/* An aggregate of `count` elements, copied from `from` or undefined. */
static struct value *
new_aggregate(struct translator *t, unsigned count, const struct value *from)
{
   struct value *v = rzalloc(t->values, struct value);
   if (!v || !(v->elements = rzalloc_array(t->values, struct value *, count))) {
      fail(t, STELA_AIR_ALLOCATION, "aggregate value allocation failed");
      return NULL;
   }
   v->aggregate_count = count;
   if (from)
      memcpy(v->elements, from->elements, count * sizeof(*v->elements));
   return v;
}

/* A pair as an aggregate, for the PHIs joining it. A member nothing reads
 * stays undefined. */
static struct value *
pair_aggregate(struct translator *t, const struct value *pair)
{
   struct value *v = new_aggregate(t, 2, NULL);
   for (unsigned i = 0; v && i < 2; i++) {
      if (!pair->members[i])
         continue;
      struct value *member = rzalloc(t->values, struct value);
      if (!member) {
         fail(t, STELA_AIR_ALLOCATION, "aggregate value allocation failed");
         return NULL;
      }
      member->def = pair->members[i];
      member->components = member->def->num_components;
      member->bits = member->def->bit_size;
      v->elements[i] = member;
   }
   return v;
}

static struct value *
aggregate_value(struct translator *t, LLVMValueRef llvm)
{
   struct value *v = find_value(t, llvm);
   if (v) {
      if (v->aggregate_count) return v;
      if (v->members[0]) return pair_aggregate(t, v);
      fail(t, STELA_AIR_TYPE, "nonaggregate value"); return NULL;
   }
   LLVMTypeRef type = LLVMTypeOf(llvm);
   unsigned count = aggregate_count(type);
   if (!count || !LLVMIsAConstant(llvm)) {
      fail(t, STELA_AIR_TYPE, "unsupported/forward aggregate"); return NULL;
   }
   if (!LLVMIsUndef(llvm)) {
      /* Constants are rematerialized at each use, as partial_value's are. */
      struct value *v = new_aggregate(t, count, NULL);
      for (unsigned i = 0; v && i < count; i++) {
         LLVMValueRef element = LLVMGetAggregateElement(llvm, i);
         v->elements[i] = aggregate_count(LLVMTypeOf(element)) ? aggregate_value(t, element) :
                                                                 partial_value(t, element);
         if (!v->elements[i])
            return NULL;
      }
      return v;
   }
   v = new_aggregate(t, count, NULL);
   if (v && !_mesa_hash_table_insert(t->values, llvm, v)) {
      fail(t, STELA_AIR_ALLOCATION, "aggregate value allocation failed"); return NULL;
   }
   return v;
}

/* `a`, of LLVM type `type`, with `leaf` at the index path, copying each
 * aggregate on the path. */
static struct value *
aggregate_insert(struct translator *t, const struct value *a, LLVMTypeRef type,
                 const unsigned *indices, unsigned depth, struct value *leaf)
{
   if (indices[0] >= a->aggregate_count) {
      fail(t, STELA_AIR_INSTRUCTION, "invalid aggregate leaf index");
      return NULL;
   }
   struct value *v = new_aggregate(t, a->aggregate_count, a);
   if (!v)
      return NULL;
   if (depth == 1) {
      v->elements[indices[0]] = leaf;
      return v;
   }
   LLVMTypeRef inner_type = aggregate_element_type(type, indices[0]);
   const struct value *inner = a->elements[indices[0]];
   if (!inner && !(inner = new_aggregate(t, aggregate_count(inner_type), NULL)))
      return NULL;
   if (!inner->aggregate_count) {
      fail(t, STELA_AIR_INSTRUCTION, "aggregate index path enters a leaf");
      return NULL;
   }
   v->elements[indices[0]] = aggregate_insert(t, inner, inner_type, indices + 1, depth - 1, leaf);
   return v->elements[indices[0]] ? v : NULL;
}

/* The leaves of a `type` aggregate at `offset` past `address`, each its own
 * access at its layout offset: the leaves of `stored`, or loaded into the
 * aggregate returned. An undefined leaf of `stored` stores nothing. */
static struct value *
aggregate_access(struct translator *t, const struct value *address, LLVMTypeRef type,
                 uint64_t offset, const struct value *stored, enum gl_access_qualifier access,
                 unsigned alignment)
{
   unsigned count = aggregate_count(type);
   struct value *loaded = stored ? NULL : new_aggregate(t, count, NULL);
   if (!stored && !loaded)
      return NULL;
   bool members = LLVMGetTypeKind(type) == LLVMStructTypeKind;
   for (unsigned i = 0; i < count; i++) {
      LLVMTypeRef element = aggregate_element_type(type, i);
      uint64_t at = offset + (members ? LLVMOffsetOfElement(t->layout, type, i) :
                                        i * LLVMABISizeOfType(t->layout, element));
      const struct value *part = stored ? stored->elements[i] : NULL;
      if (stored && !part)
         continue;
      if (aggregate_count(element)) {
         struct value *inner = aggregate_access(t, address, element, at, part, access, alignment);
         if (!inner)
            return NULL;
         if (loaded)
            loaded->elements[i] = inner;
         continue;
      }
      unsigned components, bits;
      if (!value_shape(element, &components, &bits) || bits == 1) {
         fail(t, STELA_AIR_MEMORY, "sub-byte memory access unsupported");
         return NULL;
      }
      /* The leaf's pointer, derived as a GEP to it would be. */
      struct value leaf = *address;
      leaf.def = nir_iadd_imm(&t->b, address->def, at);
      pointer_offset(&leaf, address, at, 0);
      nir_def *def = memory_access(t, &leaf, part ? whole_value(t, part) : NULL, NULL, access,
                                   components, bits, at ? MIN2(alignment, at & -at) : alignment);
      if (t->error->kind != STELA_AIR_OK)
         return NULL;
      if (loaded) {
         struct value *v = rzalloc(t->values, struct value);
         if (!v) {
            fail(t, STELA_AIR_ALLOCATION, "aggregate value allocation failed");
            return NULL;
         }
         v->def = def;
         v->components = components;
         v->bits = bits;
         loaded->elements[i] = v;
      }
   }
   return loaded ? loaded : (struct value *)stored;
}

/* A load or store of a struct or array, which memory holds leaf by leaf. */
static bool
aggregate_memory(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   LLVMValueRef value = op == LLVMStore ? LLVMGetOperand(inst, 0) : inst;
   struct value *address = pointer_value(t, LLVMGetOperand(inst, op == LLVMStore ? 1 : 0));
   unsigned required = op == LLVMStore ? STELA_AIR_WRITE : STELA_AIR_READ;
   unsigned alignment = LLVMGetAlignment(inst);
   if (!address || !(address->access & required) || LLVMGetVolatile(inst) ||
       LLVMGetOrdering(inst) != LLVMAtomicOrderingNotAtomic)
      return fail(t, STELA_AIR_MEMORY, "unsupported volatile/atomic/unbound/disallowed aggregate access");
   if (!alignment)
      return fail(t, STELA_AIR_MEMORY, "memory access lacks explicit alignment");
   LLVMTypeRef type = LLVMTypeOf(value);
   if (op == LLVMStore) {
      struct value *stored = aggregate_value(t, value);
      return stored && aggregate_access(t, address, type, 0, stored, 0, alignment);
   }
   struct value *loaded = aggregate_access(t, address, type, 0, NULL, 0, alignment);
   return loaded && (_mesa_hash_table_insert(t->values, inst, loaded) != NULL ||
                     fail(t, STELA_AIR_ALLOCATION, "aggregate load map failed"));
}

/* An aggregate's insertvalue or extractvalue. */
static bool
aggregate_instruction(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   unsigned depth = LLVMGetNumIndices(inst);
   const unsigned *indices = LLVMGetIndices(inst);
   struct value *a = aggregate_value(t, LLVMGetOperand(inst, 0));
   if (!a || !depth)
      return fail(t, STELA_AIR_INSTRUCTION, "invalid aggregate index path");
   if (op == LLVMExtractValue) {
      struct value *leaf = a;
      for (unsigned i = 0; leaf && i < depth; i++)
         leaf = indices[i] < leaf->aggregate_count ? leaf->elements[indices[i]] : NULL;
      unsigned count = aggregate_count(LLVMTypeOf(inst));
      if (count && !leaf && !(leaf = new_aggregate(t, count, NULL)))
         return false;
      if (count)
         return _mesa_hash_table_insert(t->values, inst, leaf) != NULL ||
            fail(t, STELA_AIR_ALLOCATION, "aggregate extraction map failed");
      unsigned components, bits;
      if (!value_shape(LLVMTypeOf(inst), &components, &bits))
         return fail(t, STELA_AIR_TYPE, "unsupported aggregate leaf type");
      nir_def *def = leaf ? whole_value(t, leaf) : nir_undef(&t->b, components, bits);
      return put_value(t, inst, def, 0);
   }
   LLVMValueRef inserted = LLVMGetOperand(inst, 1);
   struct value *leaf = aggregate_count(LLVMTypeOf(inserted)) ? aggregate_value(t, inserted) :
                                                                partial_value(t, inserted);
   struct value *v = leaf ? aggregate_insert(t, a, LLVMTypeOf(LLVMGetOperand(inst, 0)), indices,
                                             depth, leaf) : NULL;
   return v && (_mesa_hash_table_insert(t->values, inst, v) != NULL ||
                fail(t, STELA_AIR_ALLOCATION, "aggregate insertion map failed"));
}

/* An extractelement, insertelement or shufflevector, whose operands and
 * result may be partially defined vectors. */
static bool
vector_instruction(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   struct value *a = partial_value(t, LLVMGetOperand(inst, 0));
   if (!a) return false;
   if (op == LLVMExtractElement || op == LLVMInsertElement) {
      LLVMValueRef index = LLVMGetOperand(inst, op == LLVMInsertElement ? 2 : 1);
      if (!LLVMIsAConstantInt(index)) {
         nir_def *lane = get_value(t, index);
         if (!lane)
            return false;
         lane = nir_u2u32(&t->b, lane);
         nir_def *vector = whole_value(t, a);
         if (op == LLVMExtractElement)
            return put_value(t, inst, nir_vector_extract(&t->b, vector, lane), 0);
         nir_def *element = get_value(t, LLVMGetOperand(inst, 1));
         return element && put_value(t, inst, nir_vector_insert(&t->b, vector, element, lane), 0);
      }
      if (LLVMConstIntGetZExtValue(index) >= a->components)
         return fail(t, STELA_AIR_INSTRUCTION, "invalid vector index");
      unsigned channel = LLVMConstIntGetZExtValue(index);
      if (op == LLVMExtractElement) {
         nir_def *lane = value_lane(t, a, channel);
         return put_value(t, inst, lane ? lane : nir_undef(&t->b, 1, a->bits), 0);
      }
      nir_def *lane = get_value(t, LLVMGetOperand(inst, 1));
      if (!lane) return false;
      nir_def *lanes[4];
      for (unsigned i = 0; i < a->components; i++)
         lanes[i] = i == channel ? lane : value_lane(t, a, i);
      return put_lanes(t, inst, lanes, a->components, a->bits);
   }
   struct value *b = partial_value(t, LLVMGetOperand(inst, 1));
   unsigned n = LLVMGetNumMaskElements(inst), bits;
   /* A vector widened to a SIMD-group matrix, as Apple's compiler builds a
    * thread's elements: its elements 0 and 1 are the matrix's, the rest
    * undefined. */
   if (b && a->components == b->components && a->bits == b->bits &&
       stela_air_matrix_type(LLVMTypeOf(inst), &bits) && bits == a->bits) {
      nir_def *lanes[2];
      for (unsigned i = 0; i < 2; i++) {
         int mask = LLVMGetMaskValue(inst, i);
         unsigned channel = mask < 0 ? 0 : (unsigned)mask;
         if (mask >= 0 && channel >= a->components + b->components)
            return fail(t, STELA_AIR_INSTRUCTION, "invalid shuffle selector");
         lanes[i] = mask < 0 ? NULL : channel < a->components ? value_lane(t, a, channel) :
                                      value_lane(t, b, channel - a->components);
         if (!lanes[i])
            lanes[i] = nir_undef(&t->b, 1, bits);
      }
      return put_value(t, inst, nir_vec2(&t->b, lanes[0], lanes[1]), 0);
   }
   if (!b || a->components != b->components || a->bits != b->bits || !n || n > 4)
      return fail(t, STELA_AIR_TYPE, "unsupported shuffle shape");
   nir_def *lanes[4] = {0};
   for (unsigned i = 0; i < n; i++) {
      int mask = LLVMGetMaskValue(inst, i);
      if (mask < 0) continue; /* Undefined result lane stays symbolic. */
      unsigned channel = (unsigned)mask;
      if (channel >= a->components + b->components)
         return fail(t, STELA_AIR_INSTRUCTION, "invalid shuffle selector");
      lanes[i] = channel < a->components ? value_lane(t, a, channel) : value_lane(t, b, channel - a->components);
   }
   return put_lanes(t, inst, lanes, n, a->bits);
}

/* Whether `op` computes each element of a SIMD-group matrix from the same
 * element of its operands, as each thread's two elements compute. */
static bool
matrix_elementwise(LLVMOpcode op)
{
   return op == LLVMFNeg || op == LLVMFAdd || op == LLVMFSub || op == LLVMFMul || op == LLVMFDiv ||
          op == LLVMFRem;
}

/* A SIMD-group matrix's thread elements, its storage's elements 0 and 1,
 * read and written by extractelement and insertelement, or gathered into a
 * vector by shufflevector. */
static bool
matrix_elements(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   nir_def *matrix = get_value(t, LLVMGetOperand(inst, 0));
   if (!matrix)
      return false;
   if (op == LLVMShuffleVector) {
      unsigned n = LLVMGetNumMaskElements(inst), components, bits;
      /* A matrix of two matrices' elements, as Apple's compiler replaces a
       * thread's elements and keeps the rest: its own elements 0 and 1 are
       * those its mask names, which must be thread elements. */
      if (stela_air_matrix_type(LLVMTypeOf(inst), &bits)) {
         nir_def *other = get_value(t, LLVMGetOperand(inst, 1));
         if (!other)
            return false;
         nir_def *lanes[2];
         for (unsigned i = 0; i < 2; i++) {
            int mask = LLVMGetMaskValue(inst, i);
            unsigned element = mask < 0 ? 0 : (unsigned)mask % 64;
            if (mask >= 0 && element > 1)
               return fail(t, STELA_AIR_INSTRUCTION, "SIMD-group matrix shuffle beyond the thread elements");
            lanes[i] = mask < 0 ? nir_undef(&t->b, 1, bits) :
                       nir_channel(&t->b, mask < 64 ? matrix : other, element);
         }
         return put_value(t, inst, nir_vec2(&t->b, lanes[0], lanes[1]), 0);
      }
      if (!value_shape(LLVMTypeOf(inst), &components, &bits))
         return fail(t, STELA_AIR_TYPE, "SIMD-group matrix shuffles gather a vector of thread elements");
      nir_def *lanes[4] = {0};
      for (unsigned i = 0; i < n; i++) {
         int mask = LLVMGetMaskValue(inst, i);
         if (mask > 1)
            return fail(t, STELA_AIR_INSTRUCTION, "SIMD-group matrix shuffle beyond the thread elements");
         if (mask >= 0)
            lanes[i] = nir_channel(&t->b, matrix, (unsigned)mask);
      }
      return put_lanes(t, inst, lanes, n, bits);
   }
   LLVMValueRef index = LLVMGetOperand(inst, op == LLVMInsertElement ? 2 : 1);
   if (LLVMIsAConstantInt(index) && LLVMConstIntGetZExtValue(index) > 1)
      return fail(t, STELA_AIR_INSTRUCTION, "SIMD-group matrix index beyond the thread elements");
   nir_def *lane = get_value(t, index);
   if (!lane)
      return false;
   lane = nir_u2u32(&t->b, lane);
   if (op == LLVMExtractElement)
      return put_value(t, inst, nir_vector_extract(&t->b, matrix, lane), 0);
   nir_def *element = get_value(t, LLVMGetOperand(inst, 1));
   return element && put_value(t, inst, nir_vector_insert(&t->b, matrix, element, lane), 0);
}

/* The varying map's row for a varying. */
static const struct stela_air_varying_slot *
varying_slot(struct translator *t, const struct stela_air_stage_io *io)
{
   for (unsigned i = 0; i < t->stage_options->varying_count; i++) {
      const struct stela_air_varying_slot *v = &t->stage_options->varyings[i];
      if (v->identifier_bytes == io->identifier_bytes && !memcmp(v->identifier, io->identifier, io->identifier_bytes))
         return v;
   }
   UNREACHABLE("stage map validated before emission");
}

static unsigned
stage_location(struct translator *t, const struct stela_air_stage_io *io)
{
   switch (io->semantic) {
   case STELA_AIR_POSITION: return VARYING_SLOT_POS;
   case STELA_AIR_RENDER_TARGET: return FRAG_RESULT_DATA0 + io->location;
   case STELA_AIR_VERTEX_INPUT: return VERT_ATTRIB_GENERIC0 + io->location;
   case STELA_AIR_POINT_SIZE: return VARYING_SLOT_PSIZ;
   case STELA_AIR_LAYER: return VARYING_SLOT_LAYER;
   case STELA_AIR_VIEWPORT: return VARYING_SLOT_VIEWPORT;
   case STELA_AIR_CLIP_DISTANCE: return VARYING_SLOT_CLIP_DIST0;
   case STELA_AIR_DEPTH: return FRAG_RESULT_DEPTH;
   case STELA_AIR_SAMPLE_MASK: return FRAG_RESULT_SAMPLE_MASK;
   case STELA_AIR_STENCIL: return FRAG_RESULT_STENCIL;
   case STELA_AIR_MESH_PRIMITIVE_ID: return VARYING_SLOT_PRIMITIVE_ID;
   case STELA_AIR_PRIMITIVE_CULLED: return VARYING_SLOT_CULL_PRIMITIVE;
   default: break;
   }
   return VARYING_SLOT_VAR0 + varying_slot(t, io)->location;
}

/* Stage IO crosses the stage boundary in 32-bit lanes: floats for float and
 * half, signed or unsigned integers for the integer types. */
static bool
stage_signed(enum stela_air_primitive p)
{
   return p == STELA_AIR_CHAR || p == STELA_AIR_SHORT || p == STELA_AIR_INT;
}

static unsigned
stage_bits(enum stela_air_primitive p)
{
   return p == STELA_AIR_CHAR || p == STELA_AIR_UCHAR ? 8 :
          p == STELA_AIR_SHORT || p == STELA_AIR_USHORT || p == STELA_AIR_HALF ? 16 : 32;
}

static nir_def *
stage_narrow(nir_builder *b, nir_def *def, enum stela_air_primitive p)
{
   unsigned bits = stage_bits(p);
   if (bits == 32)
      return def;
   if (p == STELA_AIR_HALF)
      return nir_f2f16(b, def);
   return stage_signed(p) ? nir_i2iN(b, def, bits) : nir_u2uN(b, def, bits);
}

static nir_def *
stage_widen(nir_builder *b, nir_def *def, enum stela_air_primitive p)
{
   if (def->bit_size == 32)
      return def;
   if (p == STELA_AIR_HALF)
      return nir_f2f32(b, def);
   return stage_signed(p) ? nir_i2i32(b, def) : nir_u2u32(b, def);
}

/* The access of color attachment location as storage, or NULL for one the
 * blend unit writes. */
static const struct stela_air_attachment_access *
storage_attachment(const struct translator *t, unsigned location)
{
   const struct stela_air_attachment_access *attachments =
      t->stage_options ? t->stage_options->attachments : NULL;
   return attachments && location < AIR_MAX_RENDER_TARGETS &&
          attachments[location].format != PIPE_FORMAT_NONE ? &attachments[location] : NULL;
}

/* The texel of the render target under the fragment, as the input's type,
 * for which the function's interlock section loads its storage
 * attachment's texel. */
static nir_def *
attachment_input(struct translator *t, const struct stela_air_stage_io *io)
{
   nir_builder *b = &t->b;
   /* A pipeline routes every attachment it reads and has; Metal leaves the
    * value of one it lacks undefined, which reads as zero. */
   if (!storage_attachment(t, io->location))
      return nir_imm_zero(b, io->components, stage_bits(io->primitive));
   /* The interlock section loads the attachment's texel. */
   if (t->attachment_read_count == ARRAY_SIZE(t->attachment_reads)) {
      fail(t, STELA_AIR_LIMIT, "too many render target inputs");
      return NULL;
   }
   /* Unlike an undefined value, which NIR may replace by any other, each
    * use of the load stays one until the section replaces it. Only function
    * temporaries become SSA values before then. */
   unsigned bits = stage_bits(io->primitive);
   const struct glsl_type *type = glsl_vector_type(
      bits == 8 ? GLSL_TYPE_UINT8 : bits == 16 ? GLSL_TYPE_UINT16 : GLSL_TYPE_UINT, io->components);
   nir_variable *texel = nir_variable_create(b->shader, nir_var_shader_temp, type, "attachment texel");
   nir_def *placeholder = nir_load_var(b, texel);
   t->attachment_reads[t->attachment_read_count++] =
      (struct attachment_read){.io = io, .placeholder = placeholder};
   return placeholder;
}

static nir_variable *
stage_variable(struct translator *t, const struct stela_air_stage_io *io, bool output)
{
   char name[4097];
   const char *source = io->name ? io->name : "air.output";
   size_t n = io->name ? io->name_bytes : strlen(source);
   memcpy(name, source, n); name[n] = 0;
   enum glsl_base_type base = io->primitive == STELA_AIR_FLOAT || io->primitive == STELA_AIR_HALF ?
      GLSL_TYPE_FLOAT : stage_signed(io->primitive) ? GLSL_TYPE_INT : GLSL_TYPE_UINT;
   /* The indices and masks are ints, as SPIR-V's builtins are, and so is
    * whether a primitive is culled, as NAK stores it. */
   if (io->semantic == STELA_AIR_LAYER || io->semantic == STELA_AIR_VIEWPORT ||
       io->semantic == STELA_AIR_SAMPLE_MASK || io->semantic == STELA_AIR_STENCIL ||
       io->semantic == STELA_AIR_MESH_PRIMITIVE_ID || io->semantic == STELA_AIR_PRIMITIVE_CULLED)
      base = GLSL_TYPE_INT;
   /* Clip distances are a compact float array, as SPIR-V's ClipDistance is. */
   bool clip = io->semantic == STELA_AIR_CLIP_DISTANCE;
   const struct glsl_type *type = clip ? glsl_array_type(glsl_float_type(), io->clip_distances, 0) :
                                         glsl_vector_type(base, io->components);
   /* A mesh output has one element per vertex or primitive. */
   const struct stela_air_stage_entry *entry = &t->render.entry;
   bool mesh = t->stage_options->stage == STELA_AIR_MESH;
   if (mesh)
      type = glsl_array_type(type, io->per_primitive ? entry->max_primitives : entry->max_vertices, 0);
   if (output && io->semantic == STELA_AIR_STENCIL && !t->stage_options->stencil_export) {
      if (!t->stage_options->fragment_tests || !t->stage_options->raster_order_groups ||
          !t->options->bind_depth_stencil) {
         fail(t, STELA_AIR_TARGET,
              "a stencil output needs the device's shader stencil export or fragment shader interlock");
         return NULL;
      }
      t->tests_fragments = true;
   }
   nir_variable *var = nir_variable_create(t->b.shader, output ? nir_var_shader_out : nir_var_shader_in,
                                           type, name);
   if (!var) { fail(t, STELA_AIR_ALLOCATION, "render IO variable allocation failed"); return NULL; }
   var->data.location = stage_location(t, io);
   var->data.compact = clip;
   var->data.per_primitive = mesh && io->per_primitive;
   if (clip)
      t->b.shader->info.clip_distance_array_size = io->clip_distances;
   var->data.index = io->semantic == STELA_AIR_RENDER_TARGET ? io->index : 0;
   var->data.invariant = io->invariant;
   if (io->semantic == STELA_AIR_DEPTH) {
      static const enum gl_frag_depth_layout layouts[] = {
         [STELA_AIR_DEPTH_ANY] = FRAG_DEPTH_LAYOUT_ANY,
         [STELA_AIR_DEPTH_GREATER] = FRAG_DEPTH_LAYOUT_GREATER,
         [STELA_AIR_DEPTH_LESS] = FRAG_DEPTH_LAYOUT_LESS,
      };
      t->b.shader->info.fs.depth_layout = layouts[io->depth];
   }
   if (!output && io->semantic != STELA_AIR_VERTEX_INPUT) {
      var->data.interpolation = io->interpolation == STELA_AIR_PERSPECTIVE ? INTERP_MODE_SMOOTH :
                                io->interpolation == STELA_AIR_FLAT ? INTERP_MODE_FLAT :
                                INTERP_MODE_NOPERSPECTIVE;
      var->data.centroid = io->sampling == STELA_AIR_CENTROID;
      var->data.sample = io->sampling == STELA_AIR_SAMPLE;
      /* Mesh primitive data is constant over its primitive. */
      if (io->semantic == STELA_AIR_VARYING && varying_slot(t, io)->per_primitive) {
         var->data.per_primitive = true;
         var->data.interpolation = INTERP_MODE_FLAT;
         var->data.centroid = var->data.sample = false;
      }
   }
   return var;
}

/* In *def, a vertex input its pipeline fetches as 32-bit words, decoded
 * into its type; *def stays NULL for one fetched in its own format. False
 * after failing. */
static bool
vertex_input_words(struct translator *t, const struct stela_air_stage_io *io, nir_def **def)
{
   const struct stela_air_stage_options *options = t->stage_options;
   unsigned words = options->vertex_input_words && io->location < AIR_MAX_VERTEX_ATTRIBUTES ?
      options->vertex_input_words[io->location] : 0;
   *def = NULL;
   if (!words)
      return true;
   if (words > 4 || !options->decode_vertex_input)
      return fail(t, STELA_AIR_ARGUMENT, "vertex input %u has no decoding of its words",
                  io->location);
   char name[32];
   snprintf(name, sizeof(name), "air.vertex_words%u", io->location);
   nir_variable *var = nir_variable_create(t->b.shader, nir_var_shader_in,
                                           glsl_vector_type(GLSL_TYPE_UINT, words), name);
   if (!var)
      return fail(t, STELA_AIR_ALLOCATION, "render IO variable allocation failed");
   var->data.location = VERT_ATTRIB_GENERIC0 + io->location;
   nir_def *lanes = options->decode_vertex_input(&t->b, io->location, nir_load_var(&t->b, var),
                                                 options->resources.binding_data);
   if (!lanes || lanes->num_components != 4 || lanes->bit_size != 32)
      return fail(t, STELA_AIR_ARGUMENT, "vertex input %u does not decode", io->location);
   *def = stage_narrow(&t->b, nir_trim_vector(&t->b, lanes, io->components), io->primitive);
   return true;
}

/* A mesh function's shader info and outputs: its data and primitive
 * indices. */
static bool
mesh_outputs(struct translator *t)
{
   const struct stela_air_stage_entry *entry = &t->render.entry;
   if (entry->max_vertices > t->stage_options->max_mesh_vertices ||
       entry->max_primitives > t->stage_options->max_mesh_primitives)
      return fail(t, STELA_AIR_LIMIT, "mesh capacities exceed the target limits");
   shader_info *info = &t->b.shader->info;
   info->mesh.max_vertices_out = entry->max_vertices;
   info->mesh.max_primitives_out = entry->max_primitives;
   info->mesh.primitive_type = entry->topology == STELA_AIR_POINTS ? MESA_PRIM_POINTS :
                               entry->topology == STELA_AIR_LINES ? MESA_PRIM_LINES :
                               MESA_PRIM_TRIANGLES;
   for (unsigned i = 0; i < entry->output_count; i++) {
      if (!(t->outputs[i] = stage_variable(t, &t->render.outputs[i], true)))
         return false;
   }
   unsigned corners = mesa_vertices_per_prim(info->mesh.primitive_type);
   t->indices = nir_variable_create(t->b.shader, nir_var_shader_out,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT, corners), entry->max_primitives, 0),
      "air.primitive_indices");
   if (!t->indices)
      return fail(t, STELA_AIR_ALLOCATION, "mesh index variable allocation failed");
   t->indices->data.location = VARYING_SLOT_PRIMITIVE_INDICES;
   t->indices->data.per_primitive = true;
   return true;
}

/* The arguments of an object or mesh function, and a mesh function's
 * outputs. The grid or primitive count lives in threadgroup memory: Metal
 * lets any thread set it, while Vulkan takes it uniformly at the end. */
static bool
mesh_stage_arguments(struct translator *t)
{
   if (!compute_arguments(t, &t->render.arguments))
      return false;
   bool object = t->stage_options->stage == STELA_AIR_OBJECT;
   t->mesh_state = nir_variable_create(t->b.shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint_type(), object ? 3 : 1, 0), "air.mesh_state");
   if (!t->mesh_state)
      return fail(t, STELA_AIR_ALLOCATION, "mesh state allocation failed");
   return object || mesh_outputs(t);
}

/* A flat int output an amplified vertex function writes for its view. */
static nir_variable *
view_output(struct translator *t, gl_varying_slot slot, const char *name)
{
   nir_variable *var = nir_variable_create(t->b.shader, nir_var_shader_out, glsl_int_type(), name);
   if (!var)
      fail(t, STELA_AIR_ALLOCATION, "render IO variable allocation failed");
   else
      var->data.location = slot;
   return var;
}

/* A call of a post-tessellation vertex function's control point function:
 * the attributes of the control point its first operand indexes. Members
 * the function never reads stay undefined. */
static bool
control_point_call(struct translator *t, LLVMValueRef inst, LLVMValueRef callee, bool *lowered)
{
   *lowered = callee && callee == t->render.control_point_function;
   if (!*lowered)
      return true;
   const struct stela_air_stage_io *io = t->render.inputs;
   while (io->semantic != STELA_AIR_CONTROL_POINTS)
      io++;
   if (LLVMGetNumArgOperands(inst) != 2 || LLVMGetInstructionCallConv(inst) != LLVMCCallConv ||
       LLVMGetOperand(inst, 1) != LLVMGetParam(t->entry, io->argument))
      return fail(t, STELA_AIR_INSTRUCTION, "control points are read through their argument");
   nir_def *point = get_value(t, LLVMGetOperand(inst, 0));
   struct value *v = point ? new_aggregate(t, io->member_count, NULL) : NULL;
   if (!v)
      return false;
   for (unsigned i = 0; i < io->member_count; i++) {
      const struct stela_air_struct_member *member = &io->members[i];
      if (member->unused)
         continue;
      nir_def *def = stage_in_attribute(t, member->location, t->patch_ids, point, member->primitive,
                                        member->components);
      if (!def)
         return false;
      struct value *leaf = rzalloc(t->values, struct value);
      if (!leaf)
         return fail(t, STELA_AIR_ALLOCATION, "control point allocation failed");
      *leaf = (struct value){.def = def, .components = def->num_components, .bits = def->bit_size};
      v->elements[i] = leaf;
   }
   return _mesa_hash_table_insert(t->values, inst, v) != NULL ||
      fail(t, STELA_AIR_ALLOCATION, "control point map allocation failed");
}

/* An air.interpolate_<place>_<perspective>.<type> call: the interpolant
 * argument its first operand names, sampled at the pixel center, the
 * centroid, a sample or an offset. Metal measures offsets from the pixel's
 * upper-left corner and NIR from its center, as SPIRV-Cross's MSL backend
 * accounts for. */
static bool
interpolant_call(struct translator *t, LLVMValueRef inst, const char *name, size_t name_length,
                 bool *lowered)
{
   static const char prefix[] = "air.interpolate_";
   *lowered = name_length > strlen(prefix) && !memcmp(name, prefix, strlen(prefix));
   if (!*lowered)
      return true;
   enum { CENTER, CENTROID, SAMPLE, OFFSET };
   static const char *const places[] = {
      [CENTER] = "center_", [CENTROID] = "centroid_", [SAMPLE] = "sample_", [OFFSET] = "offset_",
   };
   const char *rest = name + strlen(prefix);
   size_t left = name_length - strlen(prefix);
   unsigned place = 0;
   while (place < ARRAY_SIZE(places) &&
          (left < strlen(places[place]) || memcmp(rest, places[place], strlen(places[place]))))
      place++;
   if (place == ARRAY_SIZE(places))
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported interpolation %.*s", (int)name_length, name);
   rest += strlen(places[place]);
   left -= strlen(places[place]);
   static const char perspective[] = "perspective.", linear[] = "no_perspective.";
   enum stela_air_interpolation interpolation =
      left > strlen(perspective) && !memcmp(rest, perspective, strlen(perspective)) ?
         STELA_AIR_PERSPECTIVE :
      left > strlen(linear) && !memcmp(rest, linear, strlen(linear)) ? STELA_AIR_NO_PERSPECTIVE :
                                                                        STELA_AIR_INTERPOLATION_NONE;
   const struct stela_air_stage_io *io = NULL;
   unsigned operands = place == SAMPLE || place == OFFSET ? 2 : 1;
   for (unsigned i = 0; LLVMGetNumArgOperands(inst) == operands && i < t->render.entry.input_count;
        i++) {
      if (t->render.inputs[i].interpolant &&
          LLVMGetOperand(inst, 0) == LLVMGetParam(t->entry, t->render.inputs[i].argument))
         io = &t->render.inputs[i];
   }
   if (!io)
      return fail(t, STELA_AIR_INSTRUCTION, "interpolants are read through their argument");
   LLVMTypeRef type = LLVMTypeOf(inst);
   bool vector = LLVMGetTypeKind(type) == LLVMVectorTypeKind;
   LLVMTypeKind scalar = LLVMGetTypeKind(vector ? LLVMGetElementType(type) : type);
   if (interpolation != io->interpolation || (vector ? LLVMGetVectorSize(type) : 1) != io->components ||
       scalar != (io->primitive == STELA_AIR_HALF ? LLVMHalfTypeKind : LLVMFloatTypeKind))
      return fail(t, STELA_AIR_TYPE, "interpolation of another interpolant");
   nir_variable *var =
      nir_find_variable_with_location(t->b.shader, nir_var_shader_in, stage_location(t, io));
   nir_def *deref = &nir_build_deref_var(&t->b, var)->def, *def;
   if (place == CENTROID) {
      def = nir_interp_deref_at_centroid(&t->b, io->components, 32, deref);
   } else if (place == SAMPLE) {
      nir_def *sample = get_value(t, LLVMGetOperand(inst, 1));
      if (!sample || sample->num_components != 1 || sample->bit_size != 32)
         return fail(t, STELA_AIR_TYPE, "interpolation sample is one uint");
      def = nir_interp_deref_at_sample(&t->b, io->components, 32, deref, sample);
   } else {
      nir_def *offset = place == CENTER ? nir_imm_vec2(&t->b, 0.0f, 0.0f) :
                                          get_value(t, LLVMGetOperand(inst, 1));
      if (!offset || offset->num_components != 2 || offset->bit_size != 32)
         return fail(t, STELA_AIR_TYPE, "interpolation offset is one float2");
      if (place == OFFSET)
         offset = nir_fadd_imm(&t->b, offset, -0.5);
      def = nir_interp_deref_at_offset(&t->b, io->components, 32, deref, offset);
   }
   return put_value(t, inst, stage_narrow(&t->b, def, io->primitive), 0);
}

/* A fragment function runs once per sample when, its function constants
 * folded, it still reads its sample ID or an input interpolated at its
 * sample; declaring them alone does not. A copy finds the live reads, so
 * the translation keeps its own form. */
static void
shade_per_sample(nir_shader *shader)
{
   nir_shader *live = nir_shader_clone(NULL, shader);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, live, nir_opt_constant_folding);
      NIR_PASS(progress, live, nir_opt_dead_cf);
      NIR_PASS(progress, live, nir_opt_dce);
   } while (progress);
   bool per_sample = false;
   nir_foreach_function_impl(impl, live) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
            if (intrinsic->intrinsic == nir_intrinsic_load_sample_id ||
                intrinsic->intrinsic == nir_intrinsic_load_sample_pos) {
               per_sample = true;
            } else if (intrinsic->intrinsic == nir_intrinsic_load_deref) {
               nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(intrinsic->src[0]));
               per_sample |= var && var->data.mode == nir_var_shader_in && var->data.sample;
            }
         }
      }
   }
   ralloc_free(live);
   shader->info.fs.uses_sample_shading = per_sample;
}

/* Whether instr reads or writes memory another fragment may also access:
 * a device buffer or a texture. Loads free to reorder read memory nothing
 * writes during the draw, such as the argument tables. */
static bool
shared_memory_access(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
   if (nir_intrinsic_has_access(intrinsic) && nir_intrinsic_access(intrinsic) & ACCESS_CAN_REORDER)
      return false;
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_store_global:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      return true;
   case nir_intrinsic_load_deref:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
      return nir_deref_mode_may_be(nir_src_as_deref(intrinsic->src[0]),
                                   nir_var_mem_global | nir_var_mem_ssbo | nir_var_image);
   default:
      return false;
   }
}

/* Whether src is a storage attachment read's placeholder. */
static bool
placeholder_source(nir_src *src, void *data)
{
   const struct translator *t = data;
   for (unsigned i = 0; i < t->attachment_read_count; i++) {
      if (src->ssa == t->attachment_reads[i].placeholder)
         return false;
   }
   return true;
}

/* Whether instr must follow the overlapping fragments before it: a device
 * buffer or texture access of a function with raster order groups, or a use
 * of a storage attachment's texel. */
static bool
ordered_instr(struct translator *t, nir_instr *instr)
{
   if (t->render.entry.raster_ordered && shared_memory_access(instr))
      return true;
   return t->attachment_read_count && !nir_foreach_src(instr, placeholder_source, t);
}

/* Storage attachment n's image at the cursor; a two-plane attachment's
 * second plane, `plane` 1, follows its first. */
static nir_deref_instr *
attachment_plane(struct translator *t, unsigned n, unsigned plane)
{
   nir_builder *b = &t->b;
   return nir_build_deref_array(b, nir_build_deref_var(b, t->attachment_heaps[n]),
                                nir_iadd_imm(b, t->attachment_indices[n], plane));
}

static nir_deref_instr *
attachment_image(struct translator *t, unsigned n)
{
   return attachment_plane(t, n, 0);
}

/* The texel under the fragment: its pixel, the layer it renders to and, of
 * multisampled attachments, its sample. */
static void
attachment_texel(struct translator *t, nir_def **coord, nir_def **sample)
{
   nir_builder *b = &t->b;
   nir_def *xy = nir_f2i32(b, nir_trim_vector(b, nir_load_frag_coord(b), 2));
   *coord = nir_vec4(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1), nir_load_layer_id(b),
                     nir_imm_int(b, 0));
   *sample = t->stage_options->samples > 1 ? nir_load_sample_id(b) : nir_imm_int(b, 0);
}

/* The texel bits of storage attachment n's `plane`, a 32-bit word per four
 * bytes of a texel of `format`. */
static nir_def *
load_plane(struct translator *t, unsigned n, unsigned plane, enum pipe_format format)
{
   nir_builder *b = &t->b;
   nir_def *coord, *sample;
   attachment_texel(t, &coord, &sample);
   bool multisample = t->stage_options->samples > 1;
   nir_def *texel = nir_image_deref_load(b, 4, 32, &attachment_plane(t, n, plane)->def, coord, sample,
                                         nir_imm_int(b, 0),
                                         .image_dim = multisample ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                                         .image_array = true, .format = PIPE_FORMAT_NONE,
                                         .access = ACCESS_COHERENT, .dest_type = nir_type_uint32);
   return nir_trim_vector(b, texel, DIV_ROUND_UP(util_format_get_blocksizebits(format), 32));
}

/* Storage attachment n's texel bits, a 32-bit word per four bytes. */
static nir_def *
load_attachment(struct translator *t, unsigned n)
{
   return load_plane(t, n, 0, storage_attachment(t, n)->format);
}

static void
store_plane(struct translator *t, unsigned n, unsigned plane, nir_def *bits)
{
   nir_builder *b = &t->b;
   nir_def *coord, *sample;
   attachment_texel(t, &coord, &sample);
   bool multisample = t->stage_options->samples > 1;
   nir_def *words = nir_pad_vector_imm_int(b, bits, 0, 4);
   nir_image_deref_store(b, &attachment_plane(t, n, plane)->def, coord, sample, words, nir_imm_int(b, 0),
                         .image_dim = multisample ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                         .image_array = true, .format = PIPE_FORMAT_NONE,
                         .access = ACCESS_COHERENT, .src_type = nir_type_uint32);
}

static void
store_attachment(struct translator *t, unsigned n, nir_def *bits)
{
   store_plane(t, n, 0, bits);
}

/* A two-plane storage attachment's texel as RGBA: its first plane's raw
 * byte and its second's two, converted as its texture samples them. */
static nir_def *
load_planes(struct translator *t, unsigned n)
{
   nir_builder *b = &t->b;
   const struct stela_air_attachment_access *access = storage_attachment(t, n);
   nir_def *first = nir_format_unpack_rgba(b, load_plane(t, n, 0, PIPE_FORMAT_R8_UNORM),
                                           PIPE_FORMAT_R8_UNORM);
   nir_def *second = nir_format_unpack_rgba(b, load_plane(t, n, 1, PIPE_FORMAT_R8G8_UNORM),
                                            PIPE_FORMAT_R8G8_UNORM);
   return stela_air_planes_rgba(b, first, second, nir_imm_int(b, (int)access->planes));
}

/* Writes a two-plane storage attachment's RGBA `color`: its first plane's
 * byte and its second's two. */
static void
store_planes(struct translator *t, unsigned n, nir_def *color)
{
   nir_builder *b = &t->b;
   const struct stela_air_attachment_access *access = storage_attachment(t, n);
   nir_def *first, *second;
   stela_air_rgb_planes(b, color, access->planes & STELA_AIR_PLANES_YCBCR, &first, &second);
   nir_def *zero = nir_imm_float(b, 0.0f);
   store_plane(t, n, 0,
               nir_format_pack_rgba(b, PIPE_FORMAT_R8_UNORM, nir_vec4(b, first, zero, zero, zero)));
   store_plane(t, n, 1,
               nir_format_pack_rgba(b, PIPE_FORMAT_R8G8_UNORM,
                                    nir_vec4(b, nir_channel(b, second, 0),
                                             nir_channel(b, second, 1), zero, zero)));
}

/* The bits of format's channel `rgba` (red, green, blue or alpha) in each of
 * its texel's words; zero for a channel it lacks. */
static void
channel_bits(enum pipe_format format, unsigned rgba, uint32_t words[4])
{
   const struct util_format_description *desc = util_format_description(format);
   memset(words, 0, 4 * sizeof(*words));
   for (unsigned c = 0; c < desc->nr_channels; c++) {
      if (desc->swizzle[rgba] != PIPE_SWIZZLE_X + c)
         continue;
      unsigned shift = desc->channel[c].shift, size = desc->channel[c].size;
      words[shift / 32] |= (uint32_t)BITFIELD64_MASK(size) << (shift % 32);
   }
}

/* A factor of Metal's blend equation for the color or alpha part. */
static nir_def *
blend_factor(struct translator *t, enum stela_air_blend_factor factor, bool alpha,
             nir_def *source, nir_def *destination, nir_def *constant, nir_def *source1)
{
   nir_builder *b = &t->b;
   unsigned components = alpha ? 1 : 3;
   nir_def *one = nir_imm_float(b, 1.0f);
   nir_def *part;
#define PART(v) (alpha ? nir_channel(b, (v), 3) : nir_trim_vector(b, (v), 3))
#define ALPHA(v) (nir_channel(b, (v), 3))
   switch (factor) {
   case STELA_AIR_BLEND_ZERO: part = nir_imm_float(b, 0.0f); break;
   case STELA_AIR_BLEND_ONE: part = one; break;
   case STELA_AIR_BLEND_SOURCE_COLOR: part = PART(source); break;
   case STELA_AIR_BLEND_ONE_MINUS_SOURCE_COLOR: part = nir_fsub(b, one, PART(source)); break;
   case STELA_AIR_BLEND_SOURCE_ALPHA: part = ALPHA(source); break;
   case STELA_AIR_BLEND_ONE_MINUS_SOURCE_ALPHA: part = nir_fsub(b, one, ALPHA(source)); break;
   case STELA_AIR_BLEND_DESTINATION_COLOR: part = PART(destination); break;
   case STELA_AIR_BLEND_ONE_MINUS_DESTINATION_COLOR: part = nir_fsub(b, one, PART(destination)); break;
   case STELA_AIR_BLEND_DESTINATION_ALPHA: part = ALPHA(destination); break;
   case STELA_AIR_BLEND_ONE_MINUS_DESTINATION_ALPHA: part = nir_fsub(b, one, ALPHA(destination)); break;
   case STELA_AIR_BLEND_SOURCE_ALPHA_SATURATED:
      part = alpha ? one : nir_fmin(b, ALPHA(source), nir_fsub(b, one, ALPHA(destination)));
      break;
   case STELA_AIR_BLEND_COLOR_FACTOR: part = PART(constant); break;
   case STELA_AIR_BLEND_ONE_MINUS_COLOR_FACTOR: part = nir_fsub(b, one, PART(constant)); break;
   case STELA_AIR_BLEND_ALPHA_FACTOR: part = ALPHA(constant); break;
   case STELA_AIR_BLEND_ONE_MINUS_ALPHA_FACTOR: part = nir_fsub(b, one, ALPHA(constant)); break;
   case STELA_AIR_BLEND_SOURCE1_COLOR: part = PART(source1); break;
   case STELA_AIR_BLEND_ONE_MINUS_SOURCE1_COLOR: part = nir_fsub(b, one, PART(source1)); break;
   case STELA_AIR_BLEND_SOURCE1_ALPHA: part = ALPHA(source1); break;
   case STELA_AIR_BLEND_ONE_MINUS_SOURCE1_ALPHA: part = nir_fsub(b, one, ALPHA(source1)); break;
   default: UNREACHABLE("validated blend factor");
   }
#undef PART
#undef ALPHA
   return part->num_components == components ? part : nir_replicate(b, part, components);
}

static nir_def *
blend_part(struct translator *t, const struct stela_air_attachment_access *access, bool alpha,
           nir_def *source, nir_def *destination, nir_def *constant, nir_def *source1)
{
   nir_builder *b = &t->b;
   enum stela_air_blend_op op = alpha ? access->alpha_op : access->rgb_op;
   nir_def *s = alpha ? nir_channel(b, source, 3) : nir_trim_vector(b, source, 3);
   nir_def *d = alpha ? nir_channel(b, destination, 3) : nir_trim_vector(b, destination, 3);
   if (op == STELA_AIR_BLEND_MIN)
      return nir_fmin(b, s, d);
   if (op == STELA_AIR_BLEND_MAX)
      return nir_fmax(b, s, d);
   nir_def *sf = nir_fmul(b, s, blend_factor(t, alpha ? access->source_alpha : access->source_rgb,
                                             alpha, source, destination, constant, source1));
   nir_def *df = nir_fmul(b, d, blend_factor(t, alpha ? access->destination_alpha :
                                                        access->destination_rgb,
                                             alpha, source, destination, constant, source1));
   return op == STELA_AIR_BLEND_ADD ? nir_fadd(b, sf, df) :
          op == STELA_AIR_BLEND_SUBTRACT ? nir_fsub(b, sf, df) : nir_fsub(b, df, sf);
}

/* Metal's blend of source over destination, both RGBA, for a normalized or
 * float attachment: a normalized one clamps its inputs to its range. */
static nir_def *
blend(struct translator *t, const struct stela_air_attachment_access *access, nir_def *source,
      nir_def *destination, nir_def *source1)
{
   nir_builder *b = &t->b;
   const struct util_format_description *desc = util_format_description(access->format);
   int channel = util_format_get_first_non_void_channel(access->format);
   nir_def *constant = t->blend_color;
   if (channel >= 0 && desc->channel[channel].normalized) {
      bool snorm = desc->channel[channel].type == UTIL_FORMAT_TYPE_SIGNED;
      nir_def *low = nir_imm_float(b, snorm ? -1.0f : 0.0f), *high = nir_imm_float(b, 1.0f);
      source = nir_fclamp(b, source, low, high);
      if (constant)
         constant = nir_fclamp(b, constant, low, high);
      source1 = nir_fclamp(b, source1, low, high);
   }
   nir_def *rgb = blend_part(t, access, false, source, destination, constant, source1);
   nir_def *a = blend_part(t, access, true, source, destination, constant, source1);
   return nir_vector_insert_imm(b, nir_pad_vector(b, rgb, 4), a, 3);
}

/* Apple GPUs draw points at most 511 pixels wide, as Metal's feature set
 * tables give: a larger point size draws a point that wide. NVIDIA's reach
 * 2048. */
#define STELA_AIR_MAX_POINT_SIZE 511.0f

static nir_def *
largest_point(nir_builder *b, nir_def *size)
{
   return nir_fmin(b, size, nir_imm_floatN_t(b, STELA_AIR_MAX_POINT_SIZE, size->bit_size));
}

/* The value of the extended-range code nearest v, ties to even, among codes
 * low to high: (code - 384) / 510, through the sRGB curve mirrored below
 * zero for the sRGB encodings, as metal_abi.c's xr_code and xr_value encode
 * and decode texels. */
nir_def *
stela_air_extended_range_code(nir_builder *b, nir_def *v, bool srgb, float low, float high)
{
   nir_def *zero = nir_imm_float(b, 0.0f);
   if (srgb) {
      nir_def *m = nir_fabs(b, v);
      nir_def *curve = nir_fadd_imm(
         b, nir_fmul_imm(b, nir_fpow(b, m, nir_imm_float(b, 1.0f / 2.4f)), 1.055), -0.055);
      nir_def *encoded = nir_bcsel(b, nir_fge(b, nir_imm_float(b, 0.0031308f), m),
                                   nir_fmul_imm(b, m, 12.92), curve);
      v = nir_bcsel(b, nir_flt(b, v, zero), nir_fneg(b, encoded), encoded);
   }
   nir_def *code = nir_fround_even(b, nir_fadd_imm(b, nir_fmul_imm(b, v, 510.0), 384.0));
   code = nir_fmin(b, nir_fmax(b, code, nir_imm_float(b, low)), nir_imm_float(b, high));
   v = nir_fmul_imm(b, nir_fadd_imm(b, code, -384.0), 1.0 / 510.0);
   if (!srgb)
      return v;
   nir_def *m = nir_fabs(b, v);
   nir_def *curve = nir_fpow(b, nir_fmul_imm(b, nir_fadd_imm(b, m, 0.055), 1.0 / 1.055),
                             nir_imm_float(b, 2.4f));
   nir_def *linear = nir_bcsel(b, nir_fge(b, nir_imm_float(b, 0.04045f), m),
                               nir_fmul_imm(b, m, 1.0 / 12.92), curve);
   return nir_bcsel(b, nir_flt(b, v, zero), nir_fneg(b, linear), linear);
}

/* Whether color attachment location is a blended two-plane extended-range
 * one (STELA_AIR_EXTENDED_RANGE_PLANES_HELD), whose output Apple GPUs blend
 * in halves. */
static bool
blended_in_halves(const struct translator *t, unsigned location)
{
   unsigned kinds = t->stage_options ? t->stage_options->extended_range : 0;
   unsigned kind = location < AIR_MAX_RENDER_TARGETS
                      ? kinds >> (location * STELA_AIR_EXTENDED_RANGE_BITS) &
                           BITFIELD_MASK(STELA_AIR_EXTENDED_RANGE_BITS)
                      : STELA_AIR_EXTENDED_RANGE_NONE;
   return kind == STELA_AIR_EXTENDED_RANGE_PLANES_HELD ||
          kind == STELA_AIR_EXTENDED_RANGE_PLANES_HELD_SRGB;
}

/* An output of color attachment location as its extended-range format
 * stores it: each color channel the value of its nearest code, 0 to 1023,
 * and alpha its nearest code's in [0, 1], codes 384 to 894, or for the
 * two-plane formats its nearest 8-bit value. The half-float texels that
 * hold the codes then hold that value rounded to its nearest half, which
 * reads back as the same code: NVIDIA's color output narrows by
 * truncating, which can store a value nearer the code below. An unblended
 * two-plane output is only rounded to its nearest half, as Apple GPUs'
 * tile memory holds it, and a blended one is held from that half. Other
 * attachments' outputs are returned as they are. */
static nir_def *
clamp_extended_range(struct translator *t, unsigned location, nir_def *value)
{
   unsigned kinds = t->stage_options ? t->stage_options->extended_range : 0;
   unsigned kind = location < AIR_MAX_RENDER_TARGETS
                      ? kinds >> (location * STELA_AIR_EXTENDED_RANGE_BITS) &
                           BITFIELD_MASK(STELA_AIR_EXTENDED_RANGE_BITS)
                      : STELA_AIR_EXTENDED_RANGE_NONE;
   /* The pipeline gives these attachments only floating-point outputs. */
   if (kind == STELA_AIR_EXTENDED_RANGE_NONE || value->bit_size < 16)
      return value;
   nir_builder *b = &t->b;
   bool srgb = kind == STELA_AIR_EXTENDED_RANGE_SRGB ||
               kind == STELA_AIR_EXTENDED_RANGE_PLANES_HELD_SRGB;
   bool planes = kind >= STELA_AIR_EXTENDED_RANGE_PLANES_HELD;
   nir_def *channels[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < value->num_components; i++) {
      nir_def *channel = nir_f2f32(b, nir_channel(b, value, i));
      /* A blended two-plane output is held from its half
       * (blended_in_halves). */
      if (planes && kind != STELA_AIR_EXTENDED_RANGE_PLANES && value->bit_size == 32)
         channel = nir_f2f32(b, nir_f2f16_rtne(b, channel));
      if (kind == STELA_AIR_EXTENDED_RANGE_PLANES) {
         /* Held as it is. */
      } else if (i == 3 && planes) {
         nir_def *steps = nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, channel), 255.0));
         channel = nir_fmul_imm(b, steps, 1.0 / 255.0);
      } else {
         channel = i == 3 ? stela_air_extended_range_code(b, channel, false, 384.0f, 894.0f)
                          : stela_air_extended_range_code(b, channel, srgb, 0.0f, 1023.0f);
      }
      channel = nir_f2f16_rtne(b, channel);
      channels[i] = value->bit_size == 32 ? nir_f2f32(b, channel) : channel;
   }
   return nir_vec(b, channels, value->num_components);
}

/* Writes storage attachment n with the function's output: blended, written
 * where its mask allows, the rest of the texel kept. */
static void
write_attachment(struct translator *t, unsigned n, nir_def *destination, nir_def *bits)
{
   nir_builder *b = &t->b;
   const struct stela_air_attachment_access *access = storage_attachment(t, n);
   nir_variable *output = t->attachment_outputs[n];
   unsigned mask = access->write_mask & 0xf;
   if (!output || !mask)
      return;
   bool integer = util_format_is_pure_integer(access->format);
   nir_def *value = nir_load_var(b, output);
   nir_def *defaults[4] = {
      integer ? nir_imm_int(b, 0) : nir_imm_float(b, 0.0f),
      integer ? nir_imm_int(b, 0) : nir_imm_float(b, 0.0f),
      integer ? nir_imm_int(b, 0) : nir_imm_float(b, 0.0f),
      integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f),
   };
   nir_def *channels[4];
   for (unsigned i = 0; i < 4; i++)
      channels[i] = i < value->num_components ? nir_channel(b, value, i) : defaults[i];
   nir_def *source = nir_vec(b, channels, 4);
   /* The second source, of attachment 0 alone; zero, as Vulkan's undefined
    * one, without an index(1) output. */
   nir_def *source1 = nir_imm_vec4(b, 0.0f, 0.0f, 0.0f, 0.0f);
   if (n == 0 && t->attachment_source1 && !integer) {
      nir_def *second = nir_load_var(b, t->attachment_source1);
      nir_def *parts[4];
      for (unsigned i = 0; i < 4; i++)
         parts[i] = i < second->num_components ? nir_f2f32(b, nir_channel(b, second, i)) : defaults[i];
      source1 = nir_vec(b, parts, 4);
   }
   if (access->blend && !integer) {
      /* Apple GPUs blend a two-plane extended-range output as their tile
       * memory holds it, in halves: 0.05 blends as 0.0499878, code 409. */
      if (blended_in_halves(t, n)) {
         source = nir_f2f32(b, nir_f2f16_rtne(b, source));
         source1 = nir_f2f32(b, nir_f2f16_rtne(b, source1));
      }
      source = clamp_extended_range(t, n, blend(t, access, source, destination, source1));
   }
   if (access->planes) {
      /* The planes hold the color as a whole: unwritten channels keep the
       * texel's, then both planes take the converted color. */
      nir_def *merged[4];
      for (unsigned i = 0; i < 4; i++)
         merged[i] = nir_channel(b, mask & 1u << i ? source : destination, i);
      nir_push_if(b, nir_inot(b, nir_is_helper_invocation(b, 1)));
      store_planes(t, n, nir_vec(b, merged, 4));
      nir_pop_if(b, NULL);
      return;
   }
   nir_def *packed = nir_format_pack_rgba(b, access->format, source);
   if (mask != 0xf) {
      if (access->format == PIPE_FORMAT_R9G9B9E5_FLOAT) {
         /* Its channels share their exponent: the kept ones pack again. */
         nir_def *merged[4];
         for (unsigned i = 0; i < 4; i++)
            merged[i] = nir_channel(b, mask & 1u << i ? source : destination, i);
         packed = nir_format_pack_rgba(b, access->format, nir_vec(b, merged, 4));
      } else {
         uint32_t written[4] = {0};
         for (unsigned i = 0; i < 4; i++) {
            if (!(mask & 1u << i))
               continue;
            uint32_t words[4];
            channel_bits(access->format, i, words);
            for (unsigned w = 0; w < 4; w++)
               written[w] |= words[w];
         }
         nir_def *keep[4];
         for (unsigned w = 0; w < packed->num_components; w++) {
            keep[w] = nir_ior(b, nir_iand_imm(b, nir_channel(b, packed, w), written[w]),
                              nir_iand_imm(b, nir_channel(b, bits, w), ~written[w]));
         }
         packed = nir_vec(b, keep, packed->num_components);
      }
   }
   /* A helper invocation writes nothing. */
   nir_push_if(b, nir_inot(b, nir_is_helper_invocation(b, 1)));
   store_attachment(t, n, packed);
   nir_pop_if(b, NULL);
}

/* Turns storage attachment outputs into the function's own variables, which
 * write_attachment reads where the function ends. */
static bool
attachment_outputs(struct translator *t)
{
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      if (io->absent || !t->outputs[i])
         continue;
      if ((io->semantic == STELA_AIR_DEPTH || io->semantic == STELA_AIR_STENCIL ||
           io->semantic == STELA_AIR_SAMPLE_MASK) && !t->tests_fragments)
         return fail(t, STELA_AIR_ARGUMENT,
                     "storage attachments run after the fragment tests, so their function may not write depth, stencil or coverage");
      if (io->semantic != STELA_AIR_RENDER_TARGET || !storage_attachment(t, io->location))
         continue;
      if (io->index && io->location)
         return fail(t, STELA_AIR_ARGUMENT, "only color attachment 0 blends a second source");
      nir_variable *var = t->outputs[i];
      exec_node_remove(&var->node);
      var->data.mode = nir_var_function_temp;
      exec_list_push_tail(&t->b.impl->locals, &var->node);
      if (io->index)
         t->attachment_source1 = var;
      else
         t->attachment_outputs[io->location] = var;
   }
   nir_fixup_deref_modes(t->b.shader);
   return true;
}

static bool repeat(void *translator, nir_def *count, unsigned carried, nir_def **values,
                   bool (*body)(void *data, nir_def *index, nir_def **values), void *data);
static struct value *imageblock_authority(struct translator *t, struct value *authority);

/* Refuses the outputs the fragment tests read, which run before a function
 * that writes memory in raster order. */
static bool
tested_outputs(struct translator *t)
{
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      if (!io->absent && (io->semantic == STELA_AIR_DEPTH || io->semantic == STELA_AIR_STENCIL ||
                          io->semantic == STELA_AIR_SAMPLE_MASK))
         return fail(t, STELA_AIR_ARGUMENT,
                     "imageblock data is written after the fragment tests, so its function may not write depth, stencil or coverage");
   }
   return true;
}

/* The address of sample `sample` of the pass's imageblock data at pixel,
 * of `samples` per pixel. */
static nir_def *
imageblock_address(struct translator *t, nir_def *pixel, nir_def *sample, nir_def *samples)
{
   nir_builder *b = &t->b;
   nir_def *pass = t->imageblock_pass;
   nir_def *index = nir_iadd(b, nir_imul(b, nir_channel(b, pixel, 1), nir_channel(b, pass, 1)),
                             nir_channel(b, pixel, 0));
   index = nir_iadd(b, nir_imul(b, index, samples), sample);
   nir_def *base = nir_pack_64_2x32_split(b, nir_channel(b, pass, 2), nir_channel(b, pass, 3));
   return nir_iadd(b, base, nir_imul(b, nir_u2u64(b, index), nir_u2u64(b, nir_channel(b, pass, 0))));
}

/* A fragment function's pixel and the samples it covers, for its
 * imageblock data. */
static bool
fragment_imageblock_prelude(struct translator *t)
{
   if (t->fragment_imageblock)
      return true;
   nir_builder *b = &t->b;
   nir_def *samples = device_value(t, STELA_AIR_SAMPLE_COUNT);
   if (!samples || !(t->imageblock_pass = device_value(t, STELA_AIR_IMAGEBLOCK)))
      return false;
   t->imageblock_pixel = nir_f2u32(b, nir_trim_vector(b, nir_load_frag_coord(b), 2));
   nir_def *all = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), samples), -1);
   t->imageblock_mask = nir_iand(b, nir_load_sample_mask_in(b), all);
   t->fragment_imageblock = true;
   return true;
}

static nir_def *
fragment_samples(struct translator *t)
{
   return device_value(t, STELA_AIR_SAMPLE_COUNT);
}

/* A fragment's [[imageblock_data]] input: its first covered sample's, which
 * its interlock section loads. */
static bool
imageblock_data_input(struct translator *t, LLVMValueRef param)
{
   if (!fragment_imageblock_prelude(t))
      return false;
   nir_builder *b = &t->b;
   nir_def *samples = fragment_samples(t);
   if (!samples)
      return false;
   nir_def *first = nir_umin(b, nir_find_lsb(b, t->imageblock_mask), nir_iadd_imm(b, samples, -1));
   struct value address = {
      .def = imageblock_address(t, t->imageblock_pixel, first, samples),
      .access = STELA_AIR_READ, .components = 1, .bits = 64};
   LLVMTypeRef type = LLVMTypeOf(param);
   struct value *loaded = aggregate_access(t, &address, type, 0, NULL, ACCESS_COHERENT,
                                           LLVMABIAlignmentOfType(t->layout, type));
   return loaded && (_mesa_hash_table_insert(t->values, param, loaded) != NULL ||
                     fail(t, STELA_AIR_ALLOCATION, "imageblock data map failed"));
}

struct imageblock_store {
   struct translator *t;
   const struct value *data;
   LLVMTypeRef type;
   nir_def *samples;
};

static bool
imageblock_store_sample(void *data, nir_def *sample, nir_def **values)
{
   (void)values;
   const struct imageblock_store *store = data;
   struct translator *t = store->t;
   nir_builder *b = &t->b;
   nir_def *covered = nir_i2b(b, nir_iand(b, nir_ushr(b, t->imageblock_mask, sample), nir_imm_int(b, 1)));
   struct memory_guard guard;
   if (!memory_guard_begin(t, nir_iand(b, covered, nir_inot(b, nir_is_helper_invocation(b, 1))), &guard))
      return false;
   struct value address = {
      .def = imageblock_address(t, t->imageblock_pixel, sample, store->samples),
      .access = STELA_AIR_WRITE, .components = 1, .bits = 64};
   if (!aggregate_access(t, &address, store->type, 0, store->data, ACCESS_COHERENT,
                         LLVMABIAlignmentOfType(t->layout, store->type)))
      return false;
   memory_guard_end(t, &guard, NULL);
   return true;
}

/* A fragment's [[imageblock_data]] output, which every sample it covers
 * holds once its interlock section ends. */
static bool
imageblock_data_output(struct translator *t, const struct value *data, LLVMTypeRef type)
{
   if (!data->aggregate_count)
      return fail(t, STELA_AIR_TYPE, "imageblock data of an unknown form");
   if (!fragment_imageblock_prelude(t))
      return false;
   struct imageblock_store store = {.t = t, .data = data, .type = type, .samples = fragment_samples(t)};
   return store.samples && repeat(t, store.samples, 0, NULL, imageblock_store_sample, &store);
}

/* The fragment tests' comparison `function` (MTLCompareFunction, a mask of
 * less, equal and greater) of a against b. */
static nir_def *
compare_fragment(nir_builder *b, nir_def *function, nir_def *lt, nir_def *eq, nir_def *gt)
{
   nir_def *bit = nir_imm_int(b, 0);
   nir_def *result = nir_imm_false(b);
   nir_def *outcomes[3] = {lt, eq, gt};
   for (unsigned i = 0; i < 3; i++) {
      bit = nir_i2b(b, nir_iand_imm(b, nir_ushr_imm(b, function, i), 1));
      result = nir_ior(b, result, nir_iand(b, bit, outcomes[i]));
   }
   return result;
}

/* MTLStencilOperation `op` on the 8-bit value with reference ref. */
static nir_def *
stencil_operation(nir_builder *b, nir_def *op, nir_def *value, nir_def *ref)
{
   nir_def *results[8] = {
      value,
      nir_imm_int(b, 0),
      ref,
      nir_umin(b, nir_iadd_imm(b, value, 1), nir_imm_int(b, 0xff)),
      nir_bcsel(b, nir_ieq_imm(b, value, 0), value, nir_iadd_imm(b, value, -1)),
      nir_iand_imm(b, nir_inot(b, value), 0xff),
      nir_iand_imm(b, nir_iadd_imm(b, value, 1), 0xff),
      nir_iand_imm(b, nir_iadd_imm(b, value, -1), 0xff),
   };
   nir_def *result = results[7];
   for (int i = 6; i >= 0; i--)
      result = nir_bcsel(b, nir_ieq_imm(b, op, i), results[i], result);
   return result;
}

/* Turns a fragment output into the function's own variable, whose value
 * the fragment tests read where the function ends; NULL for none. */
static nir_variable *
tested_output(struct translator *t, enum stela_air_stage_semantic semantic)
{
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      if (io->absent || !t->outputs[i] || io->semantic != semantic)
         continue;
      nir_variable *var = t->outputs[i];
      exec_node_remove(&var->node);
      var->data.mode = nir_var_function_temp;
      exec_list_push_tail(&t->b.impl->locals, &var->node);
      return var;
   }
   return NULL;
}

/* A copy of the pass's depth or stencil attachment, resolved at the cursor:
 * its heap image and descriptor index. */
struct tested_image {
   nir_variable *heap;
   nir_def *index;
};

static bool
tested_image(struct translator *t, bool stencil, struct tested_image *image)
{
   bool multisample = t->stage_options->samples > 1;
   const struct glsl_type *type =
      glsl_image_type(multisample ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D, true, GLSL_TYPE_UINT);
   nir_deref_instr *deref = t->options->bind_depth_stencil(&t->b, stencil, type, t->options->binding_data);
   nir_deref_instr *parent = deref && deref->deref_type == nir_deref_type_array ?
      nir_deref_instr_parent(deref) : NULL;
   if (!parent || parent->deref_type != nir_deref_type_var || !glsl_type_is_image(deref->type))
      return fail(t, STELA_AIR_ARGUMENT, "the pass's %s copy is not a heap image",
                  stencil ? "stencil" : "depth");
   image->heap = parent->var;
   image->index = deref->arr.index.ssa;
   return true;
}

static nir_def *
tested_load(struct translator *t, const struct tested_image *image, nir_def *coord, unsigned sample)
{
   nir_builder *b = &t->b;
   bool multisample = t->stage_options->samples > 1;
   nir_deref_instr *deref = nir_build_deref_array(b, nir_build_deref_var(b, image->heap), image->index);
   nir_def *texel = nir_image_deref_load(b, 4, 32, &deref->def, coord, nir_imm_int(b, sample),
                                         nir_imm_int(b, 0),
                                         .image_dim = multisample ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                                         .image_array = true, .format = PIPE_FORMAT_NONE,
                                         .access = ACCESS_COHERENT, .dest_type = nir_type_uint32);
   return nir_channel(b, texel, 0);
}

static void
tested_store(struct translator *t, const struct tested_image *image, nir_def *coord, unsigned sample,
             nir_def *value)
{
   nir_builder *b = &t->b;
   bool multisample = t->stage_options->samples > 1;
   nir_deref_instr *deref = nir_build_deref_array(b, nir_build_deref_var(b, image->heap), image->index);
   nir_image_deref_store(b, &deref->def, coord, nir_imm_int(b, sample),
                         nir_pad_vector_imm_int(b, value, 0, 4), nir_imm_int(b, 0),
                         .image_dim = multisample ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                         .image_array = true, .format = PIPE_FORMAT_NONE,
                         .access = ACCESS_COHERENT, .src_type = nir_type_uint32);
}

/* What a fragment function testing fragments itself takes where it starts,
 * in uniform control flow, and the outputs the tests read where it ends. */
struct fragment_tests {
   nir_def *state, *range, *positions, *frag, *z, *dzdx, *dzdy;
   struct tested_image depth_image, stencil_image;
   nir_variable *stencil_var, *depth_var, *mask_var;
};

/* The color(0) output a function testing fragments itself turns into
 * coverage, and the color outputs it gives alpha 1: storage attachments'
 * own variables, or the function's outputs; NULL for none. */
static nir_variable *
color_output(struct translator *t, unsigned location)
{
   if (location < AIR_MAX_RENDER_TARGETS && t->attachment_outputs[location])
      return t->attachment_outputs[location];
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      if (!io->absent && t->outputs[i] && io->semantic == STELA_AIR_RENDER_TARGET &&
          io->location == location && !io->index)
         return t->outputs[i];
   }
   return NULL;
}

static bool
fragment_tests_begin(struct translator *t, struct fragment_tests *tests)
{
   if (t->render.entry.early_fragment_tests)
      return fail(t, STELA_AIR_ARGUMENT, "early fragment tests precede fragment tests the function runs itself");
   nir_builder *b = &t->b;
   nir_function_impl *impl = nir_shader_get_entrypoint(b->shader);
   tests->stencil_var = tested_output(t, STELA_AIR_STENCIL);
   tests->depth_var = tested_output(t, STELA_AIR_DEPTH);
   tests->mask_var = tested_output(t, STELA_AIR_SAMPLE_MASK);
   nir_fixup_deref_modes(b->shader);
   b->cursor = nir_before_impl(impl);
   tests->state = t->options->load_device_value(b, STELA_AIR_DEPTH_STENCIL, t->options->binding_data);
   tests->range = t->options->load_device_value(b, STELA_AIR_DEPTH_RANGE, t->options->binding_data);
   tests->positions = t->options->load_device_value(b, STELA_AIR_SAMPLE_POSITIONS, t->options->binding_data);
   if (!tests->state || !tests->range || !tests->positions || tests->state->num_components != 4 ||
       tests->range->num_components != 4 || tests->positions->num_components != 4)
      return fail(t, STELA_AIR_ARGUMENT, "the fragment tests' state is not supplied");
   if (!tested_image(t, false, &tests->depth_image) || !tested_image(t, true, &tests->stencil_image))
      return false;
   tests->frag = nir_load_frag_coord(b);
   tests->z = nir_channel(b, tests->frag, 2);
   tests->dzdx = nir_ddx_fine(b, tests->z);
   tests->dzdy = nir_ddy_fine(b, tests->z);
   return true;
}

/* The samples a fragment keeps before its tests: those it covers, of its
 * sample mask output, and of alpha to coverage, color(0)'s alpha of the
 * pixel's samples, rounded to the nearest, the lowest first. */
static nir_def *
fragment_kept(struct translator *t, struct fragment_tests *tests)
{
   nir_builder *b = &t->b;
   nir_def *kept = tests->mask_var ? nir_load_var(b, tests->mask_var) : nir_imm_int(b, ~0);
   const struct stela_air_stage_options *stage = t->stage_options;
   nir_variable *color = color_output(t, 0);
   if (stage->alpha_to_coverage && color && glsl_get_base_type(glsl_without_array(color->type)) != GLSL_TYPE_INT &&
       glsl_get_base_type(glsl_without_array(color->type)) != GLSL_TYPE_UINT) {
      unsigned samples = MAX2(stage->samples, 1);
      nir_def *value = nir_load_var(b, color);
      nir_def *alpha = value->num_components == 4 ? nir_f2f32(b, nir_channel(b, value, 3)) : nir_imm_float(b, 1.0f);
      nir_def *count = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, alpha), samples)));
      /* (1 << count) - 1, of at most 32 samples. */
      nir_def *mask = nir_bcsel(b, nir_uge_imm(b, count, 32), nir_imm_int(b, ~0),
                                nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), count), -1));
      kept = nir_iand(b, kept, mask);
   }
   return kept;
}

/* Alpha to one for the storage attachments a function writes: their
 * outputs' alpha becomes 1 before they blend, as the pipeline gives the
 * outputs it writes. */
static void
storage_alpha_to_one(struct translator *t)
{
   if (!t->stage_options->alpha_to_one)
      return;
   nir_builder *b = &t->b;
   u_foreach_bit(n, t->storage_attachments) {
      nir_variable *var = t->attachment_outputs[n];
      enum glsl_base_type base = var ? glsl_get_base_type(glsl_without_array(var->type)) : GLSL_TYPE_INT;
      if (!var || base == GLSL_TYPE_INT || base == GLSL_TYPE_UINT || glsl_get_vector_elements(var->type) != 4)
         continue;
      nir_def *value = nir_load_var(b, var);
      nir_def *one = nir_imm_floatN_t(b, 1.0, value->bit_size);
      nir_store_var(b, var, nir_vector_insert_imm(b, value, one, 3), 0xf);
   }
}

/* Runs the fragment tests at the cursor, inside the function's interlock
 * section: each sample the fragment keeps takes the stencil test against
 * the reference, its stencil output's or the encoder's, then the depth test
 * at the sample's depth, and the stencil operation and depth write they
 * choose. Returns the samples that pass. The depth at a sample is the
 * fragment's plane at the sample's position, from the depth and its fine
 * derivatives the function takes where it starts, or the depth it writes. */
static nir_def *
fragment_tests_run(struct translator *t, struct fragment_tests *tests, nir_def *kept)
{
   nir_builder *b = &t->b;
   unsigned samples = MAX2(t->stage_options->samples, 1);
   nir_def *state = tests->state, *range = tests->range, *positions = tests->positions;
   nir_def *frag = tests->frag, *z = tests->z, *dzdx = tests->dzdx, *dzdy = tests->dzdy;
   nir_def *written_depth = tests->depth_var ? nir_load_var(b, tests->depth_var) : NULL;
   nir_def *covered = nir_iand(b, nir_load_sample_mask_in(b), kept);
   nir_def *depth_word = nir_channel(b, state, 0);
   nir_def *format = nir_iand_imm(b, nir_ushr_imm(b, depth_word, STELA_AIR_DEPTH_FORMAT_SHIFT), 3);
   nir_def *has_depth = nir_ine_imm(b, format, STELA_AIR_DEPTH_NONE);
   nir_def *has_stencil = nir_i2b(b, nir_iand_imm(b, depth_word, STELA_AIR_DEPTH_STENCIL_ATTACHED));
   nir_def *writes_depth = nir_iand(b, has_depth, nir_i2b(b, nir_iand_imm(b, depth_word, STELA_AIR_DEPTH_WRITE)));
   nir_def *front = nir_load_front_face(b, 1);
   nir_def *face = nir_bcsel(b, front, nir_channel(b, state, 1), nir_channel(b, state, 2));
   nir_def *references = nir_channel(b, state, 3);
   nir_def *ref = tests->stencil_var ? nir_iand_imm(b, nir_load_var(b, tests->stencil_var), 0xff) :
                  nir_bcsel(b, front, nir_iand_imm(b, references, 0xff),
                            nir_iand_imm(b, nir_ushr_imm(b, references, 8), 0xff));
   nir_def *read_mask = nir_iand_imm(b, nir_ushr_imm(b, face, 16), 0xff);
   nir_def *write_mask = nir_ushr_imm(b, face, 24);
   nir_def *near = nir_channel(b, range, 0), *far = nir_channel(b, range, 1);
   nir_def *low = nir_fmin(b, near, far), *high = nir_fmax(b, near, far);
   nir_def *clamps = nir_i2b(b, nir_iand_imm(b, depth_word, STELA_AIR_DEPTH_CLAMP));
   nir_def *xy = nir_ffloor(b, nir_trim_vector(b, frag, 2));
   nir_def *pixel = nir_f2i32(b, xy);
   nir_def *coord = nir_vec4(b, nir_channel(b, pixel, 0), nir_channel(b, pixel, 1), nir_load_layer_id(b),
                             nir_imm_int(b, 0));

   /* Zero, built ahead of the branches: their phis follow them directly,
    * and each takes it where its branch is skipped. */
   nir_def *passed = nir_imm_int(b, 0);
   /* A helper invocation, or a fragment the function discarded, tests
    * nothing. */
   nir_push_if(b, nir_inot(b, nir_is_helper_invocation(b, 1)));
   nir_def *tested = nir_imm_int(b, 0);
   for (unsigned s = 0; s < samples; s++) {
      nir_push_if(b, nir_i2b(b, nir_iand_imm(b, nir_ushr_imm(b, covered, s), 1)));
      nir_def *depth;
      if (written_depth) {
         depth = written_depth;
      } else {
         nir_def *position = nir_ubitfield_extract_imm(b, nir_channel(b, positions, s / 4), 8 * (s % 4), 8);
         nir_def *sx = nir_fmul_imm(b, nir_u2f32(b, nir_iand_imm(b, position, 0xf)), 1.0 / 16.0);
         nir_def *sy = nir_fmul_imm(b, nir_u2f32(b, nir_ushr_imm(b, position, 4)), 1.0 / 16.0);
         nir_def *dx = nir_fsub(b, nir_fadd(b, nir_channel(b, xy, 0), sx), nir_channel(b, frag, 0));
         nir_def *dy = nir_fsub(b, nir_fadd(b, nir_channel(b, xy, 1), sy), nir_channel(b, frag, 1));
         depth = nir_ffma(b, dzdy, dy, nir_ffma(b, dzdx, dx, z));
      }
      depth = nir_bcsel(b, clamps, nir_fmin(b, nir_fmax(b, depth, low), high), depth);
      depth = nir_fsat(b, depth);
      /* The sample's depth as the attachment holds it, and comparable. */
      nir_def *unorm16 = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, depth, 65535.0)));
      nir_def *unorm24 = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, depth, 16777215.0)));
      nir_def *bits = nir_bcsel(b, nir_ieq_imm(b, format, STELA_AIR_DEPTH_UNORM16), unorm16,
                                nir_bcsel(b, nir_ieq_imm(b, format, STELA_AIR_DEPTH_UNORM24), unorm24, depth));
      nir_def *held = tested_load(t, &tests->depth_image, coord, s);
      held = nir_bcsel(b, nir_ieq_imm(b, format, STELA_AIR_DEPTH_UNORM24), nir_iand_imm(b, held, 0xffffff),
                       nir_bcsel(b, nir_ieq_imm(b, format, STELA_AIR_DEPTH_UNORM16), nir_iand_imm(b, held, 0xffff),
                                 held));
      nir_def *is_float = nir_ieq_imm(b, format, STELA_AIR_DEPTH_FLOAT32);
      nir_def *lt = nir_bcsel(b, is_float, nir_flt(b, depth, held), nir_ult(b, bits, held));
      nir_def *eq = nir_bcsel(b, is_float, nir_feq(b, depth, held), nir_ieq(b, bits, held));
      nir_def *gt = nir_bcsel(b, is_float, nir_flt(b, held, depth), nir_ult(b, held, bits));
      nir_def *depth_passes = nir_ior(b, nir_inot(b, has_depth),
                                      compare_fragment(b, nir_iand_imm(b, depth_word, 7), lt, eq, gt));
      nir_def *stencil = nir_iand_imm(b, tested_load(t, &tests->stencil_image, coord, s), 0xff);
      nir_def *a = nir_iand(b, ref, read_mask), *h = nir_iand(b, stencil, read_mask);
      nir_def *stencil_passes = nir_ior(b, nir_inot(b, has_stencil),
                                        compare_fragment(b, nir_iand_imm(b, face, 7), nir_ult(b, a, h),
                                                         nir_ieq(b, a, h), nir_ult(b, h, a)));
      nir_def *op = nir_bcsel(b, stencil_passes,
                              nir_bcsel(b, depth_passes, nir_ubitfield_extract_imm(b, face, 9, 3),
                                        nir_ubitfield_extract_imm(b, face, 6, 3)),
                              nir_ubitfield_extract_imm(b, face, 3, 3));
      nir_def *updated = stencil_operation(b, op, stencil, ref);
      updated = nir_ior(b, nir_iand(b, updated, write_mask), nir_iand(b, stencil, nir_inot(b, write_mask)));
      nir_push_if(b, nir_iand(b, has_stencil, nir_ine(b, updated, stencil)));
      tested_store(t, &tests->stencil_image, coord, s, updated);
      nir_pop_if(b, NULL);
      nir_def *passes = nir_iand(b, stencil_passes, depth_passes);
      nir_push_if(b, nir_iand(b, passes, writes_depth));
      tested_store(t, &tests->depth_image, coord, s, bits);
      nir_pop_if(b, NULL);
      nir_def *bit = nir_bcsel(b, passes, nir_imm_int(b, 1u << s), nir_imm_int(b, 0));
      nir_pop_if(b, NULL);
      tested = nir_ior(b, tested, nir_if_phi(b, bit, passed));
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, tested, passed);
}

/* The samples that pass are the fragment's coverage, which its pipeline
 * writes its color outputs to; it tests no depth or stencil. */
static bool
fragment_tests_end(struct translator *t, nir_def *passed)
{
   nir_builder *b = &t->b;
   nir_variable *mask = nir_variable_create(b->shader, nir_var_shader_out, glsl_int_type(), "tested coverage");
   if (!mask)
      return fail(t, STELA_AIR_ALLOCATION, "render IO variable allocation failed");
   mask->data.location = FRAG_RESULT_SAMPLE_MASK;
   nir_store_var(b, mask, passed, 1);
   b->shader->info.fs.pixel_interlock_ordered = true;
   b->shader->info.fs.early_fragment_tests = false;
   return true;
}

/* A fragment function that tests fragments itself (tests_fragments) and
 * accesses no storage attachments, imageblocks or raster order groups: its
 * tests are its whole interlock section, where it ends. */
static bool
test_fragments(struct translator *t)
{
   if (t->storage_attachments || t->fragment_imageblock || t->render.entry.raster_ordered)
      return true;
   struct fragment_tests tests;
   if (!fragment_tests_begin(t, &tests))
      return false;
   nir_builder *b = &t->b;
   nir_function_impl *impl = nir_shader_get_entrypoint(b->shader);
   b->cursor = nir_after_block_before_jump(nir_impl_last_block(impl));
   nir_def *kept = fragment_kept(t, &tests);
   nir_variable_mode modes = nir_var_image;
   nir_begin_invocation_interlock(b);
   nir_scoped_memory_barrier(b, SCOPE_DEVICE, NIR_MEMORY_ACQUIRE, modes);
   nir_def *passed = fragment_tests_run(t, &tests, kept);
   nir_scoped_memory_barrier(b, SCOPE_DEVICE, NIR_MEMORY_RELEASE, modes);
   nir_end_invocation_interlock(b);
   if (!fragment_tests_end(t, nir_iand(b, passed, kept)))
      return false;
   nir_progress(true, impl, nir_metadata_none);
   nir_lower_vars_to_ssa(b->shader);
   return true;
}

/* A fragment function with raster order groups or storage attachments: its
 * accesses that must follow those of the overlapping fragments before it,
 * from the first to the last, form a critical section that those fragments
 * leave first, in primitive order. The section opens and closes once in the
 * function's top-level control flow, as fragment shader interlock requires,
 * around the nodes holding those accesses. It opens loading each storage
 * attachment's texel and closes storing each one the function writes, where
 * the function ends. Raster order groups are not told apart: one order is
 * also each group's. */
static bool
order_fragments(struct translator *t)
{
   bool storage = t->storage_attachments != 0, imageblock = t->fragment_imageblock;
   if (!storage && !imageblock && !t->render.entry.raster_ordered)
      return true;
   if (!t->stage_options->raster_order_groups)
      return fail(t, STELA_AIR_TARGET, "raster order groups need the device's fragment shader interlock");
   if (storage && !attachment_outputs(t))
      return false;
   if (imageblock && !tested_outputs(t))
      return false;
   /* A function testing fragments itself runs its tests where it ends, in
    * this section, then writes its storage attachments for the samples that
    * pass. */
   bool tests = t->tests_fragments;
   if (tests && imageblock)
      return fail(t, STELA_AIR_TARGET, "imageblock data is written after the fragment tests, which its function runs itself");
   struct fragment_tests fragment_tests;
   if (tests && !fragment_tests_begin(t, &fragment_tests))
      return false;
   nir_function_impl *impl = nir_shader_get_entrypoint(t->b.shader);
   nir_cursor begin = {0}, end = {0};
   bool found = false;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_block) {
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (!ordered_instr(t, instr))
               continue;
            /* A phi uses its sources where its predecessors end, in the
             * node before its block. */
            if (!found)
               begin = instr->type == nir_instr_type_phi ? nir_before_cf_node(nir_cf_node_prev(node))
                                                          : nir_before_instr(instr);
            end = nir_after_instr(instr);
            found = true;
         }
         continue;
      }
      bool accesses = false;
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block)
            accesses |= ordered_instr(t, instr);
      }
      if (!accesses)
         continue;
      if (!found)
         begin = nir_before_cf_node(node);
      end = nir_after_cf_node(node);
      found = true;
   }
   /* Imageblock data is read where the function starts and written where
    * it ends. */
   if (imageblock) {
      begin = nir_before_impl(impl);
      end = nir_after_block_before_jump(nir_impl_last_block(impl));
      found = true;
   }
   if (tests) {
      if (!found)
         begin = nir_after_block_before_jump(nir_impl_last_block(impl));
      end = nir_after_block_before_jump(nir_impl_last_block(impl));
      found = true;
   }
   /* A function whose grouped arguments go unused orders nothing. */
   if (!found && !storage)
      return true;
   if (!found)
      begin = nir_after_block_before_jump(nir_impl_last_block(impl));
   nir_variable_mode modes = nir_var_mem_global | nir_var_mem_ssbo | nir_var_image;
   nir_builder *b = &t->b;
   b->cursor = begin;
   nir_begin_invocation_interlock(b);
   nir_scoped_memory_barrier(b, SCOPE_DEVICE, NIR_MEMORY_ACQUIRE, modes);
   nir_def *bits[AIR_MAX_RENDER_TARGETS] = {0}, *texels[AIR_MAX_RENDER_TARGETS] = {0};
   u_foreach_bit(n, t->storage_attachments) {
      if (storage_attachment(t, n)->planes) {
         texels[n] = load_planes(t, n);
         continue;
      }
      bits[n] = load_attachment(t, n);
      texels[n] = nir_format_unpack_rgba(b, bits[n], storage_attachment(t, n)->format);
   }
   for (unsigned i = 0; i < t->attachment_read_count; i++) {
      const struct attachment_read *read = &t->attachment_reads[i];
      nir_def *texel = nir_trim_vector(b, texels[read->io->location], read->io->components);
      nir_def_rewrite_uses(read->placeholder, stage_narrow(b, texel, read->io->primitive));
      nir_intrinsic_instr *load = nir_instr_as_intrinsic(nir_def_instr(read->placeholder));
      nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
      nir_instr_remove(&load->instr);
      exec_node_remove(&deref->var->node);
      nir_deref_instr_remove_if_unused(deref);
   }
   nir_def *kept = NULL, *passed = NULL;
   if (storage || tests)
      b->cursor = nir_after_block_before_jump(nir_impl_last_block(impl));
   else
      b->cursor = end;
   if (tests) {
      kept = fragment_kept(t, &fragment_tests);
      passed = fragment_tests_run(t, &fragment_tests, kept);
      /* A sample the tests drop writes nothing: one invocation writes one
       * sample's texels, or the pixel's single sample. */
      nir_push_if(b, nir_ine_imm(b, passed, 0));
   }
   storage_alpha_to_one(t);
   u_foreach_bit(n, t->storage_attachments)
      write_attachment(t, n, texels[n], bits[n]);
   if (tests)
      nir_pop_if(b, NULL);
   nir_scoped_memory_barrier(b, SCOPE_DEVICE, NIR_MEMORY_RELEASE, modes);
   nir_end_invocation_interlock(b);
   t->b.shader->info.fs.pixel_interlock_ordered = true;
   if (tests && !fragment_tests_end(t, nir_iand(b, passed, kept)))
      return false;
   nir_progress(true, impl, nir_metadata_none);
   /* The fragment tests decide which fragments write the attachments and
    * the imageblock data, unless the function runs them itself. */
   if (imageblock)
      t->b.shader->info.fs.early_fragment_tests = true;
   if (storage && !tests)
      t->b.shader->info.fs.early_fragment_tests = true;
   if (storage || tests)
      nir_lower_vars_to_ssa(t->b.shader);
   return true;
}

/* "air.is_uniform.<type>", which states what make_uniform guarantees. */
static bool
uniform_call(struct translator *t, LLVMValueRef inst, const char *name, size_t name_length,
             bool *lowered)
{
   static const char uniform[] = "air.is_uniform.";
   *lowered = name_length > strlen(uniform) && !memcmp(name, uniform, strlen(uniform));
   if (!*lowered)
      return true;
   if (LLVMGetNumArgOperands(inst) != 1 ||
       LLVMTypeOf(inst) != LLVMInt1TypeInContext(LLVMGetTypeContext(LLVMTypeOf(inst))))
      return fail(t, STELA_AIR_TYPE, "is_uniform answers a bool of one value");
   return put_value(t, inst, nir_imm_true(&t->b), 0);
}

/* "air.get_num_samples.i32" and "air.get_sample_position.v2f32", whose last
 * operand is 0, from the pass's samples; an index past them reads one of the
 * 16 positions the pass holds. */
static bool
sample_call(struct translator *t, LLVMValueRef inst, const char *name, size_t name_length,
            bool *lowered)
{
   bool count = stela_air_call_named(name, name_length, "air.get_num_samples.i32");
   *lowered = count || stela_air_call_named(name, name_length, "air.get_sample_position.v2f32");
   if (!*lowered)
      return true;
   unsigned operands = count ? 1 : 2;
   LLVMValueRef last = LLVMGetNumArgOperands(inst) == operands ?
      LLVMGetOperand(inst, operands - 1) : NULL;
   if (!last || !LLVMIsAConstantInt(last) || LLVMConstIntGetZExtValue(last))
      return fail(t, STELA_AIR_INSTRUCTION, "a sample query of another kind");
   if (!t->stage_options || t->stage_options->stage != STELA_AIR_FRAGMENT)
      return fail(t, STELA_AIR_INSTRUCTION, "sample queries exist only in fragment functions");
   if (count) {
      nir_def *samples = device_value(t, STELA_AIR_SAMPLE_COUNT);
      return samples && put_value(t, inst, samples, 0);
   }
   nir_def *index = get_value(t, LLVMGetOperand(inst, 0));
   if (!index || index->num_components != 1 || index->bit_size != 32)
      return fail(t, STELA_AIR_TYPE, "a sample index is one uint");
   nir_def *positions = device_value(t, STELA_AIR_SAMPLE_POSITIONS);
   if (!positions)
      return false;
   nir_builder *b = &t->b;
   index = nir_iand_imm(b, index, 15);
   nir_def *word = nir_vector_extract(b, positions, nir_ushr_imm(b, index, 2));
   nir_def *byte = nir_ubitfield_extract(b, word, nir_imul_imm(b, nir_iand_imm(b, index, 3), 8),
                                         nir_imm_int(b, 8));
   nir_def *sixteenths = nir_vec2(b, nir_iand_imm(b, byte, 0xf), nir_ushr_imm(b, byte, 4));
   return put_value(t, inst, nir_fmul_imm(b, nir_u2f32(b, sixteenths), 1.0 / 16.0), 0);
}

/* Whether a factor of access reads the blend color. */
static bool
reads_blend_color(const struct stela_air_attachment_access *access)
{
   const enum stela_air_blend_factor factors[] = {
      access->source_rgb, access->destination_rgb, access->source_alpha, access->destination_alpha,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(factors); i++) {
      if (factors[i] >= STELA_AIR_BLEND_COLOR_FACTOR && factors[i] <= STELA_AIR_BLEND_ONE_MINUS_ALPHA_FACTOR)
         return true;
   }
   return false;
}

/* Whether the texel layout of format is one storage attachments convert: a
 * plain color layout of 1, 2, 4, 8 or 16 bytes, or a packed float one. */
static bool
storage_format(enum pipe_format format)
{
   if (format == PIPE_FORMAT_R9G9B9E5_FLOAT || format == PIPE_FORMAT_R11G11B10_FLOAT)
      return true;
   const struct util_format_description *desc = util_format_description(format);
   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->block.width != 1 ||
       desc->block.height != 1 || (desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB &&
                                   desc->colorspace != UTIL_FORMAT_COLORSPACE_SRGB))
      return false;
   if (desc->block.bits != 8 && desc->block.bits != 16 && desc->block.bits != 32 &&
       desc->block.bits != 64 && desc->block.bits != 128)
      return false;
   for (unsigned c = 0; c < desc->nr_channels; c++) {
      if (desc->channel[c].type == UTIL_FORMAT_TYPE_FIXED)
         return false;
      /* Wider layouts unpack channels of one size, the narrow ones in
       * order. */
      if (desc->block.bits > 32 && desc->channel[c].size != desc->channel[0].size)
         return false;
      if (c && desc->block.bits <= 32 &&
          desc->channel[c].shift != desc->channel[c - 1].shift + desc->channel[c - 1].size)
         return false;
   }
   return true;
}

/* Resolves color attachment n's raw texel image in the prelude, which its
 * accesses deref again where they are: the structurizer drops this deref. */
static bool
attachment_binding(struct translator *t, unsigned n, bool multisample)
{
   if (!t->options->bind_attachment)
      return fail(t, STELA_AIR_ARGUMENT, "unbound render target %u", n);
   const struct glsl_type *type =
      glsl_image_type(multisample ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D, true, GLSL_TYPE_UINT);
   nir_deref_instr *image = t->options->bind_attachment(&t->b, n, type, t->options->binding_data);
   if (!image || !binding_prelude(t))
      return fail(t, STELA_AIR_ARGUMENT, "unbound render target %u", n);
   nir_deref_instr *parent = image->deref_type == nir_deref_type_array ?
      nir_deref_instr_parent(image) : NULL;
   if (!parent || parent->deref_type != nir_deref_type_var || !glsl_type_is_image(image->type))
      return fail(t, STELA_AIR_ARGUMENT, "render target %u is not a heap image", n);
   t->attachment_heaps[n] = parent->var;
   t->attachment_indices[n] = image->arr.index.ssa;
   return true;
}

/* Resolves each storage attachment's image in the prelude, with the blend
 * color its factors read. */
static bool
attachment_prelude(struct translator *t)
{
   for (unsigned n = 0; n < AIR_MAX_RENDER_TARGETS; n++) {
      const struct stela_air_attachment_access *access = storage_attachment(t, n);
      if (!access)
         continue;
      if (!t->stage_options->raster_order_groups)
         return fail(t, STELA_AIR_TARGET, "storage attachments need the device's fragment shader interlock");
      if (!storage_format(access->format))
         return fail(t, STELA_AIR_ARGUMENT, "render target %u has no storage texel layout", n);
      if (!attachment_binding(t, n, t->stage_options->samples > 1))
         return false;
      if (access->blend && reads_blend_color(access) && !t->blend_color &&
          !(t->blend_color = device_value(t, STELA_AIR_BLEND_COLOR)))
         return false;
      t->storage_attachments |= 1u << n;
   }
   /* The two-plane extended-range attachments mark the tiles their
    * fragments reach (mark_dirty_tiles). Multisampled ones are never
    * IOSurface planes. */
   unsigned kinds = t->stage_options->extended_range;
   if (t->stage_options->samples <= 1 && t->options->bind_attachment &&
       t->options->load_device_value) {
      for (unsigned n = 0; n < AIR_MAX_RENDER_TARGETS; n++) {
         unsigned kind = kinds >> (n * STELA_AIR_EXTENDED_RANGE_BITS) &
                         BITFIELD_MASK(STELA_AIR_EXTENDED_RANGE_BITS);
         if (kind < STELA_AIR_EXTENDED_RANGE_PLANES_HELD)
            continue;
         if (!(t->storage_attachments & 1u << n) && !attachment_binding(t, n, false))
            return false;
         t->dirty_attachments |= 1u << n;
      }
      if (t->dirty_attachments) {
         t->dirty = t->options->load_device_value(&t->b, STELA_AIR_DIRTY_TILES,
                                                  t->options->binding_data);
         if (!t->dirty || t->dirty->num_components != 4 || t->dirty->bit_size != 32 ||
             !binding_prelude(t))
            return fail(t, STELA_AIR_ARGUMENT, "invalid device value");
      }
   }
   if (t->storage_attachments || t->dirty_attachments)
      t->b.shader->info.fs.origin_upper_left = true;
   return true;
}

/* Where a fragment function ends, each fragment that reached it stores its
 * pass's stamp, when it has one, at its tile's texel of the dirty layer of
 * the attachment the pass names (STELA_AIR_DIRTY_TILES): the pass's end then
 * encodes only those tiles into the IOSurface's planes. A fragment that
 * writes no color marks its tile all the same. */
static void
mark_dirty_tiles(struct translator *t)
{
   if (!t->dirty)
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(t->b.shader);
   nir_builder *b = &t->b;
   b->cursor = nir_after_block_before_jump(nir_impl_last_block(impl));
   nir_def *stamp = nir_channel(b, t->dirty, 0), *marked = nir_channel(b, t->dirty, 1);
   nir_def *tile = nir_ushr_imm(b, nir_f2u32(b, nir_trim_vector(b, nir_load_frag_coord(b), 2)),
                                STELA_AIR_DIRTY_TILE_SHIFT);
   nir_def *coord = nir_vec4(b, nir_channel(b, tile, 0), nir_channel(b, tile, 1),
                             nir_imm_int(b, STELA_AIR_DIRTY_LAYER), nir_imm_int(b, 0));
   nir_push_if(b, nir_iand(b, nir_ine_imm(b, stamp, 0), nir_inot(b, nir_is_helper_invocation(b, 1))));
   u_foreach_bit(n, t->dirty_attachments) {
      nir_push_if(b, nir_ieq_imm(b, marked, n));
      nir_image_deref_store(b, &attachment_image(t, n)->def, coord, nir_imm_int(b, 0),
                            nir_vec4(b, stamp, nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0)),
                            nir_imm_int(b, 0), .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = true,
                            .format = PIPE_FORMAT_NONE, .src_type = nir_type_uint32);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
   nir_progress(true, impl, nir_metadata_none);
}

static bool
stage_arguments(struct translator *t, LLVMValueRef fn)
{
   bool slots[UINT8_MAX + 1] = {0};
   t->fields = t->render.arguments.fields;
   if (t->stage_options->stage == STELA_AIR_FRAGMENT && !attachment_prelude(t))
      return false;
   /* Its builtins and stage-in read the patch's IDs. */
   if (t->render.entry.patch && !(t->patch_ids = device_value(t, STELA_AIR_PATCH_IDS)))
      return false;
   for (unsigned i = 0; i < t->render.entry.input_count; i++) {
      const struct stela_air_stage_io *io = &t->render.inputs[i];
      const struct stela_air_argument *argument = &t->render.arguments.arguments[i];
      LLVMValueRef param = LLVMGetParam(fn, io->argument);
      bool absent;
      if (!absent_argument(t, argument, &absent))
         return false;
      if (absent)
         continue;
      /* No stage writes it, and it takes no place among the varyings. */
      if (io->absent && io->semantic == STELA_AIR_VARYING) {
         if (!put_value(t, param, nir_undef(&t->b, io->components, stage_bits(io->primitive)), 0))
            return false;
         continue;
      }
      nir_def *def;
      unsigned slot = 0;
      if (io->semantic == STELA_AIR_STAGE_TEXTURE || io->semantic == STELA_AIR_STAGE_SAMPLER) {
         bool texture = io->semantic == STELA_AIR_STAGE_TEXTURE;
         if (!put_resource(t, param, texture ? &io->texture : NULL, &io->sampler))
            return false;
         continue;
      }
      if (io->semantic == STELA_AIR_STAGE_BUFFER) {
         /* Full stage metadata/layout validation has already completed. An
          * actually unused declaration needs no resource binding or SSA value;
          * a dead instruction using the parameter still counts as a real use. */
         if (!io->used && !LLVMGetFirstUse(param)) continue;
         if (t->options->bind_ssbo) {
            if (!t->options->bind_ssbo(&io->buffer, &slot, t->options->binding_data) ||
                slot >= t->options->ssbo_count || slots[slot])
               return fail(t, STELA_AIR_ARGUMENT, "invalid render buffer SSBO binding");
            slots[slot] = true; def = nir_imm_int64(&t->b, 0);
         } else {
            if (!t->options->bind_buffer)
               return fail(t, STELA_AIR_ARGUMENT, "unbound render buffer");
            def = t->options->bind_buffer(&t->b, &io->buffer, t->options->binding_data);
            if (!binding_prelude(t))
               return false;
            if (!def || def->num_components != 1 || def->bit_size != 64)
               return fail(t, STELA_AIR_ARGUMENT, "invalid/unbound render buffer");
         }
      } else if (io->semantic == STELA_AIR_STAGE_BUILTIN) {
         def = builtin_argument(t, io->builtin, param);
         if (!def)
            return false;
      } else if (io->semantic == STELA_AIR_POSITION) {
         t->b.shader->info.fs.origin_upper_left = true;
         t->b.shader->info.fs.pixel_center_integer = false;
         if (!io->used) continue;
         def = nir_load_frag_coord(&t->b);
      } else if (io->semantic == STELA_AIR_ATTACHMENT) {
         if (!(def = attachment_input(t, io)))
            return false;
      } else if (io->semantic == STELA_AIR_IMAGEBLOCK_DATA) {
         if (!io->used && !LLVMGetFirstUse(param))
            continue;
         if (!imageblock_data_input(t, param))
            return false;
         continue;
      } else if (io->semantic == STELA_AIR_CONTROL_POINTS) {
         continue; /* Calls of its function read it. */
      } else if (io->semantic == STELA_AIR_BARYCENTRIC) {
         if (!io->used) continue;
         enum glsl_interp_mode mode = io->interpolation == STELA_AIR_PERSPECTIVE ?
            INTERP_MODE_SMOOTH : INTERP_MODE_NOPERSPECTIVE;
         def = io->sampling == STELA_AIR_SAMPLE ?
                  nir_load_barycentric_coord_sample(&t->b, 32, .interp_mode = mode) :
               io->sampling == STELA_AIR_CENTROID ?
                  nir_load_barycentric_coord_centroid(&t->b, 32, .interp_mode = mode) :
                  nir_load_barycentric_coord_pixel(&t->b, 32, .interp_mode = mode);
         def = nir_trim_vector(&t->b, def, io->components);
      } else if (io->semantic == STELA_AIR_POSITION_IN_PATCH) {
         /* Metal's patch coordinates are Vulkan's with the upper-left domain
          * origin. */
         def = nir_trim_vector(&t->b, nir_load_tess_coord(&t->b), io->components);
      } else if (t->render.entry.patch) {
         if (!io->used) continue;
         if (!(def = stage_in_attribute(t, io->location, t->patch_ids, NULL, io->primitive,
                                        io->components)))
            return false;
      } else {
         /* A vertex input the function never reads is not fetched. */
         if (io->semantic == STELA_AIR_VERTEX_INPUT && !io->used) continue;
         if (io->semantic == STELA_AIR_VERTEX_INPUT) {
            if (!vertex_input_words(t, io, &def)) return false;
            if (def) {
               if (!put_value(t, param, def, io->buffer.access)) return false;
               continue;
            }
         }
         nir_variable *var = stage_variable(t, io, false);
         if (!var) return false;
         if (io->interpolant) continue; /* Calls interpolate it. */
         def = stage_narrow(&t->b, nir_load_var(&t->b, var), io->primitive);
      }
      if (!put_value(t, param, def, io->buffer.access)) return false;
      if (io->semantic == STELA_AIR_STAGE_BUFFER) {
         struct value *v = find_value(t, param);
         if (!argument_root(t, v, param, argument))
            return false;
         v->ssbo = t->options->bind_ssbo != NULL; v->ssbo_slot = slot;
      }
   }
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      /* Imageblock data is memory, not an output. */
      if (io->absent || io->semantic == STELA_AIR_IMAGEBLOCK_DATA) continue;
      if (!(t->outputs[i] = stage_variable(t, io, true))) return false;
      if (io->semantic == STELA_AIR_LAYER) t->layer = t->outputs[i];
      if (io->semantic == STELA_AIR_VIEWPORT) t->viewport = t->outputs[i];
   }
   if (t->stage_options->amplified) {
      if (!t->layer && !(t->layer = view_output(t, VARYING_SLOT_LAYER, "air.view_layer")))
         return false;
      if (!t->viewport && !(t->viewport = view_output(t, VARYING_SLOT_VIEWPORT, "air.view_viewport")))
         return false;
   }
   return true;
}

static bool
stage_return(struct translator *t, LLVMValueRef inst)
{
   if (!t->render.entry.output_count && !LLVMGetNumOperands(inst))
      return true;
   if (LLVMGetNumOperands(inst) != 1) return fail(t, STELA_AIR_TYPE, "render entry requires its real return value");
   LLVMValueRef returned = LLVMGetOperand(inst, 0);
   struct value *aggregate = NULL;
   if (LLVMGetTypeKind(LLVMTypeOf(returned)) == LLVMStructTypeKind) {
      aggregate = aggregate_value(t, returned);
      if (!aggregate || aggregate->aggregate_count != t->render.entry.output_count)
         return fail(t, STELA_AIR_TYPE, "render output aggregate mismatch");
   }
   nir_def *values[AIR_MAX_ARGUMENTS];
   nir_def *clip[AIR_MAX_CLIP_DISTANCES];
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      const struct value *leaf = aggregate ? aggregate->elements[i] : NULL;
      /* An output the function leaves undefined holds anything, so it stays
       * unwritten. */
      if (io->semantic == STELA_AIR_IMAGEBLOCK_DATA) {
         values[i] = NULL;
         continue;
      }
      if (io->semantic == STELA_AIR_CLIP_DISTANCE) {
         for (unsigned j = 0; j < io->clip_distances; j++) {
            /* A lone distance need not be an array. */
            const struct value *distance = leaf && leaf->aggregate_count ? leaf->elements[j] : leaf;
            clip[j] = distance ? whole_value(t, distance) : NULL;
         }
         values[i] = NULL;
      } else if (aggregate) {
         values[i] = leaf ? whole_value(t, leaf) : NULL;
      } else if (!(values[i] = get_value(t, returned))) {
         return false;
      }
   }
   /* Validate ALL leaves before emitting any output effect. */
   nir_def *layer = NULL, *viewport = NULL;
   for (unsigned i = 0; i < t->render.entry.output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      if (io->absent)
         continue;
      if (io->semantic == STELA_AIR_IMAGEBLOCK_DATA) {
         const struct value *data = aggregate ? aggregate->elements[i] : NULL;
         if (data && !imageblock_data_output(t, data, LLVMStructGetTypeAtIndex(LLVMTypeOf(returned), i)))
            return false;
         continue;
      }
      if (io->semantic == STELA_AIR_CLIP_DISTANCE) {
         nir_deref_instr *array = nir_build_deref_var(&t->b, t->outputs[i]);
         for (unsigned j = 0; j < io->clip_distances; j++) {
            if (clip[j])
               nir_store_deref(&t->b, nir_build_deref_array_imm(&t->b, array, j), clip[j], 1);
         }
         continue;
      }
      if (!values[i])
         continue;
      nir_def *value = stage_widen(&t->b, values[i], io->primitive);
      if (t->stage_options->amplified && t->outputs[i] == t->layer)
         layer = value;
      else if (t->stage_options->amplified && t->outputs[i] == t->viewport)
         viewport = value;
      else {
         if (io->semantic == STELA_AIR_RENDER_TARGET && !io->index)
            value = clamp_extended_range(t, io->location, value);
         if (io->semantic == STELA_AIR_POINT_SIZE)
            value = largest_point(&t->b, value);
         nir_store_var(&t->b, t->outputs[i], value, BITFIELD_MASK(value->num_components));
      }
   }
   if (t->stage_options->amplified) {
      nir_def *mapping = device_value(t, STELA_AIR_VIEW_MAPPING);
      if (!mapping)
         return false;
      nir_def *zero = nir_imm_int(&t->b, 0);
      nir_store_var(&t->b, t->layer, nir_iadd(&t->b, layer ? layer : zero, nir_ushr_imm(&t->b, mapping, 16)), 1);
      nir_store_var(&t->b, t->viewport,
                    nir_iadd(&t->b, viewport ? viewport : zero, nir_iand_imm(&t->b, mapping, 0xffff)), 1);
   }
   return true;
}

enum integer_count_operation {
   INTEGER_COUNT_POPULATION,
   INTEGER_COUNT_LEADING_ZEROS,
   INTEGER_COUNT_TRAILING_ZEROS,
};

/* Admit canonical LLVM overloads, never names that merely share a prefix. */
static bool
integer_count(struct translator *t, LLVMValueRef inst, LLVMValueRef callee,
              unsigned intrinsic, enum integer_count_operation operation)
{
   unsigned components, bits;
   LLVMTypeRef type = LLVMTypeOf(inst);
   if (!shape(type, &components, &bits) || bits == 1)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-count result shape");
   LLVMTypeRef expected = LLVMIntrinsicGetType(t->context, intrinsic, &type, 1);
   unsigned arity = operation == INTEGER_COUNT_POPULATION ? 1 : 2;
   if (!LLVMIsDeclaration(callee) || LLVMGetFunctionCallConv(callee) != LLVMCCallConv ||
       LLVMGetInstructionCallConv(inst) != LLVMCCallConv ||
       LLVMGlobalGetValueType(callee) != expected || LLVMGetCalledFunctionType(inst) != expected ||
       LLVMGetNumArgOperands(inst) != arity || LLVMCountParams(callee) != arity ||
       LLVMIsFunctionVarArg(expected) || LLVMGetNumOperandBundles(inst) ||
       LLVMTypeOf(LLVMGetOperand(inst, 0)) != type)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-count call signature");
   size_t canonical_length = 0, name_length = 0;
   char *canonical = LLVMIntrinsicCopyOverloadedName2(t->module, intrinsic, &type, 1,
                                                     &canonical_length);
   if (!canonical)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM integer-count overload name unavailable");
   const char *name = LLVMGetValueName2(callee, &name_length);
   bool canonical_name = name_length == canonical_length && !memcmp(name, canonical, name_length);
   LLVMDisposeMessage(canonical); /* The pinned C API uses strdup/free. */
   if (!canonical_name)
      return fail(t, STELA_AIR_INSTRUCTION, "noncanonical LLVM integer-count overload name");
   if (operation != INTEGER_COUNT_POPULATION) {
      LLVMValueRef flag = LLVMGetOperand(inst, 1);
      if (LLVMTypeOf(flag) != LLVMInt1TypeInContext(t->context) || !LLVMIsAConstantInt(flag))
         return fail(t, STELA_AIR_INSTRUCTION, "LLVM zero-count poison flag requires constant scalar i1");
      /* true with a zero input is LLVM poison. Defined inputs have the same
       * mapping for either flag; no zero-result/trap claim is made for true. */
   }
   nir_def *input = get_value(t, LLVMGetOperand(inst, 0));
   if (!input)
      return false;
   if (input->num_components != components || input->bit_size != bits)
      return fail(t, STELA_AIR_TYPE, "LLVM integer-count operand shape mismatch");
   /* NIR ufind_msb is only valid for 32/64-bit sources. Widen without sign
    * extension so the original width still determines leading-zero counts. */
   if (bits < 32)
      input = nir_u2u32(&t->b, input);
   nir_def *count;
   if (operation == INTEGER_COUNT_POPULATION) {
      count = nir_bit_count(&t->b, input);
   } else if (operation == INTEGER_COUNT_LEADING_ZEROS) {
      count = nir_isub(&t->b, nir_imm_int(&t->b, bits - 1), nir_ufind_msb(&t->b, input));
   } else {
      nir_def *lsb = nir_find_lsb(&t->b, input);
      count = nir_bcsel(&t->b, nir_ieq_imm(&t->b, lsb, -1), nir_imm_int(&t->b, bits), lsb);
   }
   return put_value(t, inst, nir_u2uN(&t->b, count, bits), 0);
}

enum integer_permutation_operation {
   INTEGER_REVERSE_BITS,
   INTEGER_SWAP_BYTES,
};

enum { AIR_BYTE_BITS = 8, AIR_HALFWORD_BITS = 16, AIR_WORD_BITS = 32 };

static nir_def *
byte_swap32(nir_builder *b, nir_def *value)
{
   const uint32_t low_bytes_of_halfwords = UINT32_C(0x00ff00ff);
   nir_def *adjacent = nir_ior(b,
      nir_ishl_imm(b, nir_iand_imm(b, value, low_bytes_of_halfwords), AIR_BYTE_BITS),
      nir_iand_imm(b, nir_ushr_imm(b, value, AIR_BYTE_BITS), low_bytes_of_halfwords));
   return nir_ior(b, nir_ishl_imm(b, adjacent, AIR_HALFWORD_BITS),
                     nir_ushr_imm(b, adjacent, AIR_HALFWORD_BITS));
}

static bool
integer_permutation(struct translator *t, LLVMValueRef inst, LLVMValueRef callee,
                    unsigned intrinsic, enum integer_permutation_operation operation)
{
   unsigned components, bits;
   LLVMTypeRef type = LLVMTypeOf(inst);
   if (!shape(type, &components, &bits) || bits == 1 ||
       (operation == INTEGER_SWAP_BYTES && bits % AIR_HALFWORD_BITS))
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-permutation result shape");
   LLVMTypeRef expected = LLVMIntrinsicGetType(t->context, intrinsic, &type, 1);
   if (!LLVMIsDeclaration(callee) || LLVMGetFunctionCallConv(callee) != LLVMCCallConv ||
       LLVMGetInstructionCallConv(inst) != LLVMCCallConv ||
       LLVMGlobalGetValueType(callee) != expected || LLVMGetCalledFunctionType(inst) != expected ||
       LLVMGetNumArgOperands(inst) != 1 || LLVMCountParams(callee) != 1 ||
       LLVMIsFunctionVarArg(expected) || LLVMGetNumOperandBundles(inst) ||
       LLVMTypeOf(LLVMGetOperand(inst, 0)) != type)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-permutation call signature");
   size_t canonical_length = 0, name_length = 0;
   char *canonical = LLVMIntrinsicCopyOverloadedName2(t->module, intrinsic, &type, 1,
                                                     &canonical_length);
   if (!canonical)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM integer-permutation overload name unavailable");
   const char *name = LLVMGetValueName2(callee, &name_length);
   bool canonical_name = name_length == canonical_length && !memcmp(name, canonical, name_length);
   LLVMDisposeMessage(canonical);
   if (!canonical_name)
      return fail(t, STELA_AIR_INSTRUCTION, "noncanonical LLVM integer-permutation overload name");
   nir_def *input = get_value(t, LLVMGetOperand(inst, 0));
   if (!input)
      return false;
   if (input->num_components != components || input->bit_size != bits)
      return fail(t, STELA_AIR_TYPE, "LLVM integer-permutation operand shape mismatch");
   if (bits < AIR_WORD_BITS)
      input = nir_u2u32(&t->b, input);
   nir_def *result;
   if (operation == INTEGER_REVERSE_BITS) {
      /* Keep native64 available to ACO's uniform scalar selector. Each actual
       * compiler owns its existing lowering; ELK intake adds its missing bit. */
      result = nir_bitfield_reverse(&t->b, input);
      if (bits < AIR_WORD_BITS)
         result = nir_ushr_imm(&t->b, result, AIR_WORD_BITS - bits);
   } else if (bits == AIR_HALFWORD_BITS) {
      result = nir_ior(&t->b, nir_ishl_imm(&t->b, input, AIR_BYTE_BITS),
                             nir_ushr_imm(&t->b, input, AIR_BYTE_BITS));
   } else if (bits == AIR_WORD_BITS) {
      result = byte_swap32(&t->b, input);
   } else {
      nir_def *low = byte_swap32(&t->b, nir_unpack_64_2x32_split_x(&t->b, input));
      nir_def *high = byte_swap32(&t->b, nir_unpack_64_2x32_split_y(&t->b, input));
      result = nir_pack_64_2x32_split(&t->b, high, low);
   }
   return put_value(t, inst, nir_u2uN(&t->b, result, bits), 0);
}

enum integer_minmax_operation {
   INTEGER_SIGNED_MINIMUM,
   INTEGER_SIGNED_MAXIMUM,
   INTEGER_UNSIGNED_MINIMUM,
   INTEGER_UNSIGNED_MAXIMUM,
};

static bool
integer_minmax(struct translator *t, LLVMValueRef inst, LLVMValueRef callee,
               unsigned intrinsic, enum integer_minmax_operation operation)
{
   unsigned components, bits;
   LLVMTypeRef type = LLVMTypeOf(inst);
   if (!shape(type, &components, &bits) || bits == 1)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-minmax result shape");
   LLVMTypeRef expected = LLVMIntrinsicGetType(t->context, intrinsic, &type, 1);
   if (!LLVMIsDeclaration(callee) || LLVMGetFunctionCallConv(callee) != LLVMCCallConv ||
       LLVMGetInstructionCallConv(inst) != LLVMCCallConv ||
       LLVMGlobalGetValueType(callee) != expected || LLVMGetCalledFunctionType(inst) != expected ||
       LLVMGetNumArgOperands(inst) != 2 || LLVMCountParams(callee) != 2 ||
       LLVMIsFunctionVarArg(expected) || LLVMGetNumOperandBundles(inst) ||
       LLVMTypeOf(LLVMGetOperand(inst, 0)) != type || LLVMTypeOf(LLVMGetOperand(inst, 1)) != type)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-minmax call signature");
   size_t canonical_length = 0, name_length = 0;
   char *canonical = LLVMIntrinsicCopyOverloadedName2(t->module, intrinsic, &type, 1,
                                                     &canonical_length);
   if (!canonical)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM integer-minmax overload name unavailable");
   const char *name = LLVMGetValueName2(callee, &name_length);
   bool canonical_name = name_length == canonical_length && !memcmp(name, canonical, name_length);
   LLVMDisposeMessage(canonical);
   if (!canonical_name)
      return fail(t, STELA_AIR_INSTRUCTION, "noncanonical LLVM integer-minmax overload name");
   nir_def *a = get_value(t, LLVMGetOperand(inst, 0));
   nir_def *b = get_value(t, LLVMGetOperand(inst, 1));
   if (!a || !b)
      return false;
   if (a->num_components != components || b->num_components != components ||
       a->bit_size != bits || b->bit_size != bits)
      return fail(t, STELA_AIR_TYPE, "LLVM integer-minmax operand shape mismatch");
   nir_def *result;
   switch (operation) {
   case INTEGER_SIGNED_MINIMUM: result = nir_imin(&t->b, a, b); break;
   case INTEGER_SIGNED_MAXIMUM: result = nir_imax(&t->b, a, b); break;
   case INTEGER_UNSIGNED_MINIMUM: result = nir_umin(&t->b, a, b); break;
   case INTEGER_UNSIGNED_MAXIMUM: result = nir_umax(&t->b, a, b); break;
   default: return fail(t, STELA_AIR_INSTRUCTION, "unknown LLVM integer-minmax operation");
   }
   return put_value(t, inst, result, 0);
}

enum integer_funnel_operation {
   INTEGER_FUNNEL_LEFT,
   INTEGER_FUNNEL_RIGHT,
};

static bool
integer_funnel(struct translator *t, LLVMValueRef inst, LLVMValueRef callee,
               unsigned intrinsic, enum integer_funnel_operation operation)
{
   unsigned components, bits;
   LLVMTypeRef type = LLVMTypeOf(inst);
   LLVMTypeRef element = LLVMGetTypeKind(type) == LLVMVectorTypeKind ? LLVMGetElementType(type) : type;
   if (LLVMGetTypeKind(element) != LLVMIntegerTypeKind ||
       !shape(type, &components, &bits) || bits == 1)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-funnel result shape");
   LLVMTypeRef expected = LLVMIntrinsicGetType(t->context, intrinsic, &type, 1);
   if (!LLVMIsDeclaration(callee) || LLVMGetFunctionCallConv(callee) != LLVMCCallConv ||
       LLVMGetInstructionCallConv(inst) != LLVMCCallConv ||
       LLVMGlobalGetValueType(callee) != expected || LLVMGetCalledFunctionType(inst) != expected ||
       LLVMGetNumArgOperands(inst) != 3 || LLVMCountParams(callee) != 3 ||
       LLVMIsFunctionVarArg(expected) || LLVMGetNumOperandBundles(inst) ||
       LLVMTypeOf(LLVMGetOperand(inst, 0)) != type ||
       LLVMTypeOf(LLVMGetOperand(inst, 1)) != type ||
       LLVMTypeOf(LLVMGetOperand(inst, 2)) != type)
      return fail(t, STELA_AIR_TYPE, "unsupported LLVM integer-funnel call signature");
   size_t canonical_length = 0, name_length = 0;
   char *canonical = LLVMIntrinsicCopyOverloadedName2(t->module, intrinsic, &type, 1,
                                                     &canonical_length);
   if (!canonical)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM integer-funnel overload name unavailable");
   const char *name = LLVMGetValueName2(callee, &name_length);
   bool canonical_name = name_length == canonical_length && !memcmp(name, canonical, name_length);
   LLVMDisposeMessage(canonical);
   if (!canonical_name)
      return fail(t, STELA_AIR_INSTRUCTION, "noncanonical LLVM integer-funnel overload name");
   nir_def *a = get_value(t, LLVMGetOperand(inst, 0));
   nir_def *b = get_value(t, LLVMGetOperand(inst, 1));
   nir_def *amount = get_value(t, LLVMGetOperand(inst, 2));
   if (!a || !b || !amount)
      return false;
   if (a->num_components != components || b->num_components != components ||
       amount->num_components != components || a->bit_size != bits ||
       b->bit_size != bits || amount->bit_size != bits)
      return fail(t, STELA_AIR_TYPE, "LLVM integer-funnel operand shape mismatch");
   /* NIR shift amounts are uint32 and masked by the data width. Truncating
    * i64 first preserves every bit relevant to modulo 64. Keep data native. */
   nir_def *k = nir_iand_imm(&t->b, nir_u2u32(&t->b, amount), bits - 1);
   bool left = operation == INTEGER_FUNNEL_LEFT;
   bool same_input = LLVMGetOperand(inst, 0) == LLVMGetOperand(inst, 1);
   const nir_shader_compiler_options *options = t->b.shader->options;
   bool rotate = (bits == 8 && options->has_rotate8) ||
                 (bits == 16 && options->has_rotate16) ||
                 (bits == 32 && options->has_rotate32);
   nir_def *result;
   if (same_input && rotate) {
      result = left ? nir_urol(&t->b, a, k) : nir_uror(&t->b, a, k);
   } else if (bits == 32 && options->has_shfr32) {
      result = nir_shfr(&t->b, a, b, left ? nir_ineg(&t->b, k) : k);
      if (left && !same_input)
         result = nir_bcsel(&t->b, nir_ieq_imm(&t->b, k, 0), a, result);
   } else {
      nir_def *negative = nir_ineg(&t->b, k);
      nir_def *shifted = left ? nir_ior(&t->b, nir_ishl(&t->b, a, k), nir_ushr(&t->b, b, negative)) :
                                nir_ior(&t->b, nir_ishl(&t->b, a, negative), nir_ushr(&t->b, b, k));
      /* At zero, the masked complementary shift is also zero. OR would be
       * wrong for unequal a/b; LLVM selects a for left and b for right. */
      result = nir_bcsel(&t->b, nir_ieq_imm(&t->b, k, 0), left ? a : b, shifted);
   }
   return put_value(t, inst, result, 0);
}

static bool same_pointer_resource(const struct value *a, const struct value *b);
static const struct value *joined_pointer(const struct value *a, LLVMValueRef b_llvm,
                                           const struct value *b);

static nir_def *
texture_operand(void *translator, LLVMValueRef value)
{
   return get_value(translator, value);
}

static const struct stela_air_texture *
texture_argument(void *translator, LLVMValueRef value, nir_def **id)
{
   const struct value *v = find_value(translator, value);
   *id = v ? v->id : NULL;
   return v && !v->array ? v->texture : NULL;
}

static bool
math_environment(void *translator)
{
   return fp_environment(translator);
}

/* A library function's access of a value at `offset` bytes from `pointer`,
 * aligned to its element; `stored` makes it a store. */
static nir_def *
call_access(struct translator *t, LLVMValueRef pointer, nir_def *offset, nir_def *stored,
            unsigned components, unsigned bits)
{
   const struct value *base = pointer_value(t, pointer);
   unsigned required = stored ? STELA_AIR_WRITE : STELA_AIR_READ;
   if (!base || !(base->access & required)) {
      fail(t, STELA_AIR_MEMORY, "function pointer operand lacks %s access", stored ? "write" : "read");
      return NULL;
   }
   struct value address = *base;
   if (offset) {
      address.def = nir_iadd(&t->b, base->def, nir_u2uN(&t->b, offset, base->def->bit_size));
      pointer_offset(&address, base, 0, bits / 8);
   }
   return memory_access(t, &address, stored, NULL, 0, components, bits, bits / 8);
}

static nir_def *
call_load(void *translator, LLVMValueRef pointer, nir_def *offset, unsigned bits)
{
   return call_access(translator, pointer, offset, NULL, 1, bits);
}

static bool
call_store(void *translator, LLVMValueRef pointer, nir_def *offset, nir_def *value)
{
   struct translator *t = translator;
   call_access(t, pointer, offset, value, value->num_components, value->bit_size);
   return t->error->kind == STELA_AIR_OK;
}

/* LLVM's conversions to, from and between half, float and bfloat, which
 * converts through single precision: exactly, but from an integer, which
 * would round twice and is refused. */
static bool
fp_conversion(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   unsigned components, bits, source_components, source_bits;
   LLVMTypeRef result = LLVMTypeOf(inst), source = LLVMTypeOf(LLVMGetOperand(inst, 0));
   bool to_float = op != LLVMFPToSI && op != LLVMFPToUI;
   bool from_float = op != LLVMSIToFP && op != LLVMUIToFP;
   bool to_bfloat = stela_air_call_bfloat(result), from_bfloat = stela_air_call_bfloat(source);
   if (to_bfloat || from_bfloat) {
      if (to_bfloat && !from_float)
         return fail(t, STELA_AIR_TYPE, "integer to bfloat conversion");
      if (!fp_environment(t))
         return false;
      nir_def *a = get_value(t, LLVMGetOperand(inst, 0));
      if (!a)
         return false;
      nir_def *single = from_bfloat ? stela_air_bfloat_to_float(&t->b, a) : nir_f2f32(&t->b, a);
      if (to_bfloat)
         return put_value(t, inst, stela_air_float_to_bfloat(&t->b, single), 0);
      if (!value_shape(result, &components, &bits) || bits == 1)
         return fail(t, STELA_AIR_TYPE, "unsupported floating-point conversion types");
      nir_def *value = op == LLVMFPToSI ? nir_f2iN(&t->b, single, bits) :
                       op == LLVMFPToUI ? nir_f2uN(&t->b, single, bits) :
                       bits == 16       ? narrow_to_half(t, single) : single;
      return put_value(t, inst, value, 0);
   }
   if (!(to_float ? float_shape(result, &components, &bits) : shape(result, &components, &bits)) ||
       !(from_float ? float_shape(source, &source_components, &source_bits) :
                      shape(source, &source_components, &source_bits)) ||
       components != source_components || (!to_float && bits == 1))
      return fail(t, STELA_AIR_TYPE, "unsupported floating-point conversion types");
   if (!fp_environment(t))
      return false;
   nir_def *a = get_value(t, LLVMGetOperand(inst, 0));
   if (!a)
      return false;
   nir_def *value;
   switch (op) {
   case LLVMFPExt:
   case LLVMFPTrunc:
      value = bits == 16 ? narrow_to_half(t, a) : nir_f2fN(&t->b, a, bits);
      break;
   case LLVMFPToSI: value = nir_f2iN(&t->b, a, bits); break;
   case LLVMFPToUI: value = nir_f2uN(&t->b, a, bits); break;
   case LLVMSIToFP:
      value = source_bits == 1 ? nir_bcsel(&t->b, a, nir_imm_floatN_t(&t->b, -1.0, bits),
                                           nir_imm_floatN_t(&t->b, 0.0, bits)) :
                                 nir_i2fN(&t->b, a, bits);
      break;
   default:
      value = source_bits == 1 ? nir_b2fN(&t->b, a, bits) : nir_u2fN(&t->b, a, bits);
      break;
   }
   return put_value(t, inst, value, 0);
}

/* A constexpr sampler's MTLResourceID, or NULL after reporting a state the
 * options give no sampler. */
static nir_def *
constexpr_sampler(struct translator *t, uint64_t state)
{
   const struct stela_air_options *options = t->options;
   for (unsigned i = 0; i < options->constexpr_sampler_count; i++) {
      if (options->constexpr_samplers[i].state == state)
         return nir_imm_int64(&t->b, options->constexpr_samplers[i].id);
   }
   fail(t, STELA_AIR_ARGUMENT, "constexpr sampler state %#" PRIx64 " has no sampler", state);
   return NULL;
}

static const struct stela_air_sampler *
sampler_argument(void *translator, LLVMValueRef value, nir_def **id)
{
   struct translator *t = translator;
   uint64_t state;
   if (stela_air_metadata_sampler_state(&t->metadata, value, &state)) {
      *id = constexpr_sampler(t, state);
      return NULL;
   }
   const struct value *v = find_value(t, value);
   *id = v ? v->id : NULL;
   return v && !v->array ? v->sampler : NULL;
}

/* The read-write word at a texture call's pointer operand. */
static const struct value *
texture_word(struct translator *t, LLVMValueRef pointer)
{
   const struct value *address = pointer_value(t, pointer);
   if (!address || (address->access & STELA_AIR_READ_WRITE) != STELA_AIR_READ_WRITE) {
      fail(t, STELA_AIR_MEMORY, "texture call lacks a bound read-write pointer");
      return NULL;
   }
   return address;
}

static nir_def *
texture_load_word(void *translator, LLVMValueRef pointer)
{
   struct translator *t = translator;
   const struct value *address = texture_word(t, pointer);
   return address ? memory_access(t, address, NULL, NULL, 0, 1, 32, 4) : NULL;
}

static bool
texture_store_word(void *translator, LLVMValueRef pointer, nir_def *value)
{
   struct translator *t = translator;
   const struct value *address = texture_word(t, pointer);
   if (address)
      memory_access(t, address, value, NULL, 0, 1, 32, 4);
   return t->error->kind == STELA_AIR_OK;
}

/* The tile's dimensions, a 32-bit uvec2: a tile function's imageblock's. */
static nir_def *
tile_extent(struct translator *t)
{
   nir_builder *b = &t->b;
   nir_def *size = nir_channel(b, t->tile, 0);
   return nir_vec2(b, nir_iand_imm(b, size, 0xffff), nir_ushr_imm(b, size, 16));
}

/* Element `pixel` of a tile function's explicit imageblock's member that
 * slice points to: the pass's data at the tile's pixel, whose elements lie
 * a stride apart from the data's base. A multisampled pass's pixels hold a
 * color a sample, which a slice does not choose among. */
static nir_def *
tile_imageblock_load(struct translator *t, const struct value *base, nir_def *pixel,
                     unsigned components, unsigned bits)
{
   if (base->pointer_root != t->imageblock_root || base->choices) {
      fail(t, STELA_AIR_MEMORY, "an imageblock slice that names no member of the imageblock");
      return NULL;
   }
   if (t->options->tile->samples > 1) {
      fail(t, STELA_AIR_INSTRUCTION, "an imageblock slice write in a multisampled tile function");
      return NULL;
   }
   nir_builder *b = &t->b;
   nir_def *pass = t->imageblock_pass;
   nir_def *data = nir_pack_64_2x32_split(b, nir_channel(b, pass, 2), nir_channel(b, pass, 3));
   nir_def *member = nir_umod(b, nir_isub(b, base->def, data), nir_u2u64(b, nir_channel(b, pass, 0)));
   nir_def *width = nir_channel(b, tile_extent(t), 0);
   nir_def *at = nir_vec2(b, nir_umod(b, pixel, width), nir_udiv(b, pixel, width));
   struct value address = *base;
   address.def = nir_iadd(b, imageblock_address(t, tile_pixel(t, at), nir_imm_int(b, 0),
                                                nir_imm_int(b, 1)),
                          member);
   address.offset_known = false;
   return memory_access(t, &address, NULL, NULL, 0, components, bits, bits / 8);
}

/* Element `pixel` of the imageblock's member that slice points to, of
 * `components` lanes of `bits`. */
static nir_def *
imageblock_load(void *translator, LLVMValueRef slice, nir_def *pixel, unsigned components,
                unsigned bits)
{
   struct translator *t = translator;
   const struct value *base = pointer_value(t, slice);
   if (base && t->tile_explicit)
      return tile_imageblock_load(t, base, pixel, components, bits);
   if (!base || !t->imageblock || base->storage != t->imageblock) {
      fail(t, STELA_AIR_MEMORY, "an imageblock slice that names no member of the imageblock");
      return NULL;
   }
   nir_builder *b = &t->b;
   struct value address = *base;
   nir_def *member = nir_umod_imm(b, base->def, t->imageblock_element);
   address.def = nir_iadd(b, member, nir_u2u64(b, nir_imul_imm(b, pixel, t->imageblock_element)));
   address.offset_known = false;
   return memory_access(t, &address, NULL, NULL, 0, components, bits, bits / 8);
}

/* Runs body count times, a loop of the flat CFG, with the 32-bit index of
 * each iteration and the values it carries, each a phi of the loop's
 * header. */
static bool
repeat(void *translator, nir_def *count, unsigned carried, nir_def **values,
       bool (*body)(void *data, nir_def *index, nir_def **values), void *data)
{
   struct translator *t = translator;
   nir_builder *b = &t->b;
   nir_block *entry = nir_cursor_current_block(b->cursor);
   nir_block *header = new_block(t), *start = new_block(t), *exit = new_block(t);
   nir_phi_instr *phis[1 + 8];
   if (carried >= ARRAY_SIZE(phis))
      return fail(t, STELA_AIR_LIMIT, "an AIR loop carries more than %zu values",
                  ARRAY_SIZE(phis) - 1);
   for (unsigned i = 0; i <= carried; i++)
      phis[i] = nir_phi_instr_create(b->shader);
   for (unsigned i = 0; i <= carried; i++)
      if (!phis[i])
         return fail(t, STELA_AIR_ALLOCATION, "AIR loop allocation failed");
   if (!header || !start || !exit)
      return fail(t, STELA_AIR_ALLOCATION, "AIR loop allocation failed");
   nir_def *initial[1 + 8] = {nir_imm_int(b, 0)};
   for (unsigned i = 0; i < carried; i++)
      initial[1 + i] = values[i];
   nir_goto(b, header);
   for (unsigned i = 0; i <= carried; i++) {
      nir_def_init(&phis[i]->instr, &phis[i]->def, initial[i]->num_components,
                   initial[i]->bit_size);
      if (i)
         values[i - 1] = &phis[i]->def;
   }
   nir_def *index = &phis[0]->def;
   b->cursor = nir_after_block(header);
   nir_goto_if(b, start, nir_ult(b, index, count), exit);
   b->cursor = nir_after_block(start);
   if (!body(data, index, values))
      return false;
   nir_block *end = nir_cursor_current_block(b->cursor);
   nir_def *next[1 + 8] = {nir_iadd_imm(b, index, 1)};
   for (unsigned i = 0; i < carried; i++)
      next[1 + i] = values[i];
   nir_goto(b, header);
   for (unsigned i = 0; i <= carried; i++) {
      nir_phi_instr_add_src(phis[i], entry, initial[i]);
      nir_phi_instr_add_src(phis[i], end, next[i]);
      nir_instr_insert(nir_before_block(header), &phis[i]->instr);
      if (i)
         values[i - 1] = &phis[i]->def;
   }
   b->cursor = nir_after_block(exit);
   return true;
}

/* A tile function's prelude: its dispatch's tile and memory, and the raw
 * texel image of each color attachment of its pass. */
static bool
tile_prelude(struct translator *t)
{
   const struct stela_air_tile *tile = t->options->tile;
   if (!(t->tile = device_value(t, STELA_AIR_TILE)))
      return false;
   for (unsigned n = 0; n < AIR_MAX_RENDER_TARGETS; n++) {
      if (tile->formats[n] == PIPE_FORMAT_NONE)
         continue;
      if (!storage_format(tile->formats[n]))
         return fail(t, STELA_AIR_ARGUMENT, "render target %u has no storage texel layout", n);
      if (!attachment_binding(t, n, tile->samples > 1))
         return false;
   }
   return true;
}

static enum glsl_sampler_dim
tile_dim(const struct translator *t)
{
   return t->options->tile->samples > 1 ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
}

/* A tile's color attachment n's image, its texel coordinate at pixel and
 * whether the attachment has that pixel: an edge tile's others hold
 * nothing. */
static nir_deref_instr *
tile_image(struct translator *t, unsigned n, nir_def *pixel, nir_def **coord, nir_def **inside)
{
   nir_builder *b = &t->b;
   nir_deref_instr *image = attachment_image(t, n);
   nir_def *size = nir_image_deref_size(b, 3, 32, &image->def, nir_imm_int(b, 0),
                                        .image_dim = tile_dim(t), .image_array = true);
   *inside = nir_ball(b, nir_ult(b, pixel, nir_trim_vector(b, size, 2)));
   *coord = nir_vec4(b, nir_channel(b, pixel, 0), nir_channel(b, pixel, 1), nir_imm_int(b, 0),
                     nir_imm_int(b, 0));
   return image;
}

/* The words of a tile's color attachment n's texel at coord's sample. */
static nir_def *
tile_load(struct translator *t, unsigned n, nir_deref_instr *image, nir_def *coord, nir_def *sample)
{
   nir_builder *b = &t->b;
   enum pipe_format format = t->options->tile->formats[n];
   nir_def *texel = nir_image_deref_load(b, 4, 32, &image->def, coord, sample, nir_imm_int(b, 0),
                                         .image_dim = tile_dim(t), .image_array = true,
                                         .format = PIPE_FORMAT_NONE, .access = ACCESS_COHERENT,
                                         .dest_type = nir_type_uint32);
   return nir_trim_vector(b, texel, DIV_ROUND_UP(util_format_get_blocksizebits(format), 32));
}

static void
tile_store(struct translator *t, nir_deref_instr *image, nir_def *coord, nir_def *sample,
           nir_def *words)
{
   nir_builder *b = &t->b;
   nir_image_deref_store(b, &image->def, coord, sample, nir_pad_vector_imm_int(b, words, 0, 4),
                         nir_imm_int(b, 0), .image_dim = tile_dim(t), .image_array = true,
                         .format = PIPE_FORMAT_NONE, .access = ACCESS_COHERENT,
                         .src_type = nir_type_uint32);
}

/* Whether every sample of every color attachment at pixel holds its first
 * sample's texel: the pixel's imageblock has one color. */
static nir_def *
tile_one_color(struct translator *t, nir_def *pixel)
{
   nir_builder *b = &t->b;
   const struct stela_air_tile *tile = t->options->tile;
   nir_def *same = nir_imm_true(b);
   for (unsigned n = 0; n < AIR_MAX_RENDER_TARGETS; n++) {
      if (tile->formats[n] == PIPE_FORMAT_NONE)
         continue;
      nir_def *coord, *inside;
      nir_deref_instr *image = tile_image(t, n, pixel, &coord, &inside);
      struct memory_guard guard;
      if (!memory_guard_begin(t, inside, &guard))
         return NULL;
      nir_def *first = tile_load(t, n, image, coord, nir_imm_int(b, 0));
      nir_def *equal = nir_imm_true(b);
      for (unsigned s = 1; s < tile->samples; s++) {
         nir_def *other = tile_load(t, n, image, coord, nir_imm_int(b, s));
         equal = nir_iand(b, equal, nir_ball_iequal(b, first, other));
      }
      /* Outside the attachment the guard reads zero, which counts as one
       * color. */
      nir_def *differ = memory_guard_end(t, &guard, nir_b2i32(b, nir_inot(b, equal)));
      if (!differ)
         return NULL;
      same = nir_iand(b, same, nir_ieq_imm(b, differ, 0));
   }
   return same;
}

/* A value of an implicit imageblock's color from its texel's words. */
static nir_def *
tile_unpack(nir_builder *b, enum pipe_format format, nir_def *words, unsigned components,
            unsigned bits, bool floating)
{
   nir_def *rgba = nir_trim_vector(b, nir_format_unpack_rgba(b, words, format), components);
   return floating ? nir_f2fN(b, rgba, bits) : nir_u2uN(b, rgba, bits);
}

/* The words of a texel holding an implicit imageblock's color value; the
 * channels it lacks are zero, and alpha one. */
static nir_def *
tile_pack(struct translator *t, enum pipe_format format, nir_def *value, bool floating)
{
   nir_builder *b = &t->b;
   bool integer = util_format_is_pure_integer(format);
   bool sint = util_format_is_pure_sint(format);
   nir_def *channels[4];
   for (unsigned i = 0; i < 4; i++) {
      if (i < value->num_components) {
         nir_def *channel = nir_channel(b, value, i);
         channels[i] = floating ? nir_f2f32(b, channel) : sint ? nir_i2i32(b, channel) :
                                                         nir_u2u32(b, channel);
      } else {
         channels[i] = integer ? nir_imm_int(b, i == 3) : nir_imm_float(b, i == 3 ? 1.0f : 0.0f);
      }
   }
   return nir_format_pack_rgba(b, format, nir_vec(b, channels, 4));
}

/* The pixel of the pass at a position in the tile. */
static nir_def *
tile_pixel(struct translator *t, nir_def *at)
{
   nir_builder *b = &t->b;
   nir_def *origin = nir_imul(b, nir_trim_vector(b, nir_load_workgroup_id(b), 2), tile_extent(t));
   return nir_iadd(b, origin, nir_u2u32(b, at));
}

/* A tile function's implicit imageblock functions: a color's load and
 * stores, which access its pass's color attachments at the tile's pixels,
 * and the samples and colors at a pixel. Each sample holds its own color
 * unless every sample of the pixel holds the same: then that is the
 * pixel's one color, and storing it stores every sample. */
static bool
tile_call(struct translator *t, LLVMValueRef inst, const char *name, size_t name_length,
          bool *lowered)
{
   bool load = named_after(name, name_length, "air.load.implicit_imageblock");
   bool masked = named_after(name, name_length, "air.store.implicit_imageblock.mask");
   bool store = !masked && named_after(name, name_length, "air.store.implicit_imageblock");
   bool samples = stela_air_call_named(name, name_length, "air.get_imageblock_samples");
   bool colors = stela_air_call_named(name, name_length, "air.get_imageblock_num_colors");
   bool coverage = stela_air_call_named(name, name_length, "air.get_color_coverage_mask");
   *lowered = load || masked || store || samples || colors || coverage;
   if (!*lowered)
      return true;
   if (!t->tile)
      return fail(t, STELA_AIR_INSTRUCTION, "%.*s outside a tile function", (int)name_length, name);
   nir_builder *b = &t->b;
   const struct stela_air_tile *tile = t->options->tile;
   unsigned operands = LLVMGetNumArgOperands(inst);
   LLVMTypeRef i16 = LLVMInt16TypeInContext(t->context);
   if (samples) {
      if (operands || LLVMTypeOf(inst) != i16)
         return fail(t, STELA_AIR_TYPE, "an imageblock's samples of an unknown form");
      return put_value(t, inst, nir_imm_intN_t(b, tile->samples, 16), 0);
   }
   /* (value,) color, position, then a rate and index or a coverage mask;
    * a position, and a color's index. */
   unsigned first = load ? 0 : 1, at_operand = colors || coverage ? 0 : first + 1;
   unsigned expected = colors ? 1 : coverage ? 2 : load || masked ? 4 : 5;
   unsigned components, bits;
   if (operands != expected ||
       !value_shape(LLVMTypeOf(LLVMGetOperand(inst, at_operand)), &components, &bits) ||
       components != 2 || bits != 16)
      return fail(t, STELA_AIR_TYPE, "%.*s of an unknown form", (int)name_length, name);
   nir_def *at = get_value(t, LLVMGetOperand(inst, at_operand));
   if (!at)
      return false;
   nir_def *pixel = tile_pixel(t, at);
   /* The two layouts count a multisampled pixel's colors apart, and the
    * call names neither. */
   if ((colors || coverage) && t->tile_explicit && t->tile_implicit && tile->samples > 1)
      return fail(t, STELA_AIR_INSTRUCTION, "%.*s in a multisampled tile function of both imageblocks",
                  (int)name_length, name);
   /* An explicit imageblock's samples each hold a color. */
   if (colors) {
      if (LLVMTypeOf(inst) != i16)
         return fail(t, STELA_AIR_TYPE, "an imageblock's colors of an unknown form");
      nir_def *count = nir_imm_intN_t(b, t->tile_explicit ? tile->samples : 1, 16);
      if (tile->samples > 1 && !t->tile_explicit) {
         nir_def *one = tile_one_color(t, pixel);
         if (!one)
            return false;
         count = nir_bcsel(b, one, count, nir_imm_intN_t(b, tile->samples, 16));
      }
      return put_value(t, inst, count, 0);
   }
   /* The samples a color covers: its own, or all of one color's. */
   if (coverage) {
      nir_def *color = get_value(t, LLVMGetOperand(inst, 1));
      if (!color || color->bit_size != 16 || LLVMTypeOf(inst) != i16)
         return fail(t, STELA_AIR_TYPE, "a color coverage mask of an unknown form");
      color = nir_u2u32(b, color);
      nir_def *mask = nir_bcsel(b, nir_ult_imm(b, color, tile->samples),
                                nir_ishl(b, nir_imm_int(b, 1), color), nir_imm_int(b, 0));
      if (tile->samples > 1 && !t->tile_explicit) {
         nir_def *one = tile_one_color(t, pixel);
         if (!one)
            return false;
         nir_def *all = nir_bcsel(b, nir_ieq_imm(b, color, 0),
                                  nir_imm_int(b, BITFIELD_MASK(tile->samples)), nir_imm_int(b, 0));
         mask = nir_bcsel(b, one, all, mask);
      }
      return put_value(t, inst, nir_u2u16(b, mask), 0);
   }
   LLVMValueRef color = LLVMGetOperand(inst, first);
   unsigned n = LLVMIsAConstantInt(color) ? (unsigned)LLVMConstIntGetZExtValue(color) : UINT_MAX;
   if (n >= AIR_MAX_RENDER_TARGETS || tile->formats[n] == PIPE_FORMAT_NONE)
      return fail(t, STELA_AIR_ARGUMENT, "an imageblock color without a color attachment");
   enum pipe_format format = tile->formats[n];
   LLVMTypeRef type = load ? LLVMTypeOf(inst) : LLVMTypeOf(LLVMGetOperand(inst, 0));
   bool floating = float_shape(type, &components, &bits);
   if ((!floating && !shape(type, &components, &bits)) || components > 4 ||
       (bits != 16 && bits != 32) || floating == util_format_is_pure_integer(format))
      return fail(t, STELA_AIR_TYPE, "an imageblock color of another type than its attachment's");
   /* The samples the access names: by a color's index, a sample's or a
    * store's coverage mask. */
   nir_def *sample = NULL, *mask = NULL;
   if (masked) {
      nir_def *coverage = get_value(t, LLVMGetOperand(inst, 3));
      if (!coverage || coverage->bit_size != 16)
         return fail(t, STELA_AIR_TYPE, "an imageblock coverage mask of an unknown form");
      mask = nir_u2u32(b, coverage);
   } else {
      LLVMValueRef rate = LLVMGetOperand(inst, first + 2);
      uint64_t sample_rate = LLVMIsAConstantInt(rate) ? LLVMConstIntGetZExtValue(rate) : 1;
      nir_def *index = get_value(t, LLVMGetOperand(inst, first + 3));
      /* The first color, a color by index or a sample. */
      if (sample_rate > 2 || !index || index->bit_size != 16)
         return fail(t, STELA_AIR_TYPE, "an imageblock access of an unknown rate or index");
      sample = nir_u2u32(b, index);
      if (store) {
         mask = nir_ishl(b, nir_imm_int(b, 1), sample);
         if (sample_rate != 2 && tile->samples > 1) {
            nir_def *one = tile_one_color(t, pixel);
            if (!one)
               return false;
            mask = nir_bcsel(b, one, nir_imm_int(b, BITFIELD_MASK(tile->samples)), mask);
         }
      }
   }
   nir_def *coord, *inside;
   nir_deref_instr *image = tile_image(t, n, pixel, &coord, &inside);
   struct memory_guard guard;
   if (load) {
      if (!memory_guard_begin(t, nir_iand(b, inside, nir_ult_imm(b, sample, tile->samples)), &guard))
         return false;
      nir_def *words = memory_guard_end(t, &guard, tile_load(t, n, image, coord, sample));
      if (!words)
         return false;
      return put_value(t, inst, tile_unpack(b, format, words, components, bits, floating), 0);
   }
   nir_def *value = get_value(t, LLVMGetOperand(inst, 0));
   if (!value)
      return false;
   nir_def *words = tile_pack(t, format, value, floating);
   for (unsigned s = 0; s < tile->samples; s++) {
      nir_def *covered = nir_i2b(b, nir_iand_imm(b, mask, 1u << s));
      if (!memory_guard_begin(t, nir_iand(b, inside, covered), &guard))
         return false;
      tile_store(t, image, coord, nir_imm_int(b, s), words);
      memory_guard_end(t, &guard, NULL);
   }
   return true;
}

/* A tile function's persistent threadgroup memory copy: each thread's
 * 16-byte chunks in turn. */
struct tile_copy {
   struct translator *t;
   nir_def *base, *count, *thread;
   unsigned threads;
   bool in;
};

static bool
tile_copy_chunk(void *data, nir_def *iteration, nir_def **values)
{
   (void)values;
   const struct tile_copy *copy = data;
   struct translator *t = copy->t;
   nir_builder *b = &t->b;
   nir_def *chunk = nir_iadd(b, nir_imul_imm(b, iteration, copy->threads), copy->thread);
   struct memory_guard guard;
   if (!memory_guard_begin(t, nir_ult(b, chunk, copy->count), &guard))
      return false;
   nir_def *offset = nir_imul_imm(b, chunk, 16);
   nir_deref_instr *byte = nir_build_deref_array(b, nir_build_deref_var(b, t->dynamic_threadgroup),
                                                offset);
   nir_deref_instr *shared = nir_build_deref_cast_with_alignment(b, &byte->def, nir_var_mem_shared,
                                                                 glsl_uvec4_type(), 16, 16, 0);
   nir_def *address = nir_iadd(b, copy->base, nir_u2u64(b, offset));
   if (copy->in)
      nir_store_deref(b, shared, nir_load_global(b, 4, 32, address, .align_mul = 16), 0xf);
   else
      nir_store_global(b, nir_load_deref(b, shared), address, .align_mul = 16);
   memory_guard_end(t, &guard, NULL);
   return true;
}

/* Copies a tile function's persistent threadgroup memory between its
 * tile's place in the pass's memory and the threadgroup's: in where every
 * invocation starts, or out where every one leaves. */
static bool
tile_memory_copy(struct translator *t, bool in)
{
   nir_builder *b = &t->b;
   const unsigned *size = t->options->workgroup_size;
   unsigned threads = size[0] * size[1] * size[2];
   unsigned capacity = glsl_get_length(t->dynamic_threadgroup->type) & ~15u;
   nir_def *bytes = nir_umin_imm(b, nir_iand_imm(b, nir_channel(b, t->tile, 1), ~15u), capacity);
   nir_def *id = nir_load_workgroup_id(b), *groups = nir_load_num_workgroups(b);
   nir_def *index = nir_iadd(b, nir_channel(b, id, 0),
                             nir_imul(b, nir_channel(b, id, 1), nir_channel(b, groups, 0)));
   nir_def *memory = nir_pack_64_2x32_split(b, nir_channel(b, t->tile, 2), nir_channel(b, t->tile, 3));
   struct tile_copy copy = {
      .t = t,
      .base = nir_iadd(b, memory, nir_imul(b, nir_u2u64(b, index), nir_u2u64(b, bytes))),
      .count = nir_ushr_imm(b, bytes, 4),
      .thread = nir_load_local_invocation_index(b),
      .threads = threads,
      .in = in,
   };
   if (!in)
      workgroup_barrier(t);
   if (!repeat(t, nir_udiv_imm(b, nir_iadd_imm(b, copy.count, threads - 1), threads), 0, NULL,
               tile_copy_chunk, &copy))
      return false;
   if (in)
      workgroup_barrier(t);
   return true;
}

/* The authority of every pointer into a function's explicit imageblock: a
 * kernel's threadgroup memory, or a tile function's pass's data. */
static struct value *
imageblock_authority(struct translator *t, struct value *authority)
{
   *authority = (struct value){.access = STELA_AIR_READ_WRITE, .pointer_root = t->imageblock_root};
   if (t->imageblock) {
      authority->storage = t->imageblock;
      authority->storage_size = t->imageblock_elements * t->imageblock_element;
   }
   return authority;
}

/* Metal's imageblock functions in a kernel, apart from the slice writes
 * texture_call lowers: an element's pointer, and the dimensions. */
static bool
imageblock_call(struct translator *t, LLVMValueRef inst, const char *name, size_t name_length,
                bool *lowered)
{
   bool data = stela_air_call_named(name, name_length, "air.imageblock_data");
   bool width = stela_air_call_named(name, name_length, "air.get_imageblock_width");
   bool height = stela_air_call_named(name, name_length, "air.get_imageblock_height");
   *lowered = data || width || height;
   if (!*lowered)
      return true;
   nir_builder *b = &t->b;
   if (!data && t->tile) {
      /* A tile function's is its pass's tile, even at the attachments'
       * edge. */
      if (LLVMGetNumArgOperands(inst) || LLVMTypeOf(inst) != LLVMInt16TypeInContext(t->context))
         return fail(t, STELA_AIR_TYPE, "an imageblock dimension of an unknown form");
      nir_def *size = nir_channel(b, t->tile, 0);
      return put_value(t, inst, nir_u2u16(b, width ? size : nir_ushr_imm(b, size, 16)), 0);
   }
   if (!t->imageblock && !t->tile_explicit)
      return fail(t, STELA_AIR_INSTRUCTION, "%.*s in a function without an explicit imageblock",
                  (int)name_length, name);
   if (!data) {
      if (LLVMGetNumArgOperands(inst) || LLVMTypeOf(inst) != LLVMInt16TypeInContext(t->context))
         return fail(t, STELA_AIR_TYPE, "an imageblock dimension of an unknown form");
      return put_value(t, inst, nir_u2u16(b, nir_channel(b, t->imageblock_extent, width ? 0 : 1)), 0);
   }
   /* The element at a position, of the first data and sample. */
   LLVMValueRef position = LLVMGetNumArgOperands(inst) == 3 ? LLVMGetOperand(inst, 0) : NULL;
   LLVMValueRef index = position ? LLVMGetOperand(inst, 1) : NULL, sample = position ? LLVMGetOperand(inst, 2) : NULL;
   unsigned components, bits;
   if (!position || !value_shape(LLVMTypeOf(position), &components, &bits) || components != 2 ||
       bits != 16 || !LLVMIsAConstantInt(index) || (!t->tile_explicit && !LLVMIsAConstantInt(sample)) ||
       LLVMGetTypeKind(LLVMTypeOf(inst)) != LLVMPointerTypeKind ||
       LLVMGetPointerAddressSpace(LLVMTypeOf(inst)) != 4)
      return fail(t, STELA_AIR_TYPE, "imageblock data of an unknown form");
   nir_def *at = get_value(t, position);
   if (!at)
      return false;
   /* A tile function's is its pass's at the tile's pixel: the first color,
    * or a color or sample by index, each sample a color. */
   if (t->tile_explicit) {
      uint64_t rate = LLVMConstIntGetZExtValue(index);
      nir_def *which = get_value(t, sample);
      if (rate > 2 || !which)
         return fail(t, STELA_AIR_TYPE, "imageblock data of an unknown rate");
      unsigned samples = t->options->tile->samples;
      which = rate ? nir_umin_imm(b, nir_u2u32(b, which), samples - 1) : nir_imm_int(b, 0);
      nir_def *address = imageblock_address(t, tile_pixel(t, at), which, nir_imm_int(b, samples));
      if (!put_value(t, inst, address, STELA_AIR_READ_WRITE))
         return false;
      struct value authority;
      pointer_authority(find_value(t, inst), imageblock_authority(t, &authority));
      return true;
   }
   if (LLVMConstIntGetZExtValue(index) || LLVMConstIntGetZExtValue(sample))
      return fail(t, STELA_AIR_INSTRUCTION, "imageblock data of a later index or sample");
   at = nir_u2u32(b, at);
   nir_def *pixel = nir_iadd(b, nir_imul(b, nir_channel(b, at, 1), nir_channel(b, t->imageblock_extent, 0)),
                             nir_channel(b, at, 0));
   nir_def *offset = nir_u2u64(b, nir_imul_imm(b, pixel, t->imageblock_element));
   if (!put_value(t, inst, offset, STELA_AIR_READ_WRITE))
      return false;
   struct value authority;
   pointer_authority(find_value(t, inst), imageblock_authority(t, &authority));
   return true;
}

/* stela_air_texture_context's two-way branches over the flat CFG: then,
 * else and merge blocks, and the then side's last block. */
static bool
texture_branch(void *translator, nir_def *condition, nir_block *blocks[4])
{
   struct memory_guard guard;
   if (!memory_guard_begin(translator, condition, &guard))
      return false;
   blocks[0] = guard.then_block;
   blocks[1] = guard.else_block;
   blocks[2] = guard.merge;
   blocks[3] = NULL;
   return true;
}

static void
texture_branch_else(void *translator, nir_block *blocks[4])
{
   struct translator *t = translator;
   blocks[3] = nir_cursor_current_block(t->b.cursor);
   nir_goto(&t->b, blocks[2]);
   t->b.cursor = nir_after_block(blocks[1]);
}

static nir_def *
texture_branch_join(void *translator, nir_block *const blocks[4], nir_def *then_value,
                    nir_def *else_value)
{
   struct translator *t = translator;
   nir_block *else_end = nir_cursor_current_block(t->b.cursor);
   nir_goto(&t->b, blocks[2]);
   t->b.cursor = nir_after_block(blocks[2]);
   nir_phi_instr *phi = nir_phi_instr_create(t->b.shader);
   if (!phi) {
      fail(t, STELA_AIR_ALLOCATION, "AIR texture branch PHI allocation failed");
      return NULL;
   }
   nir_def_init(&phi->instr, &phi->def, then_value->num_components, then_value->bit_size);
   nir_phi_instr_add_src(phi, blocks[3], then_value);
   nir_phi_instr_add_src(phi, else_end, else_value);
   nir_builder_instr_insert(&t->b, &phi->instr);
   return &phi->def;
}

/* A call of one of AIR's texture functions; false with no error when the
 * call is something else. */
static bool
texture_call(struct translator *t, LLVMValueRef inst, const char *name, size_t name_length,
             bool *lowered)
{
   const struct stela_air_texture_context context = {
      .b = &t->b, .options = t->options, .error = t->error, .translator = t,
      .operand = texture_operand, .texture = texture_argument, .sampler = sampler_argument,
      .load_word = texture_load_word, .store_word = texture_store_word, .store_at = call_store,
      .imageblock_extent = t->imageblock ? t->imageblock_extent :
                           t->tile_explicit ? tile_extent(t) : NULL,
      .imageblock_load = imageblock_load,
      .repeat = repeat, .branch = texture_branch, .branch_else = texture_branch_else,
      .branch_join = texture_branch_join,
   };
   /* A texture chosen with null textures is null where the choice is not
    * it; its other calls are the texture's, made only where it is, and
    * answer zero elsewhere as a null texture's do. */
   const struct value *chosen =
      LLVMGetNumArgOperands(inst) ? find_value(t, LLVMGetOperand(inst, 0)) : NULL;
   nir_def *present = chosen && chosen->texture ? chosen->present : NULL;
   static const char is_null[] = "air.is_null_";
   if (present && name_length > sizeof(is_null) - 1 && !memcmp(name, is_null, sizeof(is_null) - 1)) {
      *lowered = true;
      return put_value(t, inst, nir_inot(&t->b, present), 0);
   }
   struct memory_guard guard;
   if (present && !memory_guard_begin(t, present, &guard))
      return false;
   nir_def *result, *residency;
   bool pair;
   enum stela_air_texture_status status =
      stela_air_texture_call(&context, inst, name, name_length, &result, &pair, &residency);
   if (present && status == STELA_AIR_TEXTURE_LOWERED) {
      if (residency)
         return fail(t, STELA_AIR_INSTRUCTION, "the residency of a texture chosen with null ones");
      result = memory_guard_end(t, &guard, result);
   } else if (present && status == STELA_AIR_TEXTURE_OTHER) {
      return fail(t, STELA_AIR_MEMORY, "a texture chosen with null ones used other than by texture calls");
   }
   switch (status) {
   case STELA_AIR_TEXTURE_OTHER:
      *lowered = false;
      return true;
   case STELA_AIR_TEXTURE_FAILED:
      return false;
   case STELA_AIR_TEXTURE_LOWERED:
      break;
   }
   *lowered = true;
   if (!result)
      return true;
   if (!pair)
      return put_value(t, inst, result, 0);
   return put_pair(t, inst, result, residency);
}

static nir_def *
command_resource(void *translator, LLVMValueRef value, enum stela_air_field_kind kind)
{
   struct translator *t = translator;
   const struct value *v = find_value(t, value);
   if (!v || !v->object || v->object->kind != kind) {
      fail(t, STELA_AIR_ARGUMENT, "command call operand is not its argument buffer field");
      return NULL;
   }
   return v->id;
}

/* A buffer a command binds: a device or constant buffer's address, or none. */
static nir_def *
command_address(void *translator, LLVMValueRef value)
{
   struct translator *t = translator;
   if (LLVMIsAConstantPointerNull(value))
      return nir_imm_int64(&t->b, 0);
   const struct value *v = pointer_value(t, value);
   if (!v || !v->pointer_root || v->storage || v->constant_data || v->ssbo || v->array ||
       v->object || !v->def || v->def->bit_size != 64) {
      fail(t, STELA_AIR_MEMORY, "indirect commands bind device or constant buffer addresses");
      return NULL;
   }
   return v->def;
}

/* Copies bytes to encoding's destination, 16 at a time: a loop of the flat
 * CFG. */
static bool
command_copy(struct translator *t, const struct stela_air_command_encoding *encoding)
{
   nir_builder *b = &t->b;
   nir_block *entry = nir_cursor_current_block(b->cursor);
   nir_block *header = new_block(t), *body = new_block(t), *exit = new_block(t);
   nir_phi_instr *phi = nir_phi_instr_create(b->shader);
   if (!header || !body || !exit || !phi)
      return fail(t, STELA_AIR_ALLOCATION, "AIR command copy allocation failed");
   nir_def *zero = nir_imm_int(b, 0);
   nir_goto(b, header);
   /* The offset PHI heads the header once both its sources exist. */
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   b->cursor = nir_after_block(header);
   nir_goto_if(b, body, nir_ult(b, &phi->def, encoding->bytes), exit);
   b->cursor = nir_after_block(body);
   nir_def *offset = nir_u2u64(b, &phi->def);
   nir_def *chunk = encoding->source ?
      nir_load_global(b, 4, 32, nir_iadd(b, encoding->source, offset), .align_mul = 16) :
      nir_imm_zero(b, 4, 32);
   nir_store_global(b, chunk, nir_iadd(b, encoding->destination, offset), .align_mul = 16);
   nir_def *next = nir_iadd_imm(b, &phi->def, 16);
   nir_goto(b, header);
   nir_phi_instr_add_src(phi, entry, zero);
   nir_phi_instr_add_src(phi, body, next);
   nir_instr_insert(nir_before_block(header), &phi->instr);
   b->cursor = nir_after_block(exit);
   return true;
}

/* An indirect command's encoding: its stores, under guards that consecutive
 * stores with one condition share, then its copy under its own. */
static bool
command_writes(struct translator *t, const struct stela_air_command_encoding *encoding)
{
   if (encoding->store_count > STELA_AIR_COMMAND_STORES)
      return fail(t, STELA_AIR_ARGUMENT, "command callback exceeded its stores");
   for (unsigned i = 0; i < encoding->store_count;) {
      nir_def *valid = encoding->stores[i].valid;
      struct memory_guard guard;
      if (!memory_guard_begin(t, valid, &guard))
         return false;
      for (; i < encoding->store_count && encoding->stores[i].valid == valid; i++) {
         const struct stela_air_command_store *store = &encoding->stores[i];
         nir_store_global(&t->b, store->value, store->address,
                          .align_mul = store->value->bit_size / 8);
      }
      memory_guard_end(t, &guard, NULL);
   }
   if (!encoding->bytes)
      return true;
   struct memory_guard guard;
   if (!memory_guard_begin(t, encoding->copy_valid, &guard) || !command_copy(t, encoding))
      return false;
   memory_guard_end(t, &guard, NULL);
   return true;
}

/* A call of one of AIR's indirect command functions; false with no error
 * when the call is something else. */
static bool
command_call(struct translator *t, LLVMValueRef inst, LLVMValueRef callee, const char *name,
             size_t name_length, bool *lowered)
{
   const struct stela_air_command_context context = {
      .call = {.b = &t->b, .error = t->error, .translator = t, .operand = texture_operand},
      .resource_id = command_resource, .address = command_address,
   };
   struct stela_air_command command;
   bool size;
   switch (stela_air_command_call(&context, inst, callee, name, name_length, &command, &size)) {
   case STELA_AIR_CALL_OTHER:
      *lowered = false;
      return true;
   case STELA_AIR_CALL_FAILED:
      return false;
   case STELA_AIR_CALL_LOWERED:
      break;
   }
   *lowered = true;
   const struct stela_air_options *o = t->options;
   if (!o->command_count || !o->encode_command)
      return fail(t, STELA_AIR_INSTRUCTION, "indirect command functions are unbound");
   if (size) {
      nir_def *count = o->command_count(&t->b, command.buffer, o->binding_data);
      if (!count || count->num_components != 1 || count->bit_size != 32)
         return fail(t, STELA_AIR_ARGUMENT, "command count callback refused a buffer");
      return put_value(t, inst, count, 0);
   }
   struct stela_air_command_encoding encoding = {0};
   if (!o->encode_command(&t->b, &command, &encoding, o->binding_data))
      return fail(t, STELA_AIR_ARGUMENT, "command callback refused an indirect command");
   return command_writes(t, &encoding);
}

/* extractvalue of a two-member result. */
static bool
pair_member(struct translator *t, LLVMValueRef inst, const struct value *pair)
{
   if (LLVMGetNumIndices(inst) != 1)
      return fail(t, STELA_AIR_INSTRUCTION, "nested pair indexing");
   unsigned index = LLVMGetIndices(inst)[0];
   if (index > 1 || !pair->members[index])
      return fail(t, STELA_AIR_INSTRUCTION, "pair member %u is not lowered", index);
   return put_value(t, inst, pair->members[index], 0);
}

static bool atomic_form(struct translator *t, LLVMValueRef inst, unsigned form, unsigned space,
                        LLVMValueRef pointer, LLVMValueRef operand, unsigned operands,
                        LLVMTypeRef type, const char *name, size_t length);

/* The memory a set of Metal memory flags names: 1 device, 2 threadgroup,
 * 4 texture, 8 threadgroup imageblock and 16 object data memory. A kernel's
 * explicit imageblock is threadgroup memory; a tile function's implicit one
 * is its attachments' images and its explicit one device memory. */
static nir_variable_mode
memory_flag_modes(const struct translator *t, unsigned flags)
{
   bool tile = t->tile != NULL;
   return (flags & 1 || (flags & 8 && tile) ? nir_var_mem_ssbo | nir_var_mem_global : 0) |
          (flags & 2 || (flags & 8 && !tile) ? nir_var_mem_shared : 0) |
          (flags & 4 || (flags & 8 && tile) ? nir_var_image : 0) |
          (flags & 16 ? nir_var_mem_task_payload : 0);
}

/* A C++ memory order's semantics, 0 relaxed to 5 sequentially consistent:
 * consume is acquire, and sequential consistency acquire and release, as
 * SPIR-V's is. */
static nir_memory_semantics
order_semantics(unsigned order)
{
   return order == 1 || order == 2 ? NIR_MEMORY_ACQUIRE :
          order == 3 ? NIR_MEMORY_RELEASE :
          order >= 4 ? NIR_MEMORY_ACQ_REL : 0;
}

/* Metal's scope of memory: 1 threadgroup, 4 SIMD-group and otherwise
 * device (2, or Metal 1's 3). */
static mesa_scope
memory_scope(unsigned scope)
{
   return scope == 1 ? SCOPE_WORKGROUP : scope == 4 ? SCOPE_SUBGROUP : SCOPE_DEVICE;
}

/* The part of an ordered atomic's semantics a barrier on one side of it
 * implements: release before it, acquire after. */
static void
atomic_order_barrier(struct translator *t, nir_memory_semantics semantics, unsigned scope,
                     nir_variable_mode modes)
{
   if (!semantics || !modes)
      return;
   nir_barrier(&t->b, .execution_scope = SCOPE_NONE, .memory_scope = memory_scope(scope),
               .memory_semantics = semantics, .memory_modes = modes);
}

/* AIR's relaxed atomics, as Apple's compiler names them:
 *
 *   air.atomic.<global|local>.load.<type>(pointer, order, scope, volatile)
 *   air.atomic.<global|local>.<store|xchg>.<type>(pointer, value, order, scope,
 *                                                 volatile)
 *   air.atomic.<global|local>.cmpxchg.weak.<type>(pointer, expected, desired,
 *                                                 success, failure, scope,
 *                                                 volatile)
 *   air.atomic.<global|local>.<add|sub|and|or|xor|min|max>.<sign><type>(
 *       pointer, value, order, scope, volatile)
 *
 * From AIR 2.9 an i32 follows the scope: the memory flags an ordered
 * atomic orders, which a relaxed one leaves 0. The volatility, when
 * present, stays last.
 *
 * A floating-point type is bare ("f32"); an integer one carries its sign
 * ("s.i32") for arithmetic, which selects min and max. An order other than
 * relaxed is barriers around the atomic over its flags' memory: release
 * before, acquire after. Without flags (before AIR 2.9) it orders the
 * atomic's own memory. The atomic itself needs no scope, as NIR's are
 * device-scope. A
 * compare-exchange returns the old value and writes it through expected, a
 * thread pointer. The 64-bit min and max return nothing. */
/* A coherent(device) load or store: air.load.device_coherent.<type>.p1<type>
 * and air.store.device_coherent.<type>.p1<type>, which other threadgroups'
 * accesses see once a device fence orders them. */
static bool
coherent_call(struct translator *t, LLVMValueRef inst, const char *name, size_t length, bool *lowered)
{
   static const char load[] = "air.load.device_coherent.", store[] = "air.store.device_coherent.";
   bool storing = length > strlen(store) && !memcmp(name, store, strlen(store));
   *lowered = storing || (length > strlen(load) && !memcmp(name, load, strlen(load)));
   if (!*lowered)
      return true;
   unsigned arguments = LLVMGetNumArgOperands(inst);
   if (arguments != (storing ? 2u : 1u))
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported coherent device access %.*s",
                  (int)MIN2(length, 160), name);
   LLVMValueRef pointer = LLVMGetOperand(inst, storing ? 1 : 0);
   LLVMValueRef value = storing ? LLVMGetOperand(inst, 0) : inst;
   LLVMTypeRef type = LLVMTypeOf(value);
   char overload[16], expected[64];
   stela_air_call_overload(type, overload, sizeof(overload));
   snprintf(expected, sizeof(expected), "%s%s.p1", storing ? store : load, overload);
   unsigned components, bits;
   if (length <= strlen(expected) || memcmp(name, expected, strlen(expected)) ||
       !value_shape(type, &components, &bits) ||
       LLVMGetTypeKind(LLVMTypeOf(pointer)) != LLVMPointerTypeKind ||
       LLVMGetPointerAddressSpace(LLVMTypeOf(pointer)) != 1)
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported coherent device access %.*s",
                  (int)MIN2(length, 160), name);
   const struct value *address = pointer_value(t, pointer);
   if (!address || !(address->access & (storing ? STELA_AIR_WRITE : STELA_AIR_READ)))
      return fail(t, STELA_AIR_MEMORY, "coherent access lacks a device pointer");
   nir_def *stored = storing ? get_value(t, value) : NULL;
   if (storing && !stored)
      return false;
   nir_def *loaded = memory_access(t, address, stored, NULL, ACCESS_COHERENT, components, bits,
                                   LLVMABIAlignmentOfType(t->layout, type));
   return storing ? t->error->kind == STELA_AIR_OK : put_value(t, inst, loaded, 0);
}

static bool
air_atomic_call(struct translator *t, LLVMValueRef inst, LLVMValueRef callee, const char *name,
                size_t length, bool *lowered)
{
   static const char *const spaces[] = {"air.atomic.global.", "air.atomic.local."};
   unsigned space = 0;
   while (space < ARRAY_SIZE(spaces) &&
          (length <= strlen(spaces[space]) || memcmp(name, spaces[space], strlen(spaces[space]))))
      space++;
   *lowered = space < ARRAY_SIZE(spaces);
   if (!*lowered)
      return true;
   enum form { LOAD, STORE, EXCHANGE, COMPARE, ARITHMETIC };
   /* Each form's name and value operands. */
   static const struct {
      const char *name;
      unsigned operands;
   } forms[] = {
      [LOAD] = {"load.", 1}, [STORE] = {"store.", 2}, [EXCHANGE] = {"xchg.", 2},
      [COMPARE] = {"cmpxchg.weak.", 3}, [ARITHMETIC] = {"", 2},
   };
   const char *operation = name + strlen(spaces[space]);
   size_t operation_length = length - strlen(spaces[space]);
   enum form form = ARITHMETIC;
   for (unsigned i = 0; i < ARITHMETIC; i++) {
      if (operation_length > strlen(forms[i].name) &&
          !memcmp(operation, forms[i].name, strlen(forms[i].name)))
         form = i;
   }
   /* Then the orders, a compare-exchange's success and failure, the scope,
    * from AIR 2.9 its qualifier and, from Metal 2, the volatility. */
   unsigned operands = forms[form].operands, scope = operands + (form == COMPARE ? 2 : 1);
   unsigned count = LLVMGetNumArgOperands(inst);
   if (!LLVMIsAFunction(callee) || !LLVMIsDeclaration(callee) ||
       LLVMGetFunctionCallConv(callee) != LLVMCCallConv ||
       LLVMGetInstructionCallConv(inst) != LLVMCCallConv || count < scope + 1 || count > scope + 3 ||
       LLVMCountParams(callee) != count || LLVMIsFunctionVarArg(LLVMGlobalGetValueType(callee)) ||
       LLVMGetCalledFunctionType(inst) != LLVMGlobalGetValueType(callee) ||
       LLVMGetNumOperandBundles(inst))
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported AIR atomic call signature");
   LLVMValueRef pointer = LLVMGetOperand(inst, 0);
   LLVMValueRef operand = form == LOAD ? NULL : LLVMGetOperand(inst, operands - 1);
   LLVMTypeRef type = operand ? LLVMTypeOf(operand) : LLVMTypeOf(inst);
   bool returns = LLVMGetTypeKind(LLVMTypeOf(inst)) != LLVMVoidTypeKind;
   bool typed = form == STORE ? !returns :
                form == ARITHMETIC ? !returns || LLVMTypeOf(inst) == type : LLVMTypeOf(inst) == type;
   if (LLVMGetTypeKind(LLVMTypeOf(pointer)) != LLVMPointerTypeKind ||
       LLVMGetPointerAddressSpace(LLVMTypeOf(pointer)) != (space ? 3u : 1u) || !typed)
      return fail(t, STELA_AIR_INSTRUCTION, "AIR atomic types disagree with its name");
   for (unsigned i = operands; i < count; i++) {
      LLVMValueRef value = LLVMGetOperand(inst, i);
      /* The volatility is the one i1; a lone operand past the scope is it
       * only when it is one bit wide. */
      LLVMTypeRef kind = LLVMTypeOf(value);
      bool volatility = i == count - 1 && i > scope &&
                        LLVMGetTypeKind(kind) == LLVMIntegerTypeKind && LLVMGetIntTypeWidth(kind) == 1;
      bool qualifier = i == scope + 1 && !volatility;
      if (!LLVMIsAConstantInt(value) ||
          LLVMGetIntTypeWidth(LLVMTypeOf(value)) != (volatility ? 1u : 32u) ||
          (i > scope && !volatility && !qualifier))
         return fail(t, STELA_AIR_INSTRUCTION, "AIR atomic order, scope and volatility must be constants");
      uint64_t constant = LLVMConstIntGetZExtValue(value);
      if (i < scope && constant > 5)
         return fail(t, STELA_AIR_MEMORY, "unsupported AIR atomic order");
      if (qualifier && constant & ~31u)
         return fail(t, STELA_AIR_MEMORY, "unsupported AIR atomic memory flags");
   }
   /* A compare-exchange orders as its success order, which is at least as
    * strong as its failure's. */
   unsigned order = (unsigned)LLVMConstIntGetZExtValue(LLVMGetOperand(inst, operands));
   unsigned memory = (unsigned)LLVMConstIntGetZExtValue(LLVMGetOperand(inst, scope));
   bool flagged = count > scope + 1 &&
                  LLVMGetIntTypeWidth(LLVMTypeOf(LLVMGetOperand(inst, scope + 1))) == 32;
   unsigned flags = flagged ? (unsigned)LLVMConstIntGetZExtValue(LLVMGetOperand(inst, scope + 1)) :
                              space ? 2u : 1u;
   nir_memory_semantics semantics = order_semantics(order);
   nir_variable_mode modes = memory_flag_modes(t, flags);
   atomic_order_barrier(t, semantics & NIR_MEMORY_RELEASE, memory, modes);
   bool done = atomic_form(t, inst, form, space, pointer, operand, operands, type, name, length);
   if (done)
      atomic_order_barrier(t, semantics & NIR_MEMORY_ACQUIRE, memory, modes);
   return done;
}

/* An AIR atomic's access itself, once its call is validated. */
static bool
atomic_form(struct translator *t, LLVMValueRef inst, unsigned form, unsigned space,
            LLVMValueRef pointer, LLVMValueRef operand, unsigned operands, LLVMTypeRef type,
            const char *name, size_t length)
{
   enum form { LOAD, STORE, EXCHANGE, COMPARE, ARITHMETIC };
   static const char *const spaces[] = {"air.atomic.global.", "air.atomic.local."};
   static const char *const form_names[] = {
      [LOAD] = "load.", [STORE] = "store.", [EXCHANGE] = "xchg.", [COMPARE] = "cmpxchg.weak.",
      [ARITHMETIC] = "",
   };
   bool returns = LLVMGetTypeKind(LLVMTypeOf(inst)) != LLVMVoidTypeKind;
   char overload[16], expected[64];
   stela_air_call_overload(type, overload, sizeof(overload));
   bool floating = LLVMGetTypeKind(type) == LLVMFloatTypeKind;
   unsigned alignment = LLVMABIAlignmentOfType(t->layout, type);
   unsigned components, bits;
   if (form != ARITHMETIC) {
      snprintf(expected, sizeof(expected), "%s%s%s", spaces[space], form_names[form], overload);
      if (!stela_air_call_named(name, length, expected))
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported AIR atomic function");
      if (!value_shape(type, &components, &bits) || components != 1 ||
          (!floating && bits != 32 && bits != 64))
         return fail(t, STELA_AIR_TYPE, "unsupported atomic operand type");
   }
   switch (form) {
   case LOAD:
   case STORE: {
      const struct value *address = pointer_value(t, pointer);
      if (!address || !(address->access & (form == LOAD ? STELA_AIR_READ : STELA_AIR_WRITE)) ||
          thread_memory(address))
         return fail(t, STELA_AIR_MEMORY, "atomic access lacks a device or threadgroup pointer");
      nir_def *stored = operand ? get_value(t, operand) : NULL;
      if (operand && !stored)
         return false;
      /* Relaxed atomic loads and stores are single, coherent accesses. */
      nir_def *loaded = memory_access(t, address, stored, NULL, ACCESS_ATOMIC | ACCESS_COHERENT,
                                      1, bits, alignment);
      return form == STORE ? t->error->kind == STELA_AIR_OK : put_value(t, inst, loaded, 0);
   }
   case EXCHANGE:
      return atomic_access(t, inst, pointer, nir_atomic_op_xchg, false, operand, NULL, alignment);
   case COMPARE: {
      /* expected holds the value compared, and receives the old one. */
      const struct value *slot = pointer_value(t, LLVMGetOperand(inst, 1));
      if (!slot || !thread_memory(slot) ||
          (slot->access & STELA_AIR_READ_WRITE) != STELA_AIR_READ_WRITE)
         return fail(t, STELA_AIR_MEMORY, "compare-exchange expects a thread pointer");
      nir_def *compared = memory_access(t, slot, NULL, NULL, 0, 1, bits, alignment);
      nir_def *desired = compared ? get_value(t, operand) : NULL;
      if (!desired)
         return false;
      nir_def *old = atomic_old(t, pointer, nir_atomic_op_cmpxchg, false, type, desired, compared,
                                alignment);
      if (!old)
         return false;
      memory_access(t, slot, old, NULL, 0, 1, bits, alignment);
      return t->error->kind == STELA_AIR_OK && put_value(t, inst, old, 0);
   }
   case ARITHMETIC:
      break;
   }
   /* Float atomics add and subtract; the integer forms are unsigned, signed. */
   static const struct {
      const char *name;
      nir_atomic_op integer[2];
      bool floating, negate;
   } arithmetic[] = {
      {"add", {nir_atomic_op_iadd, nir_atomic_op_iadd}, true, false},
      {"sub", {nir_atomic_op_iadd, nir_atomic_op_iadd}, true, true},
      {"and", {nir_atomic_op_iand, nir_atomic_op_iand}, false, false},
      {"or", {nir_atomic_op_ior, nir_atomic_op_ior}, false, false},
      {"xor", {nir_atomic_op_ixor, nir_atomic_op_ixor}, false, false},
      {"min", {nir_atomic_op_umin, nir_atomic_op_imin}, false, false},
      {"max", {nir_atomic_op_umax, nir_atomic_op_imax}, false, false},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(arithmetic); i++) {
      if (floating && !arithmetic[i].floating)
         continue;
      for (unsigned sign = 0; sign < (floating ? 1u : 2u); sign++) {
         snprintf(expected, sizeof(expected), "%s%s.%s%s", spaces[space], arithmetic[i].name,
                  floating ? "" : sign ? "s." : "u.", overload);
         if (!stela_air_call_named(name, length, expected))
            continue;
         nir_atomic_op op = floating ? nir_atomic_op_fadd : arithmetic[i].integer[sign];
         if (returns)
            return atomic_access(t, inst, pointer, op, arithmetic[i].negate, operand, NULL, alignment);
         nir_def *data = get_value(t, operand);
         return data && atomic_old(t, pointer, op, arithmetic[i].negate, type, data, NULL, alignment);
      }
   }
   return fail(t, STELA_AIR_INSTRUCTION, "unsupported AIR atomic function");
}

static nir_def *
log_buffer(void *translator)
{
   struct translator *t = translator;
   if (!t->options->load_device_value) {
      fail(t, STELA_AIR_INSTRUCTION, "os_log has no log buffer here");
      return NULL;
   }
   return t->options->load_device_value(&t->b, STELA_AIR_LOG_BUFFER, t->options->binding_data);
}

static bool
log_guard_begin(void *translator, nir_def *condition, nir_block *blocks[3])
{
   struct memory_guard guard;
   if (!memory_guard_begin(translator, condition, &guard))
      return false;
   blocks[0] = guard.then_block;
   blocks[1] = guard.else_block;
   blocks[2] = guard.merge;
   return true;
}

static void
log_guard_end(void *translator, nir_block *const blocks[3])
{
   const struct memory_guard guard = {blocks[0], blocks[1], blocks[2]};
   memory_guard_end(translator, &guard, NULL);
}

/* A call of os_log's methods; false with no error when it is something else. */
static bool
log_call(struct translator *t, LLVMValueRef inst, LLVMValueRef callee, bool *lowered)
{
   const struct stela_air_log_context context = {
      .call = {.b = &t->b, .error = t->error, .translator = t, .operand = texture_operand,
               .fragment = t->stage_options && t->stage_options->stage == STELA_AIR_FRAGMENT},
      .buffer = log_buffer, .guard_begin = log_guard_begin, .guard_end = log_guard_end,
   };
   switch (stela_air_log_call(&context, inst, callee)) {
   case STELA_AIR_CALL_OTHER:
      *lowered = false;
      return true;
   case STELA_AIR_CALL_FAILED:
      return false;
   case STELA_AIR_CALL_LOWERED:
      break;
   }
   *lowered = true;
   return true;
}

/* Whether a barrier's call stays where it is: linked Metal 1 libraries mark it
 * noduplicate, and others convergent. */
static bool
barrier_stays(LLVMValueRef inst, LLVMValueRef callee)
{
   static const char *const attributes[] = {"convergent", "noduplicate"};
   for (unsigned i = 0; i < ARRAY_SIZE(attributes); i++) {
      unsigned kind = LLVMGetEnumAttributeKindForName(attributes[i], strlen(attributes[i]));
      if (LLVMGetEnumAttributeAtIndex(callee, LLVMAttributeFunctionIndex, kind) &&
          LLVMGetCallSiteEnumAttribute(inst, LLVMAttributeFunctionIndex, kind))
         return true;
   }
   return false;
}

/* Whether inst calls a SIMD-group asynchronous copy, whose event pointer
 * simdgroup_events leaves unused. */
static bool
async_copy_call_named(LLVMValueRef inst)
{
   LLVMValueRef callee = LLVMIsACallInst(inst) ? LLVMGetCalledValue(inst) : NULL;
   size_t length = 0;
   const char *name = callee && LLVMIsAFunction(callee) ? LLVMGetValueName2(callee, &length) : NULL;
   return name && (named_after(name, length, "air.simdgroup_async_copy_1d") ||
                   named_after(name, length, "air.simdgroup_async_copy_2d"));
}

/* The callee name of a call to a declared function, or NULL. */
static const char *
declared_callee(LLVMValueRef inst, size_t *length)
{
   LLVMValueRef callee = LLVMIsACallInst(inst) ? LLVMGetCalledValue(inst) : NULL;
   if (!callee || !LLVMIsAFunction(callee) || !LLVMIsDeclaration(callee))
      return NULL;
   return LLVMGetValueName2(callee, length);
}

/* SIMD-group asynchronous copies complete where they are made
 * (async_copy_call), so every event is null: waits have nothing to wait for
 * and go, a copy's event is not null, and the null event is. Events then
 * reach no memory, so stack slots holding them promote away. */
static bool
simdgroup_events(struct translator *t, LLVMValueRef fn)
{
   unsigned count = 0, capacity = 16;
   LLVMValueRef *calls = ralloc_array(t->cfg_scratch, LLVMValueRef, capacity);
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block && calls;
        block = LLVMGetNextBasicBlock(block)) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block); inst && calls;
           inst = LLVMGetNextInstruction(inst)) {
         size_t length;
         const char *name = declared_callee(inst, &length);
         if (!name || (!stela_air_call_named(name, length, "air.is_null_simdgroup_event") &&
                       !stela_air_call_named(name, length, "air.get_null_simdgroup_event") &&
                       !stela_air_call_named(name, length, "air.wait_simdgroup_events") &&
                       !async_copy_call_named(inst)))
            continue;
         if (count == capacity)
            calls = reralloc(t->cfg_scratch, calls, LLVMValueRef, capacity *= 2);
         if (calls)
            calls[count++] = inst;
      }
   }
   if (!calls)
      return fail(t, STELA_AIR_ALLOCATION, "SIMD-group event list allocation failed");
   LLVMBuilderRef builder = LLVMCreateBuilderInContext(LLVMGetModuleContext(t->module));
   if (!builder)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM builder allocation failed");
   LLVMTypeRef i1 = LLVMInt1TypeInContext(LLVMGetModuleContext(t->module));
   /* Null checks first, while their events are still the calls made. */
   for (unsigned i = 0; i < count; i++) {
      size_t length;
      const char *name = declared_callee(calls[i], &length);
      if (!stela_air_call_named(name, length, "air.is_null_simdgroup_event"))
         continue;
      if (LLVMGetNumArgOperands(calls[i]) != 1) {
         LLVMDisposeBuilder(builder);
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported SIMD-group event check signature");
      }
      LLVMValueRef event = LLVMGetOperand(calls[i], 0);
      LLVMValueRef answer;
      if (async_copy_call_named(event)) {
         answer = LLVMConstInt(i1, 0, false);
      } else {
         LLVMPositionBuilderBefore(builder, calls[i]);
         answer = LLVMBuildICmp(builder, LLVMIntEQ, event,
                                LLVMConstPointerNull(LLVMTypeOf(event)), "");
      }
      LLVMReplaceAllUsesWith(calls[i], answer);
      LLVMInstructionEraseFromParent(calls[i]);
      calls[i] = NULL;
   }
   LLVMDisposeBuilder(builder);
   for (unsigned i = 0; i < count; i++) {
      size_t length;
      const char *name = calls[i] ? declared_callee(calls[i], &length) : NULL;
      if (!name)
         continue;
      if (stela_air_call_named(name, length, "air.wait_simdgroup_events")) {
         LLVMInstructionEraseFromParent(calls[i]);
         continue;
      }
      LLVMTypeRef type = LLVMTypeOf(calls[i]);
      if (LLVMGetTypeKind(type) != LLVMPointerTypeKind)
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported SIMD-group event signature");
      LLVMReplaceAllUsesWith(calls[i], LLVMConstPointerNull(type));
      if (stela_air_call_named(name, length, "air.get_null_simdgroup_event"))
         LLVMInstructionEraseFromParent(calls[i]);
   }
   return true;
}

/* A SIMD-group asynchronous copy, which the SIMD-group's threads run at
 * once, element by element in turn: a destination tile of width x height
 * elements of `bytes` bytes, rows dst_row elements apart and elements
 * dst_stride apart, from the source's. Element (x, y) reads the source's
 * at (x, y) + offset, signed, which past its tile of src_width x src_height
 * elements reads as zero, or with `edge` as the nearest element of the
 * tile. */
struct async_copy {
   struct translator *t;
   struct value dst, src;
   nir_def *dst_row, *dst_stride, *src_row, *src_stride;
   nir_def *width, *src_width, *src_height, *count, *lane, *lanes;
   nir_def *offset_x, *offset_y;
   unsigned bytes;
   bool edge;
};

/* The byte offset of element (x, y), rows `row` elements apart and
 * elements `stride` apart, as a 64-bit value. */
static nir_def *
element_offset(nir_builder *b, nir_def *x, nir_def *y, nir_def *row, nir_def *stride,
               unsigned bytes)
{
   nir_def *elements = nir_iadd(b, nir_imul(b, nir_u2u64(b, y), row), nir_imul(b, nir_u2u64(b, x), stride));
   return nir_imul_imm(b, elements, bytes);
}

static struct value
offset_pointer(nir_builder *b, const struct value *base, nir_def *offset)
{
   struct value address = *base;
   address.def = nir_iadd(b, base->def, nir_u2uN(b, offset, base->def->bit_size));
   address.offset_known = false;
   return address;
}

static bool
async_copy_element(void *data, nir_def *iteration, nir_def **values)
{
   (void)values;
   struct async_copy *c = data;
   struct translator *t = c->t;
   nir_builder *b = &t->b;
   nir_def *index = nir_iadd(b, nir_imul(b, iteration, c->lanes), c->lane);
   struct memory_guard within;
   if (!memory_guard_begin(t, nir_ult(b, index, c->count), &within))
      return false;
   nir_def *y = nir_udiv(b, index, c->width);
   nir_def *x = nir_isub(b, index, nir_imul(b, y, c->width));
   nir_def *sx = nir_iadd(b, x, c->offset_x), *sy = nir_iadd(b, y, c->offset_y);
   nir_def *read;
   if (c->edge) {
      nir_def *zero = nir_imm_int(b, 0);
      sx = nir_imin(b, nir_imax(b, sx, zero), nir_iadd_imm(b, c->src_width, -1));
      sy = nir_imin(b, nir_imax(b, sy, zero), nir_iadd_imm(b, c->src_height, -1));
      read = nir_iand(b, nir_ine_imm(b, c->src_width, 0), nir_ine_imm(b, c->src_height, 0));
   } else {
      /* A coordinate before the tile is negative, so past it unsigned. */
      read = nir_iand(b, nir_ult(b, sx, c->src_width), nir_ult(b, sy, c->src_height));
   }
   unsigned bits = MIN2(c->bytes, 4) * 8, components = MAX2(c->bytes / 4, 1);
   struct memory_guard reading;
   if (!memory_guard_begin(t, read, &reading))
      return false;
   struct value from =
      offset_pointer(b, &c->src, element_offset(b, sx, sy, c->src_row, c->src_stride, c->bytes));
   nir_def *value = memory_access(t, &from, NULL, NULL, 0, components, bits,
                                  c->bytes);
   if (!value)
      return false;
   /* Zero past the source's tile. */
   value = memory_guard_end(t, &reading, value);
   if (!value)
      return false;
   struct value to =
      offset_pointer(b, &c->dst, element_offset(b, x, y, c->dst_row, c->dst_stride, c->bytes));
   memory_access(t, &to, value, NULL, 0, components, bits, c->bytes);
   memory_guard_end(t, &within, NULL);
   return t->error->kind == STELA_AIR_OK;
}

/* air.simdgroup_async_copy_2d(size, alignment, dst, dst elements per row,
 * dst element stride, dst tile, src, src elements per row, src element
 * stride, src tile, offset of the source's elements, clamp mode) and
 * air.simdgroup_async_copy_1d(size, alignment, dst, src, elements): the copy
 * runs where it is made, and the SIMD-group barrier after it shows each
 * thread's elements to the others, so its event completes at once. */
static bool
async_copy_call(struct translator *t, LLVMValueRef inst, const char *name, size_t length,
                bool *lowered)
{
   bool two_d = named_after(name, length, "air.simdgroup_async_copy_2d");
   if (!two_d && !named_after(name, length, "air.simdgroup_async_copy_1d"))
      return *lowered = false, true;
   *lowered = true;
   nir_builder *b = &t->b;
   unsigned count = LLVMGetNumArgOperands(inst);
   if (count != (two_d ? 12u : 5u))
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported SIMD-group copy signature");
   LLVMValueRef size = LLVMGetOperand(inst, 0), alignment = LLVMGetOperand(inst, 1);
   if (!LLVMIsAConstantInt(size) || !LLVMIsAConstantInt(alignment))
      return fail(t, STELA_AIR_INSTRUCTION, "SIMD-group copies of elements of a dynamic size");
   uint64_t bytes = LLVMConstIntGetZExtValue(size), align = LLVMConstIntGetZExtValue(alignment);
   if ((bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 && bytes != 16) || align < bytes)
      return fail(t, STELA_AIR_INSTRUCTION, "SIMD-group copies of %llu-byte elements aligned to %llu",
                  (unsigned long long)bytes, (unsigned long long)align);
   struct async_copy c = {.t = t, .bytes = bytes};
   const struct value *dst = pointer_value(t, LLVMGetOperand(inst, 2));
   const struct value *src = pointer_value(t, LLVMGetOperand(inst, two_d ? 6 : 3));
   if (!dst || !src)
      return fail(t, STELA_AIR_INSTRUCTION, "a SIMD-group copy's memory has no provenance");
   c.dst = *dst;
   c.src = *src;
   if (two_d) {
      LLVMValueRef offset = LLVMGetOperand(inst, 10), mode = LLVMGetOperand(inst, 11);
      if (!LLVMIsAConstantInt(mode) || LLVMConstIntGetZExtValue(mode) > 1)
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported SIMD-group copy clamp mode");
      c.edge = LLVMConstIntGetZExtValue(mode) == 1;
      nir_def *dst_tile = get_value(t, LLVMGetOperand(inst, 5));
      nir_def *src_tile = get_value(t, LLVMGetOperand(inst, 9));
      nir_def *shift = get_value(t, offset);
      c.dst_row = get_value(t, LLVMGetOperand(inst, 3));
      c.dst_stride = get_value(t, LLVMGetOperand(inst, 4));
      c.src_row = get_value(t, LLVMGetOperand(inst, 7));
      c.src_stride = get_value(t, LLVMGetOperand(inst, 8));
      if (!dst_tile || !src_tile || !c.dst_row || !c.dst_stride || !c.src_row || !c.src_stride ||
          !shift || dst_tile->num_components != 2 || src_tile->num_components != 2 ||
          shift->num_components != 2)
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported SIMD-group copy signature");
      c.width = nir_u2u32(b, nir_channel(b, dst_tile, 0));
      c.count = nir_imul(b, c.width, nir_u2u32(b, nir_channel(b, dst_tile, 1)));
      c.src_width = nir_u2u32(b, nir_channel(b, src_tile, 0));
      c.src_height = nir_u2u32(b, nir_channel(b, src_tile, 1));
      c.offset_x = nir_u2u32(b, nir_channel(b, shift, 0));
      c.offset_y = nir_u2u32(b, nir_channel(b, shift, 1));
   } else {
      nir_def *elements = get_value(t, LLVMGetOperand(inst, 4));
      if (!elements)
         return false;
      c.width = c.count = c.src_width = nir_u2u32(b, elements);
      c.src_height = nir_imm_int(b, 1);
      c.offset_x = c.offset_y = nir_imm_int(b, 0);
      c.dst_row = c.src_row = nir_u2u64(b, elements);
      c.dst_stride = c.src_stride = nir_imm_int64(b, 1);
   }
   c.dst_row = nir_u2u64(b, c.dst_row);
   c.dst_stride = nir_u2u64(b, c.dst_stride);
   c.src_row = nir_u2u64(b, c.src_row);
   c.src_stride = nir_u2u64(b, c.src_stride);
   c.lane = nir_load_subgroup_invocation(b);
   c.lanes = nir_load_subgroup_size(b);
   /* Each thread copies every lanes-th element; a zero-width tile none. */
   nir_def *rounds = nir_udiv(b, nir_iadd(b, c.count, nir_iadd_imm(b, c.lanes, -1)), c.lanes);
   rounds = nir_bcsel(b, nir_ieq_imm(b, c.width, 0), nir_imm_int(b, 0), rounds);
   if (!repeat(t, rounds, 0, NULL, async_copy_element, &c))
      return false;
   nir_barrier(b, .execution_scope = SCOPE_SUBGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared | nir_var_mem_global | nir_var_mem_ssbo);
   return true;
}

/* AIR's barriers and fences, with memory flags 1 device, 2 threadgroup, 4
 * texture and 8 imageblock memory. A barrier is (flags, scope), as Apple's compiler emits, or
 * (flags, order, scope); a fence is (flags, order, scope). Orders are C++'s,
 * 0 relaxed and 5 sequentially consistent, which NIR's acquire-release
 * barrier implements as SPIR-V's does. Scopes are 1 threadgroup, 2 device and
 * 4 SIMD-group; a barrier takes its own: threadgroup_barrier the threadgroup,
 * simdgroup_barrier the SIMD-group. */
static bool
synchronization_call(struct translator *t, LLVMValueRef inst, LLVMValueRef callee,
                     const char *name, size_t length, bool *lowered)
{
   enum { THREADGROUP_BARRIER, SIMDGROUP_BARRIER, FENCE } kind;
   if (stela_air_call_named(name, length, "air.wg.barrier"))
      kind = THREADGROUP_BARRIER;
   else if (stela_air_call_named(name, length, "air.simdgroup.barrier"))
      kind = SIMDGROUP_BARRIER;
   else if (stela_air_call_named(name, length, "air.atomic.fence"))
      kind = FENCE;
   else
      return *lowered = false, true;
   *lowered = true;
   if (kind == THREADGROUP_BARRIER) {
      if (!compute_like(t))
         return fail(t, STELA_AIR_INSTRUCTION, "threadgroup barriers exist only in compute-like functions");
      if (!shared_profile(t))
         return false;
   }
   unsigned count = LLVMGetNumArgOperands(inst);
   if ((kind == THREADGROUP_BARRIER && !t->options->max_threadgroup_bytes) ||
       !LLVMIsAFunction(callee) || !LLVMIsDeclaration(callee) ||
       LLVMGetFunctionCallConv(callee) != LLVMCCallConv ||
       LLVMGetInstructionCallConv(inst) != LLVMCCallConv ||
       (count != 3 && (kind == FENCE || count != 2)) || LLVMCountParams(callee) != count ||
       LLVMIsFunctionVarArg(LLVMGlobalGetValueType(callee)) ||
       LLVMGetCalledFunctionType(inst) != LLVMGlobalGetValueType(callee) ||
       LLVMGetNumOperandBundles(inst) || LLVMGetTypeKind(LLVMTypeOf(inst)) != LLVMVoidTypeKind ||
       (kind != FENCE && !barrier_stays(inst, callee)))
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported/unconfigured barrier or fence signature");
   unsigned values[3];
   for (unsigned i = 0; i < count; i++) {
      LLVMValueRef operand = LLVMGetOperand(inst, i);
      if (!LLVMIsAConstantInt(operand) || LLVMGetIntTypeWidth(LLVMTypeOf(operand)) != 32)
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported barrier or fence memory flags/order/scope");
      values[i] = LLVMConstIntGetZExtValue(operand);
   }
   unsigned flags = values[0], order = count == 3 ? values[1] : 5, scope = values[count - 1];
   /* Simdgroup barriers name their scope 1 before Metal 3.2. */
   /* Flag 8 is mem_threadgroup_imageblock: a kernel's explicit imageblock
    * is threadgroup memory, and a tile function's implicit one its
    * attachments' images. */
   if (flags & ~31u || (kind == FENCE ? order > 5 : order != 5) ||
       (kind == FENCE ? scope != 1 && scope != 2 && scope != 4 :
        kind == THREADGROUP_BARRIER ? scope != 1 : scope != 1 && scope != 4))
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported barrier or fence memory flags/order/scope");
   nir_variable_mode modes = memory_flag_modes(t, flags);
   bool ordered = modes && order;
   mesa_scope execution = kind == THREADGROUP_BARRIER ? SCOPE_WORKGROUP :
                          kind == SIMDGROUP_BARRIER ? SCOPE_SUBGROUP : SCOPE_NONE;
   /* A relaxed fence orders nothing. */
   if (!ordered && execution == SCOPE_NONE)
      return true;
   nir_barrier(&t->b, .execution_scope = execution,
               .memory_scope = !ordered ? SCOPE_NONE : memory_scope(scope),
               .memory_semantics = ordered ? order_semantics(order) : 0,
               .memory_modes = ordered ? modes : 0);
   t->shared_barriers++;
   return true;
}

enum { MEMORY_INTRINSIC_ACCESSES = 256 };

/* A call's alignment attribute on pointer argument `argument`, or 1. */
static unsigned
argument_alignment(LLVMValueRef call, unsigned argument)
{
   unsigned kind = LLVMGetEnumAttributeKindForName("align", strlen("align"));
   /* Attribute index 0 is the return value; arguments follow it. */
   LLVMAttributeRef attribute = LLVMGetCallSiteEnumAttribute(call, 1 + argument, kind);
   return attribute ? (unsigned)LLVMGetEnumAttributeValue(attribute) : 1;
}

/* LLVM's lifetime markers, and memset, memcpy and memmove of a constant
 * length, in chunks of the largest size the alignment allows, up to 16 bytes.
 * A copy loads every chunk before storing any, so overlap is handled too. */
static bool
memory_intrinsic(struct translator *t, LLVMValueRef inst, unsigned intrinsic, bool *lowered)
{
   enum { LIFETIME_START, LIFETIME_END, SET, COPY, MOVE };
   static const char *const names[] = {
      [LIFETIME_START] = "llvm.lifetime.start", [LIFETIME_END] = "llvm.lifetime.end",
      [SET] = "llvm.memset", [COPY] = "llvm.memcpy", [MOVE] = "llvm.memmove",
   };
   unsigned operation = ARRAY_SIZE(names);
   for (unsigned i = 0; intrinsic && i < ARRAY_SIZE(names); i++) {
      if (intrinsic == LLVMLookupIntrinsicID(names[i], strlen(names[i])))
         operation = i;
   }
   *lowered = operation < ARRAY_SIZE(names);
   /* Contents are undefined outside a lifetime, so the stored ones refine them. */
   if (!*lowered || operation <= LIFETIME_END)
      return true;
   LLVMValueRef length = LLVMGetOperand(inst, 2), is_volatile = LLVMGetOperand(inst, 3);
   if (!LLVMIsAConstantInt(length) || !LLVMIsAConstantInt(is_volatile) ||
       LLVMConstIntGetZExtValue(is_volatile) || LLVMGetNumOperandBundles(inst))
      return fail(t, STELA_AIR_MEMORY, "memory intrinsics take a constant length and no volatile");
   const struct value *destination = pointer_value(t, LLVMGetOperand(inst, 0));
   const struct value *source = operation == SET ? NULL : pointer_value(t, LLVMGetOperand(inst, 1));
   nir_def *fill = operation == SET ? get_value(t, LLVMGetOperand(inst, 1)) : NULL;
   if (t->error->kind != STELA_AIR_OK)
      return false;
   if (!destination || !(destination->access & STELA_AIR_WRITE) ||
       (operation != SET && (!source || !(source->access & STELA_AIR_READ))))
      return fail(t, STELA_AIR_MEMORY, "memory intrinsic lacks a bound writable/readable pointer");
   unsigned alignment = argument_alignment(inst, 0);
   if (source)
      alignment = MIN2(alignment, argument_alignment(inst, 1));
   uint64_t bytes = LLVMConstIntGetZExtValue(length);
   unsigned chunks[MEMORY_INTRINSIC_ACCESSES], count = 0;
   for (uint64_t offset = 0, size = MIN2(alignment, 16); offset < bytes; offset += size) {
      while (size > bytes - offset)
         size /= 2;
      if (count == ARRAY_SIZE(chunks))
         return fail(t, STELA_AIR_LIMIT, "memory intrinsic exceeds the access budget");
      chunks[count++] = size;
   }
   nir_def *values[MEMORY_INTRINSIC_ACCESSES];
   for (unsigned pass = 0; pass < 2; pass++) {
      uint64_t offset = 0;
      for (unsigned i = 0; i < count; offset += chunks[i++]) {
         unsigned components = chunks[i] >= 4 ? chunks[i] / 4 : 1;
         unsigned bits = chunks[i] >= 4 ? 32 : chunks[i] * 8;
         if (!pass && !source) {
            nir_def *pattern = nir_imul_imm(&t->b, nir_u2uN(&t->b, fill, bits),
                                            0x01010101u & BITFIELD_MASK(bits));
            values[i] = nir_replicate(&t->b, pattern, components);
            continue;
         }
         struct value at = pass ? *destination : *source;
         at.def = nir_iadd_imm(&t->b, at.def, offset);
         nir_def *loaded = memory_access(t, &at, pass ? values[i] : NULL, NULL, 0,
                                         components, bits, chunks[i]);
         if (t->error->kind != STELA_AIR_OK)
            return false;
         if (!pass)
            values[i] = loaded;
      }
   }
   return true;
}

/* The field of an argument buffer an element access at address reads: the
 * field whose elements the constant offset names in an element of an array
 * of the buffer's structure, which dynamic indices step through by whole
 * structures or by whole elements of the field. */
static const struct stela_air_field *
argument_field(struct translator *t, const struct value *address)
{
   if (!address->offset_known) {
      fail(t, STELA_AIR_MEMORY, "argument buffer field access through a pointer PHI");
      return NULL;
   }
   uint64_t offset = address->offset % address->fields.size;
   uint64_t steps = address->steps % address->fields.size ? address->steps : 0;
   for (unsigned i = 0; i < address->fields.count; i++) {
      const struct stela_air_field *field = &t->fields[address->fields.first + i];
      if (offset < field->offset || offset - field->offset >= field->size)
         continue;
      unsigned element = field->size / field->count;
      if ((offset - field->offset) % element || steps % element || (steps && field->count == 1))
         break;
      return field;
   }
   fail(t, STELA_AIR_MEMORY, "pointer load names no argument buffer element");
   return NULL;
}

/* A resource ID's value, which names its texture or sampler for texture
 * calls. */
static bool
put_id(struct translator *t, LLVMValueRef inst, const struct stela_air_texture *texture,
       const struct stela_air_sampler *sampler, nir_def *id)
{
   struct value *value = rzalloc(t->values, struct value);
   if (!value || !_mesa_hash_table_insert(t->values, inst, value))
      return fail(t, STELA_AIR_ALLOCATION, "AIR resource value allocation failed");
   value->texture = texture;
   value->sampler = sampler;
   value->id = id;
   return true;
}

/* A load of a texture or sampler array's handle: the resource ID of the
 * element its offset selects. */
static bool
array_load(struct translator *t, LLVMValueRef inst, const struct value *address)
{
   unsigned space = address->texture ? 1 : 2;
   LLVMTypeRef type = LLVMTypeOf(inst);
   if (LLVMGetPointerAddressSpace(type) != space || LLVMPointerSizeForAS(t->layout, space) != 8 ||
       LLVMGetVolatile(inst) || LLVMGetOrdering(inst) != LLVMAtomicOrderingNotAtomic)
      return fail(t, STELA_AIR_TYPE, "load differs from its texture or sampler array's handles");
   if (!address->offset_known || address->offset % 8 || address->steps % 8)
      return fail(t, STELA_AIR_MEMORY, "texture or sampler array load names no handle");
   nir_def *element = nir_ushr_imm(&t->b, address->def, 3);
   nir_def *id = t->options->resource_id(&t->b, address->texture, address->sampler, element,
                                         t->options->binding_data);
   if (!id || id->num_components != 1 || id->bit_size != 64)
      return fail(t, STELA_AIR_ARGUMENT, "resource ID callback refused a texture or sampler array");
   return put_id(t, inst, address->texture, address->sampler, id);
}

/* The address space of the pointer a load of field yields, zero for inline
 * data. */
static unsigned
field_space(const struct stela_air_field *field)
{
   switch (field->kind) {
   case STELA_AIR_FIELD_DATA:
      return 0;
   case STELA_AIR_FIELD_BUFFER:
   case STELA_AIR_FIELD_ARGUMENTS:
      return field->buffer.address_space;
   case STELA_AIR_FIELD_SAMPLER:
      return 2;
   case STELA_AIR_FIELD_TEXTURE:
   case STELA_AIR_FIELD_COMMANDS:
   case STELA_AIR_FIELD_RENDER_PIPELINE:
   case STELA_AIR_FIELD_COMPUTE_PIPELINE:
   case STELA_AIR_FIELD_FUNCTION_TABLE:
      return 1;
   }
   return 0;
}

/* A load of an address or resource ID from an argument buffer, or of a
 * texture or sampler array's handle. An address roots the pointers derived
 * from it, with its field's access; an ID names its texture or sampler for
 * texture calls, or its command buffer or pipeline state for command calls. */
static bool
argument_load(struct translator *t, LLVMValueRef inst)
{
   const struct value *address = pointer_value(t, LLVMGetOperand(inst, 0));
   if (address && address->array)
      return array_load(t, inst, address);
   if (!address || !address->fields.count || !(address->access & STELA_AIR_READ) ||
       LLVMGetVolatile(inst) || LLVMGetOrdering(inst) != LLVMAtomicOrderingNotAtomic ||
       !LLVMGetAlignment(inst))
      return fail(t, STELA_AIR_MEMORY, "pointer loads read argument buffers");
   const struct stela_air_field *field = argument_field(t, address);
   if (!field)
      return false;
   LLVMTypeRef type = LLVMTypeOf(inst);
   unsigned space = LLVMGetPointerAddressSpace(type);
   unsigned expected = field_space(field);
   if (!expected || space != expected || LLVMPointerSizeForAS(t->layout, space) != 8)
      return fail(t, STELA_AIR_TYPE, "pointer load differs from its argument buffer field");
   nir_def *loaded = memory_access(t, address, NULL, NULL, 0, 1, 64, LLVMGetAlignment(inst));
   if (!loaded)
      return false;
   switch (field->kind) {
   case STELA_AIR_FIELD_TEXTURE:
      if (t->options->texture_view_word) {
         nir_def *word = NULL;
         nir_def *named =
            t->options->texture_view_word(&t->b, loaded, &word, t->options->binding_data);
         struct memory_guard guard;
         if (!memory_guard_begin(t, named, &guard))
            return false;
         nir_def *view = nir_load_global(&t->b, 1, 64, word, .access = ACCESS_NON_WRITEABLE,
                                         .align_mul = 8);
         view = memory_guard_end(t, &guard, view);
         if (!view)
            return false;
         loaded = nir_bcsel(&t->b, named, view, loaded);
      }
      return put_id(t, inst, &field->texture, NULL, loaded);
   case STELA_AIR_FIELD_SAMPLER:
      return put_id(t, inst, NULL, &field->sampler, loaded);
   case STELA_AIR_FIELD_COMMANDS:
   case STELA_AIR_FIELD_RENDER_PIPELINE:
   case STELA_AIR_FIELD_COMPUTE_PIPELINE:
      if (!put_id(t, inst, NULL, NULL, loaded))
         return false;
      find_value(t, inst)->object = field;
      return true;
   default:
      break;
   }
   if (!put_value(t, inst, loaded, field->buffer.access))
      return false;
   struct value *value = find_value(t, inst);
   value->pointer_root = inst;
   value->declared_alignment = field->buffer.type_alignment;
   value->fields = field->arguments;
   value->offset_known = true;
   return true;
}

/* Object and mesh functions' setters of their mesh grid properties and
 * mesh, in DXMT's AIR forms: the handle, then indices, then the value. */
enum mesh_setter {
   MESH_POSITION, MESH_POINT_SIZE, MESH_CLIP_DISTANCE, MESH_VERTEX_DATA, MESH_PRIMITIVE_DATA,
   MESH_LAYER, MESH_VIEWPORT, MESH_PRIMITIVE_ID, MESH_CULLED, MESH_INDEX, MESH_INDICES,
   MESH_PRIMITIVE_COUNT, MESH_GRID,
};

static const struct {
   const char *stem;
   unsigned operands; /* Including the handle. */
   bool overload; /* The name ends with the value's overload. */
   enum stela_air_stage_semantic semantic; /* The output it sets, if any. */
} mesh_setters[] = {
   [MESH_POSITION] = {"air.set_position_mesh", 3, false, STELA_AIR_POSITION},
   [MESH_POINT_SIZE] = {"air.set_point_size_mesh", 3, false, STELA_AIR_POINT_SIZE},
   [MESH_CLIP_DISTANCE] = {"air.set_clip_distance_mesh", 4, false, STELA_AIR_CLIP_DISTANCE},
   [MESH_VERTEX_DATA] = {"air.set_vertex_data_mesh", 4, true, STELA_AIR_VARYING},
   [MESH_PRIMITIVE_DATA] = {"air.set_primitive_data_mesh", 4, true, STELA_AIR_VARYING},
   [MESH_LAYER] = {"air.set_render_target_array_index_mesh", 3, true, STELA_AIR_LAYER},
   [MESH_VIEWPORT] = {"air.set_viewport_array_index_mesh", 3, true, STELA_AIR_VIEWPORT},
   [MESH_PRIMITIVE_ID] = {"air.set_primitive_id_mesh", 3, false, STELA_AIR_MESH_PRIMITIVE_ID},
   [MESH_CULLED] = {"air.set_primitive_culled_mesh", 3, false, STELA_AIR_PRIMITIVE_CULLED},
   [MESH_INDEX] = {"air.set_index_mesh", 3},
   [MESH_INDICES] = {"air.set_indices_mesh", 3, true},
   [MESH_PRIMITIVE_COUNT] = {"air.set_primitive_count_mesh", 2},
   [MESH_GRID] = {"air.set_threadgroups_per_grid_mesh_properties", 2},
};

/* The setter a call names, or UINT_MAX. */
static unsigned
mesh_setter(LLVMValueRef inst, const char *name, size_t length)
{
   unsigned operands = LLVMGetNumArgOperands(inst);
   for (unsigned s = 0; s < ARRAY_SIZE(mesh_setters); s++) {
      char expected[96];
      size_t stem = strlen(mesh_setters[s].stem);
      memcpy(expected, mesh_setters[s].stem, stem + 1);
      if (mesh_setters[s].overload && operands) {
         expected[stem] = '.';
         stela_air_call_overload(LLVMTypeOf(LLVMGetOperand(inst, operands - 1)), expected + stem + 1,
                                 sizeof(expected) - stem - 1);
      }
      if (stela_air_call_named(name, length, expected))
         return s;
   }
   return UINT_MAX;
}

/* The output a setter sets; mesh data by its constant data index. */
static unsigned
mesh_output(struct translator *t, enum mesh_setter setter, LLVMValueRef data)
{
   const struct stela_air_stage_entry *entry = &t->render.entry;
   for (unsigned i = 0; i < entry->output_count; i++) {
      const struct stela_air_stage_io *io = &t->render.outputs[i];
      if (io->semantic != mesh_setters[setter].semantic)
         continue;
      if (io->semantic != STELA_AIR_VARYING ||
          (io->per_primitive == (setter == MESH_PRIMITIVE_DATA) && LLVMIsAConstantInt(data) &&
           LLVMConstIntGetZExtValue(data) == io->index))
         return i;
   }
   return UINT_MAX;
}

/* Stores value to element (below count), and below `inner` its index, of
 * an arrayed output, or the components `mask` names of the element; out of
 * range, the store is discarded. */
static bool
mesh_store(struct translator *t, nir_variable *var, nir_def *element, unsigned count,
           nir_def *index, unsigned inner, nir_def *value, unsigned mask)
{
   nir_builder *b = &t->b;
   nir_def *inside = nir_ult_imm(b, element, count);
   if (index)
      inside = nir_iand(b, inside, nir_ult_imm(b, index, inner));
   struct memory_guard guard;
   if (!memory_guard_begin(t, inside, &guard))
      return false;
   nir_deref_instr *deref = nir_build_deref_array(b, nir_build_deref_var(b, var), element);
   if (index)
      deref = nir_build_deref_array(b, deref, index);
   nir_store_deref(b, deref, value, mask);
   memory_guard_end(t, &guard, NULL);
   return true;
}

/* Metal's flat index list: index i is corner i % n of primitive i / n. */
static bool
mesh_index(struct translator *t, nir_def *index, nir_def *vertex)
{
   nir_builder *b = &t->b;
   unsigned corners = mesa_vertices_per_prim(t->b.shader->info.mesh.primitive_type);
   nir_def *primitive = nir_udiv_imm(b, index, corners);
   nir_def *corner = nir_umod_imm(b, index, corners);
   nir_def *value = nir_replicate(b, nir_u2u32(b, vertex), corners);
   for (unsigned c = 0; c < corners; c++) {
      /* Only the matching corner's guard passes. */
      nir_def *element = nir_bcsel(b, nir_ieq_imm(b, corner, c), primitive,
                                   nir_imm_int(b, t->render.entry.max_primitives));
      if (!mesh_store(t, t->indices, element, t->render.entry.max_primitives, NULL, 0, value,
                      1u << c))
         return false;
   }
   return true;
}

/* A call of an object or mesh setter; false with no error when the call is
 * something else. */
static bool
mesh_call(struct translator *t, LLVMValueRef inst, LLVMValueRef callee, const char *name,
          size_t length, bool *lowered)
{
   unsigned setter = mesh_setter(inst, name, length);
   *lowered = setter != UINT_MAX;
   if (!*lowered)
      return true;
   LLVMValueRef handle = setter == MESH_GRID ? t->mesh_grid : t->mesh;
   unsigned operands = mesh_setters[setter].operands;
   if (!handle || !LLVMIsAFunction(callee) || !LLVMIsDeclaration(callee) ||
       LLVMGetNumArgOperands(inst) != operands || LLVMGetOperand(inst, 0) != handle ||
       LLVMGetCalledFunctionType(inst) != LLVMGlobalGetValueType(callee) ||
       LLVMGetTypeKind(LLVMTypeOf(inst)) != LLVMVoidTypeKind)
      return fail(t, STELA_AIR_INSTRUCTION, "mesh setter without its function's mesh or grid");
   nir_def *values[4] = {NULL};
   for (unsigned i = 1; i < operands; i++) {
      if (!(values[i] = get_value(t, LLVMGetOperand(inst, i))))
         return false;
   }
   nir_builder *b = &t->b;
   nir_def *value = values[operands - 1];
   for (unsigned i = 1; i + 1 < operands; i++) {
      if (values[i]->num_components != 1 || values[i]->bit_size != 32)
         return fail(t, STELA_AIR_TYPE, "mesh setter indices are 32-bit integers");
   }
   switch (setter) {
   case MESH_GRID: {
      if (value->num_components != 3 || value->bit_size != 32)
         return fail(t, STELA_AIR_TYPE, "a mesh grid is three 32-bit integers");
      nir_deref_instr *state = nir_build_deref_var(b, t->mesh_state);
      for (unsigned i = 0; i < 3; i++)
         nir_store_deref(b, nir_build_deref_array_imm(b, state, i), nir_channel(b, value, i), 1);
      return true;
   }
   case MESH_PRIMITIVE_COUNT:
      if (value->num_components != 1 || value->bit_size != 32)
         return fail(t, STELA_AIR_TYPE, "a primitive count is a 32-bit integer");
      nir_store_deref(b, nir_build_deref_array_imm(b, nir_build_deref_var(b, t->mesh_state), 0),
                      value, 1);
      return true;
   case MESH_INDEX:
      if (value->num_components != 1 || value->bit_size != 8)
         return fail(t, STELA_AIR_TYPE, "a mesh index is an 8-bit vertex");
      return mesh_index(t, values[1], value);
   case MESH_INDICES:
      if (value->bit_size != 8)
         return fail(t, STELA_AIR_TYPE, "mesh indices are 8-bit vertices");
      for (unsigned c = 0; c < value->num_components; c++) {
         if (!mesh_index(t, nir_iadd_imm(b, values[1], c), nir_channel(b, value, c)))
            return false;
      }
      return true;
   default:
      break;
   }
   bool data = setter == MESH_VERTEX_DATA || setter == MESH_PRIMITIVE_DATA;
   unsigned output = mesh_output(t, setter, data ? LLVMGetOperand(inst, 1) : NULL);
   if (output == UINT_MAX)
      return fail(t, STELA_AIR_INSTRUCTION, "mesh setter of an output the mesh lacks");
   const struct stela_air_stage_io *io = &t->render.outputs[output];
   if (!stela_air_metadata_output_type(io, LLVMTypeOf(LLVMGetOperand(inst, operands - 1))))
      return fail(t, STELA_AIR_TYPE, "mesh setter value differs from its output");
   const struct stela_air_stage_entry *entry = &t->render.entry;
   unsigned count = io->per_primitive ? entry->max_primitives : entry->max_vertices;
   bool clip = setter == MESH_CLIP_DISTANCE;
   nir_def *element = values[clip || data ? 2 : 1];
   value = setter == MESH_CULLED ? nir_b2i32(b, value) : stage_widen(b, value, io->primitive);
   if (setter == MESH_POINT_SIZE)
      value = largest_point(b, value);
   return mesh_store(t, t->outputs[output], element, count, clip ? values[1] : NULL,
                     io->clip_distances, value, BITFIELD_MASK(value->num_components));
}

/* v with each lane it lacks zero, a freeze's one value for all its uses. */
static nir_def *
frozen_lanes(struct translator *t, struct value *v)
{
   nir_def *lanes[4];
   for (unsigned i = 0; i < v->components; i++) {
      lanes[i] = value_lane(t, v, i);
      if (!lanes[i])
         lanes[i] = nir_imm_intN_t(&t->b, 0, v->bits);
   }
   return nir_vec(&t->b, lanes, v->components);
}

/* An aggregate's frozen copy, element by element. */
static struct value *
frozen_aggregate(struct translator *t, const struct value *v, unsigned depth)
{
   struct value *copy = depth < 16 ? new_aggregate(t, v->aggregate_count, NULL) : NULL;
   if (!copy)
      return depth < 16 ? NULL : (fail(t, STELA_AIR_TYPE, "aggregate nests too deeply"), NULL);
   for (unsigned i = 0; i < v->aggregate_count; i++) {
      struct value *element = v->elements[i];
      if (!element) {
         fail(t, STELA_AIR_TYPE, "frozen aggregate lacks an element");
         return NULL;
      }
      if (element->aggregate_count) {
         if (!(copy->elements[i] = frozen_aggregate(t, element, depth + 1)))
            return NULL;
         continue;
      }
      /* A pointer keeps its proven resource; it has no undefined lane. */
      if (element->pointer_root || element->choices || !element->components) {
         copy->elements[i] = element;
         continue;
      }
      struct value *leaf = rzalloc(t->values, struct value);
      if (!leaf) {
         fail(t, STELA_AIR_ALLOCATION, "aggregate value allocation failed");
         return NULL;
      }
      *leaf = *element;
      leaf->def = frozen_lanes(t, element);
      memset(leaf->lanes, 0, sizeof(leaf->lanes));
      copy->elements[i] = leaf;
   }
   return copy;
}

/* A freeze holds one value for all its uses, so a lane with none takes zero.
 * A texture call's pair (texel and residency) is whole already; a struct or
 * array freezes element by element. */
static bool
freeze(struct translator *t, LLVMValueRef inst)
{
   LLVMValueRef operand = LLVMGetOperand(inst, 0);
   if (aggregate_count(LLVMTypeOf(operand))) {
      const struct value *pair = find_value(t, operand);
      if (pair && pair->members[0])
         return put_pair(t, inst, pair->members[0], pair->members[1]);
      struct value *v = aggregate_value(t, operand);
      struct value *copy = v ? frozen_aggregate(t, v, 0) : NULL;
      if (!copy)
         return false;
      if (!_mesa_hash_table_insert(t->values, inst, copy))
         return fail(t, STELA_AIR_ALLOCATION, "AIR value map allocation failed");
      return true;
   }
   struct value *v = partial_value(t, operand);
   if (!v)
      return false;
   return put_value(t, inst, frozen_lanes(t, v), 0);
}

/* An FDiv or FRem at Metal's precise accuracy. */
static bool
precise_division(struct translator *t, LLVMValueRef inst, LLVMOpcode op)
{
   nir_def *a = get_value(t, LLVMGetOperand(inst, 0)), *c = get_value(t, LLVMGetOperand(inst, 1));
   if (!a || !c || !fp_environment(t))
      return false;
   bool narrow = a->bit_size == 16;
   stela_air_precise_function *function =
      op == LLVMFDiv ? (narrow ? stela_air_half_divide : stela_air_precise_divide) :
                       (narrow ? stela_air_half_fmod : stela_air_precise_fmod);
   nir_def *value = stela_air_precise(&t->b, function, narrow ? nir_f2f32(&t->b, a) : a,
                                      narrow ? nir_f2f32(&t->b, c) : c);
   return put_value(t, inst, narrow ? nir_f2f16(&t->b, value) : value, 0);
}

/* A choice between places in one resource. */
static struct pointer_node *pointer_node(struct translator *t, LLVMValueRef llvm);
static nir_def *choice_index(struct translator *t, const struct pointer_choices *choices,
                             LLVMValueRef incoming);
static void choose(struct translator *t, struct value *value,
                   const struct pointer_choices *choices, nir_def *choice);

static bool
pointer_select(struct translator *t, LLVMValueRef inst)
{
   nir_def *condition = get_value(t, LLVMGetOperand(inst, 0));
   LLVMValueRef yes_llvm = LLVMGetOperand(inst, 1), no_llvm = LLVMGetOperand(inst, 2);
   /* A texture or null textures: that texture, where the choice is it. */
   const struct value *chosen = find_value(t, yes_llvm);
   if (!texture_choice_root(chosen))
      chosen = find_value(t, no_llvm);
   if (texture_choice_root(chosen) || (stela_air_null_texture(yes_llvm) && stela_air_null_texture(no_llvm))) {
      if (!texture_choice_root(chosen))
         return fail(t, STELA_AIR_MEMORY, "a texture choice between null textures only");
      nir_def *yes = condition ? texture_choice_present(t, chosen, yes_llvm) : NULL;
      nir_def *no = yes ? texture_choice_present(t, chosen, no_llvm) : NULL;
      struct value *value = no ? rzalloc(t->values, struct value) : NULL;
      if (!no)
         return false;
      if (!value || !_mesa_hash_table_insert(t->values, inst, value))
         return fail(t, STELA_AIR_ALLOCATION, "AIR texture choice allocation failed");
      pointer_authority(value, chosen);
      value->present = nir_bcsel(&t->b, condition, yes, no);
      return true;
   }
   struct pointer_node *node = pointer_node(t, inst);
   if (condition && node && node->choices) {
      nir_def *yes_index = choice_index(t, node->choices, yes_llvm);
      nir_def *no_index = yes_index ? choice_index(t, node->choices, no_llvm) : NULL;
      nir_def *yes_def = no_index ? get_value(t, yes_llvm) : NULL;
      nir_def *no_def = yes_def ? get_value(t, no_llvm) : NULL;
      if (!no_def || !put_value(t, inst, nir_bcsel(&t->b, condition, yes_def, no_def), 0))
         return false;
      choose(t, find_value(t, inst), node->choices,
             nir_bcsel(&t->b, condition, yes_index, no_index));
      return true;
   }
   const struct value *yes = pointer_value(t, yes_llvm);
   const struct value *no = yes ? pointer_value(t, no_llvm) : NULL;
   if (!condition || t->error->kind != STELA_AIR_OK)
      return false;
   const struct value *root = joined_pointer(yes, no_llvm, no);
   if (!root)
      root = joined_pointer(no, yes_llvm, yes);
   if (!root)
      return fail(t, STELA_AIR_MEMORY, "pointer select joins distinct resource origins");
   if (!put_value(t, inst, nir_bcsel(&t->b, condition, yes->def, no->def), root->access))
      return false;
   pointer_authority(find_value(t, inst), root);
   return true;
}

/* A cast between the generic address space and thread, device or
 * threadgroup memory: the same pointer, of the same roots. */
static bool
address_space_cast(struct translator *t, LLVMValueRef inst)
{
   LLVMValueRef operand = LLVMGetOperand(inst, 0);
   LLVMTypeRef from = LLVMTypeOf(operand), to = LLVMTypeOf(inst);
   unsigned spaces[2] = {pointer_space(from) ? LLVMGetPointerAddressSpace(from) : UINT_MAX,
                         pointer_space(to) ? LLVMGetPointerAddressSpace(to) : UINT_MAX};
   unsigned other = spaces[0] == 8 ? spaces[1] : spaces[1] == 8 ? spaces[0] : UINT_MAX;
   if (other != 0 && other != 1 && other != 3)
      return fail(t, STELA_AIR_MEMORY, "address space cast from %u to %u", spaces[0], spaces[1]);
   const struct value *origin = pointer_value(t, operand);
   if (!origin || !origin->access || origin->texture || origin->sampler || origin->array ||
       !put_value(t, inst, origin->def, origin->access))
      return fail(t, STELA_AIR_MEMORY, "address space cast lacks buffer provenance");
   pointer_authority(find_value(t, inst), origin);
   pointer_offset(find_value(t, inst), origin, 0, 0);
   return true;
}

/* Whether a root is device (1), threadgroup (3) or thread (0) memory. */
static bool
root_in_space(const struct value *root, unsigned space)
{
   if (root->null || root->constant_data || root->texture || root->sampler || root->array)
      return false;
   if (space == 1)
      return !root->storage;
   return root->storage &&
          root->storage->data.mode == (space == 3 ? nir_var_mem_shared : nir_var_function_temp);
}

/* `air.is_device_pointer`, `air.is_threadgroup_pointer` and
 * `air.is_thread_pointer` of a generic pointer: whether its root, or the
 * one it chooses, is that memory. */
static bool
pointer_space_call(struct translator *t, LLVMValueRef inst, const char *name, size_t length,
                   bool *lowered)
{
   static const struct {
      const char *name;
      unsigned space;
   } queries[] = {
      {"air.is_device_pointer", 1},
      {"air.is_threadgroup_pointer", 3},
      {"air.is_thread_pointer", 0},
   };
   unsigned space = UINT_MAX;
   for (unsigned i = 0; i < ARRAY_SIZE(queries); i++) {
      if (length == strlen(queries[i].name) && !memcmp(name, queries[i].name, length))
         space = queries[i].space;
   }
   *lowered = space != UINT_MAX;
   if (!*lowered)
      return true;
   if (LLVMGetNumArgOperands(inst) != 1 || LLVMTypeOf(inst) != LLVMInt1TypeInContext(t->context))
      return fail(t, STELA_AIR_INSTRUCTION, "%.*s of an unknown form", (int)length, name);
   const struct value *pointer = pointer_value(t, LLVMGetOperand(inst, 0));
   if (!pointer || !pointer->access)
      return fail(t, STELA_AIR_MEMORY, "%.*s of a pointer without provenance", (int)length, name);
   if (!pointer->choices)
      return put_value(t, inst, nir_imm_bool(&t->b, root_in_space(pointer, space)), 0);
   nir_def *in = nir_imm_false(&t->b);
   for (unsigned i = 0; i < pointer->choices->count; i++) {
      struct value root = choice_root(t, pointer->choices, i);
      if (root_in_space(&root, space))
         in = nir_ior(&t->b, in, nir_ieq_imm(&t->b, pointer->choice, i));
   }
   return put_value(t, inst, in, 0);
}

/* A rasterization rate map decoder's `map_screen_to_physical_coordinates`
 * and `map_physical_to_screen_coordinates` of a float2 at a layer, from the
 * map's parameter data: per layer, a scale and offset from screen to
 * physical coordinates for each axis, as the bundle's maps write them. */
static bool
rate_map_call(struct translator *t, LLVMValueRef inst, const char *name, size_t length,
              bool *lowered)
{
   static const char to_physical[] = "air.map_screen_to_physical_coordinates.";
   static const char to_screen[] = "air.map_physical_to_screen_coordinates.";
   bool physical = length > strlen(to_physical) && !memcmp(name, to_physical, strlen(to_physical));
   bool screen = length > strlen(to_screen) && !memcmp(name, to_screen, strlen(to_screen));
   *lowered = physical || screen;
   if (!*lowered)
      return true;
   LLVMTypeRef float2 = LLVMVectorType(LLVMFloatTypeInContext(t->context), 2);
   if (LLVMGetNumArgOperands(inst) != 3 || LLVMTypeOf(inst) != float2 ||
       LLVMTypeOf(LLVMGetOperand(inst, 0)) != float2 ||
       LLVMTypeOf(LLVMGetOperand(inst, 2)) != LLVMInt32TypeInContext(t->context))
      return fail(t, STELA_AIR_INSTRUCTION, "%.*s of an unknown form", (int)length, name);
   nir_builder *b = &t->b;
   nir_def *coordinate = get_value(t, LLVMGetOperand(inst, 0));
   nir_def *layer = get_value(t, LLVMGetOperand(inst, 2));
   const struct value *data = pointer_value(t, LLVMGetOperand(inst, 1));
   if (!coordinate || !layer)
      return false;
   if (!data || !(data->access & STELA_AIR_READ))
      return fail(t, STELA_AIR_MEMORY, "%.*s of parameter data without provenance", (int)length,
                  name);
   struct value at = *data;
   at.def = nir_iadd(b, data->def, nir_u2u64(b, nir_imul_imm(b, layer, 16)));
   nir_def *mapping = memory_access(t, &at, NULL, NULL, 0, 4, 32, 4);
   if (!mapping)
      return false;
   nir_def *scale = nir_vec2(b, nir_channel(b, mapping, 0), nir_channel(b, mapping, 2));
   nir_def *offset = nir_vec2(b, nir_channel(b, mapping, 1), nir_channel(b, mapping, 3));
   nir_def *mapped = physical ? nir_ffma(b, coordinate, scale, offset) :
                                nir_fdiv(b, nir_fsub(b, coordinate, offset), scale);
   return put_value(t, inst, mapped, 0);
}

static bool
instruction(struct translator *t, LLVMValueRef inst)
{
   LLVMOpcode op = LLVMGetInstructionOpcode(inst);
   t->b.fp_math_ctrl = fp_math_control(inst);
   unsigned components = 0, bits = 0;
   if (op == LLVMRet)
      return !compute_like(t) ? stage_return(t, inst) : LLVMGetNumOperands(inst) == 0 || fail(t, STELA_AIR_TYPE, "compute kernel return value");
   if (op == LLVMExtractValue) {
      const struct value *pair = find_value(t, LLVMGetOperand(inst, 0));
      if (pair && pair->members[0])
         return pair_member(t, inst, pair);
   }
   if (op == LLVMAtomicRMW || op == LLVMAtomicCmpXchg)
      return atomic_instruction(t, inst, op);
   if (op == LLVMFreeze)
      return freeze(t, inst);
   if (op == LLVMAlloca)
      return private_allocation(t, inst) != NULL;
   if (op == LLVMSelect && LLVMGetTypeKind(LLVMTypeOf(inst)) == LLVMPointerTypeKind)
      return pointer_select(t, inst);
   if (stela_air_read_sampler(inst) || stela_air_null_texture(inst))
      return true;
   if (op == LLVMLoad && LLVMGetTypeKind(LLVMTypeOf(inst)) == LLVMPointerTypeKind)
      return argument_load(t, inst);
   if (op == LLVMInsertValue || op == LLVMExtractValue)
      return aggregate_instruction(t, inst, op);
   if ((op == LLVMLoad && aggregate_count(LLVMTypeOf(inst))) ||
       (op == LLVMStore && aggregate_count(LLVMTypeOf(LLVMGetOperand(inst, 0)))))
      return aggregate_memory(t, inst, op);
   if ((op == LLVMShuffleVector || op == LLVMExtractElement || op == LLVMInsertElement) &&
       stela_air_matrix_type(LLVMTypeOf(LLVMGetOperand(inst, 0)), &bits))
      return matrix_elements(t, inst, op);
   if (op == LLVMShuffleVector || op == LLVMExtractElement || op == LLVMInsertElement)
      return vector_instruction(t, inst, op);
   if (op == LLVMFPExt && stela_air_log_promotion(inst))
      return true;
   if (op == LLVMAddrSpaceCast)
      return address_space_cast(t, inst);
   if (op != LLVMStore && op != LLVMGetElementPtr && op != LLVMBitCast && op != LLVMCall &&
       !value_shape(LLVMTypeOf(inst), &components, &bits) &&
       !((op == LLVMSelect || matrix_elementwise(op)) && stela_air_matrix_type(LLVMTypeOf(inst), &bits)))
      return fail(t, STELA_AIR_TYPE, "unsupported AIR instruction result type (opcode %u)", op);
   if (op == LLVMGetElementPtr) {
      const struct value *origin;
      uint64_t constant, steps;
      nir_def *def = gep(t, inst, &origin, &constant, &steps);
      if (!def || !put_value(t, inst, def, origin->access))
         return false;
      pointer_authority(find_value(t, inst), origin);
      pointer_offset(find_value(t, inst), origin, constant, steps);
      return true;
   }
   if (op == LLVMBitCast) {
      unsigned src_components, src_bits;
      LLVMTypeRef source = LLVMTypeOf(LLVMGetOperand(inst, 0));
      if (pointer_space(source) && LLVMTypeOf(inst) == source) {
         const struct value *origin = pointer_value(t, LLVMGetOperand(inst, 0));
         if (!origin || !origin->access || !put_value(t, inst, origin->def, origin->access))
            return fail(t, STELA_AIR_MEMORY, "pointer identity cast lacks buffer provenance");
         pointer_authority(find_value(t, inst), origin);
         pointer_offset(find_value(t, inst), origin, 0, 0);
         return true;
      }
      if (!value_shape(source, &src_components, &src_bits) ||
          !value_shape(LLVMTypeOf(inst), &components, &bits) || src_bits == 1 || bits == 1 ||
          src_components * src_bits != components * bits)
         return fail(t, STELA_AIR_TYPE, "bitcast between unsupported or differently sized types");
      nir_def *value = get_value(t, LLVMGetOperand(inst, 0));
      return value && put_value(t, inst, nir_bitcast_vector(&t->b, value, bits), 0);
   }
   if (op == LLVMCall) {
      LLVMValueRef callee = LLVMGetCalledValue(inst);
      bool logged;
      if (!log_call(t, inst, callee, &logged))
         return false;
      if (logged)
         return true;
      /* Inlining leaves no call of a defined function but os_log's, whose
       * varargs do not inline, and dispatch only calls through other
       * pointers than a table's. */
      if (stela_air_defined_function(callee))
         return fail(t, STELA_AIR_INSTRUCTION, "a function call was not inlined");
      if (!LLVMIsAFunction(callee))
         return fail(t, STELA_AIR_INSTRUCTION, "function pointers come only from visible function tables");
      if (LLVMIsAFunction(callee)) {
         bool lowered;
         if (!memory_intrinsic(t, inst, LLVMGetIntrinsicID(callee), &lowered))
            return false;
         if (lowered)
            return true;
         /* An assumption only permits optimization, so honoring none is
          * exact, as is ignoring a scope declaration, which only names the
          * accesses inlining found not to alias. */
         unsigned assume = LLVMLookupIntrinsicID("llvm.assume", strlen("llvm.assume"));
         if (assume && LLVMGetIntrinsicID(callee) == assume)
            return true;
         static const char scope[] = "llvm.experimental.noalias.scope.decl";
         unsigned declaration = LLVMLookupIntrinsicID(scope, strlen(scope));
         if (declaration && LLVMGetIntrinsicID(callee) == declaration)
            return true;
      }
      if (LLVMIsAFunction(callee)) {
         unsigned intrinsic = LLVMGetIntrinsicID(callee);
         static const char *const counts[] = {
            [INTEGER_COUNT_POPULATION] = "llvm.ctpop",
            [INTEGER_COUNT_LEADING_ZEROS] = "llvm.ctlz",
            [INTEGER_COUNT_TRAILING_ZEROS] = "llvm.cttz",
         };
         for (unsigned operation = 0; intrinsic && operation < ARRAY_SIZE(counts); operation++) {
            unsigned expected = LLVMLookupIntrinsicID(counts[operation], strlen(counts[operation]));
            if (expected && intrinsic == expected)
               return integer_count(t, inst, callee, intrinsic, (enum integer_count_operation)operation);
         }
      }
      if (LLVMIsAFunction(callee)) {
         unsigned intrinsic = LLVMGetIntrinsicID(callee);
         static const char *const permutations[] = {
            [INTEGER_REVERSE_BITS] = "llvm.bitreverse",
            [INTEGER_SWAP_BYTES] = "llvm.bswap",
         };
         for (unsigned operation = 0; intrinsic && operation < ARRAY_SIZE(permutations); operation++) {
            unsigned expected = LLVMLookupIntrinsicID(permutations[operation], strlen(permutations[operation]));
            if (expected && intrinsic == expected)
               return integer_permutation(t, inst, callee, intrinsic, (enum integer_permutation_operation)operation);
         }
      }
      if (LLVMIsAFunction(callee)) {
         unsigned intrinsic = LLVMGetIntrinsicID(callee);
         static const char *const minmax[] = {
            [INTEGER_SIGNED_MINIMUM] = "llvm.smin",
            [INTEGER_SIGNED_MAXIMUM] = "llvm.smax",
            [INTEGER_UNSIGNED_MINIMUM] = "llvm.umin",
            [INTEGER_UNSIGNED_MAXIMUM] = "llvm.umax",
         };
         for (unsigned operation = 0; intrinsic && operation < ARRAY_SIZE(minmax); operation++) {
            unsigned expected = LLVMLookupIntrinsicID(minmax[operation], strlen(minmax[operation]));
            if (expected && intrinsic == expected)
               return integer_minmax(t, inst, callee, intrinsic, (enum integer_minmax_operation)operation);
         }
      }
      if (LLVMIsAFunction(callee)) {
         unsigned intrinsic = LLVMGetIntrinsicID(callee);
         static const char *const funnels[] = {
            [INTEGER_FUNNEL_LEFT] = "llvm.fshl",
            [INTEGER_FUNNEL_RIGHT] = "llvm.fshr",
         };
         for (unsigned operation = 0; intrinsic && operation < ARRAY_SIZE(funnels); operation++) {
            unsigned expected = LLVMLookupIntrinsicID(funnels[operation], strlen(funnels[operation]));
            if (expected && intrinsic == expected)
               return integer_funnel(t, inst, callee, intrinsic, (enum integer_funnel_operation)operation);
         }
      }
      size_t name_length = 0;
      const char *name = LLVMGetValueName2(callee, &name_length);
      bool lowered;
      if (!pointer_space_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!rate_map_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!imageblock_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!tile_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (stela_air_call_named(name, name_length, "air.get_num_patch_control_points")) {
         unsigned points = t->render.entry.patch_control_points;
         if (!points)
            return fail(t, STELA_AIR_INSTRUCTION, "the size of a patch that declares no control points");
         return put_value(t, inst, nir_imm_int(&t->b, points), 0);
      }
      /* The prelude bound or lowered it. */
      if (stitched_resource_name(name, name_length) ||
          (stitched_builtin_name(name, name_length) && find_value(t, inst)))
         return true;
      if (!stitched_builtin_call(t, inst, callee, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!control_point_call(t, inst, callee, &lowered))
         return false;
      if (lowered)
         return true;
      if (!interpolant_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!sample_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!uniform_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!texture_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!mesh_call(t, inst, callee, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!synchronization_call(t, inst, callee, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!async_copy_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!coherent_call(t, inst, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!air_atomic_call(t, inst, callee, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      if (!command_call(t, inst, callee, name, name_length, &lowered))
         return false;
      if (lowered)
         return true;
      nir_def *math;
      const struct stela_air_call_context math_context = {
         .b = &t->b, .error = t->error, .translator = t,
         .fragment = t->stage_options && t->stage_options->stage == STELA_AIR_FRAGMENT,
         .operand = texture_operand, .fp_environment = math_environment, .load = call_load,
         .store = call_store,
      };
      switch (stela_air_matrix_call(&math_context, inst, callee, name, name_length, &math)) {
      case STELA_AIR_CALL_OTHER:
         break;
      case STELA_AIR_CALL_FAILED:
         return false;
      case STELA_AIR_CALL_LOWERED:
         return !math || put_value(t, inst, math, 0);
      }
      switch (stela_air_simd_call(&math_context, inst, callee, name, name_length, &math)) {
      case STELA_AIR_CALL_OTHER:
         break;
      case STELA_AIR_CALL_FAILED:
         return false;
      case STELA_AIR_CALL_LOWERED:
         return put_value(t, inst, math, 0);
      }
      switch (stela_air_math_call(&math_context, inst, callee, name, name_length, &math)) {
      case STELA_AIR_CALL_OTHER:
         if (!LLVMGetIntrinsicID(callee) && (name_length < 4 || memcmp(name, "air.", 4)))
            return fail(t, STELA_AIR_INSTRUCTION, "unresolved external function %.*s",
                        (int)MIN2(name_length, 160), name);
         return fail(t, STELA_AIR_INSTRUCTION, "unsupported AIR call %.*s",
                     (int)MIN2(name_length, 160), name);
      case STELA_AIR_CALL_FAILED:
         return false;
      case STELA_AIR_CALL_LOWERED:
         break;
      }
      return !math || put_value(t, inst, math, 0);
   }
   if (op == LLVMFNeg) {
      nir_def *value = get_value(t, LLVMGetOperand(inst, 0));
      /* LLVM fneg changes only the sign bit, even for NaNs/subnormals. Do not
       * route this through a target arithmetic negate that can canonicalize. */
      return value && put_value(t, inst, nir_ixor(&t->b, value,
         nir_imm_intN_t(&t->b, 1ull << (value->bit_size - 1), value->bit_size)), 0);
   }
   if (op == LLVMFPExt || op == LLVMFPTrunc || op == LLVMFPToSI || op == LLVMFPToUI ||
       op == LLVMSIToFP || op == LLVMUIToFP)
      return fp_conversion(t, inst, op);
   /* NIR divides at Vulkan's 2.5 ULP, which only fast math permits. */
   if ((op == LLVMFDiv || op == LLVMFRem) && LLVMGetFastMathFlags(inst) != LLVMFastMathAll)
      return precise_division(t, inst, op);
   if (op == LLVMFAdd || op == LLVMFSub || op == LLVMFMul || op == LLVMFDiv || op == LLVMFRem ||
       op == LLVMFCmp) {
      if (!fp_environment(t))
         return false;
      nir_def *a = get_value(t, LLVMGetOperand(inst, 0));
      if (!a)
         return false;
      nir_def *b = get_value(t, LLVMGetOperand(inst, 1));
      if (!b || !float_shape(LLVMTypeOf(LLVMGetOperand(inst, 0)), &components, &bits))
         return fail(t, STELA_AIR_TYPE, "only half and float arithmetic/comparison supported");
      nir_def *value = NULL;
      if (op == LLVMFAdd)
         value = nir_fadd(&t->b, a, b);
      else if (op == LLVMFSub)
         value = nir_fsub(&t->b, a, b);
      else if (op == LLVMFMul)
         value = nir_fmul(&t->b, a, b);
      /* Metal bounds even a fast half quotient to 1 ulp, which a half
       * reciprocal misses: divide in single precision and round once. */
      else if (op == LLVMFDiv && a->bit_size == 16)
         value = nir_f2f16_rtne(&t->b, nir_fdiv(&t->b, nir_f2f32(&t->b, a), nir_f2f32(&t->b, b)));
      else if (op == LLVMFDiv)
         value = nir_fdiv(&t->b, a, b);
      /* LLVM's frem is C's fmod. */
      else if (op == LLVMFRem)
         value = nir_frem(&t->b, a, b);
      else {
         const nir_shader_compiler_options *options = t->b.shader->options;
         /* Match the original frontend/NIR option contract. Some backends
          * lower only canonical comparisons before converting bool widths.
          * These NaN-correct recipes come from the original SPIR-V frontend
          * and NIR algebraic rules; NIR can fuse them when permitted. */
         switch (LLVMGetFCmpPredicate(inst)) {
         case LLVMRealPredicateFalse: value = nir_imm_zero(&t->b, components, 1); break;
         case LLVMRealPredicateTrue: value = nir_inot(&t->b, nir_imm_zero(&t->b, components, 1)); break;
         case LLVMRealOEQ: value = nir_feq(&t->b, a, b); break;
         case LLVMRealOGT: value = nir_flt(&t->b, b, a); break;
         case LLVMRealOGE: value = nir_fge(&t->b, a, b); break;
         case LLVMRealOLT: value = nir_flt(&t->b, a, b); break;
         case LLVMRealOLE: value = nir_fge(&t->b, b, a); break;
         case LLVMRealONE:
            value = options->has_fneo_fcmpu ? nir_fneo(&t->b, a, b) :
               nir_ior(&t->b, nir_flt(&t->b, a, b), nir_flt(&t->b, b, a));
            break;
         case LLVMRealORD:
            value = options->has_ford_funord ? nir_ford(&t->b, a, b) :
               nir_iand(&t->b, nir_feq(&t->b, a, a), nir_feq(&t->b, b, b));
            break;
         case LLVMRealUNO:
            value = options->has_ford_funord ? nir_funord(&t->b, a, b) :
               nir_ior(&t->b, nir_fneu(&t->b, a, a), nir_fneu(&t->b, b, b));
            break;
         case LLVMRealUEQ:
            value = options->has_fneo_fcmpu ? nir_fequ(&t->b, a, b) :
               nir_ior(&t->b, nir_feq(&t->b, a, b),
                       nir_ior(&t->b, nir_fneu(&t->b, a, a), nir_fneu(&t->b, b, b)));
            break;
         case LLVMRealUGT:
            value = options->has_fneo_fcmpu ? nir_fltu(&t->b, b, a) : nir_inot(&t->b, nir_fge(&t->b, b, a));
            break;
         case LLVMRealUGE:
            value = options->has_fneo_fcmpu ? nir_fgeu(&t->b, a, b) : nir_inot(&t->b, nir_flt(&t->b, a, b));
            break;
         case LLVMRealULT:
            value = options->has_fneo_fcmpu ? nir_fltu(&t->b, a, b) : nir_inot(&t->b, nir_fge(&t->b, a, b));
            break;
         case LLVMRealULE:
            value = options->has_fneo_fcmpu ? nir_fgeu(&t->b, b, a) : nir_inot(&t->b, nir_flt(&t->b, b, a));
            break;
         case LLVMRealUNE: value = nir_fneu(&t->b, a, b); break;
         default: return fail(t, STELA_AIR_INSTRUCTION, "unknown floating comparison");
         }
      }
      return put_value(t, inst, value, 0);
   }
   if (bits == 1 && (op == LLVMAdd || op == LLVMSub || op == LLVMMul ||
                    op == LLVMUDiv || op == LLVMSDiv || op == LLVMURem || op == LLVMSRem ||
                    op == LLVMShl || op == LLVMLShr || op == LLVMAShr))
      return fail(t, STELA_AIR_TYPE, "one-bit arithmetic/division/remainder/shift unsupported");
   if (op == LLVMICmp && LLVMGetTypeKind(LLVMTypeOf(LLVMGetOperand(inst, 0))) == LLVMPointerTypeKind) {
      LLVMIntPredicate predicate = LLVMGetICmpPredicate(inst);
      if (predicate != LLVMIntEQ && predicate != LLVMIntNE)
         return fail(t, STELA_AIR_TYPE, "pointer relational comparison unsupported");
      const struct value *a = pointer_value(t, LLVMGetOperand(inst, 0));
      const struct value *b = a ? pointer_value(t, LLVMGetOperand(inst, 1)) : NULL;
      if (t->error->kind != STELA_AIR_OK)
         return false;
      LLVMValueRef a_llvm = LLVMGetOperand(inst, 0), b_llvm = LLVMGetOperand(inst, 1);
      /* Threadgroup, thread and constant memory is an allocation, which no
       * address within is null. */
      bool a_null = LLVMIsAConstantPointerNull(a_llvm), b_null = LLVMIsAConstantPointerNull(b_llvm);
      const struct value *allocated = a_null ? b : b_null ? a : NULL;
      if (allocated && !allocated->null && (allocated->storage || allocated->constant_data))
         return put_value(t, inst, nir_imm_bool(&t->b, predicate == LLVMIntNE), 0);
      if ((!joined_pointer(a, b_llvm, b) && !joined_pointer(b, a_llvm, a)) || !a->def || !b->def ||
          a->def->num_components != 1 || b->def->num_components != 1 ||
          a->def->bit_size != 64 || b->def->bit_size != 64)
         return fail(t, STELA_AIR_MEMORY, "pointer equality requires one validated resource origin");
      /* Equality is invariant under the same unknown base modulo 2^64. */
      return put_value(t, inst, predicate == LLVMIntEQ ?
         nir_ieq(&t->b, a->def, b->def) : nir_ine(&t->b, a->def, b->def), 0);
   }
   if (op == LLVMLoad || op == LLVMStore) {
      struct value *address = pointer_value(t, LLVMGetOperand(inst, op == LLVMStore ? 1 : 0));
      unsigned required = op == LLVMStore ? STELA_AIR_WRITE : STELA_AIR_READ;
      LLVMAtomicOrdering ordering = LLVMGetOrdering(inst);
      bool atomic = ordering != LLVMAtomicOrderingNotAtomic;
      if (!address || !(address->access & required) || LLVMGetVolatile(inst) ||
          (atomic && ordering != LLVMAtomicOrderingUnordered && ordering != LLVMAtomicOrderingMonotonic))
         return fail(t, STELA_AIR_MEMORY, "unsupported volatile/ordered/unbound/disallowed memory access");
      if (atomic && (thread_memory(address) || address->constant_data ||
                     chooses_thread_memory(t, address)))
         return fail(t, STELA_AIR_MEMORY, "atomic access to thread memory or constant data");
      if (op == LLVMStore && !value_shape(LLVMTypeOf(LLVMGetOperand(inst, 0)), &components, &bits)) {
         char *type = LLVMPrintTypeToString(LLVMTypeOf(LLVMGetOperand(inst, 0)));
         fail(t, STELA_AIR_TYPE, "unsupported store type %s", type);
         LLVMDisposeMessage(type);
         return false;
      }
      if (bits == 1)
         return fail(t, STELA_AIR_MEMORY, "sub-byte memory access unsupported");
      if (atomic && (components != 1 || (bits != 32 && bits != 64) || LLVMGetAlignment(inst) < bits / 8))
         return fail(t, STELA_AIR_MEMORY, "relaxed atomic loads and stores take aligned 32- or 64-bit scalars");
      /* Relaxed atomic loads and stores are single, coherent accesses. */
      enum gl_access_qualifier access = atomic ? ACCESS_ATOMIC | ACCESS_COHERENT : 0;
      unsigned alignment = LLVMGetAlignment(inst);
      if (!alignment)
         return fail(t, STELA_AIR_MEMORY, "memory access lacks explicit alignment");
      if (op == LLVMStore) {
         nir_def *value = get_value(t, LLVMGetOperand(inst, 0));
         if (!value)
            return false;
         memory_access(t, address, value, NULL, access, components, bits, alignment);
         return t->error->kind == STELA_AIR_OK;
      }
      return put_value(t, inst, memory_access(t, address, NULL, NULL, access, components, bits,
                                              alignment), 0);
   }
   /* The remaining arithmetic is integer-only. Flags restricting defined
    * LLVM inputs (nuw/nsw/exact) need no extra NIR operation on those inputs;
    * all remaining opcodes are refused. LLVM
    * divisor-zero and signed MIN/-1 (including srem) inputs are undefined;
    * no result or trap contract is invented for those cases. */
   nir_def *a = NULL, *b = NULL, *def = NULL;
   /* A select of something undefined may as well always choose the other. */
   if (op == LLVMSelect && LLVMIsUndef(LLVMGetOperand(inst, 1)))
      return put_value(t, inst, get_value(t, LLVMGetOperand(inst, 2)), 0);
   if (op == LLVMSelect && LLVMIsUndef(LLVMGetOperand(inst, 2)))
      return put_value(t, inst, get_value(t, LLVMGetOperand(inst, 1)), 0);
   switch (op) {
   case LLVMAdd: case LLVMSub: case LLVMMul:
   case LLVMUDiv: case LLVMSDiv: case LLVMURem: case LLVMSRem:
   case LLVMAnd: case LLVMOr: case LLVMXor:
   case LLVMShl: case LLVMLShr: case LLVMAShr: case LLVMICmp:
   case LLVMTrunc: case LLVMZExt: case LLVMSExt:
   case LLVMSelect:
      a = get_value(t, LLVMGetOperand(inst, 0));
      if (!a)
         return false;
      break;
   default:
      return fail(t, STELA_AIR_INSTRUCTION, "unsupported AIR opcode %u", op);
   }
   if (op == LLVMTrunc && bits == 1)
      def = nir_ine_imm(&t->b, nir_iand_imm(&t->b, a, 1), 0);
   else if (op == LLVMZExt && a->bit_size == 1)
      def = nir_b2iN(&t->b, a, bits);
   else if (op == LLVMSExt && a->bit_size == 1)
      def = nir_ineg(&t->b, nir_b2iN(&t->b, a, bits));
   else if (op == LLVMTrunc || op == LLVMZExt)
      def = nir_u2uN(&t->b, a, bits);
   else if (op == LLVMSExt)
      def = nir_i2iN(&t->b, a, bits);
   else {
      b = get_value(t, LLVMGetOperand(inst, 1));
      if (!b)
         return false;
      switch (op) {
      case LLVMAdd: def = nir_iadd(&t->b, a, b); break;
      case LLVMSub: def = nir_isub(&t->b, a, b); break;
      case LLVMMul: def = nir_imul(&t->b, a, b); break;
      case LLVMUDiv: def = nir_udiv(&t->b, a, b); break;
      case LLVMSDiv: def = nir_idiv(&t->b, a, b); break;
      case LLVMURem: def = nir_umod(&t->b, a, b); break;
      case LLVMSRem: def = nir_irem(&t->b, a, b); break;
      case LLVMAnd: def = nir_iand(&t->b, a, b); break;
      case LLVMOr: def = nir_ior(&t->b, a, b); break;
      case LLVMXor: def = nir_ixor(&t->b, a, b); break;
      case LLVMShl: def = nir_ishl(&t->b, a, nir_u2u32(&t->b, b)); break;
      case LLVMLShr: def = nir_ushr(&t->b, a, nir_u2u32(&t->b, b)); break;
      case LLVMAShr: def = nir_ishr(&t->b, a, nir_u2u32(&t->b, b)); break;
      case LLVMICmp: def = compare(t, inst, a, b); break;
      case LLVMSelect: {
         nir_def *c = get_value(t, LLVMGetOperand(inst, 2));
         if (c)
            def = nir_bcsel(&t->b, a, b, c);
         break;
      }
      default: return fail(t, STELA_AIR_INSTRUCTION, "unsupported integer opcode %u", op);
      }
   }
   return def && put_value(t, inst, def, 0);
}

/* Count before and after LLVM legalization. Even unreachable input consumes
 * the caller's compilation budget; it cannot hide an unbounded dead CFG. */
static bool
instruction_budget(struct translator *t, LLVMValueRef fn)
{
   unsigned count = 0;
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
        block = LLVMGetNextBasicBlock(block)) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block); inst;
           inst = LLVMGetNextInstruction(inst)) {
         if (count == t->options->max_instructions) {
            t->error->instruction = count;
            return fail(t, STELA_AIR_LIMIT, "AIR instruction budget exceeded");
         }
         count++;
      }
   }
   return true;
}

static bool
legalize_switches(struct translator *t, LLVMValueRef fn)
{
   if (!instruction_budget(t, fn))
      return false;
   bool has_switch = false;
   unsigned cases = 0;
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
        block = LLVMGetNextBasicBlock(block)) {
      LLVMValueRef term = LLVMGetBasicBlockTerminator(block);
      if (LLVMGetInstructionOpcode(term) == LLVMSwitch) {
         unsigned count = LLVMGetNumSuccessors(term) - 1; /* Exclude default. */
         if (count > t->options->max_instructions - cases)
            return fail(t, STELA_AIR_LIMIT, "AIR switch-case budget exceeded");
         cases += count;
         has_switch = true;
      }
   }
   if (!has_switch)
      return true;

   LLVMPassBuilderOptionsRef options = LLVMCreatePassBuilderOptions();
   if (!options)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM pass options allocation failed");
   /* Only the selected entry is legalized. This is the pinned LLVM pass, not
    * target-specific switch recipes or an optimization of unrelated entries. */
   LLVMErrorRef error = LLVMRunPassesOnFunction(fn, "lower-switch", NULL, options);
   LLVMDisposePassBuilderOptions(options);
   if (error) {
      char *message = LLVMGetErrorMessage(error); /* Consumes error. */
      fail(t, STELA_AIR_CONTROL_FLOW, "LLVM switch legalization failed: %s",
           message ? message : "no diagnostic");
      LLVMDisposeErrorMessage(message);
      return false;
   }
   char *verification = NULL;
   bool invalid = LLVMVerifyModule(t->module, LLVMReturnStatusAction, &verification);
   if (invalid)
      fail(t, STELA_AIR_VERIFY, "%s", verification ? verification : "invalid legalized AIR");
   LLVMDisposeMessage(verification);
   return !invalid && t->error->kind == STELA_AIR_OK && instruction_budget(t, fn);
}

/* Whether `fn` stores a pointer, whose provenance memory cannot keep. */
static bool
stores_pointer(LLVMValueRef fn)
{
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
        block = LLVMGetNextBasicBlock(block)) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block); inst;
           inst = LLVMGetNextInstruction(inst)) {
         if (LLVMGetInstructionOpcode(inst) == LLVMStore &&
             LLVMGetTypeKind(LLVMTypeOf(LLVMGetOperand(inst, 0))) == LLVMPointerTypeKind)
            return true;
      }
   }
   return false;
}

/* The most slots of a pointer array that a load at a run-time index
 * chooses among. */
#define STELA_AIR_POINTER_SLOTS 8

/* Whether `gep` indexes slot `index` of the array its alloca is, and only
 * loads read through it. */
static bool
loaded_slot(LLVMValueRef gep, LLVMTypeRef array, LLVMValueRef *index)
{
   if (!LLVMIsAGetElementPtrInst(gep) || LLVMGetNumOperands(gep) != 3 ||
       LLVMGetGEPSourceElementType(gep) != array || !LLVMIsAConstantInt(LLVMGetOperand(gep, 1)) ||
       LLVMConstIntGetZExtValue(LLVMGetOperand(gep, 1)))
      return false;
   *index = LLVMGetOperand(gep, 2);
   for (LLVMUseRef use = LLVMGetFirstUse(gep); use; use = LLVMGetNextUse(use)) {
      LLVMValueRef user = LLVMGetUser(use);
      if (!LLVMIsALoadInst(user) || LLVMTypeOf(user) != LLVMGetElementType(array))
         return false;
   }
   return true;
}

/* Rewrites each load of a small array of pointers at a run-time index as a
 * choice among its slots' loads at constant indices, so that SROA promotes
 * the array and the pointers keep their provenance; an index past the
 * array is undefined. */
static void
choose_pointer_slots(struct translator *t, LLVMValueRef fn)
{
   LLVMBuilderRef builder = NULL;
   LLVMBasicBlockRef entry = LLVMGetEntryBasicBlock(fn);
   for (LLVMValueRef alloca = LLVMGetFirstInstruction(entry); alloca;
        alloca = LLVMGetNextInstruction(alloca)) {
      if (!LLVMIsAAllocaInst(alloca))
         continue;
      LLVMTypeRef array = LLVMGetAllocatedType(alloca);
      if (LLVMGetTypeKind(array) != LLVMArrayTypeKind ||
          LLVMGetTypeKind(LLVMGetElementType(array)) != LLVMPointerTypeKind ||
          LLVMGetArrayLength2(array) > STELA_AIR_POINTER_SLOTS)
         continue;
      unsigned slots = (unsigned)LLVMGetArrayLength2(array);
      LLVMTypeRef element = LLVMGetElementType(array);
      LLVMUseRef next;
      for (LLVMUseRef use = LLVMGetFirstUse(alloca); use; use = next) {
         next = LLVMGetNextUse(use);
         LLVMValueRef gep = LLVMGetUser(use), index;
         if (!loaded_slot(gep, array, &index) || LLVMIsAConstant(index))
            continue;
         /* The GEP's uses are loads, which go with it. */
         while (next && LLVMGetUser(next) == gep)
            next = LLVMGetNextUse(next);
         if (!builder)
            builder = LLVMCreateBuilderInContext(t->context);
         LLVMTypeRef index_type = LLVMTypeOf(index);
         LLVMValueRef load;
         while ((load = LLVMGetFirstUse(gep) ? LLVMGetUser(LLVMGetFirstUse(gep)) : NULL)) {
            LLVMPositionBuilderBefore(builder, load);
            LLVMValueRef chosen = NULL;
            for (unsigned slot = 0; slot < slots; slot++) {
               LLVMValueRef at[2] = {LLVMGetOperand(gep, 1), LLVMConstInt(index_type, slot, false)};
               LLVMValueRef address = LLVMBuildInBoundsGEP2(builder, array, alloca, at, 2, "");
               LLVMValueRef value = LLVMBuildLoad2(builder, element, address, "");
               LLVMSetAlignment(value, LLVMGetAlignment(load));
               chosen = !chosen ? value :
                        LLVMBuildSelect(builder,
                                        LLVMBuildICmp(builder, LLVMIntEQ, index, at[1], ""),
                                        value, chosen, "");
            }
            LLVMReplaceAllUsesWith(load, chosen);
            LLVMInstructionEraseFromParent(load);
         }
         LLVMInstructionEraseFromParent(gep);
      }
   }
   if (builder)
      LLVMDisposeBuilder(builder);
}

/* An unoptimized build keeps every value in memory and bars every pass from
 * its functions, which an optimizing build has no need to undo, unless it
 * left a pointer in a stack slot: only an SSA value keeps its provenance. */
static bool
promote_memory(struct translator *t, LLVMValueRef fn)
{
   LLVMRemoveEnumAttributeAtIndex(fn, LLVMAttributeFunctionIndex,
                                  LLVMGetEnumAttributeKindForName("optnone", 7));
   choose_pointer_slots(t, fn);
   LLVMPassBuilderOptionsRef options = LLVMCreatePassBuilderOptions();
   if (!options)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM pass options allocation failed");
   LLVMErrorRef error = LLVMRunPassesOnFunction(fn, "sroa", NULL, options);
   LLVMDisposePassBuilderOptions(options);
   if (error) {
      LLVMDisposeErrorMessage(LLVMGetErrorMessage(error));
      return fail(t, STELA_AIR_INSTRUCTION, "LLVM refused to promote the entry's memory");
   }
   return true;
}

/* Whether a terminator can branch along successor s: a conditional branch on
 * a constant takes one edge only, and what only the other reaches never runs
 * (folding constants leaves such branches, with poison on the dead side). */
static bool
edge_taken(LLVMValueRef term, unsigned s)
{
   if (!LLVMIsACondBrInst(term))
      return true;
   LLVMValueRef condition = LLVMGetCondition(term);
   return !LLVMIsAConstantInt(condition) ||
          (LLVMConstIntGetZExtValue(condition) != 0) == (s == 0);
}

/* Whether from can branch to to. */
static bool
edge_to(LLVMBasicBlockRef from, LLVMBasicBlockRef to)
{
   LLVMValueRef term = LLVMGetBasicBlockTerminator(from);
   unsigned successors = term ? LLVMGetNumSuccessors(term) : 0;
   for (unsigned s = 0; s < successors; s++) {
      if (LLVMGetSuccessor(term, s) == to && edge_taken(term, s))
         return true;
   }
   return false;
}

/* Refuses a call, on a path the entry can take, of a visible function the
 * module references and no linked module defines. Such a call cannot run,
 * and what it would have read is left in memory, where a pointer it was
 * passed (a thread struct of buffer addresses, as MPS kernels pass their
 * stitched functions) would otherwise be refused first. Branches on
 * constants take only their taken edge. */
static bool
linked_references(struct translator *t, LLVMValueRef fn)
{
   LLVMValueRef nodes[AIR_MAX_METADATA_OPERANDS];
   unsigned count;
   if (!stela_air_named_md(&t->metadata, "air.visible_function_references", nodes, &count))
      return false;
   LLVMValueRef unlinked[AIR_MAX_METADATA_OPERANDS];
   unsigned unlinked_count = 0;
   for (unsigned i = 0; i < count; i++) {
      LLVMValueRef fields[AIR_MAX_METADATA_OPERANDS];
      unsigned field_count;
      if (!stela_air_md_node(&t->metadata, nodes[i], fields, &field_count))
         return false;
      if (field_count == 3 && fields[1] && LLVMIsAFunction(fields[1]) &&
          LLVMIsDeclaration(fields[1]))
         unlinked[unlinked_count++] = fields[1];
   }
   if (!unlinked_count)
      return true;
   unsigned blocks = LLVMCountBasicBlocks(fn);
   LLVMBasicBlockRef *stack = malloc(sizeof(*stack) * (blocks ? blocks : 1));
   struct set *seen = _mesa_pointer_set_create(NULL);
   if (!stack || !seen) {
      free(stack);
      _mesa_set_destroy(seen, NULL);
      return fail(t, STELA_AIR_ALLOCATION, "visible function reference walk allocation failed");
   }
   unsigned depth = 0;
   stack[depth++] = LLVMGetEntryBasicBlock(fn);
   _mesa_set_add(seen, stack[0]);
   bool linked = true;
   while (linked && depth) {
      LLVMBasicBlockRef block = stack[--depth];
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block); linked && inst;
           inst = LLVMGetNextInstruction(inst)) {
         LLVMValueRef callee = LLVMIsACallInst(inst) ? LLVMGetCalledValue(inst) : NULL;
         for (unsigned i = 0; callee && i < unlinked_count; i++) {
            if (callee != unlinked[i])
               continue;
            size_t length;
            const char *name = LLVMGetValueName2(callee, &length);
            linked = fail(t, STELA_AIR_ARGUMENT, "visible function %.*s is not linked",
                          (int)MIN2(length, 160), name);
            break;
         }
      }
      LLVMValueRef terminator = LLVMGetBasicBlockTerminator(block);
      unsigned successors = terminator ? LLVMGetNumSuccessors(terminator) : 0;
      for (unsigned i = 0; i < successors; i++) {
         if (!edge_taken(terminator, i))
            continue;
         LLVMBasicBlockRef next = LLVMGetSuccessor(terminator, i);
         if (!_mesa_set_search(seen, next)) {
            _mesa_set_add(seen, next);
            stack[depth++] = next;
         }
      }
   }
   free(stack);
   _mesa_set_destroy(seen, NULL);
   return linked;
}

static struct air_block *
cfg_block(struct translator *t, LLVMBasicBlockRef llvm)
{
   struct hash_entry *entry = _mesa_hash_table_search(t->block_map, llvm);
   return entry ? entry->data : NULL;
}

/* Whether an instruction of `user` reads the constant expression `value`
 * as an instruction the translator handles: a GEP or a pointer cast; operand
 * reads handle constant bitcasts.
 * Calls keep their constant-space pointers, which name the strings and
 * tables they read. */
static bool
expandable(LLVMValueRef user, LLVMValueRef value)
{
   if (!LLVMIsAConstantExpr(value))
      return false;
   switch (LLVMGetConstOpcode(value)) {
   case LLVMGetElementPtr:
   case LLVMAddrSpaceCast:
   case LLVMPtrToInt:
   case LLVMIntToPtr:
      break;
   default:
      return false;
   }
   LLVMTypeRef type = LLVMTypeOf(value);
   bool pointer = LLVMGetTypeKind(type) == LLVMPointerTypeKind;
   if (!LLVMIsACallInst(user))
      return true;
   return pointer && LLVMGetPointerAddressSpace(type) != 2;
}

/* The instructions computing the constant expression `value`, built at
 * `builder`'s position. */
static LLVMValueRef
expand_constant(LLVMBuilderRef builder, LLVMValueRef user, LLVMValueRef value)
{
   unsigned count = LLVMGetNumOperands(value);
   LLVMValueRef operands[count ? count : 1];
   for (unsigned i = 0; i < count; i++) {
      operands[i] = LLVMGetOperand(value, i);
      if (expandable(user, operands[i]))
         operands[i] = expand_constant(builder, user, operands[i]);
   }
   LLVMOpcode opcode = LLVMGetConstOpcode(value);
   /* Constant folding writes a constant offset off a null pointer, a buffer
    * a function constant left out, as the integer it is: the translator
    * follows it as that offset off null, which reaches no memory. */
   if (opcode == LLVMIntToPtr && LLVMIsAConstantInt(operands[0])) {
      LLVMTypeRef type = LLVMTypeOf(value);
      LLVMValueRef null = LLVMConstPointerNull(type);
      LLVMValueRef base = LLVMBuildFreeze(builder, null, "");
      LLVMTypeRef byte = LLVMInt8TypeInContext(LLVMGetTypeContext(type));
      LLVMValueRef inst = LLVMBuildGEP2(builder, byte, base, &operands[0], 1, "");
      LLVMSetOperand(inst, 0, null);
      LLVMInstructionEraseFromParent(base);
      return inst;
   }
   /* The builder folds constant operands back into an expression, so the
    * instruction is built over a placeholder its own operand replaces. */
   LLVMValueRef placeholder = LLVMBuildFreeze(builder, operands[0], "");
   LLVMValueRef inst;
   if (opcode != LLVMGetElementPtr) {
      inst = LLVMBuildCast(builder, opcode, placeholder, LLVMTypeOf(value), "");
   } else {
      LLVMTypeRef source = LLVMGetGEPSourceElementType(value);
      inst = LLVMIsInBounds(value) ?
         LLVMBuildInBoundsGEP2(builder, source, placeholder, &operands[1], count - 1, "") :
         LLVMBuildGEP2(builder, source, placeholder, &operands[1], count - 1, "");
   }
   LLVMSetOperand(inst, 0, operands[0]);
   LLVMInstructionEraseFromParent(placeholder);
   return inst;
}

/* Rewrites `fn`'s constant-expression addresses and casts as instructions,
 * as an optimizer leaves a threadgroup array's constant element: values the
 * translator follows as it does any other address. A phi's operand computes
 * at the end of its incoming block, once for each block. */
static bool
expand_constant_expressions(struct translator *t, LLVMValueRef fn)
{
   LLVMBuilderRef builder = LLVMCreateBuilderInContext(t->context);
   if (!builder)
      return fail(t, STELA_AIR_ALLOCATION, "LLVM builder allocation failed");
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
        block = LLVMGetNextBasicBlock(block)) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block); inst;
           inst = LLVMGetNextInstruction(inst)) {
         bool phi = LLVMIsAPHINode(inst);
         unsigned count = LLVMGetNumOperands(inst);
         for (unsigned i = 0; i < count; i++) {
            LLVMValueRef value = LLVMGetOperand(inst, i);
            if (!expandable(inst, value))
               continue;
            LLVMValueRef expanded = NULL;
            if (phi) {
               LLVMBasicBlockRef from = LLVMGetIncomingBlock(inst, i);
               for (unsigned j = 0; j < i && !expanded; j++) {
                  if (LLVMGetIncomingBlock(inst, j) == from && !LLVMIsAConstant(LLVMGetOperand(inst, j)))
                     expanded = LLVMGetOperand(inst, j);
               }
               if (!expanded)
                  LLVMPositionBuilderBefore(builder, LLVMGetBasicBlockTerminator(from));
            } else {
               LLVMPositionBuilderBefore(builder, inst);
            }
            if (!expanded)
               expanded = expand_constant(builder, inst, value);
            LLVMSetOperand(inst, i, expanded);
         }
      }
   }
   LLVMDisposeBuilder(builder);
   return true;
}

/* Refuses a reachable call of an intrinsic of an Apple GPU ISA (llvm.agx*,
 * such as agx3's matrix-unit gemm and element-masked loads): those name
 * one hardware generation's instructions, whose operand layouts AIR does
 * not define, rather than an AIR operation. Found first, it is named
 * instead of a wider vector or PHI it feeds. */
static bool
isa_intrinsics_absent(struct translator *t)
{
   for (unsigned i = 0; i < t->reachable_count; i++) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(t->work[i]->llvm); inst;
           inst = LLVMGetNextInstruction(inst)) {
         LLVMValueRef callee = LLVMIsACallInst(inst) ? LLVMGetCalledValue(inst) : NULL;
         if (!callee || !LLVMIsAFunction(callee))
            continue;
         size_t length;
         const char *name = LLVMGetValueName2(callee, &length);
         if (length > 8 && !memcmp(name, "llvm.agx", 8))
            return fail(t, STELA_AIR_INSTRUCTION, "unsupported Apple GPU ISA intrinsic %.*s",
                        (int)MIN2(length, 160), name);
      }
   }
   return true;
}

static bool
cfg_prepare(struct translator *t, LLVMValueRef fn)
{
   if (!expand_constant_expressions(t, fn) || !legalize_switches(t, fn))
      return false;
   t->block_count = LLVMCountBasicBlocks(fn);
   if (!t->block_count || t->block_count > t->options->max_instructions)
      return fail(t, STELA_AIR_LIMIT, "AIR basic-block budget exceeded");
   t->cfg_scratch = ralloc_context(NULL);
   if (!t->cfg_scratch)
      return fail(t, STELA_AIR_ALLOCATION, "AIR CFG scratch allocation failed");
   t->blocks = rzalloc_array(t->cfg_scratch, struct air_block, t->block_count);
   t->work = ralloc_array(t->cfg_scratch, struct air_block *, t->block_count);
   t->block_map = _mesa_pointer_hash_table_create(t->cfg_scratch);
   if (!t->blocks || !t->work || !t->block_map)
      return fail(t, STELA_AIR_ALLOCATION, "AIR CFG allocation failed");
   unsigned i = 0;
   for (LLVMBasicBlockRef block = LLVMGetFirstBasicBlock(fn); block;
        block = LLVMGetNextBasicBlock(block)) {
      t->blocks[i].llvm = block;
      if (!_mesa_hash_table_insert(t->block_map, block, &t->blocks[i++]))
         return fail(t, STELA_AIR_ALLOCATION, "AIR block map allocation failed");
   }

   /* Iterative discovery does not impose a recursion/nesting limit. A block
    * enters the queue once; all dominators precede its emission regardless of
    * the input's textual block order. PHI backedge defs are declared separately. */
   t->blocks[0].reachable = true;
   t->work[t->reachable_count++] = &t->blocks[0];
   for (unsigned current = 0; current < t->reachable_count; current++) {
      LLVMValueRef term = LLVMGetBasicBlockTerminator(t->work[current]->llvm);
      LLVMOpcode op = LLVMGetInstructionOpcode(term);
      /* An entry that is unreachable runs nothing defined at all. */
      if ((op != LLVMRet && op != LLVMUnreachable && op != LLVMUncondBr && op != LLVMCondBr) ||
          (op == LLVMUnreachable && !current))
         return fail(t, STELA_AIR_CONTROL_FLOW, "unsupported AIR terminator %u", op);
      for (unsigned s = 0; s < LLVMGetNumSuccessors(term); s++) {
         struct air_block *next = cfg_block(t, LLVMGetSuccessor(term, s));
         if (!next)
            return fail(t, STELA_AIR_CONTROL_FLOW, "branch leaves AIR entry");
         if (!edge_taken(term, s))
            continue;
         if (!next->reachable) {
            next->reachable = true;
            t->work[t->reachable_count++] = next;
         }
      }
   }
   return isa_intrinsics_absent(t);
}

static bool
same_pointer_resource(const struct value *a, const struct value *b)
{
   return a && b && a->pointer_root && a->pointer_root == b->pointer_root &&
      a->access == b->access && a->ssbo == b->ssbo && a->ssbo_slot == b->ssbo_slot &&
      a->fields.first == b->fields.first && a->fields.count == b->fields.count &&
      a->storage == b->storage && a->constant_data == b->constant_data &&
      a->storage_size == b->storage_size && a->constant_base == b->constant_base;
}

/* Whether a pointer is a device or constant buffer's numeric address, which
 * no buffer has at zero, so that a null pointer can stand for it: a join or
 * comparison with null keeps its authority. Offsets into a variable, constant
 * data or a bound buffer may be zero, and a pointer off null is not null. */
static bool
numeric_address(const struct value *v)
{
   return v && v->pointer_root && !v->null && !v->storage && !v->constant_data && !v->ssbo &&
      !v->array && !v->texture && !v->sampler && !v->object && v->def &&
      v->def->num_components == 1 && v->def->bit_size == 64;
}

/* The authority `a` and `b` share, or that a null `b` leaves with `a`. */
static const struct value *
joined_pointer(const struct value *a, LLVMValueRef b_llvm, const struct value *b)
{
   if (same_pointer_resource(a, b))
      return a;
   return LLVMIsAConstantPointerNull(b_llvm) && numeric_address(a) ? a : NULL;
}

static struct pointer_node *
pointer_node(struct translator *t, LLVMValueRef llvm)
{
   struct hash_entry *entry = _mesa_hash_table_search(t->pointer_map, llvm);
   return entry ? entry->data : NULL;
}

/* Validates a provenance root. An address loaded from an argument buffer is
 * a root its load validates when translated. */
static bool
pointer_anchor(struct translator *t, LLVMValueRef llvm)
{
   if (LLVMIsALoadInst(llvm) || imageblock_data_call(llvm))
      return true;
   struct value *value = find_value(t, llvm);
   if (!value && LLVMIsAAllocaInst(llvm))
      value = private_allocation(t, llvm);
   else if (!value && (LLVMIsAGlobalVariable(llvm) || LLVMIsAConstantPointerNull(llvm)))
      value = pointer_value(t, llvm);
   if (texture_choice_root(value) && value->pointer_root == llvm)
      return true;
   return (value && value->access && value->pointer_root == llvm &&
           pointer_space(LLVMTypeOf(llvm)) && value->def &&
           value->def->num_components == 1 && value->def->bit_size == 64) ||
      fail(t, STELA_AIR_MEMORY, "pointer PHI dependency lacks validated resource origin");
}

static unsigned
pointer_dependencies(const struct pointer_node *node)
{
   LLVMOpcode op = LLVMGetInstructionOpcode(node->llvm);
   return op == LLVMPHI ? LLVMCountIncoming(node->llvm) : op == LLVMSelect ? 2 : 1;
}

static LLVMValueRef
pointer_dependency(struct translator *t, const struct pointer_node *node, unsigned index)
{
   if (LLVMGetInstructionOpcode(node->llvm) == LLVMSelect)
      return LLVMGetOperand(node->llvm, index + 1);
   if (LLVMGetInstructionOpcode(node->llvm) != LLVMPHI)
      return LLVMGetOperand(node->llvm, 0);
   LLVMBasicBlockRef from = LLVMGetIncomingBlock(node->llvm, index);
   struct air_block *pred = cfg_block(t, from);
   /* LLVM verification has already established entry-local predecessor edges. */
   return pred && pred->reachable &&
          edge_to(from, LLVMGetInstructionParent(node->llvm)) ?
          LLVMGetIncomingValue(node->llvm, index) : NULL;
}

/* Resolve one SCC only from already-proven outgoing dependencies. In
 * particular, a rooted user cannot lend authority backward to a rootless cycle. */
/* Joins a null pointer or null texture into a component's *null. */
static bool
join_null(struct translator *t, LLVMValueRef *null, LLVMValueRef dependency)
{
   if (*null && stela_air_null_texture(*null) != stela_air_null_texture(dependency))
      return fail(t, STELA_AIR_MEMORY, "pointer PHI joins a null texture with a null pointer");
   *null = dependency;
   return true;
}

/* Adds root to a component's roots, once. */
static bool
join_root(struct translator *t, LLVMValueRef *roots, unsigned *count, LLVMValueRef root)
{
   for (unsigned i = 0; i < *count; i++) {
      if (roots[i] == root)
         return true;
   }
   if (*count == POINTER_CHOICES)
      return fail(t, STELA_AIR_LIMIT, "a pointer chooses among more than %u resources",
                  POINTER_CHOICES);
   roots[(*count)++] = root;
   return true;
}

/* The choices of a component of several roots, or of one root and null
 * that a null address cannot stand for: each a resource accessed through
 * its address or offset. */
static const struct pointer_choices *
component_choices(struct translator *t, const LLVMValueRef *roots, unsigned count,
                  LLVMValueRef null)
{
   if (null && stela_air_null_texture(null)) {
      fail(t, STELA_AIR_MEMORY, "a texture choice joins distinct textures or resources");
      return NULL;
   }
   struct pointer_choices *choices = rzalloc(t->cfg_scratch, struct pointer_choices);
   if (!choices) {
      fail(t, STELA_AIR_ALLOCATION, "pointer choice allocation failed");
      return NULL;
   }
   for (unsigned i = 0; i < count; i++) {
      const struct value *root = LLVMIsALoadInst(roots[i]) ? NULL : find_value(t, roots[i]);
      if (!root || roots[i] == t->imageblock_root || root->texture || root->sampler ||
          root->array || root->object || root->fields.count || root->pointer_root != roots[i]) {
         fail(t, STELA_AIR_MEMORY, "a pointer chooses among resources it cannot address alike: "
              "loaded addresses, imageblocks, textures or argument buffers");
         return NULL;
      }
      choices->roots[choices->count++] = roots[i];
   }
   if (null && !join_root(t, choices->roots, &choices->count, null))
      return NULL;
   return choices;
}

static bool
pointer_component(struct translator *t, struct pointer_node **stack,
                  unsigned begin, unsigned end, unsigned component)
{
   LLVMValueRef roots[POINTER_CHOICES], null = NULL;
   unsigned count = 0;
   for (unsigned i = begin; i < end; i++)
      stack[i]->component = component;
   for (unsigned i = begin; i < end; i++) {
      struct pointer_node *node = stack[i];
      unsigned dependencies = pointer_dependencies(node);
      for (unsigned j = 0; j < dependencies; j++) {
         LLVMValueRef dependency = pointer_dependency(t, node, j);
         /* An undefined pointer may be any pointer: it lends no authority
          * and takes the component's, as its PHI's NIR undef does. */
         if (!dependency || LLVMIsUndef(dependency))
            continue;
         if (LLVMIsAConstantPointerNull(dependency) || stela_air_null_texture(dependency)) {
            if (!join_null(t, &null, dependency))
               return false;
            continue;
         }
         struct pointer_node *dep = pointer_node(t, dependency);
         LLVMValueRef candidate;
         if (dep) {
            if (dep->component == component)
               continue;
            if (dep->component == UINT_MAX)
               return fail(t, STELA_AIR_MEMORY, "unresolved pointer provenance component");
            for (unsigned k = 0; dep->choices && k < dep->choices->count; k++) {
               LLVMValueRef root = dep->choices->roots[k];
               if (LLVMIsAConstantPointerNull(root) ? !join_null(t, &null, root) :
                                                      !join_root(t, roots, &count, root))
                  return false;
            }
            if (dep->choices)
               continue;
            candidate = dep->root;
            if (LLVMIsAConstantPointerNull(candidate) || stela_air_null_texture(candidate)) {
               if (!join_null(t, &null, candidate))
                  return false;
               continue;
            }
         } else {
            if (!pointer_anchor(t, dependency))
               return false;
            /* Every pointer into the imageblock has one origin. */
            candidate = imageblock_data_call(dependency) ? t->imageblock_root : dependency;
         }
         if (!join_root(t, roots, &count, candidate))
            return false;
      }
   }
   LLVMValueRef root = count ? roots[0] : null;
   const struct value *rooted = count && !LLVMIsALoadInst(root) ? find_value(t, root) : NULL;
   /* A join of null and a texture is a texture choice, and a numeric
    * address's null is its own; null and an offset choose. */
   bool chooses = count > 1 ||
                  (count && null && !(stela_air_null_texture(null) ? texture_choice_root(rooted) :
                                      numeric_address(rooted)));
   const struct pointer_choices *choices =
      chooses ? component_choices(t, roots, count, null) : NULL;
   if (chooses && !choices)
      return false;
   if (!root)
      return fail(t, STELA_AIR_MEMORY, "pointer PHI has a rootless dependency cycle");
   for (unsigned i = begin; i < end; i++) {
      stack[i]->root = choices ? NULL : root;
      stack[i]->choices = choices;
   }
   return true;
}

static bool
pointer_solve(struct translator *t, struct pointer_node **dfs, struct pointer_node **stack)
{
   unsigned serial = 0, component = 0, depth = 0, active = 0;
   for (unsigned i = 0; i < t->pointer_count; i++) {
      struct pointer_node *node = &t->pointers[i];
      node->index = node->component = UINT_MAX;
      node->next = 0;
      node->active = false;
   }
   for (unsigned i = 0; i < t->pointer_count; i++) {
      if (t->pointers[i].index != UINT_MAX)
         continue;
      dfs[depth++] = &t->pointers[i];
      while (depth) {
         struct pointer_node *node = dfs[depth - 1];
         if (node->index == UINT_MAX) {
            node->index = node->low = serial++;
            node->active = true;
            stack[active++] = node;
         }
         unsigned count = pointer_dependencies(node);
         if (node->next < count) {
            LLVMValueRef llvm = pointer_dependency(t, node, node->next++);
            struct pointer_node *dep = llvm ? pointer_node(t, llvm) : NULL;
            if (!dep)
               continue; /* Roots are checked when the SCC is resolved. */
            if (dep->index == UINT_MAX) {
               dfs[depth++] = dep;
            } else if (dep->active) {
               node->low = MIN2(node->low, dep->index);
            }
            continue;
         }
         if (node->low == node->index) {
            unsigned end = active;
            do {
               stack[--active]->active = false;
            } while (stack[active] != node);
            if (!pointer_component(t, stack, active, end, component++))
               return false;
         }
         depth--;
         if (depth)
            dfs[depth - 1]->low = MIN2(dfs[depth - 1]->low, node->low);
      }
   }
   return true;
}

static bool
pointer_provenance(struct translator *t)
{
   for (unsigned i = 0; i < t->reachable_count; i++) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(t->work[i]->llvm); inst;
           inst = LLVMGetNextInstruction(inst)) {
         if (LLVMGetTypeKind(LLVMTypeOf(inst)) == LLVMPointerTypeKind)
            t->pointer_count++;
      }
   }
   t->pointer_map = _mesa_pointer_hash_table_create(t->cfg_scratch);
   if (!t->pointer_map)
      return fail(t, STELA_AIR_ALLOCATION, "pointer provenance map allocation failed");
   if (!t->pointer_count)
      return true;
   t->pointers = rzalloc_array(t->cfg_scratch, struct pointer_node, t->pointer_count);
   struct pointer_node **dfs = ralloc_array(t->cfg_scratch, struct pointer_node *, t->pointer_count);
   struct pointer_node **stack = ralloc_array(t->cfg_scratch, struct pointer_node *, t->pointer_count);
   if (!t->pointers || !dfs || !stack)
      return fail(t, STELA_AIR_ALLOCATION, "pointer provenance scratch allocation failed");
   unsigned next = 0;
   for (unsigned i = 0; i < t->reachable_count; i++) {
      for (LLVMValueRef inst = LLVMGetFirstInstruction(t->work[i]->llvm); inst;
           inst = LLVMGetNextInstruction(inst)) {
         if (LLVMGetTypeKind(LLVMTypeOf(inst)) != LLVMPointerTypeKind)
            continue;
         LLVMOpcode op = LLVMGetInstructionOpcode(inst);
         if (imageblock_data_call(inst) && !t->imageblock_root)
            t->imageblock_root = inst;
         if (op == LLVMAlloca || op == LLVMLoad || stela_air_read_sampler(inst) ||
             stela_air_null_texture(inst) || stitched_resource_call(inst) ||
             imageblock_data_call(inst) || async_copy_call_named(inst))
            continue; /* Roots, like arguments and globals; a copy's event is unused. */
         if (op != LLVMPHI && op != LLVMSelect && op != LLVMGetElementPtr && op != LLVMBitCast &&
             op != LLVMAddrSpaceCast)
            return fail(t, STELA_AIR_TYPE, "unsupported pointer provenance producer %d", op);
         if (!pointer_space(LLVMTypeOf(inst)) ||
             (op != LLVMPHI && op != LLVMSelect && op != LLVMAddrSpaceCast &&
              LLVMTypeOf(LLVMGetOperand(inst, 0)) != LLVMTypeOf(inst)))
            return fail(t, STELA_AIR_MEMORY, "unsupported pointer provenance address space");
         struct pointer_node *node = &t->pointers[next++];
         node->llvm = inst;
         if (!_mesa_hash_table_insert(t->pointer_map, inst, node))
            return fail(t, STELA_AIR_ALLOCATION, "pointer provenance insertion failed");
      }
   }
   t->pointer_count = next; /* Allocations are roots, not nodes. */
   return pointer_solve(t, dfs, stack);
}

static bool
cfg_phi_shape(struct translator *t, LLVMTypeRef type, unsigned *leaves)
{
   unsigned components, bits;
   *leaves = 0;
   if (value_shape(type, &components, &bits) || stela_air_matrix_type(type, &bits) ||
       pointer_space(type))
      return true;
   if (LLVMGetTypeKind(type) != LLVMStructTypeKind || LLVMIsOpaqueStruct(type) ||
       !LLVMCountStructElementTypes(type) || LLVMCountStructElementTypes(type) > AIR_MAX_ARGUMENTS)
      return fail(t, STELA_AIR_TYPE, "PHI requires a numeric value, proven pointer or flat struct");
   *leaves = LLVMCountStructElementTypes(type);
   for (unsigned i = 0; i < *leaves; i++) {
      if (!leaf_shape(LLVMStructGetTypeAtIndex(type, i)))
         return fail(t, STELA_AIR_TYPE, "aggregate PHI requires flat scalar/vector leaves");
   }
   return true;
}

static struct value *
cfg_phi_leaf(struct translator *t, LLVMTypeRef type)
{
   unsigned components, bits;
   if (pointer_space(type)) {
      components = 1;
      bits = 64;
   } else if (stela_air_matrix_type(type, &bits)) {
      components = 2;
   } else if (!value_shape(type, &components, &bits)) {
      fail(t, STELA_AIR_TYPE, "unsupported PHI leaf type");
      return NULL;
   }
   struct value *value = rzalloc(t->values, struct value);
   nir_phi_instr *phi = nir_phi_instr_create(t->b.shader);
   if (!value || !phi) {
      fail(t, STELA_AIR_ALLOCATION, "AIR PHI leaf allocation failed");
      return NULL;
   }
   nir_def_init(&phi->instr, &phi->def, components, bits);
   nir_builder_instr_insert(&t->b, &phi->instr);
   value->def = &phi->def;
   value->components = components;
   value->bits = bits;
   return value;
}

/* A PHI choosing a texture or null textures: that texture, with where the
 * choice is it. */
static bool
texture_choice_phi(struct translator *t, LLVMValueRef inst, const struct value *root)
{
   struct value *value = rzalloc(t->values, struct value);
   nir_phi_instr *phi = nir_phi_instr_create(t->b.shader);
   if (!value || !phi)
      return fail(t, STELA_AIR_ALLOCATION, "AIR texture PHI allocation failed");
   nir_def_init(&phi->instr, &phi->def, 1, 1);
   nir_builder_instr_insert(&t->b, &phi->instr);
   pointer_authority(value, root);
   value->present = &phi->def;
   return _mesa_hash_table_insert(t->values, inst, value) != NULL ||
      fail(t, STELA_AIR_ALLOCATION, "AIR PHI value map allocation failed");
}

static nir_def *
texture_choice_present(struct translator *t, const struct value *target, LLVMValueRef incoming)
{
   if (stela_air_null_texture(incoming))
      return nir_imm_false(&t->b);
   if (LLVMIsUndef(incoming))
      return nir_undef(&t->b, 1, 1);
   const struct value *v = find_value(t, incoming);
   if (!texture_choice_root(v) || v->pointer_root != target->pointer_root) {
      fail(t, STELA_AIR_MEMORY, "a texture choice joins distinct textures");
      return NULL;
   }
   return v->present ? v->present : nir_imm_true(&t->b);
}

/* Makes value a pointer chosen among choices by choice: accessible as every
 * root is. */
static void
choose(struct translator *t, struct value *value, const struct pointer_choices *choices,
       nir_def *choice)
{
   value->choices = choices;
   value->choice = choice;
   value->pointer_root = NULL;
   value->access = STELA_AIR_READ_WRITE;
   for (unsigned i = 0; i < choices->count; i++)
      value->access &= choice_root(t, choices, i).access;
}

/* The index among choices of the root of `incoming`, or of the one it
 * chooses; NULL after reporting why not. */
static nir_def *
choice_index(struct translator *t, const struct pointer_choices *choices, LLVMValueRef incoming)
{
   if (LLVMIsUndef(incoming))
      return nir_undef(&t->b, 1, 32);
   const struct value *v = LLVMIsAConstantPointerNull(incoming) ? NULL : find_value(t, incoming);
   const struct pointer_choices *from = v ? v->choices : NULL;
   unsigned count = from ? from->count : 1;
   nir_def *index = NULL;
   for (unsigned j = 0; j < count; j++) {
      LLVMValueRef root = from ? from->roots[j] : v ? v->pointer_root : incoming;
      unsigned k = 0;
      while (k < choices->count && choices->roots[k] != root &&
             !(LLVMIsAConstantPointerNull(choices->roots[k]) && root &&
               LLVMIsAConstantPointerNull(root)))
         k++;
      if (k == choices->count) {
         fail(t, STELA_AIR_MEMORY, "a pointer chooses a root its provenance does not");
         return NULL;
      }
      nir_def *chosen = nir_imm_int(&t->b, k);
      index = index ? nir_bcsel(&t->b, nir_ieq_imm(&t->b, v->choice, j), chosen, index) : chosen;
   }
   return index;
}

static bool
cfg_declare_phi(struct translator *t, LLVMValueRef inst)
{
   LLVMTypeRef type = LLVMTypeOf(inst);
   unsigned leaves;
   if (!cfg_phi_shape(t, type, &leaves))
      return false;
   if (pointer_space(type)) {
      struct pointer_node *node = pointer_node(t, inst);
      const struct value *root = node && node->root && !LLVMIsALoadInst(node->root) ?
                                    find_value(t, node->root) : NULL;
      if (texture_choice_root(root))
         return texture_choice_phi(t, inst, root);
   }
   struct value *value;
   if (leaves) {
      value = rzalloc(t->values, struct value);
      if (!value || !(value->elements = rzalloc_array(t->values, struct value *, leaves)))
         return fail(t, STELA_AIR_ALLOCATION, "aggregate PHI wrapper allocation failed");
      value->aggregate_count = leaves;
      for (unsigned i = 0; i < leaves; i++) {
         value->elements[i] = cfg_phi_leaf(t, LLVMStructGetTypeAtIndex(type, i));
         if (!value->elements[i])
            return false;
      }
   } else {
      value = cfg_phi_leaf(t, type);
      if (!value)
         return false;
   }
   if (pointer_space(type)) {
      struct pointer_node *node = pointer_node(t, inst);
      if (node && node->choices) {
         nir_phi_instr *phi = nir_phi_instr_create(t->b.shader);
         if (!phi)
            return fail(t, STELA_AIR_ALLOCATION, "AIR pointer choice PHI allocation failed");
         nir_def_init(&phi->instr, &phi->def, 1, 32);
         nir_builder_instr_insert(&t->b, &phi->instr);
         choose(t, value, node->choices, &phi->def);
         return _mesa_hash_table_insert(t->values, inst, value) != NULL ||
            fail(t, STELA_AIR_ALLOCATION, "AIR PHI value map allocation failed");
      }
      if (node && LLVMIsALoadInst(node->root))
         return fail(t, STELA_AIR_MEMORY, "pointer PHI of an address loaded from an argument buffer");
      struct value authority;
      struct value *root = !node ? NULL :
                           node->root && node->root == t->imageblock_root ? imageblock_authority(t, &authority) :
                                                              pointer_value(t, node->root);
      if (!root)
         return fail(t, STELA_AIR_MEMORY, "pointer PHI lacks completed provenance proof");
      pointer_authority(value, root);
   }
   return _mesa_hash_table_insert(t->values, inst, value) != NULL ||
      fail(t, STELA_AIR_ALLOCATION, "AIR PHI value map allocation failed");
}

/* A workgroup execution barrier ordering threadgroup memory. */
static void
workgroup_barrier(struct translator *t)
{
   nir_barrier(&t->b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
}

/* Where every thread of an object or mesh function leaves it: an object
 * function launches the grid its threads set, carrying its payload; a mesh
 * function outputs the primitives its threads set, up to its capacity. */
static void
mesh_epilogue(struct translator *t)
{
   nir_builder *b = &t->b;
   b->cursor = nir_after_block(t->exit);
   workgroup_barrier(t);
   nir_deref_instr *state = nir_build_deref_var(b, t->mesh_state);
   if (t->stage_options->stage == STELA_AIR_OBJECT) {
      nir_def *grid[3];
      for (unsigned i = 0; i < 3; i++)
         grid[i] = nir_load_deref(b, nir_build_deref_array_imm(b, state, i));
      unsigned payload = t->payload ? glsl_get_length(t->payload->type) : 0;
      nir_launch_mesh_workgroups(b, nir_vec(b, grid, 3), .base = 0, .range = payload);
   } else {
      const struct stela_air_stage_entry *entry = &t->render.entry;
      nir_def *count = nir_umin(b, nir_load_deref(b, nir_build_deref_array_imm(b, state, 0)),
                                nir_imm_int(b, entry->max_primitives));
      nir_set_vertex_and_primitive_count(b, nir_imm_int(b, entry->max_vertices), count,
                                         nir_undef(b, 1, 32));
   }
   nir_goto(b, b->impl->end_block);
}

static bool
cfg_declare(struct translator *t)
{
   nir_block *entry = nir_start_block(t->b.impl);
   if (!binding_prelude(t))
      return false;
   t->b.impl->structured = false;
   t->b.cursor = nir_after_block(entry);
   t->exit = t->b.impl->end_block;
   /* The device value belongs to the straight-line prelude. */
   nir_def *shape = grid_bounded(t) || t->options->dynamic_shape ? running_shape(t) : NULL;
   if ((grid_bounded(t) || t->options->dynamic_shape) && !shape)
      return false;
   /* A tile's padding threads hold no position of its shape. */
   nir_def *position = shape && t->options->tile ? local_position(t) : NULL;
   if (shape && t->options->tile && !position)
      return false;
   if (t->mesh_state) {
      /* Every thread clears the state before any sets it. */
      nir_deref_instr *state = nir_build_deref_var(&t->b, t->mesh_state);
      for (unsigned i = 0; i < glsl_get_length(t->mesh_state->type); i++)
         nir_store_deref(&t->b, nir_build_deref_array_imm(&t->b, state, i), nir_imm_int(&t->b, 0), 1);
      workgroup_barrier(t);
      if (!(t->exit = new_block(t)))
         return false;
   }
   /* A tile function's threadgroup memory holds what its pass's earlier
    * dispatches left in the tile's. */
   if (t->tile && t->dynamic_threadgroup) {
      if (!tile_memory_copy(t, true) || !(t->exit = new_block(t)))
         return false;
      t->tile_memory = true;
      entry = nir_cursor_current_block(t->b.cursor);
   }
   /* Invocations past the threads a threadgroup runs, at a dispatchThreads
    * grid's edge or past a dynamic kernel's shape, skip the function. */
   nir_def *runs = NULL;
   if (position) {
      runs = nir_ball(&t->b, nir_ult(&t->b, nir_trim_vector(&t->b, position, 2),
                                     nir_trim_vector(&t->b, shape, 2)));
   } else if (shape) {
      runs = nir_ult(&t->b, nir_load_local_invocation_index(&t->b), volume(&t->b, shape));
   }
   if (runs) {
      nir_block *body = new_block(t);
      if (!body)
         return false;
      nir_goto_if(&t->b, body, runs, t->exit);
      entry = body;
   }
   t->work[0]->start = entry;
   for (unsigned i = 1; i < t->reachable_count; i++) {
      t->work[i]->start = new_block(t);
      if (!t->work[i]->start)
         return false;
   }
   for (unsigned i = 0; i < t->reachable_count; i++) {
      struct air_block *block = t->work[i];
      t->b.cursor = nir_before_block(block->start);
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block->llvm);
           inst && LLVMGetInstructionOpcode(inst) == LLVMPHI; inst = LLVMGetNextInstruction(inst)) {
         if (!cfg_declare_phi(t, inst))
            return false;
      }
   }
   return true;
}

/* The innermost loop holding a block, or NULL. */
static nir_loop *
innermost_loop(nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(node);
   }
   return NULL;
}

/* Whether a block lies inside a loop that does not also hold `within`. */
static bool
leaves_loop(nir_block *block, nir_block *within)
{
   nir_loop *loop = innermost_loop(block);
   if (!loop)
      return false;
   for (nir_cf_node *node = within->cf_node.parent; node; node = node->parent) {
      if (node == &loop->cf_node)
         return false;
   }
   return true;
}

/* What hoisting may run for more invocations than the branch it is in. */
struct hoisting {
   /* Each instruction moved, and the highest floor it moved toward. */
   struct hash_table *floors;
   /* The function writes memory or orders its accesses, so a read may see
    * a different value where it runs earlier. */
   bool writes;
};

/* Whether an intrinsic writes memory another read could see, or orders the
 * function's accesses. */
static bool
writes_memory(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
      return !nir_deref_mode_is_one_of(nir_src_as_deref(intrinsic->src[0]),
                                       nir_var_function_temp | nir_var_shader_temp |
                                          nir_var_shader_out);
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
   case nir_intrinsic_begin_invocation_interlock:
   case nir_intrinsic_end_invocation_interlock:
   case nir_intrinsic_barrier:
      return true;
   default:
      return false;
   }
}

/* Whether an instruction may run for more invocations than the branch it is
 * in: it computes from its sources alone, cannot fault, and reads memory
 * nothing changes while the function runs. Texture reads and queries cannot
 * fault. A buffer read can, so it moves only where translation proved it
 * reads mapped memory in every invocation (ACCESS_CAN_SPECULATE: a field of
 * a bound buffer at a fixed offset within its declared type, or of the
 * argument tables). Being free to reorder (a read-only buffer) is no such
 * proof: its index may be one the branch rejects. A read differentiated or
 * sampled with implicit derivatives at the hoisted point takes them from
 * the whole quad, as it would outside the branch. */
static bool
speculatable(nir_instr *instr, const struct hoisting *h)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_deref:
      return true;
   case nir_instr_type_tex:
      switch (nir_instr_as_tex(instr)->op) {
      case nir_texop_txs:
      case nir_texop_query_levels:
      case nir_texop_texture_samples:
      case nir_texop_lod:
         return true;
      case nir_texop_tex:
      case nir_texop_txb:
      case nir_texop_txl:
      case nir_texop_txd:
      case nir_texop_txf:
      case nir_texop_txf_ms:
      case nir_texop_tg4:
         return !h->writes;
      default:
         return false;
      }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
      switch (intrinsic->intrinsic) {
      case nir_intrinsic_load_input:
      case nir_intrinsic_load_interpolated_input:
      case nir_intrinsic_load_barycentric_pixel:
      case nir_intrinsic_load_barycentric_centroid:
      case nir_intrinsic_load_barycentric_sample:
      case nir_intrinsic_load_barycentric_at_offset:
      case nir_intrinsic_load_barycentric_at_sample:
      case nir_intrinsic_load_frag_coord:
      case nir_intrinsic_load_pixel_coord:
      case nir_intrinsic_load_push_constant:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_deref_samples:
      case nir_intrinsic_image_deref_levels:
         return true;
      case nir_intrinsic_load_global:
      case nir_intrinsic_load_global_constant: {
         enum gl_access_qualifier access = nir_intrinsic_access(intrinsic);
         /* A read nothing writes (CAN_REORDER) sees the same value earlier
          * even when the function writes other memory. */
         return (access & ACCESS_CAN_SPECULATE) && ((access & ACCESS_CAN_REORDER) || !h->writes);
      }
      case nir_intrinsic_image_deref_load:
         return !h->writes;
      default:
         return false;
      }
   }
   default:
      return false;
   }
}

static bool
hoist_source(nir_src *src, void *data)
{
   nir_instr *parent = nir_def_instr(src->ssa);
   nir_block **early = data;
   if (nir_block_dominates(*early, parent->block))
      *early = parent->block;
   return true;
}

static nir_block *hoist_speculatable(nir_instr *instr, nir_block *floor, const struct hoisting *h);

struct hoisting_operand {
   nir_block *floor;
   const struct hoisting *h;
};

static bool
hoist_operand(nir_src *src, void *data)
{
   const struct hoisting_operand *operand = data;
   hoist_speculatable(nir_def_instr(src->ssa), operand->floor, operand->h);
   return true;
}

/* Moves a speculatable instruction and the speculatable ones it computes
 * from up to `floor`, or to the first block below it where all their
 * sources exist. Floor dominates the use the move is for, so a value a use
 * needs no higher stays where it is. Returns its block. */
static nir_block *
hoist_speculatable(nir_instr *instr, nir_block *floor, const struct hoisting *h)
{
   if (nir_block_dominates(instr->block, floor) || instr->type == nir_instr_type_phi ||
       !speculatable(instr, h))
      return instr->block;
   struct hash_entry *entry = _mesa_hash_table_search(h->floors, instr);
   if (entry && nir_block_dominates(entry->data, floor))
      return instr->block;
   if (entry)
      entry->data = floor;
   else
      _mesa_hash_table_insert(h->floors, instr, floor);
   nir_foreach_src(instr, hoist_operand, &(struct hoisting_operand){floor, h});
   nir_block *early = floor;
   nir_foreach_src(instr, hoist_source, &early);
   if (early != instr->block && !leaves_loop(early, instr->block))
      nir_instr_move(nir_after_block_before_jump(early), instr);
   return instr->block;
}

/* Whether some invocations of a subgroup may take a branch, or an iteration
 * of a loop, that others leave. Divergence analysis proves the rest uniform:
 * all of a quad's invocations that reach them take them together. */
static bool
diverges(nir_cf_node *node)
{
   if (node->type == nir_cf_node_if)
      return nir_src_is_divergent(&nir_cf_node_as_if(node)->condition);
   return node->type == nir_cf_node_loop && nir_loop_is_divergent(nir_cf_node_as_loop(node));
}

/* The block before the outermost divergent branch or loop holding block:
 * the latest point where each invocation of a quad that reaches block is
 * still active. NULL when no divergent branch holds block. */
static nir_block *
converged_before(nir_block *block)
{
   nir_cf_node *outer = NULL;
   for (nir_cf_node *node = block->cf_node.parent; node && node->type != nir_cf_node_function;
        node = node->parent) {
      if (diverges(node))
         outer = node;
   }
   return outer ? nir_cf_node_as_block(nir_cf_node_prev(outer)) : NULL;
}

/* Whether control flow node holds block. */
static bool
holds(const nir_cf_node *node, const nir_block *block)
{
   for (const nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n == node)
         return true;
   }
   return false;
}

/* Whether a divergent branch or loop holds `use` but not `def`, so that an
 * invocation of a quad may reach def and not use. */
static bool
branches_between(const nir_block *def, const nir_block *use)
{
   for (nir_cf_node *node = use->cf_node.parent; node && node->type != nir_cf_node_function;
        node = node->parent) {
      if (holds(node, def))
         return false;
      if (diverges(node))
         return true;
   }
   return false;
}

/* Where a value's derivative is taken: at the end of floor when the value
 * exists there, which keeps the derivative's live range short, or else right
 * after the value (after the phis of the block a phi heads). */
static nir_cursor
derivative_point(nir_instr *instr, nir_block *floor)
{
   if (nir_block_dominates(instr->block, floor))
      return nir_after_block_before_jump(floor);
   return instr->type == nir_instr_type_phi ? nir_after_phis(instr->block) : nir_after_instr(instr);
}

/* A texture instruction with implicit derivatives in a branch its
 * coordinate is computed outside of (or can be): a level-of-detail query
 * moves to where its sources exist, and a sample takes explicit gradients,
 * the derivatives of its coordinate taken there. A bias scales both
 * gradients by 2^bias, which adds it to the level of detail. */
static bool
hoist_implicit_derivatives(nir_builder *b, nir_tex_instr *tex, const struct hoisting *h)
{
   int index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (index < 0)
      return false;
   nir_block *block = tex->instr.block;
   /* Outside any divergent branch, every invocation that reached the
    * coordinate reaches the instruction. */
   nir_block *floor = converged_before(block);
   if (!floor)
      return false;
   nir_instr *operand = nir_def_instr(tex->src[index].src.ssa);
   nir_block *at = hoist_speculatable(operand, floor, h);
   if (!branches_between(at, block) || leaves_loop(at, block))
      return false;
   if (tex->op == nir_texop_lod)
      return hoist_speculatable(&tex->instr, floor, h) != block;
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;
   b->cursor = derivative_point(operand, floor);
   nir_def *coord = nir_trim_vector(b, tex->src[index].src.ssa, tex->coord_components - tex->is_array);
   nir_def *ddx = nir_ddx(b, coord), *ddy = nir_ddy(b, coord);
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias) {
      nir_def *scale = nir_fexp2(b, nir_f2fN(b, bias, ddx->bit_size));
      ddx = nir_fmul(b, ddx, scale);
      ddy = nir_fmul(b, ddy, scale);
   }
   tex->op = nir_texop_txd;
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, ddx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, ddy);
   return true;
}

/* Apple GPUs take a derivative, and a texture instruction's implicit level
 * of detail, from the values each pixel of the quad holds, even where control
 * flow left some of them out of the branch that takes it: a shader may stop
 * shading pixels outside a shape and differentiate, or sample at, a value
 * the others go on to compute. NVIDIA's quad shuffles read nothing from
 * inactive lanes. So a derivative inside a divergent branch or loop has its
 * speculatable computation moved up to just before the outermost one, or to
 * where its sources exist below that, and is taken there, where every lane
 * of the quad that reached the branch is active; a sample in such a branch
 * takes explicit gradients taken there, and a level-of-detail query moves
 * there. A branch divergence analysis proves uniform keeps a quad's lanes
 * together, so nothing leaves it: moving values further up only lengthens
 * their live ranges (Core Animation's generic shader spilled 1473 times).
 * Nothing moves into a loop, so a value a loop leaves keeps its last
 * iteration's derivative. A value computed from a read that may fault, such
 * as a buffer element at a per-pixel index, stays where it is, and so does
 * its derivative. */
static bool
hoist_derivatives(nir_shader *shader)
{
   bool progress = false;
   nir_builder b;
   /* The structurizer routes paths through phis of constants and undefined
    * values, which divergence analysis takes as divergent after any
    * divergent branch. Folding them lets it see the uniform branches. */
   while (nir_opt_remove_phis(shader))
      progress = true;
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_dominance | nir_metadata_divergence);
      struct hoisting h = {.floors = _mesa_pointer_hash_table_create(NULL)};
      if (!h.floors)
         return progress;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                writes_memory(nir_instr_as_intrinsic(instr)))
               h.writes = true;
         }
      }
      b = nir_builder_create(impl);
      bool moved = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (nir_tex_instr_has_implicit_derivative(tex) && tex->op != nir_texop_tg4)
                  moved |= hoist_implicit_derivatives(&b, tex, &h);
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_ddx: case nir_intrinsic_ddx_fine: case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy: case nir_intrinsic_ddy_fine: case nir_intrinsic_ddy_coarse:
               break;
            default:
               continue;
            }
            nir_block *floor = converged_before(block);
            if (!floor)
               continue;
            nir_instr *operand = nir_def_instr(nir_instr_as_intrinsic(instr)->src[0].ssa);
            nir_block *at = hoist_speculatable(operand, floor, &h);
            if (!branches_between(at, block) || leaves_loop(at, block))
               continue;
            nir_instr_move(derivative_point(operand, floor), instr);
            moved = true;
         }
      }
      moved |= h.floors->entries != 0;
      _mesa_hash_table_destroy(h.floors, NULL);
      /* Moves keep the control flow; the divergence found is not kept up
       * to date by the passes after this one. */
      nir_progress(true, impl, nir_metadata_control_flow);
      progress |= moved;
   }
   return progress;
}

static bool
cfg_emit(struct translator *t)
{
   if (!pointer_provenance(t) || !cfg_declare(t))
      return false;
   for (unsigned i = 0; i < t->reachable_count; i++) {
      struct air_block *block = t->work[i];
      t->b.cursor = nir_after_block(block->start);
      LLVMValueRef term = LLVMGetBasicBlockTerminator(block->llvm);
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block->llvm); inst != term;
           inst = LLVMGetNextInstruction(inst)) {
         if (LLVMGetInstructionOpcode(inst) != LLVMPHI && !instruction(t, inst))
            return false;
         t->error->instruction++;
      }
      LLVMOpcode op = LLVMGetInstructionOpcode(term);
      if (op == LLVMRet) {
         if (!instruction(t, term))
            return false;
         nir_goto(&t->b, t->exit);
      } else if (op == LLVMUnreachable) {
         /* Reaching it is undefined, so the invocation may as well end. */
         nir_goto(&t->b, t->exit);
      } else {
         struct air_block *yes = cfg_block(t, LLVMGetSuccessor(term, 0));
         if (op == LLVMUncondBr || !edge_taken(term, 1)) {
            nir_goto(&t->b, yes->start);
         } else if (!edge_taken(term, 0)) {
            nir_goto(&t->b, cfg_block(t, LLVMGetSuccessor(term, 1))->start);
         } else {
            nir_def *condition = get_value(t, LLVMGetCondition(term));
            if (!condition || condition->num_components != 1 || condition->bit_size != 1)
               return fail(t, STELA_AIR_TYPE, "branch requires scalar i1");
            struct air_block *no = cfg_block(t, LLVMGetSuccessor(term, 1));
            if (yes == no)
               nir_goto(&t->b, yes->start);
            else
               nir_goto_if(&t->b, yes->start, condition, no->start);
         }
      }
      /* SSBO/shared bounds checks may have split the LLVM block. Its actual
       * final jump, not its original NIR entry, supplies every outgoing PHI. */
      block->end = nir_cursor_current_block(t->b.cursor);
      t->error->instruction++;
   }

   for (unsigned i = 0; i < t->reachable_count; i++) {
      struct air_block *block = t->work[i];
      for (LLVMValueRef inst = LLVMGetFirstInstruction(block->llvm);
           inst && LLVMGetInstructionOpcode(inst) == LLVMPHI; inst = LLVMGetNextInstruction(inst)) {
         struct value *target = find_value(t, inst);
         unsigned sources = 0;
         for (unsigned s = 0; s < LLVMCountIncoming(inst); s++) {
            struct air_block *pred = cfg_block(t, LLVMGetIncomingBlock(inst, s));
            if (!pred)
               return fail(t, STELA_AIR_CONTROL_FLOW, "PHI predecessor leaves AIR entry");
            /* No NIR edge exists from an unreachable LLVM block, or along a
             * branch its constant condition never takes. */
            if (!pred->reachable || !edge_to(pred->llvm, block->llvm))
               continue;
            LLVMValueRef incoming = LLVMGetIncomingValue(inst, s);
            if (pred->incoming_phi == inst) {
               if (pred->incoming_value != incoming)
                  return fail(t, STELA_AIR_VERIFY, "different PHI values on duplicate predecessor edges");
               continue;
            }
            pred->incoming_phi = inst;
            pred->incoming_value = incoming;
            /* Constants are materialized before the predecessor's jump. Uses
             * of loop-carried PHIs retain their original simultaneous defs. */
            t->b.cursor = nir_before_instr(nir_block_last_instr(pred->end));
            if (target->present) {
               nir_def *present = texture_choice_present(t, target, incoming);
               if (!present)
                  return false;
               nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(target->present));
               nir_phi_src *source = nir_phi_instr_add_src(phi, pred->end, present);
               list_addtail(&source->src.use_link, &present->uses);
               sources++;
               continue;
            }
            if (target->choices) {
               nir_def *index = choice_index(t, target->choices, incoming);
               if (!index)
                  return false;
               nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(target->choice));
               nir_phi_src *source = nir_phi_instr_add_src(phi, pred->end, index);
               list_addtail(&source->src.use_link, &index->uses);
            }
            if (target->pointer_root && !LLVMIsUndef(incoming) &&
                !joined_pointer(target, incoming, find_value(t, incoming)))
               return fail(t, STELA_AIR_MEMORY, "pointer PHI incoming authority differs from proof");
            struct value *aggregate = target->aggregate_count ? aggregate_value(t, incoming) : NULL;
            if (target->aggregate_count &&
                (!aggregate || aggregate->aggregate_count != target->aggregate_count))
               return fail(t, STELA_AIR_TYPE, "aggregate PHI incoming shape mismatch");
            unsigned leaves = target->aggregate_count ? target->aggregate_count : 1;
            for (unsigned leaf = 0; leaf < leaves; leaf++) {
               nir_def *def = target->aggregate_count ? target->elements[leaf]->def : target->def;
               /* An undefined incoming value may be any value, as NIR's undef is. */
               const struct value *element = aggregate ? aggregate->elements[leaf] : NULL;
               nir_def *value = element ? whole_value(t, element) :
                  aggregate || LLVMIsUndef(incoming) ?
                  nir_undef(&t->b, def->num_components, def->bit_size) : get_value(t, incoming);
               if (!value)
                  return false;
               if (value->num_components != def->num_components || value->bit_size != def->bit_size)
                  return fail(t, STELA_AIR_TYPE, "PHI incoming leaf shape mismatch");
               nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(def));
               /* Leaves are already inserted; attach each late use as well. */
               nir_phi_src *source = nir_phi_instr_add_src(phi, pred->end, value);
               list_addtail(&source->src.use_link, &value->uses);
            }
            sources++;
         }
         if (sources != nir_block_num_preds(block->start))
            return fail(t, STELA_AIR_CONTROL_FLOW, "PHI sources differ from emitted predecessor edges");
      }
   }
   if (t->mesh_state)
      mesh_epilogue(t);
   if (t->tile_memory) {
      t->b.cursor = nir_after_block(t->exit);
      if (!tile_memory_copy(t, false))
         return false;
      nir_goto(&t->b, t->b.impl->end_block);
   }
   nir_sort_unstructured_blocks(t->b.impl);
   nir_validate_shader(t->b.shader, "AIR flat CFG before Mesa structurization");
   nir_lower_goto_ifs(t->b.shader);
   /* The structurizer leaves a return where a loop exits the function, which
    * the backends' passes do not take, as SPIR-V's lowering leaves none. */
   nir_lower_returns(t->b.shader);
   nir_lower_vars_to_ssa(t->b.shader);
   /* Its flag's derefs outlive the flag's loads and stores. */
   nir_remove_dead_derefs(t->b.shader);
   hoist_derivatives(t->b.shader);
   nir_validate_shader(t->b.shader, "AIR structured CFG after Mesa lowering");
   return true;
}

/* The thread variables nothing accesses, which take no thread memory:
 * not those of any deref, nor storage attachments', which the fragment
 * tests read after the layout. */
struct unused_thread_variables {
   const struct translator *t;
   struct set *referenced;
};

static bool
unused_thread_variable(nir_variable *var, void *data)
{
   const struct unused_thread_variables *unused = data;
   for (unsigned i = 0; i < AIR_MAX_RENDER_TARGETS; i++) {
      if (unused->t->attachment_outputs[i] == var)
         return false;
   }
   return var != unused->t->attachment_source1 && !_mesa_set_search(unused->referenced, var);
}

/* Removes the thread variables nothing accesses: the structurizer's path
 * variables, now SSA values. One only written stays, as written. */
static void
remove_unused_thread_variables(struct translator *t)
{
   struct unused_thread_variables unused = {t, _mesa_pointer_set_create(NULL)};
   /* Without the set every variable stays, as without the removal. */
   if (!unused.referenced)
      return;
   nir_foreach_function_impl(impl, t->b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref &&
                nir_instr_as_deref(instr)->deref_type == nir_deref_type_var)
               _mesa_set_add(unused.referenced, nir_instr_as_deref(instr)->var);
         }
      }
   }
   const nir_remove_dead_variables_options options = {.can_remove_var = unused_thread_variable,
                                                      .can_remove_var_data = &unused};
   nir_remove_dead_variables(t->b.shader, nir_var_function_temp, &options);
   _mesa_set_destroy(unused.referenced, NULL);
}

/* A backend with no fused operation of a size takes the weak one, which
 * fuses where its hardware can. */
static bool
weaken_fma(nir_builder *b, nir_alu_instr *alu, void *data)
{
   const nir_shader_compiler_options *options = b->shader->options;
   nir_float_muladd_support support = alu->def.bit_size == 16 ? options->float_mul_add16 :
                                      alu->def.bit_size == 32 ? options->float_mul_add32 :
                                                                options->float_mul_add64;
   if (alu->op != nir_op_ffma || (support & nir_float_muladd_support_has_ffma))
      return false;
   alu->op = nir_op_ffma_weak;
   return true;
}

static nir_shader *
translate(const void *bitcode, size_t size, const char *entry,
                 const nir_shader_compiler_options *nir_options,
                 const struct stela_air_options *options,
                 struct stela_air_error *error, const struct stela_air_stage_options *stage,
                 bool *folded_early, enum stela_air_backend_profile profile)
{
   struct stela_air_error local_error = {0};
   if (!error)
      error = &local_error;
   *error = (struct stela_air_error){0};
   struct translator t = {.options = options, .error = error, .stage_options = stage};
   bool compute = compute_like(&t);
   const bool software_half = profile != STELA_AIR_BACKEND_NATIVE;
   if (!bitcode || size < 4 || size > AIR_MAX_BITCODE_BYTES ||
       memcmp(bitcode, "BC\xc0\xde", 4) || !entry || !*entry || !nir_options ||
       !options || !options->max_instructions || (options->constant_count && !options->constants) ||
       (options->constexpr_sampler_count && !options->constexpr_samplers) ||
       (compute && (!options->workgroup_size[0] ||
       !options->workgroup_size[1] || !options->workgroup_size[2]))) {
      fail(&t, STELA_AIR_INPUT, "invalid AIR bitcode/options");
      return NULL;
   }
   if (software_half && (options->tile ||
       (stage && stage->stage != STELA_AIR_VERTEX && stage->stage != STELA_AIR_FRAGMENT))) {
      fail(&t, STELA_AIR_TARGET, "software binary16 supports kernels and ordinary vertex/fragment stages");
      return NULL;
   }
   if (software_half && stage &&
       (stage->amplified || stage->self_tests || stage->attachments || stage->samples > 1 ||
        stage->extended_range || stage->vertex_input_words)) {
      fail(&t, STELA_AIR_TARGET, "software vertex/fragment specialization requires ordinary single-sample render IO");
      return NULL;
   }
   if (stage && (stage->stage < STELA_AIR_VERTEX || stage->stage > STELA_AIR_MESH ||
       (!compute && (options->workgroup_size[0] || options->workgroup_size[1] ||
                     options->workgroup_size[2] || options->max_threadgroup_bytes ||
                     options->bind_threadgroup || stage->launched_by_grid)) ||
       (stage->amplified && stage->stage != STELA_AIR_VERTEX) ||
       stage->max_varying_slots > VARYING_SLOT_MAX - VARYING_SLOT_VAR0)) {
      fail(&t, STELA_AIR_INPUT, "stage options must suit the function's stage");
      return NULL;
   }
   for (unsigned i = 0; i < 3; i++) {
      if (options->workgroup_size[i] > UINT16_MAX) {
         fail(&t, STELA_AIR_INPUT, "workgroup dimension exceeds NIR representation");
         return NULL;
      }
   }
   if ((options->bind_buffer && options->bind_ssbo) ||
       (options->bind_ssbo && (!options->ssbo_count || options->ssbo_count > UINT8_MAX)) ||
       (!options->bind_ssbo && options->ssbo_count)) {
      fail(&t, STELA_AIR_INPUT, "invalid/mixed global and SSBO binding ABI");
      return NULL;
   }
   if (options->bind_threadgroup && !options->max_threadgroup_bytes) {
      fail(&t, STELA_AIR_INPUT, "threadgroup callback requires an explicit shared-byte limit");
      return NULL;
   }
   if ((options->dynamic_shape && (stage || options->workgroup_size[1] != 1 ||
                                   options->workgroup_size[2] != 1 || !options->load_device_value)) ||
       (options->dynamic_threadgroup_bytes &&
        (!options->dynamic_shape || !options->threadgroup_offset ||
         options->dynamic_threadgroup_bytes % 16))) {
      fail(&t, STELA_AIR_INPUT, "dynamic threadgroups are a linear kernel's, with their callbacks");
      return NULL;
   }
   if (options->tile && (stage || !options->dynamic_shape ||
                         !util_is_power_of_two_nonzero(options->tile->samples) ||
                         options->tile->samples > 8)) {
      fail(&t, STELA_AIR_INPUT, "a tile function is a dynamic kernel of a pass of 1 to 8 samples");
      return NULL;
   }
   if (options->fp_profile != STELA_AIR_FP_IEEE_DEFAULT &&
       options->fp_profile != STELA_AIR_FP_METAL) {
      fail(&t, STELA_AIR_INPUT, "unknown AIR floating-point profile");
      return NULL;
   }
   unsigned major, minor, patch;
   LLVMGetVersion(&major, &minor, &patch);
   if (major != 23) {
      fail(&t, STELA_AIR_LLVM_VERSION, "requires LLVM 23 C ABI, found %u.%u.%u", major, minor, patch);
      return NULL;
   }
   t.context = LLVMContextCreate();
   LLVMContextSetDiagnosticHandler(t.context, diagnostic, &t);
   LLVMMemoryBufferRef buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(bitcode, size, entry);
   LLVMBool parse_failed = LLVMParseBitcodeInContext2(t.context, buffer, &t.module);
   LLVMDisposeMemoryBuffer(buffer);
   nir_shader *result = NULL;
   if (parse_failed || error->kind != STELA_AIR_OK) {
      fail(&t, STELA_AIR_BITCODE, "LLVM refused AIR bitcode");
      goto done;
   }
   char *verification = NULL;
   if (LLVMVerifyModule(t.module, LLVMReturnStatusAction, &verification)) {
      fail(&t, STELA_AIR_VERIFY, "%s", verification ? verification : "invalid LLVM module");
      LLVMDisposeMessage(verification);
      goto done;
   }
   LLVMDisposeMessage(verification);
   const char *target = LLVMGetTarget(t.module);
   size_t asm_size;
   LLVMGetModuleInlineAsm(t.module, &asm_size);
   if (strncmp(target, "air64", 5) || asm_size ||
       LLVMGetFirstGlobalAlias(t.module) ||
       LLVMGetFirstGlobalIFunc(t.module)) {
      fail(&t, STELA_AIR_TARGET, "unsupported target/globals/inline assembly");
      goto done;
   }
   t.metadata = (struct stela_air_metadata){.module = t.module, .error = error, .render = stage != NULL};
   if (!stela_air_metadata_layout(&t.metadata) || !stela_air_metadata_validate(&t.metadata) ||
       !stela_air_specialize_constants(&t.metadata, options->constants, options->constant_count) ||
       !stela_air_construct_constants(&t.metadata) ||
       !stela_air_link_modules(&t.metadata, options->constants, options->constant_count,
                               options->modules, options->module_count))
      goto done;
   LLVMValueRef specialized = NULL;
   /* Calls through tables may depend on inlining, and logging admission uses
    * the use lists of its globals. Keep those modules on the original path. */
   if (folded_early && stage && stage->stage == STELA_AIR_FRAGMENT && options->constant_count &&
       !options->module_count && !options->linked_function_count &&
       stela_air_metadata_target(&t.metadata, LLVMGetTarget(t.module)) &&
       LLVMGetNamedMetadataNumOperands(t.module, "air.function_constants") &&
       !LLVMGetNamedMetadataNumOperands(t.module, "air.visible") &&
       !LLVMGetNamedMetadataNumOperands(t.module, "air.visible_function_references") &&
       !LLVMGetNamedFunction(t.module, "air.get_size_visible_function_table") &&
       !LLVMGetNamedFunction(t.module, "air.is_null_visible_function_table") &&
       !LLVMGetNamedFunction(t.module, "air.get_null_visible_function_table") &&
       !LLVMGetNamedFunction(t.module, "air.get_function_pointer_visible_function_table") &&
       !LLVMGetNamedFunction(t.module, "air.os_log"))
      specialized = LLVMGetNamedFunction(t.module, entry);
   /* A table's pointer may pass through functions to the call it makes. */
   if (!stela_air_inline_specialized_calls(&t.metadata, specialized, folded_early) ||
       !stela_air_dispatch_tables(&t.metadata, options->linked_functions,
                                  options->linked_function_count) ||
       /* Walk again only for calls the first pass left or table lowering added. */
       (stela_air_has_defined_calls(t.module) &&
        !stela_air_inline_calls(&t.metadata, LLVMGetNamedFunction(t.module, entry),
                                options->max_call_stack_depth, options->max_instructions)))
      goto done;
   t.layout = t.metadata.layout;
   t.air_minor = t.metadata.air_minor;
   t.language_major = t.metadata.language_major;
   t.language_minor = t.metadata.language_minor;
   t.max_device_buffers = t.metadata.max_device_buffers;
   t.max_threadgroup_buffers = t.metadata.max_threadgroup_buffers;
   t.denorms_disable = t.metadata.denorms_disable;
   t.fast_math = t.metadata.fast_math;
   if (profile == STELA_AIR_BACKEND_SOFTWARE && stage && !t.denorms_disable) {
      fail(&t, STELA_AIR_TARGET, "software vertex/fragment requires FP32 flush-to-zero metadata");
      goto done;
   }
   bool threadgroup_globals;
   if (!stela_air_metadata_globals(&t.metadata, &threadgroup_globals))
      goto done;
   /* Linking sets the triple again. */
   target = LLVMGetTarget(t.module);
   if (!stela_air_metadata_target(&t.metadata, target)) {
      fail(&t, STELA_AIR_TARGET, "AIR version and target triple disagree");
      goto done;
   }
   LLVMValueRef fn = LLVMGetNamedFunction(t.module, entry);
   if (!fn || LLVMIsDeclaration(fn) || LLVMGetFunctionCallConv(fn) != LLVMCCallConv ||
       (!stage && LLVMGetTypeKind(LLVMGetReturnType(LLVMGlobalGetValueType(fn))) != LLVMVoidTypeKind) ||
       LLVMIsFunctionVarArg(LLVMGlobalGetValueType(fn))) {
      fail(&t, STELA_AIR_ARGUMENT, "entry must be a defined nonvariadic void C-convention kernel");
      goto done;
   }
   if (!simdgroup_events(&t, fn) ||
       ((t.metadata.unoptimized || stores_pointer(fn)) && !promote_memory(&t, fn)) ||
       !stela_air_fold_constants(&t.metadata, fn))
      goto done;
   if (stage && (!stela_air_metadata_stage(&t.metadata, fn, stage->stage, &t.render) ||
                 !stela_air_metadata_stage_map(&t.metadata, &t.render, stage->varyings,
                                               stage->varying_count, stage->max_varying_slots) ||
                 !stela_air_fold_absent(&t.metadata, fn, t.render.arguments.arguments,
                                        t.render.arguments.count)))
      goto done;
   if (software_half && stage) {
      if (t.render.entry.patch) {
         fail(&t, STELA_AIR_TARGET, "software binary16 does not support tessellation");
         goto done;
      }
      if (t.render.entry.imageblock_bytes) {
         fail(&t, STELA_AIR_TARGET, "software vertex/fragment does not support imageblocks");
         goto done;
      }
      /* AIR can represent narrow stage IO as 32-bit NIR variables followed
       * by conversions. Check the actual AIR interface before it widens. */
      for (unsigned i = 0; i < t.render.entry.input_count; i++) {
         const struct stela_air_stage_io *io = &t.render.inputs[i];
         if (!io->absent && io->used &&
             (io->semantic == STELA_AIR_ATTACHMENT || io->semantic == STELA_AIR_BARYCENTRIC ||
              io->per_primitive)) {
            fail(&t, STELA_AIR_TARGET, "software fragment does not support framebuffer fetch, barycentrics or per-primitive inputs");
            goto done;
         }
         if ((io->semantic == STELA_AIR_VERTEX_INPUT || io->semantic == STELA_AIR_VARYING) &&
             stage_bits(io->primitive) != 32) {
            fail(&t, STELA_AIR_TARGET, "software vertex/fragment requires 32-bit stage inputs");
            goto done;
         }
      }
      for (unsigned i = 0; i < t.render.entry.output_count; i++) {
         const struct stela_air_stage_io *io = &t.render.outputs[i];
         /* The primary color reaches a 32-bit NIR variable. Its exact half
          * to float widening is lowered along with the shader's arithmetic. */
         bool half_color = stage->stage == STELA_AIR_FRAGMENT &&
                           io->semantic == STELA_AIR_RENDER_TARGET &&
                           io->primitive == STELA_AIR_HALF && io->index == 0;
         if (stage_bits(io->primitive) != 32 && !half_color) {
            fail(&t, STELA_AIR_TARGET, "software stage requires 32-bit outputs or a half color target");
            goto done;
         }
      }
   }
   if (!linked_references(&t, fn) || !cfg_prepare(&t, fn))
      goto done;
   t.entry = fn;
   if (!workgroup_within(&t, t.render.entry.max_threads))
      goto done;
   static const mesa_shader_stage stages[] = {
      [STELA_AIR_VERTEX] = MESA_SHADER_VERTEX, [STELA_AIR_FRAGMENT] = MESA_SHADER_FRAGMENT,
      [STELA_AIR_OBJECT] = MESA_SHADER_TASK, [STELA_AIR_MESH] = MESA_SHADER_MESH,
   };
   /* Post-tessellation vertex functions are evaluation shaders. */
   enum stela_air_patch domain = t.render.entry.patch;
   t.b = nir_builder_init_simple_shader(!stage ? MESA_SHADER_COMPUTE : domain ? MESA_SHADER_TESS_EVAL :
                                        stages[stage->stage], nir_options, "%s", entry);
   if (domain)
      t.b.shader->info.tess._primitive_mode = domain == STELA_AIR_PATCH_QUAD ? TESS_PRIMITIVE_QUADS :
                                                                              TESS_PRIMITIVE_TRIANGLES;
   if (stage && stage->stage == STELA_AIR_FRAGMENT)
      t.b.shader->info.fs.early_fragment_tests = t.render.entry.early_fragment_tests;
   if (stage && stage->stage == STELA_AIR_FRAGMENT)
      t.b.shader->info.fs.post_depth_coverage = t.render.entry.post_depth_coverage;
   t.b.fp_math_ctrl = nir_fp_no_fast_math;
   if (options->fp_profile != STELA_AIR_FP_IEEE_DEFAULT && !fp_environment(&t))
      goto done;
   for (unsigned i = 0; i < 3; i++)
      t.b.shader->info.workgroup_size[i] = options->workgroup_size[i];
   t.b.shader->info.num_ssbos = options->ssbo_count;
   t.values = _mesa_pointer_hash_table_create(t.cfg_scratch);
   if (!t.values || !(!stage ? kernel_arguments(&t, fn) :
                      compute ? mesh_stage_arguments(&t) : stage_arguments(&t, fn)) ||
       !stitched_resources(&t, fn)) {
      fail(&t, STELA_AIR_ALLOCATION, "AIR shader/value map allocation failed");
      goto done;
   }
   if (!cfg_emit(&t))
      goto done;
   /* Texture calls' loops and branches use the derefs made before them;
    * each use takes its own copy in its block, as backends take them. */
   nir_rematerialize_derefs_in_use_blocks_impl(t.b.impl);
   if (!software_half)
      nir_shader_alu_pass(t.b.shader, weaken_fma, nir_metadata_all, NULL);
   /* Metal keeps half denormals in every mode. Only half users declare it, so
    * backends without FP16 controls still take FP32-only shaders. */
   if (t.fp_checked && half_arithmetic(t.b.shader))
      t.b.shader->info.float_controls_execution_mode |=
         FLOAT_CONTROLS_DENORM_PRESERVE_FP16 | FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP16;
   if (t.payload) {
      /* Both stages lay the payload out alike: one variable at offset zero. */
      nir_lower_vars_to_explicit_types(t.b.shader, nir_var_mem_task_payload,
                                       glsl_get_natural_size_align_bytes);
      nir_lower_explicit_io(t.b.shader, nir_var_mem_task_payload, nir_address_format_32bit_offset);
      nir_remove_dead_variables(t.b.shader, nir_var_mem_task_payload, NULL);
   }
   /* Each layout drops the variables it placed: a driver lays out what
    * variables remain after what the shader's size already holds, so
    * keeping them would place the same memory twice. */
   if (t.shared_resources || t.mesh_state) {
      nir_lower_vars_to_explicit_types(t.b.shader, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
      if (t.b.shader->info.shared_size > options->max_threadgroup_bytes) {
         fail(&t, STELA_AIR_LIMIT, "aligned shared layout exceeds target/backend byte limit");
         goto done;
      }
      nir_lower_explicit_io(t.b.shader, nir_var_mem_shared, nir_address_format_32bit_offset);
      nir_remove_dead_variables(t.b.shader, nir_var_mem_shared, NULL);
   }
   if (t.private_allocations) {
      remove_unused_thread_variables(&t);
      nir_lower_vars_to_explicit_types(t.b.shader, nir_var_function_temp, glsl_get_natural_size_align_bytes);
      if (t.b.shader->scratch_size > options->max_private_bytes) {
         fail(&t, STELA_AIR_LIMIT, "aligned thread memory layout exceeds the target limit");
         goto done;
      }
      nir_lower_explicit_io(t.b.shader, nir_var_function_temp, nir_address_format_32bit_offset);
      nir_remove_dead_variables(t.b.shader, nir_var_function_temp, NULL);
   }
   if (stage && stage->stage == STELA_AIR_FRAGMENT && stage->self_tests) {
      if (!stage->fragment_tests || !stage->raster_order_groups || !options->bind_depth_stencil) {
         fail(&t, STELA_AIR_TARGET, "fragment tests a function runs itself need the device's fragment shader interlock");
         goto done;
      }
      t.tests_fragments = true;
   }
   if (stage && stage->stage == STELA_AIR_FRAGMENT && t.tests_fragments && !test_fragments(&t))
      goto done;
   if (stage && stage->stage == STELA_AIR_FRAGMENT && !order_fragments(&t))
      goto done;
   if (stage && stage->stage == STELA_AIR_FRAGMENT)
      mark_dirty_tiles(&t);
   if (stage && stage->stage == STELA_AIR_FRAGMENT) {
      shade_per_sample(t.b.shader);
      /* A function running once per sample orders the fragments of each
       * sample, not of each pixel: one invocation accesses one sample's
       * texels, and NVIDIA's pixel conflict detection faults the GPU for
       * sample-rate shading (NVK forces sample detection only for the
       * pipeline's own sample shading). */
      struct shader_info *info = &t.b.shader->info;
      if (info->fs.pixel_interlock_ordered && info->fs.uses_sample_shading) {
         info->fs.pixel_interlock_ordered = false;
         info->fs.sample_interlock_ordered = true;
      }
   }

   if (software_half) {
      const char *reason;
      if (!stela_air_lower_software_half_profile(t.b.shader, profile, &reason)) {
         fail(&t, STELA_AIR_TARGET, "unsupported software binary16: %s", reason);
         goto done;
      }
      nir_shader_alu_pass(t.b.shader, weaken_fma, nir_metadata_all, NULL);
   }
   if (stage || t.shared_resources || t.shared_barriers)
      nir_shader_gather_info(t.b.shader, nir_shader_get_entrypoint(t.b.shader));
   nir_validate_shader(t.b.shader, stage ? "AIR bounded render translation" : "AIR bounded compute translation");
   /* The lowering map is compiler scratch, not shader state. Drop LLVM keys
    * and render descriptor/lane wrappers before their input/stack owners end. */
   ralloc_free(t.values);
   t.values = NULL;
   result = t.b.shader;
   t.b.shader = NULL;
   error->instruction = 0;

done:
   ralloc_free(t.cfg_scratch);
   ralloc_free(t.b.shader);
   if (t.module)
      LLVMDisposeModule(t.module);
   LLVMContextDispose(t.context);
   return result;
}

nir_shader *
stela_air_to_nir(const void *bitcode, size_t size, const char *entry,
                  const nir_shader_compiler_options *nir_options,
                  const struct stela_air_options *options, struct stela_air_error *error)
{
   return translate(bitcode, size, entry, nir_options, options, error, NULL, NULL, STELA_AIR_BACKEND_NATIVE);
}

nir_shader *
stela_air_software_to_nir(const void *bitcode, size_t size, const char *entry,
                         const nir_shader_compiler_options *nir_options,
                         const struct stela_air_options *options,
                         enum stela_air_backend_profile profile, struct stela_air_error *error)
{
   return translate(bitcode, size, entry, nir_options, options, error, NULL, NULL, profile);
}

nir_shader *
stela_air_stage_to_nir(const void *bitcode, size_t size, const char *entry,
                        const nir_shader_compiler_options *nir_options,
                        const struct stela_air_stage_options *options, struct stela_air_error *error)
{
   if (!options) {
      if (error) *error = (struct stela_air_error){.kind = STELA_AIR_INPUT};
      return NULL;
   }
   return translate(bitcode, size, entry, nir_options, &options->resources, error, options, NULL,
                    STELA_AIR_BACKEND_NATIVE);
}

static nir_shader *
metal_stage_to_nir(const void *bitcode, size_t size, const char *entry,
                    const nir_shader_compiler_options *nir_options,
                    const struct stela_air_stage_options *options,
                    struct stela_air_error *error, enum stela_air_backend_profile profile)
{
   if (!options)
      return stela_air_stage_to_nir(bitcode, size, entry, nir_options, options, error);
   struct stela_air_error local_error;
   if (!error)
      error = &local_error;
   bool folded_early = false;
   nir_shader *shader = translate(bitcode, size, entry, nir_options, &options->resources, error,
                                  options, &folded_early, profile);
   /* Only a fragment that attempted the early pass can retry. Its temporary
    * LLVM module and NIR are already gone, and the second call cannot fold
    * early again. Avoid extra work after an allocation refusal. */
   if (!shader && folded_early && error->kind != STELA_AIR_ALLOCATION)
      return translate(bitcode, size, entry, nir_options, &options->resources, error,
                        options, NULL, profile);
   return shader;
}

nir_shader *
stela_air_metal_stage_to_nir(const void *bitcode, size_t size, const char *entry,
                             const nir_shader_compiler_options *nir_options,
                             const struct stela_air_stage_options *options,
                             struct stela_air_error *error)
{
   return metal_stage_to_nir(bitcode, size, entry, nir_options, options, error, STELA_AIR_BACKEND_NATIVE);
}

nir_shader *
stela_air_metal_software_stage_to_nir(const void *bitcode, size_t size, const char *entry,
                                      const nir_shader_compiler_options *nir_options,
                                      const struct stela_air_stage_options *options,
                                      enum stela_air_backend_profile profile,
                                      struct stela_air_error *error)
{
   return metal_stage_to_nir(bitcode, size, entry, nir_options, options, error, profile);
}

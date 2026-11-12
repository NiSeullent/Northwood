/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "air_soft_half.h"
#include "compiler/nir/nir_builder.h"

static nir_def *
unpack_half(nir_builder *b, nir_def *half)
{
   nir_def *bits = nir_u2u32(b, half);
   nir_def *sign = nir_ishl_imm(b, nir_iand_imm(b, bits, 0x8000), 16);
   nir_def *exp = nir_iand_imm(b, nir_ushr_imm(b, bits, 10), 31);
   nir_def *frac = nir_iand_imm(b, bits, 1023);
   nir_def *normal = nir_ior(b, nir_ishl_imm(b, nir_iadd_imm(b, exp, 112), 23),
                             nir_ishl_imm(b, frac, 13));
   nir_def *special = nir_ior_imm(b, nir_ishl_imm(b, frac, 13), 0x7f800000);
   /* Every half denormal is a small integer times 2^-24, exact in FP32. */
   nir_def *denormal = nir_fmul_imm(b, nir_u2f32(b, frac), 0x1p-24f);
   nir_def *magnitude = nir_bcsel(b, nir_ieq_imm(b, exp, 0), denormal,
      nir_bcsel(b, nir_ieq_imm(b, exp, 31), special, normal));
   return nir_ior(b, sign, magnitude);
}

/* Round an FP32 bit pattern to a half held as uint16, ties to even. */
static nir_def *
pack_half(nir_builder *b, nir_def *value)
{
   nir_def *sign = nir_iand_imm(b, nir_ushr_imm(b, value, 16), 0x8000);
   nir_def *mag = nir_iand_imm(b, value, 0x7fffffff);
   nir_def *exp = nir_ushr_imm(b, mag, 23);
   nir_def *frac = nir_iand_imm(b, mag, 0x7fffff);
   nir_def *normal = nir_ushr_imm(b,
      nir_iadd(b, nir_iadd_imm(b, mag, -0x38000000 + 0xfff),
               nir_iand_imm(b, nir_ushr_imm(b, mag, 13), 1)), 13);

   /* Only exponents 102..112 reach this result; clamping makes the
    * discarded arms' shift well-defined too. */
   nir_def *shift = nir_isub(b, nir_imm_int(b, 126),
                            nir_uclamp(b, exp, nir_imm_int(b, 102), nir_imm_int(b, 112)));
   nir_def *significand = nir_ior_imm(b, frac, 0x800000);
   nir_def *tail = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), nir_iadd_imm(b, shift, -1)), -1);
   nir_def *even = nir_iand_imm(b, nir_ushr(b, significand, shift), 1);
   nir_def *denormal = nir_ushr(b, nir_iadd(b, nir_iadd(b, significand, tail), even), shift);
   nir_def *finite = nir_bcsel(b, nir_uge_imm(b, mag, 0x38800000), normal,
      nir_bcsel(b, nir_ult_imm(b, mag, 0x33000000), nir_imm_int(b, 0), denormal));
   nir_def *quiet_nan = nir_ior_imm(b, nir_ushr_imm(b, frac, 13), 0x7e00);
   nir_def *out = nir_bcsel(b, nir_ult_imm(b, mag, 0x47800000), finite,
      nir_bcsel(b, nir_ugt_imm(b, mag, 0x7f800000), quiet_nan, nir_imm_int(b, 0x7c00)));
   return nir_u2u16(b, nir_ior(b, sign, out));
}

/* A finite half product is exact in FP32, with at most 22 significant bits
 * and a minimum nonzero magnitude of 2^-48. TwoSum recovers the exact
 * residual of the addition. An FP32 sum can round across a half boundary
 * only when it lands on that boundary, including underflow and overflow.
 * Move a midpoint one FP32 ULP toward the residual before narrowing. */
static nir_def *
fused_half(nir_builder *b, nir_def *x, nir_def *y, nir_def *z)
{
   nir_def *product = nir_fmul(b, x, y);
   nir_def *sum = nir_fadd(b, product, z);
   nir_def *rounded_z = nir_fsub(b, sum, product);
   nir_def *error = nir_fadd(b,
      nir_fsub(b, product, nir_fsub(b, sum, rounded_z)),
      nir_fsub(b, z, rounded_z));
   nir_def *exp = nir_iand_imm(b, nir_ushr_imm(b, sum, 23), 255);
   nir_def *cut = nir_isub(b, nir_imm_int(b, 126),
                           nir_uclamp(b, exp, nir_imm_int(b, 102), nir_imm_int(b, 113)));
   nir_def *one = nir_imm_int(b, 1);
   nir_def *mask = nir_iadd_imm(b, nir_ishl(b, one, cut), -1);
   nir_def *midpoint = nir_ishl(b, one, nir_iadd_imm(b, cut, -1));
   nir_def *significand = nir_ior_imm(b, nir_iand_imm(b, sum, 0x7fffff), 0x800000);
   nir_def *at_midpoint = nir_iand(b, nir_uge_imm(b, exp, 102), nir_ule_imm(b, exp, 142));
   at_midpoint = nir_iand(b, at_midpoint, nir_ieq(b, nir_iand(b, significand, mask), midpoint));
   nir_def *adjust = nir_iand(b, at_midpoint, nir_fneu_imm(b, error, 0.0f));
   nir_def *same_sign = nir_ieq_imm(b, nir_iand_imm(b, nir_ixor(b, sum, error), 0x80000000), 0);
   nir_def *next = nir_iadd(b, sum, nir_bcsel(b, same_sign, one, nir_imm_int(b, -1)));
   return nir_bcsel(b, adjust, next, sum);
}

static bool
float_alu(const nir_alu_instr *alu, unsigned bits)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   if (alu->def.bit_size == bits && nir_alu_type_get_base_type(info->output_type) == nir_type_float)
      return true;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      if (nir_src_bit_size(alu->src[i].src) == bits &&
          nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
         return true;
   }
   return false;
}

static bool
supported_alu(const nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
   case nir_op_ffma:
      return alu->def.bit_size == 16;
   case nir_op_f2f16:
   case nir_op_f2f16_rtne:
      return nir_src_bit_size(alu->src[0].src) == 32;
   case nir_op_f2f32:
      return nir_src_bit_size(alu->src[0].src) == 16;
   default:
      return false;
   }
}

static bool
preserving_fp32_alu(nir_op op)
{
   switch (op) {
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
   case nir_op_feq:
   case nir_op_fneu:
   case nir_op_fge:
   case nir_op_flt:
      return true;
   default:
      return false;
   }
}

static bool
no_narrow_src(nir_src *src, void *data)
{
   (void)data;
   return nir_src_bit_size(*src) != 16;
}

static bool
narrow_type(const glsl_type *type)
{
   if (glsl_type_is_array(type))
      return narrow_type(glsl_get_array_element(type));
   if (glsl_type_is_struct_or_ifc(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); ++i) {
         if (narrow_type(glsl_get_struct_field(type, i)))
            return true;
      }
   }
   return glsl_type_is_16bit(type);
}

static bool
float_intrinsic(const nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_has_atomic_op(intr) &&
       nir_atomic_op_type(nir_intrinsic_atomic_op(intr)) == nir_type_float)
      return true;
   if (nir_intrinsic_has_reduction_op(intr) &&
       nir_alu_type_get_base_type(nir_op_infos[nir_intrinsic_reduction_op(intr)].output_type) ==
       nir_type_float)
      return true;
   return (nir_intrinsic_has_src_type(intr) &&
           nir_alu_type_get_base_type(nir_intrinsic_src_type(intr)) == nir_type_float) ||
          (nir_intrinsic_has_dest_type(intr) &&
           nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) == nir_type_float);
}

static bool
interpolated_input(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_sample:
   case nir_intrinsic_interp_deref_at_offset:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_barycentric_coord_centroid:
   case nir_intrinsic_load_barycentric_coord_pixel:
   case nir_intrinsic_load_barycentric_coord_sample:
      return true;
   default:
      return false;
   }
}

static const char *
unsupported_half(nir_shader *shader, enum stela_air_backend_profile profile)
{
   mesa_shader_stage stage = shader->info.stage;
   if (stage != MESA_SHADER_COMPUTE && stage != MESA_SHADER_VERTEX && stage != MESA_SHADER_FRAGMENT)
      return "software binary16 supports kernels and ordinary vertex/fragment stages";
   unsigned control = shader->info.float_controls_execution_mode;
   if (control & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16)
      return "round-toward-zero half execution mode";
   if (control & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32)
      return "round-toward-zero FP32 execution mode";
   if (control & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16)
      return "flush-to-zero half execution mode";
   const bool flush_fp32 = control & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   const bool preserve_requested = control & FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   const bool patched_preserve = profile == STELA_AIR_BACKEND_PATCHED_LAVAPIPE && preserve_requested;
   if (patched_preserve && flush_fp32)
      return "conflicting FP32 denorm modes";
   /* Generic draw and raster flush FP32 subnormals. Compute can still
    * transport bits and run half operations with normal FP32 intermediates. */
   if (stage != MESA_SHADER_COMPUTE && !patched_preserve && (!flush_fp32 || preserve_requested))
      return "software vertex/fragment requires FP32 flush-to-zero";
   if (stage == MESA_SHADER_FRAGMENT &&
       (shader->info.fs.pixel_interlock_ordered || shader->info.fs.pixel_interlock_unordered ||
        shader->info.fs.sample_interlock_ordered || shader->info.fs.sample_interlock_unordered))
      return "fragment shader interlock";
   nir_foreach_variable_in_shader(var, shader) {
      if (narrow_type(var->type))
         return "16-bit shader variable ABI";
      if (patched_preserve && var->data.mode == nir_var_image)
         return "lavapipe FP32 preservation does not support storage images";
      if (var->data.mode == nir_var_shader_in || var->data.mode == nir_var_shader_out) {
         const glsl_type *type = glsl_without_array(var->type);
         if (!glsl_type_is_numeric(type) || glsl_get_bit_size(type) != 32)
            return "software vertex/fragment requires 32-bit stage IO";
         if (var->data.per_primitive)
            return "per-primitive stage IO";
         if (patched_preserve && stage == MESA_SHADER_FRAGMENT &&
             var->data.mode == nir_var_shader_in && glsl_get_base_type(type) == GLSL_TYPE_FLOAT)
            return "lavapipe FP32 preservation does not support floating-point fragment inputs";
      }
   }
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            switch (instr->type) {
            case nir_instr_type_alu: {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (float_alu(alu, 16)) {
                  if (!supported_alu(alu))
                     return nir_op_infos[alu->op].name;
               } else if (alu->op == nir_op_ffma) {
                  return "fused FP32 multiply-add";
               } else if (!flush_fp32 && float_alu(alu, 32)) {
                  if (!patched_preserve)
                     return "FP32 arithmetic requires denorm preservation";
                  if (!preserving_fp32_alu(alu->op))
                     return nir_op_infos[alu->op].name;
               }
               break;
            }
            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_begin_invocation_interlock ||
                   intr->intrinsic == nir_intrinsic_end_invocation_interlock)
                  return "fragment shader interlock";
               if (nir_intrinsic_has_cmat_desc(intr) || nir_intrinsic_has_cmat_signed_mask(intr))
                  return "cooperative matrix ABI";
               if ((nir_intrinsic_infos[intr->intrinsic].has_dest && intr->def.bit_size == 16) ||
                   !nir_foreach_src(instr, no_narrow_src, NULL))
                  return "16-bit intrinsic ABI";
               if (patched_preserve && nir_intrinsic_has_image_dim(intr))
                  return "lavapipe FP32 preservation does not support storage images";
               if (patched_preserve && interpolated_input(intr->intrinsic))
                  return "lavapipe FP32 preservation does not support interpolation";
               if (!flush_fp32 && float_intrinsic(intr))
                  return "FP32 intrinsic requires denorm preservation";
               break;
            }
            case nir_instr_type_tex:
               if (nir_instr_as_tex(instr)->def.bit_size == 16 ||
                   !nir_foreach_src(instr, no_narrow_src, NULL))
                  return "16-bit texture ABI";
               if (!flush_fp32)
                  return "texture operations require denorm preservation";
               break;
            case nir_instr_type_deref:
               if (narrow_type(nir_instr_as_deref(instr)->type))
                  return "16-bit memory dereference";
               break;
            case nir_instr_type_call:
               return "uninlined function call";
            default:
               break;
            }
         }
      }
   }
   return NULL;
}

static bool
is_half_alu(const nir_instr *instr, const void *data)
{
   (void)data;
   return instr->type == nir_instr_type_alu && float_alu(nir_instr_as_alu(instr), 16);
}

static nir_def *
lower_half_alu(nir_builder *b, nir_instr *instr, void *data)
{
   (void)data;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   const nir_op_info *info = &nir_op_infos[alu->op];
   assert(supported_alu(alu));
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *srcs[3];
   for (unsigned i = 0; i < info->num_inputs; ++i)
      srcs[i] = nir_ssa_for_alu_src(b, alu, i);

   nir_def *lanes[NIR_MAX_VEC_COMPONENTS];
   for (unsigned lane = 0; lane < alu->def.num_components; ++lane) {
      nir_def *x = nir_channel(b, srcs[0], lane);
      if (alu->op == nir_op_f2f16 || alu->op == nir_op_f2f16_rtne) {
         lanes[lane] = pack_half(b, x);
         continue;
      }
      x = unpack_half(b, x);
      if (alu->op == nir_op_f2f32) {
         lanes[lane] = x;
         continue;
      }
      nir_def *y = unpack_half(b, nir_channel(b, srcs[1], lane));
      nir_def *out;
      switch (alu->op) {
      case nir_op_fadd: out = nir_fadd(b, x, y); break;
      case nir_op_fsub: out = nir_fsub(b, x, y); break;
      case nir_op_fmul: out = nir_fmul(b, x, y); break;
      case nir_op_ffma: out = fused_half(b, x, y, unpack_half(b, nir_channel(b, srcs[2], lane))); break;
      default: UNREACHABLE("checked binary16 operation");
      }
      lanes[lane] = pack_half(b, out);
   }
   return nir_vec(b, lanes, alu->def.num_components);
}

bool
stela_air_lower_software_half_profile(nir_shader *shader, enum stela_air_backend_profile profile,
                                     const char **reason)
{
   *reason = unsupported_half(shader, profile);
   if (*reason)
      return false;
   nir_shader_lower_instructions(shader, is_half_alu, lower_half_alu, NULL);
   shader->info.float_controls_execution_mode &=
      ~(FLOAT_CONTROLS_DENORM_PRESERVE_FP16 | FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP16);
   if (profile != STELA_AIR_BACKEND_PATCHED_LAVAPIPE)
      shader->info.float_controls_execution_mode &= ~FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   return true;
}

bool
stela_air_lower_software_half(nir_shader *shader, const char **reason)
{
   return stela_air_lower_software_half_profile(shader, STELA_AIR_BACKEND_SOFTWARE, reason);
}

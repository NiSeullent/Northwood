/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "../air_soft_half.h"
#include "compiler/nir/nir_builder.h"
#include "util/ralloc.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static const nir_shader_compiler_options options = {0};

static nir_def *
half(nir_builder *b, unsigned bits)
{
   return nir_imm_intN_t(b, bits, 16);
}

static nir_def *
arithmetic(nir_builder *b, nir_op op, unsigned x, unsigned y, unsigned z)
{
   nir_def *values[] = {half(b, x), half(b, y), half(b, z)};
   return nir_build_alu_src_arr(b, op, values);
}

static void
check_result(nir_op op, unsigned x, unsigned y, unsigned z, uint32_t expected)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "software half");
   nir_def *value;
   if (op == nir_op_f2f32)
      value = nir_f2f32(&b, half(&b, x));
   else if (op == nir_op_f2f16_rtne)
      value = nir_f2f16_rtne(&b, nir_imm_int(&b, x));
   else
      value = arithmetic(&b, op, x, y, z);
   nir_intrinsic_instr *store = nir_store_global(&b, nir_u2u32(&b, value),
                                                 nir_imm_int64(&b, 0x1000), .align_mul = 4);
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP16 | FLOAT_CONTROLS_DENORM_PRESERVE_FP16;
   const char *reason = "untouched";
   assert(stela_air_lower_software_half(b.shader, &reason) && !reason);
   assert(!b.shader->info.float_controls_execution_mode);
   nir_validate_shader(b.shader, "software half");
   while (nir_opt_constant_folding(b.shader))
      ;
   assert(nir_src_is_const(store->src[0]));
   uint32_t got = nir_src_as_uint(store->src[0]);
   if (got != expected)
      fprintf(stderr, "%s %08x %04x %04x: got %08x, expected %08x\n",
              nir_op_infos[op].name, x, y, z, got, expected);
   assert(got == expected);
   ralloc_free(b.shader);
}

static void
supported_cases(void)
{
   check_result(nir_op_fadd, 0x3c00, 0x1000, 0, 0x3c00); /* halfway, even */
   check_result(nir_op_fadd, 0x3c01, 0x1000, 0, 0x3c02); /* halfway, odd */
   check_result(nir_op_fadd, 0x8000, 0x8000, 0, 0x8000); /* -0 plus -0 */
   check_result(nir_op_fsub, 0x7bff, 0x7bff, 0, 0);
   check_result(nir_op_fmul, 0x0001, 0x3800, 0, 0);
   check_result(nir_op_fmul, 0x8001, 0x3800, 0, 0x8000);
   check_result(nir_op_fmul, 0x0003, 0x3800, 0, 0x0002);
   check_result(nir_op_ffma, 0x57ff, 0x3c01, 0x0401, 0x5801); /* FP32 sum ties down */
   check_result(nir_op_ffma, 0xd7ff, 0x3c01, 0x8401, 0xd801);
   check_result(nir_op_ffma, 0x7bff, 0x3c00, 0x4c00, 0x7c00); /* 65520 */
   check_result(nir_op_ffma, 0x7bff, 0x3c00, 0x4bff, 0x7bff);
   check_result(nir_op_f2f32, 0x0001, 0, 0, 0x33800000);
   check_result(nir_op_f2f32, 0x8000, 0, 0, 0x80000000);
   check_result(nir_op_f2f16_rtne, 0x33000000, 0, 0, 0); /* 2^-25 */
   check_result(nir_op_f2f16_rtne, 0x33000001, 0, 0, 1);
   check_result(nir_op_f2f16_rtne, 0x7f800000, 0, 0, 0x7c00);
   check_result(nir_op_f2f16_rtne, 0x7fc02000, 0, 0, 0x7e01);
}

static void
vector_and_dynamic_values(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "half swizzle");
   nir_def *x = nir_vec2(&b, half(&b, 0x57ff), half(&b, 0xd7ff));
   nir_def *y = nir_vec2(&b, half(&b, 0x3c01), half(&b, 0x3c01));
   nir_def *z = nir_vec2(&b, half(&b, 0x8401), half(&b, 0x0401));
   nir_def *fma = nir_ffma(&b, x, y, nir_swizzle(&b, z, (unsigned[]){1, 0}, 2));
   nir_intrinsic_instr *store = nir_store_global(&b, nir_u2u32(&b, fma),
                                                 nir_imm_int64(&b, 0x1000), .align_mul = 4);
   const char *reason;
   assert(stela_air_lower_software_half(b.shader, &reason));
   nir_validate_shader(b.shader, "half swizzle");
   while (nir_opt_constant_folding(b.shader))
      ;
   const nir_const_value *values = nir_src_as_const_value(store->src[0]);
   assert(values && values[0].u32 == 0x5801 && values[1].u32 == 0xd801);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "dynamic half");
   nir_def *input = nir_load_global(&b, 2, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
   x = nir_f2f16_rtne(&b, input);
   y = nir_fadd(&b, x, nir_swizzle(&b, x, (unsigned[]){1, 0}, 2));
   nir_store_global(&b, nir_f2f32(&b, y), nir_imm_int64(&b, 0x1010), .align_mul = 4);
   assert(stela_air_lower_software_half(b.shader, &reason));
   nir_validate_shader(b.shader, "dynamic half");
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         const nir_op_info *info = &nir_op_infos[alu->op];
         if (nir_alu_type_get_base_type(info->output_type) == nir_type_float)
            assert(alu->def.bit_size == 32 &&
                   alu->fp_math_ctrl == nir_op_valid_fp_math_ctrl(alu->op, nir_fp_no_fast_math));
         for (unsigned i = 0; i < info->num_inputs; ++i) {
            if (nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
               assert(nir_src_bit_size(alu->src[i].src) == 32);
         }
      }
   }
   ralloc_free(b.shader);
}

static void
render_stage(mesa_shader_stage stage)
{
   nir_builder b = nir_builder_init_simple_shader(stage, &options, "render half arithmetic");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_float_type(), "input");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_float_type(), "output");
   in->data.location = stage == MESA_SHADER_VERTEX ? VERT_ATTRIB_GENERIC0 : VARYING_SLOT_VAR0;
   out->data.location = stage == MESA_SHADER_VERTEX ? VARYING_SLOT_VAR0 : FRAG_RESULT_DATA0;
   nir_def *input = nir_f2f16_rtne(&b, nir_load_var(&b, in));
   nir_def *add = nir_fadd(&b, input, half(&b, 0x3000));
   nir_store_var(&b, out, nir_f2f32(&b, add), 1);

   const char *reason;
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "software vertex/fragment requires FP32 flush-to-zero") == 0);
   assert(nir_def_as_alu(add)->op == nir_op_fadd && add->bit_size == 16);
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   assert(stela_air_lower_software_half(b.shader, &reason) && !reason);
   assert(b.shader->info.float_controls_execution_mode == FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32);
   nir_validate_shader(b.shader, "render half arithmetic");
   ralloc_free(b.shader);
}

static void
refusals_do_not_mutate(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "half arithmetic");
   nir_def *add = nir_fadd(&b, half(&b, 0x3c00), half(&b, 0x3c00));
   nir_fsqrt(&b, add);
   const char *reason;
   assert(!stela_air_lower_software_half(b.shader, &reason) && strcmp(reason, "fsqrt") == 0);
   assert(nir_def_as_alu(add)->op == nir_op_fadd && add->bit_size == 16);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "half memory");
   nir_store_global(&b, half(&b, 0x3c00), nir_imm_int64(&b, 0x1000), .align_mul = 2);
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "16-bit intrinsic ABI") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &options, "half output");
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   nir_variable_create(b.shader, nir_var_shader_out, glsl_float16_t_type(), "output");
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "16-bit shader variable ABI") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "half flush to zero");
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16;
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "flush-to-zero half execution mode") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "FP32 denorm preservation");
   nir_def *input = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_def *product = nir_fmul_imm(&b, input, 2.0f);
   nir_store_global(&b, product, nir_imm_int64(&b, 0x1010), .align_mul = 4);
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "FP32 arithmetic requires denorm preservation") == 0);
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   assert(stela_air_lower_software_half(b.shader, &reason) && !reason);
   assert(b.shader->info.float_controls_execution_mode == FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32 &&
          nir_def_as_alu(product)->op == nir_op_fmul);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "FP32 reduction");
   input = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_reduce(&b, input, .reduction_op = nir_op_fadd);
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "FP32 intrinsic requires denorm preservation") == 0);
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   assert(!stela_air_lower_software_half_profile(b.shader, STELA_AIR_BACKEND_PATCHED_LAVAPIPE, &reason) &&
          strcmp(reason, "FP32 intrinsic requires denorm preservation") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "fused FP32 with flush");
   input = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_ffma(&b, input, input, input);
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "fused FP32 multiply-add") == 0);
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   assert(!stela_air_lower_software_half_profile(b.shader, STELA_AIR_BACKEND_PATCHED_LAVAPIPE, &reason) &&
          strcmp(reason, "fused FP32 multiply-add") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_TESS_EVAL, &options, "unsupported render stage");
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "software binary16 supports kernels and ordinary vertex/fragment stages") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "fragment interlock");
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   b.shader->info.fs.pixel_interlock_ordered = true;
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "fragment shader interlock") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "64-bit input");
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   nir_variable_create(b.shader, nir_var_shader_in, glsl_double_type(), "input");
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "software vertex/fragment requires 32-bit stage IO") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "per-primitive input");
   b.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_float_type(), "input");
   in->data.per_primitive = true;
   assert(!stela_air_lower_software_half(b.shader, &reason) &&
          strcmp(reason, "per-primitive stage IO") == 0);
   ralloc_free(b.shader);
}

static void
lavapipe_preservation_refusals(void)
{
   const enum stela_air_backend_profile profile = STELA_AIR_BACKEND_PATCHED_LAVAPIPE;
   const unsigned preserve = FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   const char *reason;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "conflicting FP32 modes");
   const unsigned conflicting = preserve | FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   b.shader->info.float_controls_execution_mode = conflicting;
   assert(!stela_air_lower_software_half_profile(b.shader, profile, &reason) &&
          strcmp(reason, "conflicting FP32 denorm modes") == 0);
   assert(b.shader->info.float_controls_execution_mode == conflicting);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "unqualified FP32 division");
   b.shader->info.float_controls_execution_mode = preserve;
   nir_def *input = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_fdiv(&b, input, input);
   assert(!stela_air_lower_software_half_profile(b.shader, profile, &reason) && strcmp(reason, "fdiv") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "unqualified storage image");
   b.shader->info.float_controls_execution_mode = preserve;
   nir_variable_create(b.shader, nir_var_image, glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT), "image");
   assert(!stela_air_lower_software_half_profile(b.shader, profile, &reason) &&
          strcmp(reason, "lavapipe FP32 preservation does not support storage images") == 0);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "unqualified bindless image");
   b.shader->info.float_controls_execution_mode = preserve;
   nir_bindless_image_load(&b, 4, 32, nir_imm_int(&b, 0), nir_imm_zero(&b, 4, 32),
                           nir_imm_int(&b, 0), nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
                           .format = PIPE_FORMAT_R32_UINT, .dest_type = nir_type_uint32);
   assert(!stela_air_lower_software_half_profile(b.shader, profile, &reason) &&
          strcmp(reason, "lavapipe FP32 preservation does not support storage images") == 0);
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   supported_cases();
   vector_and_dynamic_values();
   render_stage(MESA_SHADER_VERTEX);
   render_stage(MESA_SHADER_FRAGMENT);
   refusals_do_not_mutate();
   lavapipe_preservation_refusals();
   glsl_type_singleton_decref();
   puts("software binary16: 19 arithmetic/conversion, 2 render and 19 refusal cases passed");
   return 0;
}

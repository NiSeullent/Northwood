/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef STELA_AIR_SOFT_HALF_H
#define STELA_AIR_SOFT_HALF_H

#include "compiler/nir/nir.h"

enum stela_air_backend_profile {
   STELA_AIR_BACKEND_NATIVE,
   STELA_AIR_BACKEND_SOFTWARE,
   STELA_AIR_BACKEND_PATCHED_LAVAPIPE,
};

/* Lower the supported binary16 arithmetic to FP32 and integer bit operations.
 * Half SSA values remain 16-bit integer bit containers. Call before any
 * target-specific lowering weakens fused operations. The generic software
 * profile requires explicit FP32 flush-to-zero for vertex/fragment. The
 * patched lavapipe profile can preserve FP32 in qualified arithmetic. On
 * refusal, the input is unchanged and reason names the unsupported boundary. */
bool stela_air_lower_software_half(nir_shader *shader, const char **reason);
bool stela_air_lower_software_half_profile(nir_shader *shader, enum stela_air_backend_profile profile,
                                          const char **reason);

#endif

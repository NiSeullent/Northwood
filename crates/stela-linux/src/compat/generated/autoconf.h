/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Stela's Kconfig. Linux code runs on 64-bit x86 for one owner at a time,
 * without preemption or interrupts (see stela/linux.h): Linux's
 * uniprocessor, non-preemptible configuration of Stela's own arch. Options
 * not listed are disabled, so debug, tracing, KUnit and fault-injection
 * hooks compile out exactly as in a kernel built without them. */
#define CONFIG_64BIT 1
/* Kconfig normally records the selected Clang's version. Linux uses it to
 * enable the matching compiler diagnostics, including syscall alias pragmas. */
#ifdef __clang__
#define CONFIG_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif
/* Select native architecture ownership hooks in the shared Linux sources. */
#define CONFIG_STELA 1
#define CONFIG_DEFAULT_HOSTNAME "(none)"
#define CONFIG_MMU 1
#define CONFIG_PCI 1
#define CONFIG_PCI_MSI 1
#define CONFIG_PRINTK 1
#define CONFIG_MESSAGE_LOGLEVEL_DEFAULT 4
#define CONFIG_FW_LOADER 1
/* Real generic IEEE/CRC32C library; no architecture dispatch. */
#define CONFIG_CRC32 1
/* Original XArray ordered/multi-index algorithms, not a layout substitute. */
#define CONFIG_XARRAY_MULTI 1
#define CONFIG_BUG 1
/* x86's WARN flags; asm/bug.h reports each site without a bug table. */
#define CONFIG_GENERIC_BUG 1
#define CONFIG_TINY_RCU 1
#define CONFIG_TINY_SRCU 1
/* Record balanced atomic/RCU sections; this enables no scheduling. */
#define CONFIG_PREEMPT_NONE 1
#define CONFIG_PREEMPT_COUNT 1
#define CONFIG_RT_MUTEXES 1
/* DMA fence array/chain callbacks use the deferred hard-IRQ worker service. */
#define CONFIG_IRQ_WORK 1
/* Original task_struct embeds each retained continuation's thread_info. */
#define CONFIG_THREAD_INFO_IN_TASK 1
#define CONFIG_ARCH_DMA_ADDR_T_64BIT 1
/* Each device's owner supplies its dma_map_ops. Stela's arch is coherent, as
 * x86 is, so no DMA sync hooks are configured. */
#define CONFIG_HAS_DMA 1
#define CONFIG_ARCH_HAS_DMA_OPS 1
#define CONFIG_NEED_DMA_MAP_STATE 1
#define CONFIG_HAS_IOMEM 1
#define CONFIG_PHYS_ADDR_T_64BIT 1
#define CONFIG_NR_CPUS 1
#define CONFIG_HZ 1000
#define CONFIG_PAGE_SHIFT 12
/* The architecture represents five levels and folds P4D at runtime when the
 * original native root has four. This is an ABI shape, not LA57 admission. */
#define CONFIG_PGTABLE_LEVELS 5
#define CONFIG_ILLEGAL_POINTER_VALUE 0xdead000000000000
/* nouveau selects the I2C core for its buses and encoders. */
#define CONFIG_I2C 1
#define CONFIG_I2C_ALGOBIT 1
/* nouveau and what its Kconfig entry selects. */
#define CONFIG_DRM 1
#define CONFIG_DRM_NOUVEAU 1
#define CONFIG_DRM_TTM 1
#define CONFIG_DRM_TTM_HELPER 1
#define CONFIG_DRM_EXEC 1
#define CONFIG_DRM_GPUVM 1
#define CONFIG_DRM_SCHED 1
#define CONFIG_DRM_KMS_HELPER 1
#define CONFIG_DRM_DISPLAY_HELPER 1
#define CONFIG_DRM_DISPLAY_DP_HELPER 1
#define CONFIG_DRM_DISPLAY_HDMI_HELPER 1
#define CONFIG_DMA_SHARED_BUFFER 1
#define CONFIG_SYNC_FILE 1

/* The drivers' parameter and scheduling defaults, as Linux's Kconfig sets them. */
#define CONFIG_DRM_XE_FORCE_PROBE ""
#define CONFIG_DRM_I915_REQUEST_TIMEOUT 20000
#define CONFIG_DRM_I915_FENCE_TIMEOUT 10000
#define CONFIG_DRM_I915_USERFAULT_AUTOSUSPEND 250
#define CONFIG_DRM_I915_HEARTBEAT_INTERVAL 2500
#define CONFIG_DRM_I915_PREEMPT_TIMEOUT 640
#define CONFIG_DRM_I915_PREEMPT_TIMEOUT_COMPUTE 7500
#define CONFIG_DRM_I915_MAX_REQUEST_BUSYWAIT 8000
#define CONFIG_DRM_I915_STOP_TIMEOUT 100
#define CONFIG_DRM_I915_TIMESLICE_DURATION 1
#define CONFIG_DRM_XE_JOB_TIMEOUT_MAX 10000
#define CONFIG_DRM_XE_JOB_TIMEOUT_MIN 1
#define CONFIG_DRM_XE_TIMESLICE_MAX 10000000
#define CONFIG_DRM_XE_TIMESLICE_MIN 1
#define CONFIG_DRM_XE_PREEMPT_TIMEOUT 640000
#define CONFIG_DRM_XE_PREEMPT_TIMEOUT_MAX 10000000
#define CONFIG_DRM_XE_PREEMPT_TIMEOUT_MIN 1
#define CONFIG_NOUVEAU_DEBUG 5
#define CONFIG_NOUVEAU_DEBUG_DEFAULT 3

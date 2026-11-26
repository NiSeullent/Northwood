<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Legacy GPU compatibility

Northwood 1.0.0b2 uses the existing Nouveau/NVK Metal stack. When the PC
firmware exposes a Graphics Output Protocol framebuffer, Northwood can
draw its boot menu and status screens on that display. Accelerated macOS
requires a matching Vulkan device, the features required by the Metal
backend and a successful first GPU submission.
[Northwood NVIDIA admission](../crates/stela-driver-gpu/src/metal/nvidia.rs) · [Metal feature requirements](../crates/stela-driver-gpu/src/metal/vulkan/backend.rs)

| Requested hardware | What the available drivers cover |
| --- | --- |
| AMD GCN 2 (Sea Islands / GFX7) and newer | Mesa RADV supports GFX7 Vulkan 1.3 on `amdgpu`. Stambha's experimental image explicitly excludes the kernel's Southern Islands and Sea Islands build options because those Vulkan capabilities do not satisfy its Metal backend. The donor links its Metal RADV path for GFX8 and later. Work on 32-lane Metal subgroup semantics for wave64 hardware remains experimental. [Mesa RADV](https://docs.mesa3d.org/drivers/radv.html) · [Stambha kernel configuration](https://github.com/garvitsurana271/stambha/blob/2585a5381281891e7da7940758d3e01d86ff955a/crates/stela-linux/src/compat/generated/autoconf.h) · [Wave64 PR #10](https://github.com/garvitsurana271/stambha/pull/10) |
| NVIDIA GeForce GTX 700 / Kepler and newer | Current upstream Mesa NVK lists Kepler. The Northwood and Stambha Metal paths are built around the GSP firmware used on Turing and later and reject pre-Turing chipsets. A GTX 700 board needs the matching pre-Turing kernel path and Metal feature qualification. [Mesa NVK](https://docs.mesa3d.org/drivers/nvk.html) · [Stambha NVIDIA admission](https://github.com/garvitsurana271/stambha/blob/2585a5381281891e7da7940758d3e01d86ff955a/crates/stela-driver-gpu/src/metal/nvidia.rs) |
| Intel HD Graphics 4000 / Ivy Bridge and newer | Stambha's experimental compiler build uses ANV for Gen9+ and leaves out the ELK compiler and older HasVK Vulkan path. Ivy Bridge is Gen7. The donor contains Gen7 command and compiler tests, while its linked Intel Metal stack uses ANV. [Stambha Vulkan build](https://github.com/garvitsurana271/stambha/blob/2585a5381281891e7da7940758d3e01d86ff955a/xtask/src/gpu/nvk.rs) · [Intel HasVK options](https://docs.mesa3d.org/envvars.html) |

The Stambha 0.6.1 experimental image and Northwood use the same Linux and
Mesa source revisions. The experimental image applies a different GPU build
configuration, patch stack, firmware collection, driver integration and
runtime memory policy. Stela-Mods also publishes experimental graphics
branches. The remaining work needs a complete linked driver image and
qualification on each target generation, including a display mode,
GPU submission, macOS installer boot and accelerated desktop.
[Stambha experimental source](https://github.com/garvitsurana271/stambha/tree/2585a5381281891e7da7940758d3e01d86ff955a) · [Stela-Mods graphics branches](https://github.com/reece-builds/Stela-Mods/branches)

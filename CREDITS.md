<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Northwood credits

Northwood builds on Stela by tylertms and the Stela contributors. Stela provides the ARM64-to-x86-64 translator, execution engine, Apple silicon hardware model, UEFI boot program and guest Metal driver. [Stela](https://github.com/tylertms/Stela)

## Code and data used in the project

- Linux contributors provide kernel code, driver ports and the basis for the Linux compatibility layer. FreeBSD contributors provide the e1000 driver lineage, and MediaTek contributors provide Wi-Fi and Bluetooth driver sources. Asahi Linux contributors provide the `hci_bcm4377` Bluetooth driver used by the Stela port. [Third-party inventory and source revisions](NOTICE)
- Mesa contributors provide NVK, NIR, NAK and the Vulkan runtime. LLVM and libc++ contributors provide compiler and C++ libraries. The graphics stack also uses libdrm headers and Khronos Vulkan headers. [Third-party inventory and source revisions](NOTICE)
- Mbed TLS, smoltcp, RustCrypto, Arm AARCHMRS, Peter Cawley's `corsix/amx`, Frédéric Cambus's Spleen font and Apple Open Source XNU headers provide cryptography, networking, CPU data, architecture references, console glyphs and guest kernel interfaces. [Third-party inventory and source revisions](NOTICE)

## Design and hardware references

- Acidanthera's OpenCorePkg informed the configuration layout and terminology. Acidanthera and the contributors to VoodooI2C, VoodooPS2, AppleALC, WhateverGreen and itlwm document device behavior relevant to ACPI and hardware compatibility work. [OpenCorePkg](https://github.com/acidanthera/OpenCorePkg)
- Asahi Linux contributors document Apple silicon hardware. The Lenovo IdeaPad 1 15IGL7 owner contributed reference configuration and the 32 GiB swap precedent from their `NMXSwapReserve.kext` setup. [Asahi Linux documentation](https://asahilinux.org/docs/)
- NullMoth Systems' NVIDIA macOS driver and Golden Gate work informed the NVIDIA compatibility investigation. NullMoth identifies separate terms for its own driver components, the LGPL-3.0-or-later translator based on Anees Iqbal's `metal2vulkan`, and the MIT-licensed NVK work. Any v3 kit assessment needs the notices shipped in the exact artifact. [NullMoth component notices](https://github.com/nullmoth/nvidia-macos-driver/blob/v3.0.0-beta.1/NOTICE)
- NVIDIA publishes the Linux Open GPU Kernel Modules and supplies GSP firmware under separate terms. Linux-firmware contributors and device vendors supply additional firmware under the applicable device-specific terms. [NVIDIA module notices](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/COPYING) · [Firmware source notices](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/tree/WHENCE)

See the [licensing guide](LICENSING.md) for the program-specific distribution terms and third-party contribution policy.

# Lenovo IdeaPad 1 15IGL7 (82V7)

What Stela drives on this PC today, and what it does not. Observed on a
Celeron N4120 unit with 4 GB of RAM (the compact memory profile) running
macOS 15.8.1 natively through OpenCore, whose kext list and device
properties name the parts below.

| Part | PCI / bus identity | Stela driver | Status |
| --- | --- | --- | --- |
| CPU: Celeron N4120, Gemini Lake Refresh, 4 cores, no AVX | — | translator (x86-64-v2) | runs; slow |
| RAM: 4 GB | — | `host_memory` compact profile | boots with about 2.1 GB of guest DRAM |
| GPU: Intel UHD Graphics 600 | `8086:3185` rev 06, subsystem `17aa:3e77` | none: the Metal driver is NVIDIA only | **no GPU acceleration**; the firmware framebuffer shows macOS through `GopScanout` |
| Wi-Fi: Intel Wireless-AC 9560/9462 CNVi | `8086:31dc` | none: `wifi::intel` runs AX200 and newer | **not driven**; the install needs a USB Ethernet adapter (ASIX or RTL815x) |
| Bluetooth: Intel, USB | `8087:xxxx` | `usb::bluetooth::intel` | host side only; the guest Mac's Bluetooth is hidden |
| Audio: HD Audio controller, Realtek codec (AppleALC layout 100) | `8086:3198` | `audio::hda` with `hda::realtek` | driven |
| Touchpad: I2C-HID on LPSS I2C | `8086:31ba` and siblings `31ac`..`31b8` | `bus::lpss` (`GeminiLake` profile), `input::hid_i2c` | driven |
| Keyboard: PS/2 | 8042 | `input::ps2` | driven |
| Storage: NVMe M.2 2242 SSD | class `01 08 02` | `storage::nvme` | driven |
| Storage: eMMC / SD (some units) | `8086:31cc` / `31ca`, class `08 05` | `storage::sdhci`, `storage::mmc` | driven |
| Card reader: Realtek RTS52xx (Sinetek-rtsx) | `10ec:5227`, `522a` or `5229` | `storage::rtsx` | driven for those three chips |
| USB: xHCI | `8086:31a8`, class `0c 03 30` | `usb::xhci` | driven; USB devices (webcam, sticks) reach macOS |
| SMBus | `8086:31d4` | `bus::smbus::i801` | driven |
| Battery, lid, AC, EC | ACPI | `platform::{battery, lid, ac, ec}` | driven |
| Ethernet | none on this model | — | use a USB adapter |

## What remains to port

Two drivers, both large:

1. **Intel UHD 600 (Gen9 LP) Metal backend.** Stela's Metal driver
   (`stela-driver-gpu`) compiles AIR to NIR and runs it through Mesa's NVK on
   a nouveau kernel port. An Intel backend needs the i915 kernel driver (GTT,
   stolen memory, forcewake, execlists, display) under `stela-linux`, Mesa's
   ANV and the Intel shader compiler in the GPU toolchain, and the Metal glue
   that `metal/nvidia` provides for NVK. `src/intel/` holds Gen7..Gen12
   command packers used by hosted tests, not a driver. Months of work.
2. **Intel Wireless-AC 9000 series (`31dc`, `30dc`, `9df0`, `a370`, `2526`,
   `271b`).** `wifi::intel` (38 000 lines) implements the gen2/gen3 PCIe
   transport (context info, AX200 and newer). The 9000 series uses the legacy
   transport: secured CPU1/CPU2 section loading through the FH, firmware
   paging, gen1 TFDs with scheduler byte-count tables, and the multi-queue RX
   of its generation. Weeks of work, and it needs the laptop to test on.

Until then: boot with a USB Ethernet adapter for the install's network, and
expect macOS without GPU acceleration.

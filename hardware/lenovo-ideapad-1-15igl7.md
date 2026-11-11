# Lenovo IdeaPad 1 15IGL7 (82V7)

What Stela drives on this PC today, and what it does not. Observed on a
Celeron N4120 unit with 4 GB of RAM running
macOS 15.8.1 natively through OpenCore, whose kext list and device
properties name the parts below.

| Part | PCI / bus identity | Stela driver | Status |
| --- | --- | --- | --- |
| CPU: Celeron N4120, Gemini Lake Refresh, 4 cores, no AVX | — | translator (x86-64-v2) | A 4 GiB QEMU fixture with AVX, BMI, and LA57 disabled reaches XNU. The N4120 itself still needs a boot run. |
| RAM: 4 GB | — | `host_memory` | The field report measured about 2.7 GiB of usable host RAM and the original loader refused the boot. A new hardware run is needed to verify the reclaimed RAM and guest DRAM. |
| GPU: Intel UHD Graphics 600 | `8086:3185` rev 06, subsystem `17aa:3e77` | Firmware GOP; Northbridge CPU Metal under development | GOP displays the boot UI. The restore bundle reaches XNU. Graphical installation and WindowServer on this laptop remain unverified. |
| Wi-Fi: Intel Wireless-AC 9560/9462 CNVi | `8086:31dc` | Intel legacy transport under development | The internal adapter has no verified association or data path. The remote-install route can use supported ASIX or RTL815x USB Ethernet. Emulated xHCI/RNDIS passed host DHCP and a local HTTP transfer. |
| Bluetooth: Intel, USB | `8087:xxxx` | `usb::bluetooth::intel` | host side only; the guest Mac's Bluetooth is hidden |
| Audio: HD Audio controller, Realtek codec (AppleALC layout 100) | `8086:3198` | `audio::hda` with `hda::realtek` | driven |
| Touchpad: I2C-HID on LPSS I2C | `8086:31ba` and siblings `31ac`..`31b8` | `bus::lpss` (`GeminiLake` profile), `input::hid_i2c` | The 82V7 profile applies the `SYNA2BA6` descriptor override. The EFI stops this attach at LPSS ownership. Physical input remains unverified. |
| Keyboard: PS/2 | 8042 | `input::ps2` | The driver is available. Verify input on the actual 82V7. |
| Storage: NVMe M.2 2242 SSD | class `01 08 02` | `storage::nvme` | The laptop firmware selected a 512 GB KIOXIA KBG6AZNV512G in three saved boots. A fresh boot must verify native NVMe after ExitBootServices. |
| Storage: eMMC / SD (some units) | `8086:31cc` / `31ca`, class `08 05` | `storage::sdhci`, `storage::mmc` | The same laptop exposes 125 GB eMMC. The installer cannot obtain its serial and refuses it as an install target. |
| Card reader: Realtek RTS52xx (Sinetek-rtsx) | `10ec:5227`, `522a` or `5229` | `storage::rtsx` | driven for those three chips |
| USB: xHCI | `8086:31a8`, class `0c 03 30` | `usb::xhci` | driven; USB devices (webcam, sticks) reach macOS |
| SMBus | `8086:31d4` | `bus::smbus::i801` | driven |
| Battery, lid, AC, EC | ACPI | `platform::{battery, lid, ac, ec}` | driven |
| Ethernet | none on this model | — | use a USB adapter |

## Memory boot validation

The original EFI refused a 4 GiB, four-vCPU QEMU run after reading the
484 MiB macOS 26B123 restore bundle. This reproduced the low-memory boot
path in a VM. The diagnostic candidate with SHA-256
`f9db6bc3e38b0323abea5213e2b3817ed92fc6680f4c2ccee4eec15e5cb71960`
measured 3476 MiB usable after reclaim. When firmware could not allocate
its 1222 MiB contiguous heap, the loader retried with a 640 MiB heap.
The engine received 2674 MiB of backed guest DRAM, loaded the 21 bundle
regions, started three secondary host processors, and selected the guest
boot PE. The timed run ended before XNU console output. The Lenovo memory
map, kernel boot, and installation still require an on-device run.

In the single-guest-worker QEMU fixture, this same candidate preserved the
existing Compact and Standard engine budgets at 4096 and 8192 MiB of
installed memory. Both fixture boots reached idle. The guest gained 10 MiB
of backed DRAM in each run, from 2506 to 2516 MiB and from 5390 to 5400 MiB
respectively.

The optimized diagnostic EFI with SHA-256
`4e4ca84082be92fe82a939221860bd879d6dce8d8c463b940cfcc645aa48cdab`
ran the same 26B123 restore bundle in 4096 MiB QEMU with four workers and a
disposable NVMe target. It used the 640 MiB heap fallback, reported 2706 MiB
of backed DRAM, and printed the Darwin 27.2.0 kernel banner from XNU.
This result covers guest kernel entry in the VM. The hardware support and
installation checks below still require the corresponding real devices.

A subsequent optimized EFI with SHA-256
`2ca3df16f87428999aee938b3835b14dfbe190b76fce273972dce9faf367985a`
also reached the Darwin 27.2.0 kernel banner in the default execution path
with 4096 MiB, four host workers, and LA57, AVX, AVX2, FMA, F16C, BMI1 and
BMI2 disabled. The runtime reported 2706 MiB of backed DRAM and qualified
Intel's cached scalar line atomicity without claiming 16-byte load atomicity.
The stock QEMU 10.1.3 test set `abm=off` because that TCG version gates its
LZCNT decoder on BMI1 while the virtual CPU can still advertise LZCNT. A
separate QEMU build with corrected CPUID checks reached the same Darwin
banner with ABM enabled, BMI1 and BMI2 disabled, and 2706 MiB of backed DRAM.
Both runs used the same EFI binary. [Intel SDM §9.1.1](https://cdrdv2-public.intel.com/812386/253668-sdm-vol-3a.pdf)
[QEMU 10.1.3 decoder](https://github.com/qemu/qemu/blob/v10.1.3/target/i386/tcg/decode-new.c.inc#L515-L528)
.

The EFI with the hardware-profile matcher
`4a3815b500b70b10470cae01ba2e0187ca30a4adf3ac462ef5b4f631d4e05693`
also reached Darwin 27.2.0 with the same 4 GiB, no-AVX, ABM-enabled QEMU
configuration and 2706 MiB of guest DRAM. A separate synthetic hardware test
selected the Lenovo profile only when the SMBIOS system identity, PCI ID,
and a present device's primary ACPI HID all matched. A boot entry overrode
the profile's swap settings. The automatic match and input drivers still
need a run on the actual laptop.

.

### RAMDisk staging

The 26B123 restore bundle's 356 MiB RAMDisk contributes 354 MiB of
2 MiB-aligned backing directly to guest RAM. Its two unaligned ends total
2 MiB and are copied into the guest. The same EFI was booted with and without
this placement. Both paths reached Darwin 27.2.0 in each VM, with identical
memory profiles and reported DRAM equal to backed DRAM.

| QEMU RAM | Profile | Backed with staging | Backed with direct placement |
| --- | --- | ---: | ---: |
| 4 GiB | Paged | 2706 MiB | 3058 MiB |
| 8 GiB | Standard | 4136 MiB | 4488 MiB |

Before starting the guest CPUs, the 4 GiB run read the entire RAMDisk from
guest memory and matched its SHA-256 to the manifest. A changed byte in a
copy of the bundle caused the loader to refuse it and return the original
allocation before ExitBootServices. The standalone test kernel retained
2526 MiB under Compact and 5446 MiB under Standard.

.

Splitting the same bundle's BootKC into three groups directly placed another
114 MiB of its 119.53 MiB. Combined direct backing was 468 MiB, with
16.14 MiB of logical input left staged. The same EFI reached Darwin with
and without direct placement at both memory sizes. At 4 GiB, Paged guest
backing went from 2706 to 3166 MiB. At 8 GiB, Standard went from 4136 to
4596 MiB. The 4 GiB QMP readback matched both full source file hashes and
all seven region hashes. A corrupt BootKC byte caused all three aligned
allocations to be returned before ExitBootServices.

.

## Touchpad validation

The 4 GiB QEMU fixture supplies a synthetic `SYNA2BA6` SSDT with
`PNP0C50` compatibility, an I²C resource and an intentionally malformed
`_DSM`. PCI `00:06.0` is the synthetic Intel graphics identity and
`00:07.0` supplies the `8086:31ba` LPSS identity. The 82V7 profile applies
the descriptor override and resolves the ACPI controller against the PCI
scan before requesting ownership. The correct LPSS address reaches the
missing stop receipt. Pointing the same ACPI node at the graphics address
fails controller identification. The different SMBIOS product fails at
the malformed `_DSM`. The synthetic PCI endpoint has no LPSS register
or I²C-HID model. A 15IGL7 firmware ACPI or HID descriptor capture is still
needed. .

A historical LiberNT boot of this 15IGL7 enumerated LPSS I²C7 at `00:17.3`
(`8086:31ba` rev 06) with two 4 KiB BARs and IRQ 34. Its driver reported
64-entry transmit and receive FIFOs. The same boot started HID device
`06cb:ce2d` and read its 665-byte report descriptor. The matching driver
source shows that startup requires a successful descriptor read at the
register returned by `_DSM`. The log does not print that register, the
HID namespace path, or the bus connecting the HID device to a controller.
.

The I²C-HID model now verifies mouse-mode selection for a Precision
Touchpad with a dedicated Input Mode feature and a relative Mouse
collection. It checks the feature transaction and polled movement, click
and release. The model runs independently of the native LPSS controller.

[Lenovo specifications](https://psrefstuff.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_1_15IGL7/IdeaPad_1_15IGL7_Spec.pdf)
[Precision Touchpad configuration](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/touchpad-configuration-collection)

The GLK LPSS reset stage checks a contained, powered requester and its
original owned BAR0, then asserts and verifies function and iDMA reset.
If the platform independently certifies that prior firmware, interrupt
and work users have retired, the completion stage rechecks containment
and reset, then uses the original root's DMA completion before binding.
It returns the same BAR with the bound requester and keeps both owners
on failure. The firmware diagnostic reports the PCI command bits, BAR0
availability and I²C protocols on the controller handle as observations.
Real firmware and power exclusion, original-root DMA completion and
native I²C-HID input remain to be established on the laptop.

[Linux reset registers](https://github.com/torvalds/linux/blob/cc9de5d4b5ab6459f5774946aa3195ad51236fa8/drivers/mfd/intel-lpss.c#L41-L70)
[Linux GLK I²C IDs](https://github.com/torvalds/linux/blob/cc9de5d4b5ab6459f5774946aa3195ad51236fa8/drivers/mfd/intel-lpss-pci.c#L293-L301)

### Capturing the LPSS boundary on a physical 82V7

Capture the serial log from one normal boot of a diagnostics-enabled EFI
with host input enabled. Run `stela report --boot-log boot.txt` on the copied
log. For the BDF and PCI ID named by the `SYNA2BA6` ACPI record, compare these
observations from the same boot.

- `FW LPSS-I2C` gives the firmware command bits and I²C Master/Host protocol
  presence on the same PCI_IO handle. `FW LPSS-I2C BAR0` gives the bus and
  host addresses and the length from that handle's validated BAR descriptor.
- `HOST LPSS-I2C PCI` and `CAPS` give the parked function's configuration
  after ExitBootServices. `HOST LPSS-I2C BAR0` checks the first 4 KiB against
  the function's original PCI root and the sealed memory map. Compare the
  full range when the firmware descriptor also covers 4 KiB. A different
  firmware length leaves the full range unverified.
- `HOST LPSS-I2C ACPI` gives the two namespace paths, the I²C resource
  address, speed and interrupt type, and the effective HID descriptor
  register after any board override. If the record is absent, keep the
  profile, ACPI and PCI containment messages and the `PLATFORM FAILED`
  line from the same boot.

A matching BAR and ACPI address establish where the device is. Native
attachment still requires exclusion of firmware and other register and
power users, retirement of prior interrupt and work producers, a stable
D0/clock state, and DMA completion through the original requester root.
The retained 82V7 boot-log copy contains no LPSS or SYNA2BA6 observation.

## USB rehearsal

The physical ESP backup contains three boot logs where the KIOXIA NVMe
was eligible and each chosen disk and partition matched the saved
`disk.cfg`. Each boot stopped at the original RAM check. On a new
boot, open Diagnostics before confirming installation. With a Diagnostics
EFI that supports `CHECK install-target`, a saved choice is checked against
the native disk serial and GPT partition after ExitBootServices. A missing
choice skips this check. The `DISK Nvme` and `PARTITION` lines are available
for checking the current disk and GPT before making a new choice.
.

On November 12, the physical STELA stick received the `dcdd1e13…` EFI for
the first diagnostics boot. The file matched after the physical ESP was
unmounted and imaged again. All seven accompanying EFI files and the 17
installation assets matched their original hashes. This stick uses the
previous menu configuration, which leaves guest input disabled. The first
laptop test uses Diagnostics and can capture the firmware keyboard, NVMe
inventory and the memory report before selecting an installation target.
.

An image reproduced the attached STELA stick's partition offsets and
partition GUIDs with a fresh disk GUID. Its 17 staged data files matched the
source by SHA-256. The test image uses the new EFI, removes saved Wi-Fi
credentials, and has fresh boot logs. QEMU booted it read-only over xHCI.
A USB keyboard selected Install macOS and opened the disk picker. QEMU
reported zero writes to the blank NVMe before the VM stopped.

.

A separate copy of the image booted its 26B123 recovery bundle to Darwin
27.2.0 in a 4 GiB VM. Injected Lenovo hardware identities selected the 82V7
profile, and the native drivers found the emulated NVMe, USB keyboard and
mouse. The same EFI passed host DHCP and a local HTTP transfer over an
emulated xHCI/RNDIS adapter using a synthetic invariant TSC. Graphical
installation and these devices still need verification on the laptop.

.

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

The single-user boot path can be used for kernel and memory diagnostics on
this machine. The graphical installer and desktop require a Metal backend.
A supported USB Ethernet adapter provides an independent network path for
installation once graphical boot is available.

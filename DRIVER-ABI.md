# Northwood static driver registry ABI 1.0

The `.nwdrv` module ID names a group of statically linked host driver registrations. `crates/stela-drivers/src/probe/registry.rs` contains the compiled PCI registration table. Registration priority is the order of the table within a probe stage. `IntelEth`, for example, has separate entries for `igc`, `e1000e`, the integrated PCH transport, and `igb`.

## Manifest fields

| Field | Meaning |
| --- | --- |
| `id` | Case-sensitive module name used to group related implementations |
| `implementation` | Name of this registration's specific implementation or probe stage |
| `class` | Host subsystem served by the driver |
| `status` | `Implemented` for an attachment path and `CheckOnly` for identification, inventory, or handoff |
| `source` | Repository-relative path to the primary implementation. See `NOTICE` and the source headers for donor revisions, attribution, and license details |
| `license` | SPDX expression of the primary implementation and license tier A or B. The static registration type admits these two tiers |
| `firmware` | Known required image names or the device profile used to determine them. `stela-loader` and the driver's `Provider` request remain authoritative for the selected image and its integrity checks |

## Match and attach contract

The match callback receives a copied PCI `Identity` containing the address, vendor, device, and class. It selects candidates without reading PCI configuration space, mapping BARs, or writing registers. `pci_candidates()` exposes these matches to reporting tools in scan order. A candidate can still be refused by network configuration, hardware ownership, native service availability, firmware identity, or device validation.

Early scan registrations handle display inventory, platform discovery, and USB firmware handoff. Immediate attachment proceeds through the selected native or legacy driver. EHCI, Broadcom PCIe Bluetooth, and some USB4 controllers attach at their existing deferred stages. Pre-ExitBootServices resource discovery checks every registration marked for early claims, including overlapping matches. Early claims establish which mappings to prepare.

NVModern is a post-scan candidate for NVIDIA display functions. The existing NVIDIA path claims the function, reads the chipset from the owned register window, selects the pinned GSP-RM images, and starts nouveau and NVK on a build with NVK enabled. The PCI identity is enough to list the candidate. The live chipset determines its firmware.

Native drivers use `NativeServices` to claim the original stopped requester and obtain its DMA domain and owned BAR mappings. Attach code validates the live device and keeps any owner that must outlive a failed attachment in the retained `Devices` state. Legacy attachment requires the mount to permit legacy PCI writes. It receives the post-ExitBootServices function, handoff, timer, DMA allocator, firmware provider and devices accumulator through `legacy::Context`.

## Policy evaluation

`resolve_pci()` and `probe_with_policy()` use the same policy selection. Auto keeps the original PCI priority. Safe admits storage, input, and USB host attachments. Manual admits the enabled module IDs. Disable takes precedence over enable. Block applies to an exact vendor and device pair. Force selects a named attachment only when its existing predicate admits the identity and the registration can attach at that stage. `Arguments` is parsed and retained for future module consumers.

The boot engine applies the selected policy to the main PCI probe and to post-scan NVModern claims. Display inventory, USB firmware release, and existing non-PCI class selection remain independent of the PCI attachment policy. When policy declines xHCI, the mount leaves Intel port routing alone. When policy declines EHCI, the mount skips the native controller attach. Physical verification of policy behavior remains pending.

## Catalog generation

`cargo xtask catalog` writes `DRIVERS.md` and `catalog.json`. `cargo xtask catalog --check` compares both files to the compiled table. JSON records `abi: [1, 0]`, an ordered `modules` array with one row per compiled registration, and `nvmodern_chipsets` with the GSP-RM requests from the pinned recipes. Build features can affect the registration set. Use the Rust `pci_candidates()` API to evaluate an actual PCI identity against the same driver predicates used by the mount.

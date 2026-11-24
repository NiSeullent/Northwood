# Intel MacBook Pro with Apple T2

Northwood can report the T2 storage and Buffer Copy Engine PCI functions exposed by firmware. Native T2 storage, the T2 virtual USB bus, and Touch Bar operation await implementation and hardware validation. Treat installation to the T2 internal SSD as unsupported.

| Function | Identity | Northwood behavior |
| --- | --- | --- |
| T2 internal SSD | PCI `106b:2005` | Diagnostic EFI reports `FW APPLE T2 NVME` and whether firmware advertises NVMe Pass Thru. The standard native NVMe driver refuses attachment before changing power state, configuring DMA, or resetting the controller. EFI payload staging also refuses a whole disk whose firmware path positively identifies this PCI controller. |
| T2 Buffer Copy Engine | PCI `106b:1801` | Diagnostic EFI reports `FW APPLE T2 BCE` when the function is visible to firmware PCI enumeration. |
| iBridge Touch Bar | USB `05ac:8302` display and `05ac:8102` backlight | Requires the T2 virtual USB transport and Touch Bar HID support. |

Linux uses a 128-byte I/O submission stride with 64-byte admin submissions and 16-byte completions for PCI `106b:2005`. It keeps the standard CC.IOSQES register value, uses a single I/O queue and interrupt vector, reserves the first 32 command IDs for admin traffic, omits CID generation bits and limits Identify to CNS values 0 and 1. A T2 storage port needs model tests for queue wrap, simultaneous admin and I/O commands, completion after abort, CID reuse after reset, and namespace discovery. Validate read, write, flush and reset on a dedicated test installation before enabling the SSD as an install target. EFI Pass Thru availability describes the firmware handle before ExitBootServices.

The upstream BCE implementation transports virtual USB and audio over the T2's Buffer Copy Engine. Use its public protocol implementation to establish the BCE queue lifecycle and virtual USB enumeration first. Touch Bar validation then needs display interface discovery, key events, and backlight control on MacBook Pro hardware.

BridgePatcher's host probe uses the real model to infer T2 presence when configuring OpenCore and Apple drivers on native Intel macOS. Northwood uses the model to report the expected hardware and checks live PCI identity for native storage decisions.

- Linux [NVMe PCI driver](https://github.com/torvalds/linux/blob/8d7fe9885c62fc9a95bf55d9db244d9672a8aadc/drivers/nvme/host/pci.c#L4518-L4532)
- t2linux [BCE device table](https://github.com/t2linux/apple-bce-drv/blob/7330e638b9a32b4ae9ea97857f33838b5613cad3/pci.c#L368-L371) and [transport overview](https://github.com/t2linux/apple-bce-drv/blob/7330e638b9a32b4ae9ea97857f33838b5613cad3/README.md)
- Linux [Touch Bar HID driver](https://github.com/torvalds/linux/blob/8d7fe9885c62fc9a95bf55d9db244d9672a8aadc/drivers/hid/hid-appletb-kbd.c) and [USB identities](https://github.com/torvalds/linux/blob/8d7fe9885c62fc9a95bf55d9db244d9672a8aadc/drivers/hid/hid-ids.h)
- BridgePatcher [T2 host probe](https://github.com/bridgepatcher/BridgePatcher/blob/87546e715ef490be1124671604b00c005c7d7a51/bridgepatcher/detections/device_probe.py#L933-L941) and [model list](https://github.com/bridgepatcher/BridgePatcher/blob/87546e715ef490be1124671604b00c005c7d7a51/bridgepatcher/datasets/model_array.py#L421-L440)

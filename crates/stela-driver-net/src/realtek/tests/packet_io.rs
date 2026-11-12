// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! Packet io cases.
use super::*;

/// An RTL8169S over `registers` and `memory`, its rings in `memory`.
fn modelled(registers: &mut [u32], memory: &mut [u64], signalled: bool) -> Realtek {
    use crate::bus::dma::RegionAllocator;
    let start = memory.as_mut_ptr() as u64;
    // SAFETY: the caller owns both allocations for the driver's lifetime.
    let (device, mut dma) = unsafe {
        (
            Mmio::new(registers.as_mut_ptr() as u64, size_of_val(&registers[..])),
            RegionAllocator::new(start..start + (memory.len() * 8) as u64),
        )
    };
    let revision = revision::detect(0x0080_0000).unwrap();
    Realtek {
        device,
        clock: crate::timing::model(Clock {
            ticks_per_microsecond: 1,
        }),
        receive: Ring::new(&mut dma, true).unwrap(),
        transmit: Ring::new(&mut dma, true).unwrap(),
        mac: Mac([2, 0, 0, 0, 0, 1]),
        link: RuntimeLink::default(),
        revision,
        warning: None,
        function: 0,
        location: "model".into(),
        signalled,
    }
}

/// The acknowledgement an empty receive makes writes back the whole status
/// it read, as Linux's `rtl_ack_events` does: the receive ring's exhaustion
/// (RxDescUnavail, bit 4) and the receive FIFO's overflow (RxFIFOOver,
/// bit 6) among the events, so a receiver that ran out starts again. The
/// 2.5G chips' status is 32 bits wide. Nothing is written when no event is
/// pending, or when the function has left the bus and reads all ones.
#[test]
fn acknowledge_writes_back_the_whole_status_at_its_width() {
    let time = Time::default();
    for (version, register, width, pending) in [
        (
            46,
            INTERRUPT_STATUS,
            2,
            u32::from(RX_OK | LINK_CHANGE | 1 << 2 | 1 << 4 | 1 << 6),
        ),
        (61, NBASE_INTERRUPT_STATUS, 4, 0x0021_0051),
    ] {
        let model = Model::ready(&time);
        let status = |value: u32| {
            let mut s = model.state.borrow_mut();
            s.bytes[register..register + width].copy_from_slice(&value.to_le_bytes()[..width]);
            s.writes.clear();
        };
        let hardware = Hardware::new(&model, &time, revision(version), 0);
        status(pending);
        assert_eq!(hardware.acknowledge(), pending as u16, "{version}");
        assert_eq!(
            model.state.borrow().writes,
            [(register, pending, width as u8)],
            "{version}"
        );
        for quiet in [0, u32::MAX] {
            status(quiet);
            assert_eq!(hardware.acknowledge(), 0, "{version} {quiet:#x}");
            assert!(
                model.state.borrow().writes.is_empty(),
                "{version} {quiet:#x}"
            );
        }
    }
}

#[test]
fn transmit_without_carrier_is_refused_and_leaves_the_ring_untouched() {
    let mut registers = vec![0u32; 64];
    let mut memory = vec![0u64; 262_144];
    let mut net = modelled(&mut registers, &mut memory, false);
    net.transmit.initialize(false);
    let descriptors = net.transmit.descriptors.bytes().to_vec();
    assert_eq!(net.transmit(&[0; 60]), Err(LinkError::NoCarrier));
    assert_eq!(net.transmit.descriptors.bytes(), descriptors);
    assert_eq!(registers[TX_POLL / 4], 0);
}

/// The stack's path refills its own buffer in place across the ring wrap,
/// and leaves it unchanged when nothing arrived or the link failed.
#[test]
fn receive_reusing_refills_the_callers_buffer_across_the_wrap() {
    let mut registers = vec![0u32; 64];
    let mut memory = vec![0u64; 262_144];
    let mut net = modelled(&mut registers, &mut memory, false);
    net.receive.initialize(true);
    let mut reused = Vec::with_capacity(MAX_FRAME);
    let storage = reused.as_ptr();
    let mut last = Vec::new();
    for sequence in 0..(DESCRIPTORS * 2 + 3) {
        let frame = vec![sequence as u8; 60 + sequence % 64];
        net.receive.deliver(&frame);
        if sequence % 2 == 0 {
            assert_eq!(net.receive(), Ok(Some(frame.clone())));
        } else {
            assert_eq!(net.receive_reusing(&mut reused), Ok(true));
            assert_eq!(reused, frame);
            assert_eq!(reused.as_ptr(), storage);
            last = frame;
        }
    }
    assert_eq!(net.receive_reusing(&mut reused), Ok(false));
    assert_eq!(reused, last);
    net.link.failure = Some(BlockError::Timeout);
    assert!(net.receive_reusing(&mut reused).is_err());
    assert_eq!(reused, last);
}

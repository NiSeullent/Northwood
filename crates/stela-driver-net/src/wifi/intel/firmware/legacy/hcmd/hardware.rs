// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2003-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Intel 9000 FH/SCD initialization from pinned iwlwifi gen1_2/tx.c,
//! iwl-scd.h, iwl-fh.h and iwl-prph.h at 347246472df8.

use super::{
    BYTE_COUNT_BYTES, ENTRIES, Error, FH_DMA_LIMIT, FH_TX_CHANNELS, FH_TX_CONFIG, FH_TX_STRIDE,
    KEEP_WARM_BYTES, Layout, QUEUES, RING_BYTES, SCD_TXFACT, TFD_BYTES, WRITE_POINTER, service,
};
use crate::wifi::intel::power::Csr;

const KEEP_WARM: u32 = 0x197c;
const CBBC_0: u32 = 0x19d0;
const CBBC_16: u32 = 0x1bf0;
const CBBC_20: u32 = 0x1b20;
const SCD_SRAM_BASE: u32 = 0xa02c00;
const SCD_DRAM_BASE: u32 = 0xa02c08;
const SCD_QUEUE0_WRITE: u32 = 0xa02c18;
const SCD_QUEUE0_READ: u32 = 0xa02c68;
const SCD_QUEUE_CHAIN: u32 = 0xa02ce8;
const SCD_QUEUE0_STATUS: u32 = 0xa02d0c;
const SCD_CONTROL: u32 = 0xa02da8;
const SCD_AGGREGATION: u32 = 0xa02e48;
const SCD_QUEUE_ENABLE: u32 = 0xa02e54;
const SCD_AUTO_ACTIVE_AND_31: u32 = (1 << 18) | 1;
const SCD_ALL_QUEUES: u32 = (1 << QUEUES) - 1;
const SCD_ALL_FIFOS: u32 = (1 << FH_TX_CHANNELS) - 1;
const SCD_CONTEXT_OFFSET: u32 = 0x600;
const SCD_TRANSLATION_END: u32 = (0x7e0 + QUEUES as u32 * 2) & !3;
const SCD_FRAME_LIMIT: u32 = 64;
const SCD_COMMAND_CONTEXT: u32 = SCD_FRAME_LIMIT << 16 | SCD_FRAME_LIMIT;
const SCD_COMMAND_FIFO: u32 = 7;
const SCD_STATUS_ACTIVE: u32 = 1 << 3;
const SCD_STATUS_WSL: u32 = 1 << 4;
const SCD_STATUS_SET_ACTIVE: u32 = 1 << 19;
const SCD_STATUS_WRITE_MASK: u32 = 0x017f_0000;
const FH_DMA_ENABLE: u32 = 1 << 31;
const FH_DMA_CREDIT: u32 = 1 << 3;
const FH_DMA_MODE: u32 = 3 << 30;
const FH_TX_CHICKEN: u32 = 0x1e98;
const FH_SCD_AUTO_RETRY: u32 = 1 << 1;
const MEM_READ_ADDRESS: u32 = 0x40c;
const MEM_WRITE_ADDRESS: u32 = 0x410;
const MEM_WRITE_DATA: u32 = 0x418;
const MEM_READ_DATA: u32 = 0x41c;

/// Install the stopped engine's keep-warm buffer and all 31 TFD bases.
/// Caller holds NIC access and has marked this exact layout's pins exposed.
pub(in crate::wifi::intel::firmware::legacy) fn prepare(
    io: &mut impl Csr,
    layout: Layout,
) -> Result<(), Error> {
    validate(layout)?;
    write_prph(io, SCD_TXFACT, 0, SCD_ALL_FIFOS)?;
    require_paused(io)?;
    write_csr(io, KEEP_WARM, (layout.keep_warm >> 4) as u32)?;
    for (queue, ring) in layout.rings.into_iter().enumerate() {
        write_csr(io, cbbc(queue as u32), (ring >> 8) as u32)?;
    }
    let control = service::read_prph(io, SCD_CONTROL)? | SCD_AUTO_ACTIVE_AND_31;
    write_prph(io, SCD_CONTROL, control, SCD_AUTO_ACTIVE_AND_31)
}

/// Activate queue zero on command FIFO seven after this requester's INIT ALIVE.
/// Caller holds NIC access and keeps every staged DMA pin until teardown.
pub(in crate::wifi::intel::firmware::legacy) fn activate(
    io: &mut impl Csr,
    layout: Layout,
    alive_scd_base: u32,
) -> Result<(), Error> {
    validate(layout)?;
    verify_layout(io, layout)?;
    let base = service::read_prph(io, SCD_SRAM_BASE)?;
    if base == 0
        || !base.is_multiple_of(4)
        || base.checked_add(SCD_TRANSLATION_END).is_none()
        || (alive_scd_base != 0 && base != alive_scd_base)
    {
        return Err(Error::SchedulerSram {
            register: base,
            alive: alive_scd_base,
        });
    }
    write_prph(io, SCD_QUEUE_ENABLE, 0, SCD_ALL_QUEUES)?;
    let context = base + SCD_CONTEXT_OFFSET;
    let count = (SCD_TRANSLATION_END - SCD_CONTEXT_OFFSET) / 4;
    write_sram(io, context, core::iter::repeat_n(0, count as usize))?;
    write_prph(
        io,
        SCD_DRAM_BASE,
        (layout.byte_count >> 10) as u32,
        u32::MAX,
    )?;

    write_prph(
        io,
        SCD_QUEUE0_STATUS,
        SCD_STATUS_SET_ACTIVE,
        SCD_STATUS_ACTIVE,
    )?;
    verify_prph(io, SCD_QUEUE_CHAIN, 0, 1)?;
    let aggregation = service::read_prph(io, SCD_AGGREGATION)? & !1;
    write_prph(io, SCD_AGGREGATION, aggregation, 1)?;
    io.write32(WRITE_POINTER, 0)?;
    verify_prph(io, SCD_QUEUE0_WRITE, 0, 0xff)?;
    write_prph(io, SCD_QUEUE0_READ, 0, 0xff)?;
    write_sram(io, context, [0, SCD_COMMAND_CONTEXT])?;
    let status = SCD_STATUS_ACTIVE | SCD_STATUS_WSL | SCD_COMMAND_FIFO;
    // The high mask selects writable status fields. Its readback is not used.
    write_prph(io, SCD_QUEUE0_STATUS, status | SCD_STATUS_WRITE_MASK, 0x1f)?;
    write_prph(io, SCD_QUEUE_ENABLE, 1, SCD_ALL_QUEUES)?;

    write_prph(io, SCD_TXFACT, SCD_ALL_FIFOS, SCD_ALL_FIFOS)?;
    for channel in 0..FH_TX_CHANNELS {
        write_csr(
            io,
            FH_TX_CONFIG + channel * FH_TX_STRIDE,
            FH_DMA_ENABLE | FH_DMA_CREDIT,
        )?;
    }
    let chicken = service::read(io, FH_TX_CHICKEN)? | FH_SCD_AUTO_RETRY;
    io.write32(FH_TX_CHICKEN, chicken)?;
    expect(
        FH_TX_CHICKEN,
        service::read(io, FH_TX_CHICKEN)?,
        chicken,
        FH_SCD_AUTO_RETRY,
    )
}

fn cbbc(queue: u32) -> u32 {
    match queue {
        0..16 => CBBC_0 + 4 * queue,
        16..20 => CBBC_16 + 4 * (queue - 16),
        _ => CBBC_20 + 4 * (queue - 20),
    }
}

fn validate(layout: Layout) -> Result<(), Error> {
    if layout.entries != ENTRIES
        || layout.entry_bytes != TFD_BYTES
        || layout.ring != layout.rings[0]
    {
        return Err(Error::Address);
    }
    let mut spans = [(0, 0); QUEUES + 2];
    let regions = layout
        .rings
        .into_iter()
        .map(|ring| (ring, RING_BYTES, 256))
        .chain([
            (layout.keep_warm, KEEP_WARM_BYTES, 4096),
            (layout.byte_count, BYTE_COUNT_BYTES, 1024),
        ]);
    for (index, (start, bytes, alignment)) in regions.enumerate() {
        let end = start.checked_add(bytes as u64).ok_or(Error::Address)?;
        if start == 0
            || !start.is_multiple_of(alignment)
            || end > FH_DMA_LIMIT
            || spans[..index]
                .iter()
                .any(|&(old_start, old_end)| start < old_end && old_start < end)
        {
            return Err(Error::Address);
        }
        spans[index] = (start, end);
    }
    Ok(())
}

fn verify_layout(io: &mut impl Csr, layout: Layout) -> Result<(), Error> {
    verify_prph(io, SCD_TXFACT, 0, SCD_ALL_FIFOS)?;
    require_paused(io)?;
    verify_prph(
        io,
        SCD_CONTROL,
        SCD_AUTO_ACTIVE_AND_31,
        SCD_AUTO_ACTIVE_AND_31,
    )?;
    expect(
        KEEP_WARM,
        service::read(io, KEEP_WARM)?,
        (layout.keep_warm >> 4) as u32,
        u32::MAX,
    )?;
    for (queue, ring) in layout.rings.into_iter().enumerate() {
        let register = cbbc(queue as u32);
        expect(
            register,
            service::read(io, register)?,
            (ring >> 8) as u32,
            u32::MAX,
        )?;
    }
    Ok(())
}

fn require_paused(io: &mut impl Csr) -> Result<(), Error> {
    for channel in 0..FH_TX_CHANNELS {
        let register = FH_TX_CONFIG + channel * FH_TX_STRIDE;
        expect(register, service::read(io, register)?, 0, FH_DMA_MODE)?;
    }
    Ok(())
}

fn write_sram(
    io: &mut impl Csr,
    address: u32,
    values: impl IntoIterator<Item = u32, IntoIter: Clone>,
) -> Result<(), Error> {
    let values = values.into_iter();
    io.write32(MEM_WRITE_ADDRESS, address)?;
    for value in values.clone() {
        io.write32(MEM_WRITE_DATA, value)?;
    }
    io.write32(MEM_READ_ADDRESS, address)?;
    for (offset, expected) in values.enumerate() {
        expect(
            address + offset as u32 * 4,
            service::read(io, MEM_READ_DATA)?,
            expected,
            u32::MAX,
        )?;
    }
    Ok(())
}

fn write_csr(io: &mut impl Csr, register: u32, value: u32) -> Result<(), Error> {
    io.write32(register, value)?;
    expect(register, service::read(io, register)?, value, u32::MAX)
}

fn write_prph(io: &mut impl Csr, register: u32, value: u32, mask: u32) -> Result<(), Error> {
    service::write_prph(io, register, value)?;
    verify_prph(io, register, value, mask)
}

fn verify_prph(io: &mut impl Csr, register: u32, value: u32, mask: u32) -> Result<(), Error> {
    expect(register, service::read_prph(io, register)?, value, mask)
}

fn expect(register: u32, observed: u32, expected: u32, mask: u32) -> Result<(), Error> {
    if observed & mask != expected & mask {
        Err(Error::Rejected { register, observed })
    } else {
        Ok(())
    }
}

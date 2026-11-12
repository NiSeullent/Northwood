// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2014, 2018-2019, 2021, 2025 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Gen1 secure paging from Linux iwlwifi fw/paging.c, fw/api/paging.h,
//! and fw/img.h at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{FH_DMA_LIMIT, Firmware, LoadPlan};
use crate::{
    bus::dma::{
        mapping::{Direction, DmaDomain},
        memory::PageSource,
    },
    wifi::intel::{
        command::Id,
        dma::{self, Chunk},
    },
};
use alloc::{sync::Arc, vec, vec::Vec};

const PAGE_SHIFT: u32 = 12;
const PAGE_BYTES: usize = 1 << PAGE_SHIFT;
const BLOCK_SHIFT: u32 = 15;
const BLOCK_BYTES: usize = 1 << BLOCK_SHIFT;
const MAX_BLOCKS: usize = 32;
const FLAGS: u32 = (1 << 9) | (1 << 8);

pub(super) const PAYLOAD_BYTES: usize = (3 + 1 + MAX_BLOCKS) * 4;
// The API v1 paging body is sent with the donor's bare wide ID (version 0).
pub(super) const ID: Id = Id {
    group: 1,
    opcode: 0x4f,
    version: 0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    MissingApi,
    UnsupportedApi { command: u8, notification: u8 },
    Layout,
    Address,
    Dma(dma::Error),
}

impl From<dma::Error> for Error {
    fn from(error: dma::Error) -> Self {
        Self::Dma(error)
    }
}

pub(super) struct Layout<'a> {
    css: &'a [u8],
    data: &'a [u8],
}

impl<'a> Layout<'a> {
    pub(super) fn from_plan(firmware: &Firmware<'a>, plan: &LoadPlan<'a>) -> Result<Self, Error> {
        let version = firmware
            .command_version(ID.group, ID.opcode)
            .ok_or(Error::MissingApi)?;
        if (version.command_version, version.notification_version) != (1, 0) {
            return Err(Error::UnsupportedApi {
                command: version.command_version,
                notification: version.notification_version,
            });
        }
        // Gen1's paging loader consumes one CSS section and one contiguous
        // paging image. The transfer plan has already removed the delimiters.
        let [section] = plan.paging.as_slice() else {
            return Err(Error::Layout);
        };
        let css = plan.paging_css.data;
        let data = section.data;
        if !(1..=PAGE_BYTES).contains(&css.len())
            || !(1..=MAX_BLOCKS * BLOCK_BYTES).contains(&data.len())
            || !data.len().is_multiple_of(PAGE_BYTES)
        {
            return Err(Error::Layout);
        }
        Ok(Self { css, data })
    }
}

/// Permanent firmware-visible pages. A successful command only retires the
/// command descriptor. The CSS and data stay live until the requester stops.
pub(super) struct Pages {
    chunks: Vec<Chunk>,
    payload: [u8; PAYLOAD_BYTES],
}

impl Pages {
    pub(super) fn prepare(
        layout: &Layout<'_>,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
    ) -> Result<Self, Error> {
        let blocks = layout.data.chunks(BLOCK_BYTES);
        let block_count = blocks.len();
        let last_pages = blocks.clone().last().ok_or(Error::Layout)?.len() / PAGE_BYTES;
        let mut chunks = Vec::with_capacity(block_count + 1);
        let mut payload = [0; PAYLOAD_BYTES];
        let (header, addresses) = payload.split_at_mut(12);
        for (word, value) in header.as_chunks_mut::<4>().0.iter_mut().zip([
            FLAGS | last_pages as u32,
            BLOCK_SHIFT,
            block_count as u32,
        ]) {
            *word = value.to_le_bytes();
        }

        let mut bytes = vec![0; BLOCK_BYTES];
        for ((data, size), word) in core::iter::once((layout.css, PAGE_BYTES))
            .chain(blocks.map(|data| (data, BLOCK_BYTES)))
            .zip(addresses.as_chunks_mut::<4>().0)
        {
            let buffer = &mut bytes[..size];
            buffer[..data.len()].copy_from_slice(data);
            buffer[data.len()..].fill(0);
            let chunk = Chunk::allocate_bounded(
                source,
                domain.clone(),
                buffer,
                Direction::Bidirectional,
                FH_DMA_LIMIT - 1,
            )?;
            let address = chunk.region().address;
            if address == 0 || !address.is_multiple_of(PAGE_BYTES as u64) {
                return Err(Error::Address);
            }
            let page = u32::try_from(address >> PAGE_SHIFT).map_err(|_| Error::Address)?;
            *word = page.to_le_bytes();
            chunks.push(chunk);
        }
        Ok(Self { chunks, payload })
    }

    pub(super) fn expose(&mut self) -> &[u8; PAYLOAD_BYTES] {
        for chunk in &mut self.chunks {
            chunk.expose();
        }
        &self.payload
    }

    /// # Safety
    /// This requester has stopped and its posted writes are drained.
    pub(super) unsafe fn release_after_stop(&mut self) -> Result<(), Error> {
        for chunk in &mut self.chunks {
            // SAFETY: the caller has proved requester-wide quiescence.
            unsafe { chunk.release().map_err(dma::Error::from)? };
        }
        Ok(())
    }
}

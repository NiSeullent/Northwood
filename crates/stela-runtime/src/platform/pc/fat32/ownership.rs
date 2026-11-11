// SPDX-License-Identifier: GPL-2.0-or-later
//! Verify that each allocated cluster belongs to exactly one live FAT entry.

use super::{
    ATTRIBUTE_DIRECTORY, ATTRIBUTE_LONG_NAME, ATTRIBUTE_VOLUME_ID, BAD_CLUSTER, BOOT_SIGNATURE,
    CLUSTER_MASK, END_OF_CHAIN, ENTRY_BYTES, ENTRY_END, ENTRY_FREE, ENTRY_KANJI_E5,
    FAT_ENTRY_BYTES, FAT_WINDOW, FIRST_CLUSTER, Fat32, FatError, ReadAt, Vec, format, le16, le32,
};

// The proof is optional on boot media. Refuse volumes whose metadata would
// require a large bitmap, too much directory I/O, or too many random FAT reads.
const MAX_CLUSTERS: u64 = 1 << 24;
const MAX_FAT_BYTES: u64 = 128 << 20;
const MAX_CHAIN_READ_BYTES: u64 = 64 << 20;
const MAX_DIRECTORY_BYTES: u64 = 2 << 20;
const MAX_DIRECTORY_READ_BYTES: u64 = 32 << 20;
const MAX_PENDING_DIRECTORIES: usize = 1 << 16;
const DOT: &[u8; 11] = b".          ";
const DOT_DOT: &[u8; 11] = b"..         ";

#[derive(Clone, Copy)]
struct Directory {
    first: u32,
    parent: Option<u32>,
}

struct Allocation<'a, R> {
    fat: &'a mut Fat32<R>,
    fat_bytes: u64,
    visited: Vec<u8>,
    pending: Vec<Directory>,
    chain_reads: u64,
    directory_reads: u64,
}

impl<R: ReadAt> Fat32<R> {
    /// Check full-cluster ownership using the live directory tree and all
    /// mirrored FATs. The caller must keep the volume's metadata and geometry
    /// stable throughout verification and until the resulting mapping is no
    /// longer used. This check performs no writes and conveys no device lease.
    pub(crate) fn verify_allocation(&mut self) -> Result<(), FatError> {
        let mut boot = [0u8; 512];
        self.reader.read_at(0, &mut boot).map_err(io)?;
        let sector_bytes = u64::from(le16(&boot, 11));
        let cluster_sectors = u64::from(boot[13]);
        let reserved = u64::from(le16(&boot, 14));
        let copies = u64::from(boot[16]);
        let sectors = u64::from(le32(&boot, 32));
        let fat_bytes = sector_bytes * u64::from(le32(&boot, 36));
        let data_offset = reserved * sector_bytes + copies * fat_bytes;
        if le16(&boot, 510) != BOOT_SIGNATURE
            || !matches!(sector_bytes, 512 | 1024 | 2048 | 4096)
            || !cluster_sectors.is_power_of_two()
            || cluster_sectors * sector_bytes != self.cluster_bytes
            || reserved == 0
            || !matches!(copies, 1 | 2)
            || le16(&boot, 17) != 0
            || le16(&boot, 19) != 0
            || le16(&boot, 22) != 0
            || le16(&boot, 40) != 0
            || le16(&boot, 42) != 0
            || le32(&boot, 44) != self.root
            || reserved * sector_bytes != self.fat_offset
            || data_offset != self.data_offset
            || sectors * sector_bytes > self.reader.len()
            || (sectors * sector_bytes)
                .checked_sub(data_offset)
                .map(|bytes| bytes / self.cluster_bytes)
                != Some(self.clusters)
            || !(0xf0..=0xff).contains(&boot[21])
            || fat_bytes < (self.clusters + 2) * FAT_ENTRY_BYTES
        {
            return Err(invalid("unsupported or changed FAT32 allocation geometry"));
        }
        if self.clusters > MAX_CLUSTERS || fat_bytes > MAX_FAT_BYTES {
            return Err(invalid(
                "FAT32 allocation verification exceeds its volume limit",
            ));
        }

        let backup = u64::from(le16(&boot, 50));
        if backup != 0 && backup != u64::from(u16::MAX) {
            if backup >= reserved {
                return Err(invalid(
                    "backup FAT32 boot sector is outside the reserved area",
                ));
            }
            let mut alternate = [0u8; 512];
            self.reader
                .read_at(backup * sector_bytes, &mut alternate)
                .map_err(io)?;
            if le16(&alternate, 510) != BOOT_SIGNATURE || alternate[11..52] != boot[11..52] {
                return Err(invalid("FAT32 boot sector and its backup disagree"));
            }
        }

        let bytes = (self.clusters as usize).div_ceil(8);
        let mut visited = Vec::new();
        visited
            .try_reserve_exact(bytes)
            .map_err(|_| invalid("cannot allocate FAT32 ownership bitmap"))?;
        visited.resize(bytes, 0);
        // Ignore reads cached by a previous path lookup.
        self.window_offset = None;
        let root = self.root;
        let mut allocation = Allocation {
            fat: self,
            fat_bytes,
            visited,
            pending: Vec::new(),
            chain_reads: MAX_CHAIN_READ_BYTES,
            directory_reads: MAX_DIRECTORY_READ_BYTES,
        };
        allocation.directory(Directory {
            first: root,
            parent: None,
        })?;
        while let Some(directory) = allocation.pending.pop() {
            allocation.directory(directory)?;
        }
        allocation.check_fats(copies, boot[21])
    }
}

fn invalid(reason: &'static str) -> FatError {
    FatError::NotFat32(reason)
}

fn io(error: impl core::fmt::Debug) -> FatError {
    FatError::Io(format!("{error:?}"))
}

impl<R: ReadAt> Allocation<'_, R> {
    /// Mark the whole cluster, even for the last, partially used file cluster.
    /// A second reference is either a cross-link or a cycle in a chain.
    fn visit(&mut self, cluster: u32) -> Result<Option<u32>, FatError> {
        if !self.fat.valid(cluster) {
            return Err(invalid("FAT32 chain leaves the data area"));
        }
        let index = (cluster - FIRST_CLUSTER) as usize;
        let bit = 1 << (index % 8);
        let word = &mut self.visited[index / 8];
        if *word & bit != 0 {
            return Err(invalid("FAT32 cluster is cross-linked or part of a loop"));
        }
        *word |= bit;

        let offset = u64::from(cluster) * FAT_ENTRY_BYTES;
        let base = offset - offset % FAT_WINDOW as u64;
        if self.fat.window_offset != Some(base) {
            let bytes = (self.fat_bytes - base).min(FAT_WINDOW as u64);
            self.chain_reads = self
                .chain_reads
                .checked_sub(bytes)
                .ok_or_else(|| invalid("FAT32 chain reads exceed the verification budget"))?;
            self.fat
                .reader
                .read_at(
                    self.fat.fat_offset + base,
                    &mut self.fat.window[..bytes as usize],
                )
                .map_err(io)?;
            self.fat.window_offset = Some(base);
        }
        let next = le32(&self.fat.window, (offset - base) as usize) & CLUSTER_MASK;
        if next >= END_OF_CHAIN {
            Ok(None)
        } else if self.fat.valid(next) {
            Ok(Some(next))
        } else {
            Err(invalid("FAT32 chain names a free, bad, or invalid cluster"))
        }
    }

    fn file(&mut self, mut cluster: u32, bytes: u32) -> Result<(), FatError> {
        if bytes == 0 {
            return if cluster == 0 {
                Ok(())
            } else {
                Err(invalid("empty FAT32 file has a chain"))
            };
        }
        let expected = u64::from(bytes).div_ceil(self.fat.cluster_bytes);
        for index in 1..=expected {
            match (self.visit(cluster)?, index == expected) {
                (None, true) => return Ok(()),
                (Some(next), false) => cluster = next,
                _ => return Err(invalid("FAT32 file chain does not match its size")),
            }
        }
        unreachable!("a nonempty file needs at least one cluster")
    }

    fn directory(&mut self, directory: Directory) -> Result<(), FatError> {
        let mut names = Vec::new();
        let mut dot = false;
        let mut dot_dot = false;
        let mut end = false;
        let mut cluster = directory.first;
        let mut bytes = 0;
        loop {
            let next = self.visit(cluster)?;
            bytes += self.fat.cluster_bytes;
            if bytes > MAX_DIRECTORY_BYTES {
                return Err(invalid("FAT32 directory exceeds its verification limit"));
            }
            if !end {
                let start = self.fat.cluster_offset(cluster);
                let sector_bytes = self.fat.directory_sector.len();
                for offset in (0..self.fat.cluster_bytes).step_by(sector_bytes) {
                    self.directory_reads = self
                        .directory_reads
                        .checked_sub(sector_bytes as u64)
                        .ok_or_else(|| {
                            invalid("FAT32 directory reads exceed the verification budget")
                        })?;
                    self.fat
                        .reader
                        .read_at(start + offset, &mut self.fat.directory_sector)
                        .map_err(io)?;
                    for at in (0..sector_bytes).step_by(ENTRY_BYTES) {
                        let entry = *self.fat.directory_sector[at..]
                            .first_chunk::<ENTRY_BYTES>()
                            .expect("whole directory entry");
                        if entry[0] == ENTRY_END {
                            end = true;
                            break;
                        }
                        if let Some(name) = self.entry(directory, &entry, &mut dot, &mut dot_dot)? {
                            names
                                .try_reserve(1)
                                .map_err(|_| invalid("cannot allocate FAT32 directory names"))?;
                            names.push(name);
                        }
                    }
                    if end {
                        break;
                    }
                }
            }
            let Some(next) = next else { break };
            cluster = next;
        }

        if directory.parent.is_some() && !(dot && dot_dot) {
            return Err(invalid("FAT32 subdirectory is missing its dot entries"));
        }
        names.sort_unstable();
        if names.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid("FAT32 directory has duplicate short names"));
        }
        Ok(())
    }

    fn entry(
        &mut self,
        directory: Directory,
        entry: &[u8; ENTRY_BYTES],
        dot: &mut bool,
        dot_dot: &mut bool,
    ) -> Result<Option<[u8; 11]>, FatError> {
        if entry[0] == ENTRY_FREE {
            return Ok(None);
        }
        let attributes = entry[11];
        if attributes == ATTRIBUTE_LONG_NAME {
            // A VFAT continuation has no independent allocation. Its linked
            // short entry below is always audited, regardless of its name.
            if entry[12] != 0 || le16(entry, 26) != 0 {
                return Err(invalid("FAT32 long name contains allocation fields"));
            }
            return Ok(None);
        }
        let first = u32::from(le16(entry, 20)) << 16 | u32::from(le16(entry, 26));
        let bytes = le32(entry, 28);
        if attributes & ATTRIBUTE_VOLUME_ID != 0 {
            if attributes != ATTRIBUTE_VOLUME_ID
                || directory.parent.is_some()
                || first != 0
                || bytes != 0
            {
                return Err(invalid("malformed FAT32 volume label entry"));
            }
            return Ok(None);
        }
        let name = entry.first_chunk::<11>().expect("FAT32 short name");
        let is_directory = attributes & ATTRIBUTE_DIRECTORY != 0;
        if attributes & 0xc0 != 0 || (is_directory && bytes != 0) {
            return Err(invalid("malformed FAT32 directory entry"));
        }
        if name == DOT || name == DOT_DOT {
            let parent = directory
                .parent
                .ok_or_else(|| invalid("FAT32 root has dot entries"))?;
            let (seen, valid) = if name == DOT {
                (dot, first == directory.first)
            } else {
                (
                    dot_dot,
                    first == parent || (parent == self.fat.root && first == 0),
                )
            };
            if !is_directory || *seen || !valid {
                return Err(invalid("malformed FAT32 dot entry"));
            }
            *seen = true;
            return Ok(None);
        }
        if is_directory {
            if !self.fat.valid(first) {
                return Err(invalid("FAT32 directory has an invalid start cluster"));
            }
            if self.pending.len() >= MAX_PENDING_DIRECTORIES {
                return Err(invalid("too many pending FAT32 directories"));
            }
            self.pending
                .try_reserve(1)
                .map_err(|_| invalid("cannot allocate FAT32 directory stack"))?;
            self.pending.push(Directory {
                first,
                parent: Some(directory.first),
            });
        } else {
            self.file(first, bytes)?;
        }
        let mut normalized = *name;
        normalized.make_ascii_uppercase();
        if normalized[0] == ENTRY_KANJI_E5 {
            normalized[0] = ENTRY_FREE;
        }
        Ok(Some(normalized))
    }

    fn check_fats(&mut self, copies: u64, media: u8) -> Result<(), FatError> {
        let mut primary = [0u8; FAT_WINDOW];
        let mut mirror = [0u8; FAT_WINDOW];
        for offset in (0..self.fat_bytes).step_by(FAT_WINDOW) {
            let bytes = (self.fat_bytes - offset).min(FAT_WINDOW as u64) as usize;
            let at = self.fat.fat_offset + offset;
            self.fat
                .reader
                .read_at(at, &mut primary[..bytes])
                .map_err(io)?;
            if copies == 2 {
                self.fat
                    .reader
                    .read_at(at + self.fat_bytes, &mut mirror[..bytes])
                    .map_err(io)?;
                if primary[..bytes] != mirror[..bytes] {
                    return Err(invalid("FAT32 mirror copies disagree"));
                }
            }
            let first = offset / FAT_ENTRY_BYTES;
            let end = (self.fat.clusters + 2)
                .saturating_sub(first)
                .min(bytes as u64 / FAT_ENTRY_BYTES);
            for index in 0..end {
                let value = le32(&primary, (index * FAT_ENTRY_BYTES) as usize) & CLUSTER_MASK;
                match first + index {
                    0 if value != (0x0fff_ff00 | u32::from(media)) => {
                        return Err(invalid("FAT32 media descriptor disagrees with the FAT"));
                    }
                    // The two high bits may carry clean shutdown and I/O status.
                    1 if value & 0x03ff_ffff != 0x03ff_ffff => {
                        return Err(invalid("malformed FAT32 reserved entry"));
                    }
                    0 | 1 => {}
                    cluster => {
                        let index = (cluster - u64::from(FIRST_CLUSTER)) as usize;
                        let visited = self.visited[index / 8] & (1 << (index % 8)) != 0;
                        if !visited && value != 0 && value != BAD_CLUSTER {
                            return Err(invalid(
                                "FAT32 has an allocated cluster without a directory owner",
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

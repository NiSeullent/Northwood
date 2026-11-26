// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from Tyler Smith's Stambha commit 28d1adaf0.

//! Classify UEFI handles using the `FileInfo` directory attribute. Some
//! firmware answers `GetPosition` with success for directories.
use stela_boot_core::file_kind::{self, Answers, Kind, Probe};
use uefi::{
    Status,
    proto::media::file::{Directory, File as _, FileHandle, FileInfo, RegularFile},
};

struct Handle<'a>(&'a mut FileHandle);

// EFI_FILE_INFO requires 8-byte alignment. This buffer holds FAT filenames
// without allocating; a filesystem with longer names can request more.
#[repr(align(8))]
struct InfoBuffer([u8; 1024]);

impl Probe for Handle<'_> {
    fn attribute(&mut self) -> Result<u64, Status> {
        let mut buffer = InfoBuffer([0; 1024]);
        match self.0.get_info::<FileInfo>(&mut buffer.0) {
            Ok(info) => Ok(info.attribute().bits()),
            Err(error) if error.status() == Status::BUFFER_TOO_SMALL => self
                .0
                .get_boxed_info::<FileInfo>()
                .map(|info| info.attribute().bits())
                .map_err(|error| error.status()),
            Err(error) => Err(error.status()),
        }
    }

    fn position(&mut self) -> Status {
        match self.0.is_regular_file() {
            Ok(true) => Status::SUCCESS,
            Ok(false) => Status::UNSUPPORTED,
            Err(error) => error.status(),
        }
    }
}

/// `handle` as a directory, else what it answered.
pub fn directory(mut handle: FileHandle) -> Result<Directory, Answers> {
    file_kind::expect(&mut Handle(&mut handle), Kind::Directory)?;
    // SAFETY: the handle's FileInfo (or GetPosition) says it is a directory.
    Ok(unsafe { Directory::new(handle) })
}

/// `handle` as a regular file, else what it answered.
pub fn regular(mut handle: FileHandle) -> Result<RegularFile, Answers> {
    file_kind::expect(&mut Handle(&mut handle), Kind::Regular)?;
    // SAFETY: the handle's FileInfo (or GetPosition) says it is a file.
    Ok(unsafe { RegularFile::new(handle) })
}

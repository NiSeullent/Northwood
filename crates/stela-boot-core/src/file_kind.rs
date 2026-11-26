// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from Tyler Smith's Stambha commit 28d1adaf0.

//! Whether a handle the firmware's file system opened is a directory or a
//! regular file. The answer is the `EFI_FILE_DIRECTORY` bit of the handle's
//! `EFI_FILE_INFO.Attribute` (UEFI 2.10 13.5.16), which every file system
//! driver must report. uefi-rs instead asks `GetPosition`, which the
//! specification has fail with `EFI_UNSUPPORTED` on a directory, and takes
//! any other answer for "neither": a FAT driver that answers a directory's
//! `GetPosition` with success or another error made Stela take a volume's
//! root for a file. `GetPosition` decides only when `GetInfo` fails.
use core::fmt;
use uefi_raw::Status;

/// `EFI_FILE_DIRECTORY` in `EFI_FILE_INFO.Attribute`.
pub const DIRECTORY: u64 = 0x10;

/// The two questions asked of an open file handle.
pub trait Probe {
    /// `EFI_FILE_INFO.Attribute`, or the status `GetInfo` answered.
    fn attribute(&mut self) -> Result<u64, Status>;
    /// The status `GetPosition` answered.
    fn position(&mut self) -> Status;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Directory,
    Regular,
}

/// What the handle answered, for a refusal's words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Answers {
    /// `EFI_FILE_INFO.Attribute`, or `GetInfo`'s failure.
    pub info: Result<u64, Status>,
    /// `GetPosition`'s status, asked only when it decided or on a refusal.
    pub position: Option<Status>,
}

impl fmt::Display for Answers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.info {
            Ok(attribute) => write!(f, "attribute={attribute:#x}")?,
            Err(status) => write!(f, "GetInfo={status:?}")?,
        }
        match self.position {
            Some(status) => write!(f, " GetPosition={status:?}"),
            None => Ok(()),
        }
    }
}

/// The handle's kind, or `None` when neither `GetInfo` nor `GetPosition`
/// tells, with what it answered.
pub fn classify(file: &mut impl Probe) -> (Option<Kind>, Answers) {
    match file.attribute() {
        Ok(attribute) => (
            Some(if attribute & DIRECTORY != 0 {
                Kind::Directory
            } else {
                Kind::Regular
            }),
            Answers {
                info: Ok(attribute),
                position: None,
            },
        ),
        Err(info) => {
            let position = file.position();
            let kind = match position {
                Status::SUCCESS => Some(Kind::Regular),
                Status::UNSUPPORTED => Some(Kind::Directory),
                _ => None,
            };
            (
                kind,
                Answers {
                    info: Err(info),
                    position: Some(position),
                },
            )
        }
    }
}

/// `Ok` when the handle is of kind `wanted`, else what it answered, with
/// `GetPosition`'s status asked too.
pub fn expect(file: &mut impl Probe, wanted: Kind) -> Result<(), Answers> {
    match classify(file) {
        (Some(kind), _) if kind == wanted => Ok(()),
        (_, mut answers) => {
            answers.position.get_or_insert_with(|| file.position());
            Err(answers)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file protocol whose answers are set per handle.
    struct Mock {
        info: Result<u64, Status>,
        position: Status,
        asked_position: bool,
    }

    impl Probe for Mock {
        fn attribute(&mut self) -> Result<u64, Status> {
            self.info
        }
        fn position(&mut self) -> Status {
            self.asked_position = true;
            self.position
        }
    }

    fn mock(info: Result<u64, Status>, position: Status) -> Mock {
        Mock {
            info,
            position,
            asked_position: false,
        }
    }

    #[test]
    fn a_directory_whose_get_position_succeeds_is_a_directory() {
        // The FAT driver that lost every host firmware file: a volume root
        // answers GetPosition with success, its FileInfo says directory.
        let mut root = mock(Ok(DIRECTORY), Status::SUCCESS);
        assert_eq!(expect(&mut root, Kind::Directory), Ok(()));
        assert!(!root.asked_position);
        // A driver that answers a directory's GetPosition with another error.
        let mut root = mock(Ok(DIRECTORY | 0x1), Status::DEVICE_ERROR);
        assert_eq!(classify(&mut root).0, Some(Kind::Directory));
    }

    #[test]
    fn a_regular_file_is_regular_whatever_get_position_says() {
        let mut file = mock(Ok(0x20), Status::SUCCESS);
        assert_eq!(expect(&mut file, Kind::Regular), Ok(()));
        let mut file = mock(Ok(0x1), Status::UNSUPPORTED);
        assert_eq!(classify(&mut file).0, Some(Kind::Regular));
    }

    #[test]
    fn get_position_decides_only_when_get_info_fails() {
        let mut directory = mock(Err(Status::UNSUPPORTED), Status::UNSUPPORTED);
        assert_eq!(classify(&mut directory).0, Some(Kind::Directory));
        let mut file = mock(Err(Status::DEVICE_ERROR), Status::SUCCESS);
        assert_eq!(classify(&mut file).0, Some(Kind::Regular));
        let mut neither = mock(Err(Status::DEVICE_ERROR), Status::NO_MEDIA);
        assert_eq!(
            classify(&mut neither),
            (
                None,
                Answers {
                    info: Err(Status::DEVICE_ERROR),
                    position: Some(Status::NO_MEDIA)
                }
            )
        );
    }

    #[test]
    fn a_mismatch_names_both_answers() {
        let mut file = mock(Ok(0x20), Status::SUCCESS);
        let answers = expect(&mut file, Kind::Directory).unwrap_err();
        assert_eq!(answers.to_string(), "attribute=0x20 GetPosition=SUCCESS");
        let mut broken = mock(Err(Status::DEVICE_ERROR), Status::NO_MEDIA);
        assert_eq!(
            expect(&mut broken, Kind::Regular).unwrap_err().to_string(),
            "GetInfo=DEVICE_ERROR GetPosition=NO_MEDIA"
        );
    }
}

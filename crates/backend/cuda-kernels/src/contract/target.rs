//! Hardware identity and the kernel profile compiled into this binary.

/// Compute capability of the device the kernels run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    Sm70,
    Sm75,
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm100,
    Sm103,
    Sm120,
    Other { major: u32, minor: u32 },
}

impl Arch {
    /// Map a driver-reported compute capability to a known profile.
    #[must_use]
    pub const fn from_sm(major: u32, minor: u32) -> Self {
        match (major, minor) {
            (7, 0) => Self::Sm70,
            (7, 5) => Self::Sm75,
            (8, 0) => Self::Sm80,
            (8, 6) => Self::Sm86,
            (8, 9) => Self::Sm89,
            (9, 0) => Self::Sm90,
            (10, 0) => Self::Sm100,
            (10, 3) => Self::Sm103,
            (12, 0) => Self::Sm120,
            _ => Self::Other { major, minor },
        }
    }

    /// `sm_<major><minor>` spelling used by artifacts and capability reports.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Sm70 => "sm_70".to_owned(),
            Self::Sm75 => "sm_75".to_owned(),
            Self::Sm80 => "sm_80".to_owned(),
            Self::Sm86 => "sm_86".to_owned(),
            Self::Sm89 => "sm_89".to_owned(),
            Self::Sm90 => "sm_90".to_owned(),
            Self::Sm100 => "sm_100".to_owned(),
            Self::Sm103 => "sm_103".to_owned(),
            Self::Sm120 => "sm_120".to_owned(),
            Self::Other { major, minor } => format!("sm_{major}{minor}"),
        }
    }
}

/// Kernel build profile compiled into this binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// cuTile kernels compiled by the CUDA 13.x toolchain.
    Tile,
    /// Precompiled SASS loaded through the driver; no CUDA toolkit at build time.
    Aot,
}

/// Device and profile pair resolved once at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub arch: Arch,
    pub profile: Profile,
}

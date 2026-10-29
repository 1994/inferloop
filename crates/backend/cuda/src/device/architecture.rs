//! CUDA architecture identities reported by the device boundary.

pub const SM120_NAME: &str = "sm_120";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaArchitecture {
    Sm90,
    Sm100,
    Sm103,
    Sm120,
    Other,
}

impl CudaArchitecture {
    pub fn from_name(name: &str) -> Self {
        match name {
            "sm_90" => Self::Sm90,
            "sm_100" => Self::Sm100,
            "sm_103" => Self::Sm103,
            SM120_NAME => Self::Sm120,
            _ => Self::Other,
        }
    }
}

pub fn is_sm120(name: &str) -> bool {
    name == SM120_NAME
}

#[cfg(test)]
#[path = "../../tests/unit/device_architecture.rs"]
mod tests;

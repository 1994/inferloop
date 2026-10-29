//! Load-only nvJPEG diagnostic. This module does not create CUDA contexts or decode images.
use super::library_names::NVJPEG;
use infer_core::{Error, ErrorCode, Result};
use libloading::Library;

#[cfg(test)]
#[path = "../../tests/unit/nvjpeg_probe.rs"]
mod tests;

/// ABI major version expected by the CUDA backend's nvJPEG diagnostic.
const SUPPORTED_ABI_MAJOR: u32 = 13;
/// nvJPEG's successful C status value.
const NVJPEG_SUCCESS: i32 = 0;
/// CUDA library property selector values defined by `library_types.h`.
const MAJOR_VERSION: i32 = 0;
const MINOR_VERSION: i32 = 1;
const PATCH_LEVEL: i32 = 2;
/// Major, minor and patch components returned by the version property queries.
const VERSION_COMPONENTS: usize = 3;

type VersionQuery = unsafe extern "C" fn(i32, *mut i32) -> i32;

/// An nvJPEG or build-time CUDA runtime version reported by the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

/// Library and build-time CUDA runtime versions, not a device capability report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LibraryInfo {
    /// Version of the loaded nvJPEG library.
    nvjpeg: Version,
    /// CUDA runtime version against which nvJPEG was built, not the installed driver version.
    cudart: Version,
}

struct Api {
    _library: Library,
    nvjpeg: VersionQuery,
    cudart: VersionQuery,
}

/// Dynamically query the version of an installed nvJPEG 13 library.
///
/// This only resolves and calls nvJPEG's documented version queries; it does not open a CUDA
/// device, determine decoder support, bind a stream, allocate device buffers, or decode images.
/// The CUDA runtime version reported is the one used to build nvJPEG. This diagnostic is not a
/// serving capability check.
///
/// # Errors
/// Returns unsupported when the library, required symbols, or supported ABI are unavailable;
/// returns a backend error when a version query fails or returns an invalid version.
fn library_info() -> Result<LibraryInfo> {
    Api::load()?.info()
}

impl Api {
    fn load() -> Result<Self> {
        Self::load_names(&NVJPEG)
    }

    #[expect(
        unsafe_code,
        reason = "The dynamic library and documented nvJPEG version-query symbols are held together"
    )]
    fn load_names(names: &[&str]) -> Result<Self> {
        let mut loaded = None;
        let mut errors = Vec::with_capacity(names.len());
        for name in names {
            // SAFETY: loading an installed library by soname; the handle remains owned by Api.
            match unsafe { Library::new(name) } {
                Ok(library) => {
                    loaded = Some(library);
                    break;
                }
                Err(error) => errors.push(format!("{name}: {error}")),
            }
        }
        let library = loaded.ok_or_else(|| {
            Error::unsupported(format!(
                "nvJPEG library is unavailable: {}",
                errors.join("; ")
            ))
        })?;
        // SAFETY: `nvjpeg.h` declares this function as
        // nvjpegStatus_t(libraryPropertyType, int *); it is used while the library lives.
        let nvjpeg = *unsafe { library.get::<VersionQuery>(b"nvjpegGetProperty\0") }
            .map_err(|error| Error::unsupported(format!("nvJPEG version query symbol: {error}")))?;
        // SAFETY: the CUDA runtime property query has the same documented signature, and the
        // owning library remains live while this function pointer is used.
        let cudart = *unsafe { library.get::<VersionQuery>(b"nvjpegGetCudartProperty\0") }
            .map_err(|error| {
                Error::unsupported(format!("nvJPEG CUDA version query symbol: {error}"))
            })?;
        Ok(Self {
            _library: library,
            nvjpeg,
            cudart,
        })
    }

    fn info(&self) -> Result<LibraryInfo> {
        version_info(self.nvjpeg, self.cudart)
    }
}

fn version_info(nvjpeg: VersionQuery, cudart: VersionQuery) -> Result<LibraryInfo> {
    let nvjpeg = version(nvjpeg, "nvJPEG")?;
    if nvjpeg.major != SUPPORTED_ABI_MAJOR {
        return Err(Error::unsupported(format!(
            "nvJPEG ABI major {} is not supported; expected {SUPPORTED_ABI_MAJOR}",
            nvjpeg.major
        )));
    }
    Ok(LibraryInfo {
        nvjpeg,
        cudart: version(cudart, "nvJPEG build CUDA runtime")?,
    })
}

#[expect(
    unsafe_code,
    reason = "The documented query writes one i32 to a live, uniquely borrowed local value"
)]
fn version(query: VersionQuery, library: &str) -> Result<Version> {
    let mut values = [0_i32; VERSION_COMPONENTS];
    for (value, property) in values
        .iter_mut()
        .zip([MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL])
    {
        // SAFETY: the function pointer has the exact ABI from nvjpeg.h, and value points to a
        // live, initialized, exclusive i32 slot for the duration of this synchronous call.
        let status = unsafe { query(property, value) };
        if status != NVJPEG_SUCCESS {
            return Err(Error::new(
                ErrorCode::Backend,
                format!("{library} version query failed with status {status}"),
            ));
        }
    }
    let [major, minor, patch] = values;
    let invalid = || {
        Error::new(
            ErrorCode::Backend,
            format!("{library} returned invalid version"),
        )
    };
    let major = u32::try_from(major).map_err(|_| invalid())?;
    if major == 0 {
        return Err(invalid());
    }
    Ok(Version {
        major,
        minor: u32::try_from(minor).map_err(|_| invalid())?,
        patch: u32::try_from(patch).map_err(|_| invalid())?,
    })
}

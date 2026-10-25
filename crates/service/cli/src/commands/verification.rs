//! Verification commands.
#[cfg(native_backend)]
use super::host_quality;
use super::{VerifyOptions, backend, print, read_json};
use infer_core::{Error, Result};

pub fn verify(options: VerifyOptions, backend_choice: &backend::Selection) -> Result<()> {
    let VerifyOptions {
        reference,
        candidate,
        atol,
        rtol,
        package,
        golden,
        host_memory_mib,
    } = options;
    if package.is_none()
        && let (Some(a), Some(b)) = (&reference, &candidate)
    {
        let report = infer_quality::compare(
            &read_json::<Vec<f32>>(a)?,
            &read_json::<Vec<f32>>(b)?,
            atol,
            rtol,
        )?;
        print(&report)?;
        return if report.passed {
            Ok(())
        } else {
            Err(Error::invariant("verification failed"))
        };
    }
    #[cfg(native_backend)]
    {
        if let Some(package) = package {
            let report = host_quality::verify_package(
                &package,
                golden
                    .as_deref()
                    .ok_or_else(|| Error::invariant("validated argument"))?,
                host_memory_mib,
                atol,
                rtol,
                backend_choice,
            )?;
            print(&report)?;
            return if report["passed"] == true {
                Ok(())
            } else {
                Err(Error::invariant("package golden verification failed"))
            };
        }
        {
            Err(Error::invalid(
                "device backend verification requires --package and --golden",
            ))
        }
    }
    #[cfg(not(native_backend))]
    {
        if package.is_none() {
            return Err(Error::invalid(
                "device backend verification requires --package and --golden",
            ));
        }
        let _ = (golden, host_memory_mib, backend_choice);
        Err(Error::unsupported(
            "GPU verification requires a supported device backend",
        ))
    }
}

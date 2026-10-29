use crate::device::CudaDevice;
use infer_core::{DeviceId, Result};
use infer_ir::{
    DeviceBackend, DeviceCapabilities, NvidiaArchitecture, NvidiaCapabilities,
    SpeculationCapability,
};

pub(super) fn query(device: &CudaDevice) -> Result<DeviceCapabilities> {
    let capability = device.compute_capability()?;
    let major = capability.major;
    let minor = capability.minor;
    Ok(DeviceCapabilities {
        backend: DeviceBackend::Cuda(NvidiaCapabilities {
            architecture: NvidiaArchitecture {
                compute_major: major,
                compute_minor: minor,
                multiprocessors: device.multiprocessor_count()?,
            },
            tensor_core_generation: capability.tensor_core_generation(),
            warp_size: device.warp_size()?,
            graphs: true,
            // Advertise only features exposed by this provider, not untested hardware paths.
            tma: false,
            clusters: false,
            pinned_transfer: false,
            cuda_ipc: false,
            nvlink: false,
            gpu_direct: false,
        }),
        device: DeviceId::new(device.stream.device().ordinal() as u64 + 1)?,
        compute_dtypes: device.target().compute_dtypes().to_vec(),
        memory_bytes: device.memory_info()?.1,
        unified_memory: false,
        profiling: false,

        speculation: SpeculationCapability::default(),
    })
}

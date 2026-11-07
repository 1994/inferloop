//! Whether a device address is memory this device's context may hand a vendor library.
#![expect(
    unsafe_code,
    reason = "Read-only driver pointer queries into live stack slots of each attribute's cuda.h type"
)]
use super::Device;
use crate::tile::backend_error;
use cuda_core::sys::{
    CUcontext, CUmemoryPool, CUmemorytype, CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
    CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_CONTEXT,
    CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
    CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, cuCtxGetCurrent,
    cuPointerGetAttribute, cuStreamGetCtx, cudaError_enum_CUDA_SUCCESS,
};
use infer_core::Result;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::result::Result as StdResult;

impl Device {
    /// The context that owns this device's stream, as an opaque identity.
    /// # Errors
    /// Returns the driver's failure to report it.
    pub fn context_identity(&self) -> Result<usize> {
        let mut context: CUcontext = null_mut();
        // SAFETY: the stream is live and `context` is one live CUcontext-sized slot.
        let status = unsafe { cuStreamGetCtx(self.stream().cu_stream(), &raw mut context) };
        if status == cudaError_enum_CUDA_SUCCESS {
            Ok(context as usize)
        } else {
            Err(backend_error(format!(
                "stream context CUDA status {status}"
            )))
        }
    }

    /// Accept `address` when it was allocated in this device's stream context, or is a
    /// stream-ordered pool allocation on this device visible through that context.
    ///
    /// The CUDA programming guide (4.3.5.4) specifies that stream-ordered pool allocations answer
    /// the context query successfully with a null owner. That kind is admitted only after
    /// proving a device allocation on this ordinal, through the current context, in a real
    /// pool: a null owner alone grants nothing.
    /// # Errors
    /// Returns, as text, what the driver reported about the address.
    pub fn check_operand(&self, address: u64) -> StdResult<(), String> {
        let primary = self.context_identity().map_err(|error| error.to_string())?;
        let mut owner: CUcontext = null_mut();
        // SAFETY: CONTEXT writes one CUcontext into this live slot.
        let code = unsafe {
            cuPointerGetAttribute(
                (&raw mut owner).cast::<c_void>(),
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_CONTEXT,
                address,
            )
        };
        if code == cudaError_enum_CUDA_SUCCESS
            && (owner as usize == primary
                || (owner.is_null() && self.pool_allocation(primary, address)))
        {
            return Ok(());
        }
        Err(format!(
            "context_status={code}, owner={owner:p}, expected=0x{primary:x}, {}",
            self.diagnostics(address)
        ))
    }

    fn pool_allocation(&self, primary: usize, address: u64) -> bool {
        let mut current: CUcontext = null_mut();
        // SAFETY: `current` is one live CUcontext-sized slot.
        if unsafe { cuCtxGetCurrent(&raw mut current) } != cudaError_enum_CUDA_SUCCESS
            || current as usize != primary
        {
            return false;
        }
        let mut ordinal = -1_i32;
        let mut memory_type: CUmemorytype = 0;
        let mut accessible: u64 = 0;
        let mut pool: CUmemoryPool = null_mut();
        for (attribute, destination) in [
            (
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                (&raw mut ordinal).cast::<c_void>(),
            ),
            (
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                (&raw mut memory_type).cast::<c_void>(),
            ),
            (
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                (&raw mut accessible).cast::<c_void>(),
            ),
            (
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                (&raw mut pool).cast::<c_void>(),
            ),
        ] {
            // SAFETY: each destination matches its attribute's signed-int, enum, u64 or pool type.
            if unsafe { cuPointerGetAttribute(destination, attribute, address) }
                != cudaError_enum_CUDA_SUCCESS
            {
                return false;
            }
        }
        usize::try_from(ordinal).is_ok_and(|ordinal| ordinal == self.ordinal())
            && memory_type == CUmemorytype_enum_CU_MEMORYTYPE_DEVICE
            && accessible == address
            && !pool.is_null()
    }

    /// What the driver reports about `address`, for failure messages only.
    fn diagnostics(&self, address: u64) -> String {
        let mut current: CUcontext = null_mut();
        // SAFETY: `current` is one live CUcontext-sized slot.
        let current_status = unsafe { cuCtxGetCurrent(&raw mut current) };
        let mut ordinal = -1_i32;
        // SAFETY: DEVICE_ORDINAL writes one signed int.
        let ordinal_status = unsafe {
            cuPointerGetAttribute(
                (&raw mut ordinal).cast::<c_void>(),
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                address,
            )
        };
        let mut memory_type: CUmemorytype = 0;
        // SAFETY: MEMORY_TYPE writes one CUmemorytype enum.
        let memory_status = unsafe {
            cuPointerGetAttribute(
                (&raw mut memory_type).cast::<c_void>(),
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                address,
            )
        };
        let mut accessible: u64 = 0;
        // SAFETY: DEVICE_POINTER writes one u64, valid for the current context.
        let accessible_status = unsafe {
            cuPointerGetAttribute(
                (&raw mut accessible).cast::<c_void>(),
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                address,
            )
        };
        let mut pool: CUmemoryPool = null_mut();
        // SAFETY: MEMPOOL_HANDLE writes one CUmemoryPool opaque pointer.
        let pool_status = unsafe {
            cuPointerGetAttribute(
                (&raw mut pool).cast::<c_void>(),
                CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                address,
            )
        };
        format!(
            "pointer=0x{address:x}, stream={:p}, device_context={:p}, current_status={current_status}, current={current:p}, ordinal_status={ordinal_status}, ordinal={ordinal}, memory_status={memory_status}, memory_type={memory_type}, accessible_status={accessible_status}, accessible=0x{accessible:x}, pool_status={pool_status}, pool={pool:p}",
            self.stream().cu_stream(),
            self.stream().device().cu_ctx()
        )
    }
}

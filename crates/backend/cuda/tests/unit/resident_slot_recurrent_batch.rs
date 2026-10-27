//! Bitwise comparison against the existing per-slot fused kernel at the checkpoint geometry.
use super::{upload, values};
use std::error::Error;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;
use crate::{
    device::CudaDevice,
    resident::slot_recurrent::{ValueDevice, fused, value_split},
};
use cutile::prelude::*;
use std::slice::from_ref;

const KH: usize = 16;
const VH: usize = 48;
const D: usize = 128;
const QKV: usize = (2 * KH + VH) * D;

struct Fixture {
    block: usize,
    state: Tensor<f32>,
    qkv: Tensor<f32>,
    beta: Tensor<f32>,
    alpha: Tensor<f32>,
    rate: Tensor<f32>,
    bias: Tensor<f32>,
    parameters: Tensor<f32>,
    metadata: Arc<Tensor<i32>>,
    original: Vec<Tensor<f32>>,
    candidate: Tensor<f32>,
}

impl Fixture {
    fn new(device: &CudaDevice, slots: usize) -> TestResult<Self> {
        let lanes = slots * 3;
        let metadata = (0..lanes)
            .flat_map(|lane| {
                let slot = lane / 3;
                let active = lane % 3 < slot % 4;
                let position = if active {
                    i32::try_from(lane).unwrap()
                } else {
                    -1
                };
                [position, 0, position, 0]
            })
            .collect();
        let original = (0..lanes)
            .map(|_| upload(device, vec![0.0; VH * D], &[VH, 1, D]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        Ok(Self {
            block: D / value_split(ValueDevice::new(device)?, VH, D, slots)?,
            state: upload(
                device,
                values(slots * VH * D * D, 23),
                &[slots * VH * D * D],
            )?,
            qkv: upload(device, values(lanes * QKV, 17), &[lanes * QKV])?,
            beta: upload(device, values(lanes * VH, 13), &[lanes * VH])?,
            alpha: upload(device, values(lanes * VH, 7), &[lanes * VH])?,
            rate: upload(device, values(VH, 3), &[VH])?,
            bias: upload(device, values(VH, 5), &[VH])?,
            parameters: upload(device, vec![0.0; 2 * VH], &[2, VH])?,
            metadata: device.upload(metadata, &[lanes * 4])?,
            original,
            candidate: upload(device, vec![0.0; lanes * VH * D], &[slots, 3, VH * D])?,
        })
    }

    fn record_original(&mut self, scope: &Scope) -> Result<(), DeviceError> {
        for (slot, output) in self.original.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let [o0, o1, o2] = output;
            let qkv = lane_rows(&self.qkv, slot, QKV)?;
            let beta = lane_rows(&self.beta, slot, VH)?;
            let alpha = lane_rows(&self.alpha, slot, VH)?;
            let metadata = lane_rows(&self.metadata, slot, 4)?;
            let state_range = slot * VH * D * D..(slot + 1) * VH * D * D;
            let state = self.state.slice(from_ref(&state_range))?;
            scope.record(
                fused::delta_verify3(
                    o0.partition([1, 1, self.block]),
                    o1.partition([1, 1, self.block]),
                    o2.partition([1, 1, self.block]),
                    &state.view(&[VH, D, D])?,
                    &qkv[0],
                    &qkv[1],
                    &qkv[2],
                    &beta[0],
                    &beta[1],
                    &beta[2],
                    &alpha[0],
                    &alpha[1],
                    &alpha[2],
                    &self.rate,
                    &self.bias,
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                )
                .generics(
                    [KH, VH, D, self.block]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
        }
        Ok(())
    }

    fn prepare_parameters(&mut self, device: &CudaDevice) -> TestResult {
        fused::pack_parameters(
            (&mut self.parameters).partition([2, 1]),
            &self.rate,
            &self.bias,
        )
        .generics(vec![VH.to_string()])
        .sync_on(&device.stream)?;
        let mut expected_parameters = values(VH, 3);
        expected_parameters.extend(values(VH, 5));
        assert_eq!(device.read_borrowed(&self.parameters)?, expected_parameters);
        Ok(())
    }

    fn compare(&mut self, device: &CudaDevice, slots: usize) -> TestResult {
        self.prepare_parameters(device)?;
        let original = CudaGraph::scope(&device.stream, |scope| self.record_original(scope))?;
        let candidate = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                fused::delta_verify_batch(
                    (&mut self.candidate).partition([1, 3, self.block]),
                    &self.state.view(&[slots * VH, D, D])?,
                    &self.qkv,
                    &self.beta,
                    &self.alpha,
                    &self.parameters.view(&[2 * VH])?,
                    &self.metadata,
                )
                .generics(
                    [KH, VH, D, self.block, 4]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
            Ok(())
        })?;
        original.launch().sync_on(&device.stream)?;
        candidate.launch().sync_on(&device.stream)?;
        let actual = device.read_borrowed(&self.candidate)?;
        for (lane, reference) in self.original.iter().enumerate() {
            let expected = device.read_borrowed(reference)?;
            for (column, (&a, &b)) in actual[lane * VH * D..(lane + 1) * VH * D]
                .iter()
                .zip(&expected)
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "slots={slots} lane={lane} column={column}: {a} != {b}"
                );
            }
        }
        let actual_state = device.read_borrowed(&self.state)?;
        assert_eq!(
            actual_state,
            values(slots * VH * D * D, 23),
            "verification mutated the state"
        );
        Ok(())
    }

    fn record_original_commit(
        &self,
        scope: &Scope,
        states: &mut [Tensor<f32>],
        counts: &[i32],
    ) -> Result<(), DeviceError> {
        for (slot, state) in states.iter_mut().enumerate() {
            let qkv = lane_rows(&self.qkv, slot, QKV)?;
            let beta = lane_rows(&self.beta, slot, VH)?;
            let alpha = lane_rows(&self.alpha, slot, VH)?;
            let metadata = lane_rows(&self.metadata, slot, 4)?;
            scope.record(
                fused::delta_commit3(
                    state.partition([1, D, self.block]),
                    &qkv[0],
                    &qkv[1],
                    &qkv[2],
                    &beta[0],
                    &beta[1],
                    &beta[2],
                    &alpha[0],
                    &alpha[1],
                    &alpha[2],
                    &self.rate,
                    &self.bias,
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                    counts[slot],
                )
                .generics(
                    [KH, VH, D, self.block]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
        }
        Ok(())
    }

    fn compare_commits(&mut self, device: &CudaDevice, slots: usize) -> TestResult {
        self.prepare_parameters(device)?;
        let initial = device.read_borrowed(&self.state)?;
        let size = VH * D * D;
        let mut original = (0..slots)
            .map(|slot| {
                upload(
                    device,
                    initial[slot * size..(slot + 1) * size].to_vec(),
                    &[VH, D, D],
                )
            })
            .collect::<infer_core::Result<Vec<_>>>()?;
        let mut candidate = upload(device, initial, &[slots, VH * D, D])?;
        let mut counts = api::zeros::<i32>(&[slots]).sync_on(&device.stream)?;
        for shift in 0..4 {
            let accepted = (0..slots)
                .map(|slot| i32::try_from((slot + shift) % 4).unwrap())
                .collect::<Vec<_>>();
            device.copy_host_slice(&mut counts, &accepted)?;
            let original_graph = CudaGraph::scope(&device.stream, |scope| {
                self.record_original_commit(scope, &mut original, &accepted)
            })?;
            let candidate_graph = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    fused::delta_commit_batch(
                        (&mut candidate).partition([1, D, self.block]),
                        &self.qkv,
                        &self.beta,
                        &self.alpha,
                        &self.parameters.view(&[2 * VH])?,
                        &self.metadata,
                        &counts,
                    )
                    .generics(
                        [KH, VH, D, self.block, 4]
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                    ),
                )?;
                Ok(())
            })?;
            original_graph.launch().sync_on(&device.stream)?;
            candidate_graph.launch().sync_on(&device.stream)?;
            let actual = device.read_borrowed(&candidate)?;
            for (slot, state) in original.iter().enumerate() {
                let expected = device.read_borrowed(state)?;
                for (index, (&a, &b)) in actual[slot * size..(slot + 1) * size]
                    .iter()
                    .zip(&expected)
                    .enumerate()
                {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "slots={slots} shift={shift} slot={slot} element={index}: {a} != {b}"
                    );
                }
            }
        }
        Ok(())
    }
}

fn lane_rows<T: DType>(
    tensor: &Tensor<T>,
    slot: usize,
    width: usize,
) -> Result<[TensorView<'_, T>; 3], DeviceError> {
    let row = |offset| {
        let begin = (slot * 3 + offset) * width;
        tensor
            .slice(from_ref(&(begin..begin + width)))
            .map_err(DeviceError::from)
    };
    Ok([row(0)?, row(1)?, row(2)?])
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn batch_verify_matches_per_slot_bits_at_model_geometry() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for slots in [2, 5, 16] {
        Fixture::new(&device, slots)?.compare(&device, slots)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn batch_commit_matches_per_slot_bits_at_model_geometry() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for slots in [2, 5, 16] {
        Fixture::new(&device, slots)?.compare_commits(&device, slots)?;
    }
    Ok(())
}

#[path = "resident_slot_recurrent_wave.rs"]
mod wave;

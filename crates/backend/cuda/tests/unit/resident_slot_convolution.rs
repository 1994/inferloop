use super::*;
use crate::{device::CudaDevice, resident::slot_recurrent::fused};
use cutile::cutile_compiler::compile_api::KernelCompiler;
use infer_cuda_kernels::tile::runtime::Graph;
use std::{error::Error, slice::from_ref};

type TestResult<T = ()> = StdResult<T, Box<dyn Error>>;

#[test]
fn pooled_convolution_kernel_compiles_without_a_device() -> TestResult {
    for channels in [128, 129, 512, 10240] {
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "verify")
            .target("sm_120")
            .generics(
                [channels, i32::try_from(METADATA_FIELDS)?]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&[
                ("out", &[3 * channels, channels, 1]),
                ("input", &[3 * channels, channels, 1]),
                ("h0", &[channels, 1]),
                ("h1", &[channels, 1]),
                ("h2", &[channels, 1]),
                ("weight", &[4, 1]),
                ("metadata", &[1]),
            ])
            .compile()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn pooled_convolution_preserves_per_slot_bits() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for slots in [2, 5, 16] {
        for channels in [128, 129, 512, 10240] {
            Fixture::new(&device, slots, channels)?.check(&device)?;
        }
    }
    Ok(())
}

struct Fixture {
    slots: usize,
    channels: usize,
    input: Arc<Tensor<f32>>,
    history: [Arc<Tensor<f32>>; 3],
    weight: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
}

impl Fixture {
    fn new(device: &CudaDevice, slots: usize, channels: usize) -> TestResult<Self> {
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|i| (f32::from(u16::try_from((i + seed) % 97).unwrap()) - 48.0) / 19.0)
                .collect()
        };
        let history = |seed| device.upload(values(slots * channels, seed), &[slots * channels]);
        let lanes = slots * LANES;
        let words: Vec<_> = (0..lanes)
            .flat_map(|lane| {
                let slot = lane / LANES;
                let active =
                    slot % 5 != 1 && (!slot.is_multiple_of(3) || lane.is_multiple_of(LANES));
                [17, 0, if active { 31 } else { -1 }, 0]
            })
            .collect();
        Ok(Self {
            slots,
            channels,
            input: device.upload(values(lanes * channels, 0), &[lanes * channels])?,
            history: [history(13)?, history(29)?, history(71)?],
            weight: device.upload(
                values(channels * CONV_KERNEL_SIZE, 37),
                &[channels, CONV_KERNEL_SIZE],
            )?,
            metadata: device.upload(words, &[lanes * METADATA_FIELDS])?,
        })
    }

    fn check(&self, device: &CudaDevice) -> TestResult {
        let expected = self.reference(device)?;
        let channels = self.channels;
        let mut actual =
            api::zeros::<f32>(&[self.slots, LANES, channels]).sync_on(&device.stream)?;
        kernels::verify(
            (&mut actual).partition([1, LANES, CONV_KERNEL_TILE]),
            &self.input.view(&[self.slots, LANES, channels])?,
            &self.history[0].view(&[self.slots, channels])?,
            &self.history[1].view(&[self.slots, channels])?,
            &self.history[2].view(&[self.slots, channels])?,
            &self.weight,
            &self.metadata,
        )
        .generics(
            [channels, METADATA_FIELDS]
                .iter()
                .map(ToString::to_string)
                .collect(),
        )
        .sync_on(&device.stream)?;
        let actual = device.read_borrowed(&actual)?;
        for (lane, expected) in expected.iter().enumerate() {
            let expected = device.read_borrowed(expected)?;
            for (channel, (&a, &b)) in actual[lane * channels..(lane + 1) * channels]
                .iter()
                .zip(&expected)
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "slots={} channels={channels} lane={lane} channel={channel}: {a} != {b}",
                    self.slots
                );
            }
        }
        Ok(())
    }

    fn reference(&self, device: &CudaDevice) -> TestResult<Vec<Tensor<f32>>> {
        let channels = self.channels;
        let mut outputs = (0..self.slots * LANES)
            .map(|_| api::zeros::<f32>(&[channels]).sync_on(&device.stream))
            .collect::<StdResult<Vec<_>, _>>()?;
        let reference = Graph::scope(&device.stream, |scope| {
            for (slot, [first, second, third]) in
                outputs.as_chunks_mut::<LANES>().0.iter_mut().enumerate()
            {
                let input = |lane: usize| {
                    self.input
                        .slice(from_ref(&(lane * channels..(lane + 1) * channels)))
                };
                let metadata = |lane: usize| {
                    self.metadata.slice(from_ref(
                        &(lane * METADATA_FIELDS..(lane + 1) * METADATA_FIELDS),
                    ))
                };
                let state = slot * channels..(slot + 1) * channels;
                let lane = slot * LANES;
                scope.record(fused::conv_verify3(
                    first.partition([CONV_KERNEL_TILE]),
                    second.partition([CONV_KERNEL_TILE]),
                    third.partition([CONV_KERNEL_TILE]),
                    &self.history[0].slice(from_ref(&state))?,
                    &self.history[1].slice(from_ref(&state))?,
                    &self.history[2].slice(from_ref(&state))?,
                    &input(lane)?,
                    &input(lane + 1)?,
                    &input(lane + 2)?,
                    &self.weight,
                    &metadata(lane)?,
                    &metadata(lane + 1)?,
                    &metadata(lane + 2)?,
                ))?;
            }
            Ok(())
        })?;
        reference.launch().sync_on(&device.stream)?;
        Ok(outputs)
    }
}

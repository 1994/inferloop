use super::*;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_ir::Sampling;

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn greedy_matches_host_total_order_across_tiles_and_padding() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for vocabulary in [1, 3, 255, 257, 4097, 248_321] {
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for row in 0..6 {
            let mut logits: Vec<_> = (0..vocabulary)
                .map(|i| match row {
                    0 => -3.0,
                    1 => {
                        if i % 2 == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    2 => f32::from_bits(u32::try_from(i % 1000).unwrap() + 1),
                    3 => f32::from_bits(0x8000_0000 | (u32::try_from(i % 1000).unwrap() + 1)),
                    _ => f32::from(i16::try_from(i * 17 % 257).unwrap() - 128),
                })
                .collect();
            if row == 4 {
                logits[0] = f32::MAX;
                logits[vocabulary - 1] = f32::MAX;
            }
            expected.push(infer_workloads::sample(
                &logits,
                &Sampling::default(),
                0,
                0,
            )?);
            inputs.extend(logits);
        }
        let logits = device.upload(inputs, &[6, vocabulary])?;
        for tile in TILE_SEARCH {
            let greedy = Greedy::new(&device, 6, vocabulary, tile)?;
            let graph = Graph::scope(&device.stream, |scope| greedy.record(scope, &logits))
                .map_err(device_error)?;
            graph
                .launch()
                .sync_on(&device.stream)
                .map_err(device_error)?;
            assert_eq!(
                greedy.read(&device)?,
                expected,
                "vocabulary={vocabulary} tile={tile}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn greedy_rejects_nonfinite_values_outside_the_winning_block() -> Result<()> {
    let device = CudaDevice::new(0)?;
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![-1.0; 4097];
        values[0] = f32::MAX;
        values[4096] = invalid;
        let logits = device.upload(values, &[1, 4097])?;
        for tile in TILE_SEARCH {
            let greedy = Greedy::new(&device, 1, 4097, tile)?;
            let graph = Graph::scope(&device.stream, |scope| greedy.record(scope, &logits))
                .map_err(device_error)?;
            graph
                .launch()
                .sync_on(&device.stream)
                .map_err(device_error)?;
            assert!(greedy.read(&device).is_err());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn greedy_scattered_rows_preserve_offsets_and_invalid_inactive_rows() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let vocabulary = 257;
    let logits: Vec<_> = [17_u16, 129, 256]
        .into_iter()
        .map(|winner| {
            let mut values = vec![-2.0; vocabulary];
            values[usize::from(winner)] = 7.0;
            device.upload(values, &[1, vocabulary])
        })
        .collect::<Result<_>>()?;
    for tile in TILE_SEARCH {
        let mut selectors = (0..3)
            .map(|_| Greedy::new(&device, 1, vocabulary, tile))
            .collect::<Result<Vec<_>>>()?;
        let output = api::zeros::<i32>(&[7, 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let graph = Graph::scope(&device.stream, |scope| {
            for (index, (selector, input)) in selectors.iter_mut().zip(&logits).enumerate() {
                selector.record_into(scope, input, &output, index * 3)?;
            }
            Ok(())
        })
        .map_err(device_error)?;
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let pairs = device.read_borrowed(&output).map_err(device_error)?;
        for (index, expected) in [17_u32, 129, 256].into_iter().enumerate() {
            let row = index * 3;
            assert_eq!(
                Greedy::decode(&pairs[row * 2..row * 2 + 2], vocabulary)?,
                expected
            );
        }
        for row in [1, 2, 4, 5] {
            assert_eq!(&pairs[row * 2..row * 2 + 2], &[0, 0]);
        }
        assert!(
            Graph::scope(&device.stream, |scope| {
                selectors[0].record_into(scope, &logits[0], &output, 7)
            })
            .is_err()
        );
    }
    let mut values = vec![-1.0; 3 * vocabulary];
    values[17] = 9.0;
    values[vocabulary] = f32::NAN;
    values[2 * vocabulary + 256] = 10.0;
    let logits = device.upload(values, &[3, vocabulary])?;
    let selector = Greedy::new(&device, 3, vocabulary, 256)?;
    let graph = Graph::scope(&device.stream, |scope| selector.record(scope, &logits))
        .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let pairs = device
        .read_borrowed(&selector.tokens)
        .map_err(device_error)?;
    assert_eq!(Greedy::decode(&pairs[..2], vocabulary)?, 17);
    assert!(Greedy::decode(&pairs[2..4], vocabulary).is_err());
    assert_eq!(Greedy::decode(&pairs[4..6], vocabulary)?, 256);
    Ok(())
}

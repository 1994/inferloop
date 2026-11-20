use super::*;
use objc::rc::autoreleasepool;
#[test]
fn tiled_prefill_and_decode_cover_partial_rows_columns_and_k_tiles() -> Result<()> {
    if Device::system_default().is_none() {
        return Ok(());
    }
    autoreleasepool(|| {
        let gpu = MetalDevice::open()?;
        let dummy = gpu.zeros(1)?;
        for inner in [31, 32, 33, 63, 64, 65] {
            for (rows, columns) in [(1, 3), (3, 4), (4, 5), (5, 7), (7, 5)] {
                let x = values((rows + 2) * inner, 17)?;
                let w = values(columns * inner, 13)?;
                let input = gpu.upload(&x)?;
                let weight = gpu.upload(&w)?;
                let output = gpu.zeros(rows * columns)?;
                let command = gpu.queue.new_command_buffer();
                gpu.encode(
                    command,
                    "linear",
                    &Bindings {
                        inputs: &[&input, &weight],
                        state: &dummy,
                        output: &output,
                        dummy: &dummy,
                        page_table: &dummy,
                        tokens: &dummy,
                    },
                    Params {
                        n: u32::try_from(columns).map_err(|_| Error::invalid("test columns"))?,
                        a: u32::try_from(inner).map_err(|_| Error::invalid("test inner"))?,
                        rows: u32::try_from(rows).map_err(|_| Error::invalid("test rows"))?,
                        start_row: 2,
                        ..Default::default()
                    },
                    rows * columns,
                )?;
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                let actual = MetalDevice::read_idle(&output, rows * columns)?;
                let expected: Vec<f32> = (0..rows)
                    .flat_map(|row| (0..columns).map(move |col| (row, col)))
                    .map(|(row, col)| {
                        (0..inner)
                            .map(|k| x[(row + 2) * inner + k] * w[col * inner + k])
                            .sum()
                    })
                    .collect();
                assert_eq!(actual, expected, "shape {rows}x{columns}x{inner}");
            }
        }
        Ok(())
    })
}
fn values(count: usize, modulus: usize) -> Result<Vec<f32>> {
    (0..count)
        .map(|i| {
            Ok(
                f32::from(u16::try_from(i % modulus).map_err(|_| Error::invalid("test value"))?)
                    - 6.0,
            )
        })
        .collect()
}

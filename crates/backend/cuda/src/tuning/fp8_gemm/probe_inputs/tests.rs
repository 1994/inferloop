use super::*;
#[test]
#[ignore = "requires CUDA, release, safe-run; validates actual probe input bits and temporary ownership"]
fn readonly_probe_inputs_keep_fixture_bits_and_scope() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let outer = Scope::new(&device)?;
    for (rows, columns) in [(1, 3), (2, 131), (64, 5120), (256, 17408)] {
        for fixture in 0..super::super::FIXTURES {
            let expected = super::super::inputs(rows, columns, fixture)?;
            let first = upload(&device, [rows, columns], fixture, || Ok(expected.clone()))?;
            let second = upload(&device, [rows, columns], fixture, || {
                panic!("reused numerical probe regenerated its input")
            })?;
            assert_eq!(first.identity(), second.identity());
            let actual = device.runtime().read(&first)?;
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().all(|value| value.is_finite()));
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
    let retained = upload(&device, [1, 3], 0, || panic!("outer input lost"))?;
    {
        let _nested = Scope::new(&device)?;
        let fresh = upload(&device, [1, 3], 0, || super::super::inputs(1, 3, 0))?;
        assert_ne!(retained.identity(), fresh.identity());
        INPUTS.with(|cache| {
            let mut cache = cache.borrow_mut();
            let cache = cache.as_mut().expect("nested cache");
            cache.inputs.clear();
            cache.bytes = 0;
            cache.limit = 16;
        });
        let fits = upload(&device, [1, 4], 0, || Ok(vec![1.0; 4]))?;
        let shared = upload(&device, [1, 4], 0, || panic!("bounded cache missed"))?;
        assert_eq!(fits.identity(), shared.identity());
        let a = upload(&device, [1, 5], 0, || Ok(vec![2.0; 5]))?;
        let b = upload(&device, [1, 5], 0, || Ok(vec![2.0; 5]))?;
        assert_ne!(a.identity(), b.identity());
        INPUTS.with(|cache| assert_eq!(cache.borrow().as_ref().expect("cache").bytes, 16));
        let other_stream = CudaDevice::new(0)?;
        let separate = upload(&other_stream, [1, 4], 0, || Ok(vec![1.0; 4]))?;
        assert_ne!(fits.identity(), separate.identity());
    }
    let restored = upload(&device, [1, 3], 0, || {
        panic!("nested scope lost outer cache")
    })?;
    assert_eq!(retained.identity(), restored.identity());
    drop(outer);
    INPUTS.with(|cache| assert!(cache.borrow().is_none()));
    let uncached = upload(&device, [1, 3], 0, || super::super::inputs(1, 3, 0))?;
    assert_ne!(retained.identity(), uncached.identity());
    device.drain()?;
    println!(
        "all original fixture bits, sharing, byte cap, nested restoration, stream isolation and release passed"
    );
    Ok(())
}

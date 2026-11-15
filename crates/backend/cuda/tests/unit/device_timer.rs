use super::*;
use infer_ir::DeviceTimingMode;
use infer_ir::TimedPhase::{Decode, Draft, Other, Verify};

#[test]
fn each_stretch_belongs_to_the_phase_that_owned_its_start() {
    let marks = [
        (0, Other),
        (10, Draft),
        (40, Verify),
        (40, Verify),
        (90, Draft),
        (95, Other),
        (95, Other),
    ];
    let spans: Vec<_> = intervals(&marks)
        .iter()
        .map(|i| (i.phase, i.start_ns, i.duration_ns))
        .collect();
    assert_eq!(
        spans,
        [
            (Other, 0, 10),
            (Draft, 10, 30),
            (Verify, 40, 50),
            (Draft, 90, 5)
        ]
    );
}

#[test]
fn a_nested_return_to_the_same_phase_merges_into_one_interval() {
    let spans = intervals(&[(0, Decode), (5, Decode), (9, Other), (12, Other)]);
    assert_eq!(spans.len(), 2);
    assert_eq!((spans[0].phase, spans[0].duration_ns), (Decode, 9));
    assert_eq!(spans.iter().map(|i| i.duration_ns).sum::<u64>(), 12);
}

#[test]
fn readings_convert_to_nanoseconds_and_clamp_negative() {
    assert_eq!(nanos(1.5).unwrap(), 1_500_000);
    assert_eq!(nanos(-1.0).unwrap(), 0);
}

#[test]
#[ignore = "requires CUDA"]
fn sampled_steps_cover_nested_phases_and_unsampled_steps_record_nothing() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let mut timer = DeviceTimer::new(DeviceTimingConfig {
        mode: DeviceTimingMode::Graph,
        sample_every: 2,
    });
    for step in 0..2 {
        timer.begin_step(&device, 3)?;
        timer.enter(&device, Verify)?;
        timer.enter(&device, Draft)?;
        timer.exit(&device)?;
        timer.exit(&device)?;
        timer.end_step(&device)?;
        device.synchronize()?;
        let sample = timer.sample()?;
        if step == 1 {
            assert!(sample.is_none());
            continue;
        }
        let sample = sample.ok_or_else(|| Error::invariant("first step is sampled"))?;
        assert_eq!(sample.rows, 3);
        let mut end = 0;
        for interval in &sample.intervals {
            assert_eq!(interval.start_ns, end);
            end = interval.end_ns();
        }
        assert_eq!(end, sample.total_ns());
    }
    Ok(())
}

use super::{grouped_parts, grouped_parts_for, partitions};
#[test]
fn partition_tree_obeys_hardware_and_window_without_storage_capacity() {
    assert_eq!(partitions(16, Some(512), 170), 1);
    assert_eq!(partitions(16, None, 8), 1);
    assert_eq!(partitions(16, None, 64), 4);
    assert_eq!(partitions(16, None, 170), 8);
    assert_eq!(partitions(8, None, 170), 16);
    assert_eq!(partitions(24, None, 170), 8);
}

#[test]
fn grouped_partitions_scale_with_hardware_and_stay_inside_admission_bound() {
    // Preserve the measured 5090 layout while avoiding 64-way splits on small devices.
    assert_eq!(grouped_parts(32, 4, 170), Some(64));
    assert_eq!(grouped_parts(32, 4, 16), Some(8));
    assert_eq!(grouped_parts(32, 4, 1), Some(1));
    assert_eq!(grouped_parts(32, 4, 1000), Some(64));
    for sms in 1..512 {
        for kv_heads in [1, 2, 4, 8, 16] {
            let parts = grouped_parts(kv_heads * 8, kv_heads, sms).unwrap();
            assert!(parts.is_power_of_two());
            assert!((1..=super::MAX_GROUPED_PARTS).contains(&parts));
        }
    }
}

#[test]
fn grouped_partitions_reject_invalid_or_unrepresentable_head_groups() {
    for (heads, kv_heads) in [(0, 4), (32, 0), (31, 4), (68, 4)] {
        assert_eq!(grouped_parts(heads, kv_heads, 170), None);
    }
}

#[test]
fn pooled_launch_shares_the_wave_target_across_slots() {
    // 170 SMs: 255 CTAs over 4 KV heads; four slots per launch need a quarter of the parts.
    assert_eq!(grouped_parts_for(24, 4, 170, 1), grouped_parts(24, 4, 170));
    assert_eq!(grouped_parts_for(24, 4, 170, 4), Some(16));
    assert_eq!(grouped_parts_for(24, 4, 170, 64), Some(1));
    assert_eq!(grouped_parts_for(24, 4, 170, 0), None);
}

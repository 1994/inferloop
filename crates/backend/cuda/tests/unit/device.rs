use super::*;

fn profile(total_memory_bytes: u64) -> DeviceProfile {
    DeviceProfile {
        name: "NVIDIA GeForce RTX 5090".to_owned(),
        architecture: "sm_120".to_owned(),
        multiprocessors: 170,
        total_memory_bytes,
        memory_clock_khz: 14_001_000,
        memory_bus_width_bits: 512,
        l2_cache_bytes: 128 * 1024 * 1024,
    }
}

#[test]
fn profile_derives_bandwidth_headroom_and_identity() {
    let device = profile(32 * 1024 * MIB);
    // 14.001 GHz memory clock, double data rate, 512-bit bus: 1.792 TB/s.
    assert_eq!(
        device.memory_bandwidth_bytes_per_second(),
        1_792_128_000_000
    );
    // A 32 GiB device keeps exactly 1 GiB of driver headroom.
    assert_eq!(device.device_headroom_bytes(), 1024 * MIB);
    assert_eq!(device.arena_budget_bytes(), 1024 * MIB);
    assert_eq!(profile(4 * 1024 * MIB).arena_budget_bytes(), 128 * MIB);
    assert_eq!(profile(80 * 1024 * MIB).arena_budget_bytes(), 1024 * MIB);
    // Small devices keep the floor, huge ones stay bounded.
    assert_eq!(device_headroom(4 * 1024 * MIB), MIN_DEVICE_HEADROOM);
    assert_eq!(device_headroom(1024 * 1024 * MIB), MAX_DEVICE_HEADROOM);
    assert_eq!(device.identity(), "NVIDIA GeForce RTX 5090|sm_120|32768MiB");
    // Identities of different devices never collide.
    assert_ne!(device.identity(), profile(24 * 1024 * MIB).identity());
}

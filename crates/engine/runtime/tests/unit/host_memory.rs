use super::{snapshot, staging_bytes};
#[cfg(target_os = "linux")]
use std::io::Result as IoResult;

#[test]
fn staging_capacity_is_derived_from_the_memory_snapshot() {
    let memory = snapshot().expect("host memory snapshot");
    assert!(memory.available <= memory.total);
    assert!(staging_bytes(memory.total) >= staging_bytes(memory.available));
}

#[cfg(target_os = "linux")]
#[test]
fn nested_cgroup_uses_the_tightest_total_and_available_ancestor_limits() -> IoResult<()> {
    use super::cgroup_v2_limits_at;
    use std::env::temp_dir;
    use std::fs::{create_dir_all, remove_dir_all, write};
    use std::path::PathBuf;
    use std::process::id;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TemporaryDirectory(PathBuf);
    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.0);
        }
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let root = TemporaryDirectory(temp_dir().join(format!("infer-memory-{}-{stamp}", id())));
    let child = root.0.join("parent").join("child");
    create_dir_all(&child)?;
    write(root.0.join("memory.max"), "max")?;
    write(root.0.join("memory.current"), "0")?;
    write(root.0.join("parent/memory.max"), "1024")?;
    write(root.0.join("parent/memory.current"), "900")?;
    write(child.join("memory.max"), "400")?;
    write(child.join("memory.current"), "100")?;
    assert_eq!(
        cgroup_v2_limits_at(&root.0, "/parent/child".as_ref()),
        Some((400, 124))
    );
    assert_eq!(
        cgroup_v2_limits_at(&root.0, "/parent/../child".as_ref()),
        None
    );
    Ok(())
}

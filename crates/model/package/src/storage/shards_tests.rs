use crate::storage::package::load_shards;
use std::env::temp_dir;
use std::error::Error;
use std::fs::create_dir_all;
use std::fs::remove_dir_all;
use std::fs::write;
use std::process::id;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[test]
fn shard_size_accepts_only_exact_payload_or_file_conventions() -> Result<(), Box<dyn Error>> {
    let root = temp_dir().join(format!(
        "infer-index-size-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&root)?;
    let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(1.0f32.to_le_bytes());
    write(root.join("model.safetensors"), &bytes)?;
    for size in [4, bytes.len(), 5, bytes.len() + 1] {
        let index = serde_json::json!({
            "metadata": {"total_size": size},
            "weight_map": {"weight": "model.safetensors"},
        });
        write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&index)?,
        )?;
        assert_eq!(load_shards(&root).is_ok(), size == 4 || size == bytes.len());
    }
    remove_dir_all(root)?;
    Ok(())
}

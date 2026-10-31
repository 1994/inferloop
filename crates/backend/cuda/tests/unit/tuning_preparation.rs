use super::*;
use std::env::temp_dir;
use std::error::Error;
use std::fs::{create_dir_all, remove_dir_all};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn stamp_needs_its_table_and_identity() -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = temp_dir().join(format!("infer-stamp-{}-{unique}", id()));
    create_dir_all(&root)?;
    let table = root.join("tuning.json");
    let stamp = stamp_path(&table, &[b"first".to_vec()]);
    assert_ne!(stamp, stamp_path(&table, &[b"second".to_vec()]));
    write_stamp(&stamp, 17)?;
    assert_eq!(read_stamp(&table, &stamp), None);
    write(&table, "{}")?;
    assert_eq!(read_stamp(&table, &stamp), Some(17));
    write(&stamp, "0")?;
    assert_eq!(read_stamp(&table, &stamp), None);
    write(&stamp, "garbage")?;
    assert_eq!(read_stamp(&table, &stamp), None);
    remove_dir_all(root)?;
    Ok(())
}

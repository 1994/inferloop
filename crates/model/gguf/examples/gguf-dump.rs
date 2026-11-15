//! Print a GGUF container's header, metadata and tensor-type histogram.
use infer_gguf::GgufFile;
use std::collections::BTreeMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: gguf-dump <model.gguf>")?;
    let file = GgufFile::open(path)?;
    println!(
        "version={} architecture={:?} tensors={} data_start={} file_len={}",
        file.version(),
        file.architecture(),
        file.tensors().len(),
        file.data_start(),
        file.file_len()
    );
    for key in file.metadata().keys() {
        println!("metadata {key}");
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut bytes = 0u64;
    for tensor in file.tensors().values() {
        *counts.entry(tensor.ty().name()).or_insert(0) += 1;
        bytes = bytes
            .checked_add(tensor.byte_len())
            .ok_or("GGUF tensor byte total overflow")?;
    }
    println!("payload_bytes={bytes}");
    for (name, count) in counts {
        println!("type {name}: {count}");
    }
    Ok(())
}

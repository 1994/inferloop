//! Open a native mmproj and print its vision inventory.
use infer_models::GgufVisionPackage;
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mmproj = args().nth(1).ok_or("usage: gguf-vision <mmproj.gguf>")?;
    let package = GgufVisionPackage::open(mmproj)?;
    println!(
        "slots={} patch={} deepstack={}",
        package.slots().len(),
        package.patch().iter().flatten().count(),
        package.deepstack().len()
    );
    for slice in package.patch().iter().flatten() {
        println!(
            "patch_slice={} shape={:?} dtype={:?}",
            slice.source, slice.shape, slice.dtype
        );
    }
    let patch = package.patch_weight(u64::MAX)?;
    println!("patch_values={} first={}", patch.len(), patch[0]);
    if let Some(slot) = package.slots().values().next() {
        let values = package.dense(slot, u64::MAX)?;
        println!(
            "slot={} dtype={:?} values={}",
            slot.slot,
            slot.dtype,
            values.len()
        );
    }
    Ok(())
}

//! Print the paired evidence behind every tuning record: one line per decided shape.
//!
//! Reads the machine-local table, or the table named as the only argument.
#[cfg(cuda_backend)]
use std::error::Error as StdError;

#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use infer_backend_cuda::tuning::{TilingTable, search, table_path};
    use std::env::args_os;
    use std::path::PathBuf;
    let path = args_os()
        .nth(1)
        .map(PathBuf::from)
        .or_else(table_path)
        .ok_or("no tuning table path: pass one, or set a user cache directory")?;
    println!("{}", search::report(&TilingTable::load(&path)?));
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    eprintln!("This diagnostic requires Linux and the cuda feature.");
}

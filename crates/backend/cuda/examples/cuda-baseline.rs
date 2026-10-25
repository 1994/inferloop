#[cfg(cuda_backend)]
use std::env::args as env_args;
#[cfg(cuda_backend)]
use std::error::Error;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn Error>> {
    use infer_backend_cuda::{
        benchmark::dense_baseline,
        device::CudaDevice,
        strategy::{DecodeSearch, LinearStrategy, LinearTiling},
    };
    let args: Vec<String> = env_args().skip(1).collect();
    if args.len() == 2 && args[0] == "--catalog" {
        use infer_models::ProjectionCatalog;
        let model = infer_models::QuantizedPackage::open(&args[1], infer_core::ModelId::ONE)?;
        println!("{}", serde_json::to_string_pretty(&model.projections())?);
        return Ok(());
    }
    if args.len() != 2 && args.len() != 4 {
        return Err(
            "usage: cuda-baseline ROWS COLUMNS [TILE_ROWS TILE_COLUMNS] (JSON on stdout)".into(),
        );
    }
    let rows = args[0].parse()?;
    let columns = args[1].parse()?;
    let device = CudaDevice::new(0)?;
    let strategy: Box<dyn LinearStrategy> = if args.len() == 4 {
        Box::new(LinearTiling::new(args[2].parse()?, args[3].parse()?)?)
    } else {
        Box::new(DecodeSearch)
    };
    let report = dense_baseline(&device, rows, columns, strategy.as_ref())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA baselines require Linux");
    exit(1);
}

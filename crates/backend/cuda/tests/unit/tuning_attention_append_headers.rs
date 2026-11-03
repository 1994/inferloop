use super::*;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use infer_cuda_kernels::tile::attention::resident_attention_prefill as kernels;
use std::error::Error as StdError;

fn append_generics(s: Shape, capacity: usize) -> Vec<String> {
    vec![
        f8e4m3fn::DTYPE.as_str().into(),
        s.dim.to_string(),
        capacity.to_string(),
        "1".into(),
        KV_BLOCK_TOKENS.to_string(),
        s.lanes
            .div_ceil(KV_BLOCK_TOKENS)
            .next_power_of_two()
            .to_string(),
    ]
}

// A generic-signature mismatch here silently disables the production MMA path.
#[test]
fn canonical_append_headers_compile_without_a_device() -> StdResult<(), Box<dyn StdError>> {
    for lanes in [64, 256, 1024, 2048] {
        let s = Shape {
            heads: 24,
            kv_heads: 4,
            dim: 256,
            lanes,
            window: 0,
        };
        let capacity = 32768;
        let dim = i32::try_from(s.dim)?;
        let row = i32::try_from(s.kv_heads * s.dim)?;
        let cache = i32::try_from(capacity * s.dim)?;
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "append")
            .target("sm_120")
            .generics(append_generics(s, capacity))
            .strides(&[
                ("keys", &[cache, dim, 1]),
                ("values", &[cache, dim, 1]),
                ("k", &[row, dim, 1]),
                ("v", &[row, dim, 1]),
                ("metadata", &[1]),
                ("table", &[1]),
            ])
            .compile()?;
    }
    Ok(())
}

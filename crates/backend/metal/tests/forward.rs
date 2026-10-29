#![cfg(target_os = "macos")]
use infer_backend_metal::{MetalBackend, MetalConfig, MetalKernels};
use infer_core::*;
use infer_ir::*;
use infer_kernel_api::KernelRegistry;
use infer_models::{ModelPackage, SafetensorsFile, TensorDtype, TensorHeader};
use infer_spi::BackendProvider;
use std::env::temp_dir;
use std::error::Error;
use std::fs::copy;
use std::fs::create_dir_all;
use std::fs::read;
use std::fs::remove_dir_all;
use std::fs::write;
use std::process::id as process_id;
use std::result::Result as StdResult;
use std::thread::sleep;
use std::{
    collections::BTreeMap, path::Path, path::PathBuf, sync::atomic::AtomicU64,
    sync::atomic::Ordering, time::Duration, time::Instant,
};

static NEXT: AtomicU64 = AtomicU64::new(0);
type TestResult<T = ()> = StdResult<T, Box<dyn Error>>;
fn original(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples")
        .join(name)
}
fn fixture(name: &str, dtype: TensorDtype) -> TestResult<PathBuf> {
    let root = temp_dir().join(format!(
        "infer-native-{}-{}",
        process_id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&root)?;
    copy(original(name).join("config.json"), root.join("config.json"))?;
    let mut source = SafetensorsFile::open(original(name).join("model.safetensors"))?;
    let tensors = source.tensors.clone();
    let mut shards = [BTreeMap::new(), BTreeMap::new()];
    let mut payloads = [Vec::new(), Vec::new()];
    let mut weight_map = BTreeMap::new();
    for (i, (name, tensor)) in tensors.into_iter().enumerate() {
        let at = i % 2;
        let data = source.read_f32(&name, 1 << 20)?.data;
        let payload = &mut payloads[at];
        let start = payload.len() as u64;
        for value in data {
            payload.extend_from_slice(&native_bits(value, dtype)?);
        }
        shards[at].insert(
            name.clone(),
            TensorHeader {
                dtype,
                shape: tensor.shape,
                data_offsets: [start, payload.len() as u64],
            },
        );
        weight_map.insert(name, format!("weights-{at}.safetensors"));
    }
    for (at, (headers, payload)) in shards.into_iter().zip(&payloads).enumerate() {
        let mut header = serde_json::to_vec(&headers)?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        write(root.join(format!("weights-{at}.safetensors")), bytes)?;
    }
    let index = serde_json::json!({"metadata":{"total_size":payloads.iter().map(Vec::len).sum::<usize>()},"weight_map":weight_map});
    write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index)?,
    )?;
    Ok(root)
}
fn native_bits(value: f32, dtype: TensorDtype) -> TestResult<[u8; 2]> {
    let bits = value.to_bits();
    if dtype == TensorDtype::BF16 {
        return Ok(u16::try_from(bits >> 16)?.to_le_bytes());
    }
    let sign = (bits >> 16) & 0x8000;
    let exponent = i32::try_from((bits >> 23) & 0xff)? - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    let half = if exponent <= -10 {
        sign
    } else if exponent <= 0 {
        sign | ((mantissa | 0x0080_0000) >> u32::try_from(14 - exponent)?)
    } else {
        if exponent >= 31 {
            return Err("fixture F16 overflow".into());
        }
        sign | (u32::try_from(exponent)? << 10) | (mantissa >> 13)
    };
    Ok(u16::try_from(half)?.to_le_bytes())
}
fn program(backend: &MetalBackend) -> Result<ExecutionProgram> {
    let mut registry = KernelRegistry::default();
    registry.register(&MetalKernels)?;
    infer_compiler::compile(
        ProgramId::ONE,
        infer_compiler::lower(
            backend.model(),
            backend.execution_graph(backend.model())?,
            PrecisionPlan::f32(),
        )?,
        &registry,
        &backend.capabilities(),
        1 << 20,
    )
}
fn execute(
    backend: &mut MetalBackend,
    program: &ExecutionProgram,
    tokens: &[u32],
    added: usize,
) -> TestResult<ModelOutput> {
    let role = if added == 1 {
        ExecutionRole::Decode
    } else {
        ExecutionRole::Prefill
    };
    let step = StepPlan {
        id: StepId::new(tokens.len() as u64)?,
        decision: DecisionId::ONE,
        program: program.id,
        role,
        work: vec![PlannedWork {
            request: RequestId::ONE,
            state: StateId::ONE,
            token_count: added,
            role,
        }],
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    };
    let mut ticket = backend.submit(
        program,
        &step,
        vec![ExecutionTask {
            plain_greedy: false,
            publish_prefix: false,
            request: RequestId::ONE,
            state: StateId::ONE,
            tokens: tokens.to_vec().into(),

            sampling: None,

            media: None,
        }],
    )?;
    let limit = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(mut outputs) = backend.poll(&mut ticket)? {
            return Ok(outputs.remove(0).output);
        }
        assert!(Instant::now() < limit, "GPU completion timeout");
        sleep(Duration::from_micros(100));
    }
}
fn compare(expected: &ModelOutput, actual: &ModelOutput) -> TestResult {
    for (expected, actual) in [
        (expected.logits.clone(), actual.logits.clone()),
        (
            expected.hidden.iter().flatten().copied().collect(),
            actual.hidden.iter().flatten().copied().collect(),
        ),
    ] {
        let metric = infer_quality::compare(&expected, &actual, 2e-6, 2e-5)?;
        assert!(metric.passed, "forward mismatch: {metric:?}");
    }
    Ok(())
}
#[test]
fn sharded_bf16_and_f16_remain_native_and_match_f32_execution() -> TestResult {
    if !MetalBackend::available() {
        return Ok(());
    }
    for name in ["qwen-hybrid-tiny", "qwen-hybrid-grouped"] {
        for dtype in [TensorDtype::BF16, TensorDtype::F16] {
            let root = fixture(name, dtype)?;
            let mut package = ModelPackage::open(&root, ModelId::ONE)?;
            let config = MetalConfig {
                prefix_cache_bytes: 0,
                upload_staging_bytes: 12,
                ..Default::default()
            };
            let mut native = MetalBackend::from_package(&mut package, config.clone())?;
            assert_eq!(
                native.load_plan().resident_bytes * 2,
                package.manifest.host_f32_bytes
            );
            assert_eq!(native.load_plan().staging_bytes, 12);
            assert!(native.load_plan().tensors.iter().all(|t| t.dtype == dtype));
            let mut expected = MetalBackend::new(
                package.imported.model.clone(),
                package.load_host_weights(1 << 20)?,
                config,
            )?;
            let program = program(&native)?;
            native.reserve_state(StateId::ONE, 32)?;
            expected.reserve_state(StateId::ONE, 32)?;
            for (tokens, added) in [
                (vec![1, 2, 3, 5, 8, 13], 6),
                (vec![1, 2, 3, 5, 8, 13, 2], 1),
            ] {
                compare(
                    &execute(&mut expected, &program, &tokens, added)?,
                    &execute(&mut native, &program, &tokens, added)?,
                )?;
            }
            let mut fresh = native.fresh()?;
            assert_eq!(fresh.load_plan(), native.load_plan());
            fresh.reserve_state(StateId::ONE, 32)?;
            let mut independent = expected.fresh()?;
            independent.reserve_state(StateId::ONE, 32)?;
            let tokens = [1, 2, 3, 5, 8, 13, 2];
            compare(
                &execute(&mut independent, &program, &tokens, tokens.len())?,
                &execute(&mut fresh, &program, &tokens, tokens.len())?,
            )?;
            remove_dir_all(root)?;
        }
    }
    Ok(())
}
#[test]
fn layer_major_chunks_match_official_golden_and_reduce_dispatches() -> TestResult {
    if !MetalBackend::available() {
        return Ok(());
    }
    for name in ["qwen-hybrid-tiny", "qwen-hybrid-grouped"] {
        let golden: serde_json::Value =
            serde_json::from_slice(&read(original(name).join("golden.json"))?)?;
        let prefix = &golden["prefixes"][5];
        let tokens: Vec<u32> = serde_json::from_value(prefix["tokens"].clone())?;
        let expected = ModelOutput {
            next_token: None,
            logits: serde_json::from_value(prefix["logits"].clone())?,
            hidden: serde_json::from_value(prefix["hidden"].clone())?,

            tokens: Vec::new(),
        };
        let mut dispatches = Vec::new();
        for chunk in [1, 2, 3, 32] {
            let mut package = ModelPackage::open(original(name), ModelId::ONE)?;
            let mut backend = MetalBackend::from_package(
                &mut package,
                MetalConfig {
                    prefix_cache_bytes: 0,
                    prefill_chunk_tokens: chunk,
                    ..Default::default()
                },
            )?;
            let program = program(&backend)?;
            backend.reserve_state(StateId::ONE, 32)?;
            compare(
                &expected,
                &execute(&mut backend, &program, &tokens, tokens.len())?,
            )?;
            let profile = backend.profile();
            dispatches.push(
                profile["traceEvents"][0]["args"]["encoded_dispatches"]
                    .as_u64()
                    .ok_or("dispatch count")?,
            );
            let heads: Vec<_> = backend
                .traces()
                .iter()
                .filter(|t| {
                    program
                        .operations
                        .iter()
                        .any(|op| op.kernel == t.kernel && op.op.operation == Operation::LmHead)
                })
                .collect();
            assert_eq!(
                heads.len(),
                1,
                "LM head should execute only for final readout"
            );
            assert_eq!(heads[0].position, tokens.len() - 1);
        }
        assert!(dispatches[3] * 4 < dispatches[0], "batched {dispatches:?}");
    }
    Ok(())
}

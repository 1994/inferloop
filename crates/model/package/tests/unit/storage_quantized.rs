use super::*;
#[test]
fn checkpoint_kv_storage_is_declared_independently_of_weights() -> Result<()> {
    assert_eq!(declared_kv_dtype(&serde_json::Value::Null)?, None);
    let mut scheme = serde_json::json!({"num_bits":8,"type":"float","strategy":"tensor","dynamic":false,"symmetric":true});
    assert_eq!(declared_kv_dtype(&scheme)?, Some(TensorDtype::F8E4m3));
    scheme["dynamic"] = true.into();
    assert!(declared_kv_dtype(&scheme).is_err());
    scheme["dynamic"] = false.into();
    scheme["type"] = "int".into();
    assert!(declared_kv_dtype(&scheme).is_err());
    Ok(())
}

#[test]
fn packed_expert_views_are_bounded_and_match_individual_weights() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let mut ordinary = QuantizedPackage::open(root.join("qwen-moe-tiny"), ModelId::ONE)?;
    let mut packed = QuantizedPackage::open(root.join("qwen-moe-packed-tiny"), ModelId::ONE)?;
    for slot in [
        "layers.0.mlp.experts.1.gate_proj.weight",
        "layers.1.mlp.experts.3.up_proj.weight",
        "layers.0.mlp.experts.2.down_proj.weight",
    ] {
        let first = ordinary.weights[slot].data.clone();
        let second = packed.weights[slot].data.clone();
        assert_eq!(ordinary.read(&first, 1024)?, packed.read(&second, 1024)?);
        assert!(packed.read(&second, second.bytes - 1).is_err());
        let mut invalid = second;
        invalid.view.as_mut().unwrap().element_offset = u64::MAX;
        assert!(packed.read(&invalid, 1024).is_err());
    }
    Ok(())
}

#[test]
fn fused_linear_projection_rejects_out_of_range_gather_rows() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-next-moe-tiny");
    let mut package = QuantizedPackage::open(root, ModelId::ONE)?;
    let mut source = package.weights["layers.0.linear_attn.in_proj_z.weight"]
        .data
        .clone();
    assert_eq!(package.read(&source, 1024)?.len() as u64, source.bytes);
    assert!(package.read_view(&source, 1024).is_err());
    source.view.as_mut().unwrap().rows[0] = usize::MAX;
    assert!(package.read(&source, 1024).is_err());
    Ok(())
}

#[test]
fn column_views_gather_each_row_slice() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-moe-tiny");
    let mut package = QuantizedPackage::open(root, ModelId::ONE)?;
    let source = package
        .weights
        .values()
        .find(|weight| {
            weight.encoding == WeightEncoding::Float
                && weight.data.view.is_none()
                && weight.shape.len() == 2
                && weight.shape[1] >= 4
        })
        .map(|weight| weight.data.clone())
        .expect("a dense matrix weight");
    let [rows, columns] = source.shape[..] else {
        unreachable!()
    };
    let element = usize::try_from(source.dtype.bytes()).unwrap();
    let whole = package.read(&source, u64::MAX)?;
    let (first, count) = (1, columns / 2);
    let mut slice = source.clone();
    slice.shape = vec![rows, count];
    slice.bytes = (rows * count * element) as u64;
    slice.view = Some(WeightView {
        stored_shape: source.shape.clone(),
        element_offset: 0,
        rows: Vec::new(),
        columns: Some([first, count]),
    });
    let expected: Vec<u8> = whole
        .chunks(columns * element)
        .flat_map(|row| row[first * element..(first + count) * element].to_vec())
        .collect();
    assert_eq!(package.read(&slice, u64::MAX)?, expected);
    assert!(package.read_view(&slice, u64::MAX).is_err());
    let mut outside = slice;
    outside.view.as_mut().unwrap().columns = Some([columns - count + 1, count]);
    assert!(package.read(&outside, u64::MAX).is_err());
    Ok(())
}

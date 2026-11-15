use super::encoding_for;
use infer_format::GgufType;
use infer_ir::WeightEncoding;

#[test]
fn maps_tensor_types_to_weight_encodings() {
    assert_eq!(
        encoding_for(GgufType::Q4K).unwrap(),
        WeightEncoding::GroupedInt4
    );
    assert_eq!(
        encoding_for(GgufType::Q6K).unwrap(),
        WeightEncoding::GroupedInt6
    );
    assert_eq!(encoding_for(GgufType::F16).unwrap(), WeightEncoding::Dense);
    assert!(encoding_for(GgufType::Q5K).is_err());
}

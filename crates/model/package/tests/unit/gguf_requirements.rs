use super::encoding_for;
use infer_format::GgufType;
use infer_ir::ExecutableEncoding;

#[test]
fn maps_tensor_types_to_weight_encodings() {
    assert_eq!(
        encoding_for(GgufType::Q4K).unwrap(),
        ExecutableEncoding::GroupedInt4
    );
    assert_eq!(
        encoding_for(GgufType::Q6K).unwrap(),
        ExecutableEncoding::GroupedInt6
    );
    assert_eq!(
        encoding_for(GgufType::F16).unwrap(),
        ExecutableEncoding::Dense
    );
    assert!(encoding_for(GgufType::Q5K).is_err());
}

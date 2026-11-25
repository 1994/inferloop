//! Tool executor contract validation.
use infer_core::{ErrorCode, Result};
use infer_spi::{
    ResponsesExtensions, TOOLS_SPI_VERSION, ToolEffect, ToolExecutor, ToolFuture, ToolInvocation,
    ToolName, ToolOutput, ToolSpec,
};
use std::sync::Arc;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(1);
const OUTPUT_BYTES: usize = 1024;

struct Executor {
    tools: Vec<ToolSpec>,
    version: u32,
}

impl ToolExecutor for Executor {
    fn spi_version(&self) -> u32 {
        self.version
    }
    fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }
    fn call<'a>(&'a self, invocation: ToolInvocation<'a>) -> ToolFuture<'a> {
        Box::pin(async move { ToolOutput::Text(invocation.arguments.to_owned()) })
    }
}

fn spec(name: &str, effect: ToolEffect) -> Result<ToolSpec> {
    ToolSpec::new(
        ToolName::new(name)?,
        "test tool",
        r#"{"type":"object"}"#,
        effect,
        TIMEOUT,
        OUTPUT_BYTES,
    )
}

fn validate(tools: Vec<ToolSpec>, version: u32) -> Result<()> {
    ResponsesExtensions::new()
        .with_tools(Arc::new(Executor { tools, version }))
        .validate()
}

#[test]
fn tool_names_follow_the_function_name_rule() -> Result<()> {
    assert_eq!(ToolName::new("read_file-2")?.as_str(), "read_file-2");
    for bad in [
        "",
        "has space",
        "dotted.name",
        &"x".repeat(ToolName::MAX_BYTES + 1),
    ] {
        assert_eq!(
            ToolName::new(bad).map_err(|error| error.code).err(),
            Some(ErrorCode::InvalidInput),
            "{bad:?}"
        );
    }
    Ok(())
}

#[test]
fn specs_need_a_schema_timeout_and_output_limit() -> Result<()> {
    let name = ToolName::new("read")?;
    let schema = r#"{"type":"object"}"#;
    let reject = |parameters: &str, timeout: Duration, bytes: usize| {
        ToolSpec::new(
            name.clone(),
            "",
            parameters,
            ToolEffect::ReadOnly,
            timeout,
            bytes,
        )
        .map_err(|error| error.code)
        .err()
    };
    assert_eq!(
        reject(" ", TIMEOUT, OUTPUT_BYTES),
        Some(ErrorCode::InvalidInput)
    );
    assert_eq!(
        reject(schema, Duration::ZERO, OUTPUT_BYTES),
        Some(ErrorCode::InvalidInput)
    );
    assert_eq!(reject(schema, TIMEOUT, 0), Some(ErrorCode::InvalidInput));
    assert_eq!(reject(schema, TIMEOUT, OUTPUT_BYTES), None);
    Ok(())
}

#[test]
fn extensions_accept_unique_read_only_tools() -> Result<()> {
    validate(
        vec![
            spec("read", ToolEffect::ReadOnly)?,
            spec("search", ToolEffect::ReadOnly)?,
        ],
        TOOLS_SPI_VERSION,
    )
}

#[test]
fn extensions_reject_duplicates_side_effects_and_other_versions() -> Result<()> {
    let code = |result: Result<()>| result.map_err(|error| error.code).err();
    let duplicate = vec![
        spec("read", ToolEffect::ReadOnly)?,
        spec("read", ToolEffect::ReadOnly)?,
    ];
    assert_eq!(
        code(validate(duplicate, TOOLS_SPI_VERSION)),
        Some(ErrorCode::InvalidInput)
    );
    let writes = vec![spec("write", ToolEffect::SideEffect)?];
    assert_eq!(
        code(validate(writes, TOOLS_SPI_VERSION)),
        Some(ErrorCode::Unsupported)
    );
    let read = vec![spec("read", ToolEffect::ReadOnly)?];
    assert_eq!(
        code(validate(read, TOOLS_SPI_VERSION + 1)),
        Some(ErrorCode::Unsupported)
    );
    Ok(())
}

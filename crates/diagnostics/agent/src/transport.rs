use crate::constants::INITIAL_FRAME_CAPACITY;
use crate::constants::JSONRPC_PARSE_ERROR;
use crate::constants::JSONRPC_SERVER_ERROR;
use crate::{AgentBackend, AgentDispatcher, protocol::RpcError, protocol::error_response};
use infer_core::{Error, ErrorCode, Result};
use serde_json::Value;
use std::io::Error as IoError;
use std::{io::BufRead, io::Write};

pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Bounded NDJSON framing. Oversized frames are discarded before any command runs.
/// # Errors
/// Returns input/output errors; malformed messages get RPC errors and the stream continues.
pub fn serve<B: AgentBackend>(
    service: &mut AgentDispatcher<B>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    let mut frame = Vec::with_capacity(INITIAL_FRAME_CAPACITY);
    loop {
        frame.clear();
        let mut oversized = false;
        loop {
            let chunk = input.fill_buf().map_err(|e| io_error(&e))?;
            if chunk.is_empty() {
                break;
            }
            let end = chunk.iter().position(|b| *b == b'\n');
            let count = end.map_or(chunk.len(), |index| index + 1);
            if frame.len().saturating_add(count) > MAX_FRAME_BYTES {
                oversized = true;
            }
            if !oversized {
                frame.extend_from_slice(&chunk[..count]);
            }
            input.consume(count);
            if end.is_some() {
                break;
            }
        }
        if frame.is_empty() && !oversized {
            return Ok(());
        }
        let response = if oversized {
            Some(error_response(
                Value::Null,
                &RpcError::new(JSONRPC_SERVER_ERROR, "agent frame exceeds 1 MiB"),
            ))
        } else {
            match serde_json::from_slice(&frame) {
                Ok(value) => service.handle(value),
                Err(error) => Some(error_response(
                    Value::Null,
                    &RpcError::new(JSONRPC_PARSE_ERROR, error.to_string()),
                )),
            }
        };
        if let Some(response) = response {
            serde_json::to_writer(&mut *output, &response)
                .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?;
            writeln!(output).map_err(|e| io_error(&e))?;
            output.flush().map_err(|e| io_error(&e))?;
        }
    }
}

fn io_error(error: &IoError) -> Error {
    Error::new(ErrorCode::Backend, error.to_string())
}

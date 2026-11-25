use infer_core::{Error, ErrorCode, Result};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::iter::once;
use std::{collections::BTreeMap, collections::BTreeSet};

/// Commands declare whether they inspect, mutate live state, or run isolated engines.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandEffect {
    ReadOnly,
    Control,
    IsolatedExecution,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandDescriptor {
    pub method: &'static str,
    pub aliases: Vec<&'static str>,
    pub description: &'static str,
    pub effect: CommandEffect,
    pub params: Value,
}

trait Handler<C> {
    fn call(&self, context: &mut C, params: Value) -> Result<Value>;
}

/// Infallible queries and fallible controls share one dispatch contract.
pub trait CommandOutput {
    /// # Errors
    /// Preserves any parameter or execution error returned by the command.
    fn into_outcome(self) -> Result<Value>;
}

impl CommandOutput for Value {
    fn into_outcome(self) -> Result<Value> {
        Ok(self)
    }
}
impl CommandOutput for Result<Value> {
    fn into_outcome(self) -> Result<Value> {
        self
    }
}

struct TypedHandler<C, P, R> {
    execute: fn(&mut C, P) -> R,
}

impl<C, P: DeserializeOwned, R: CommandOutput> Handler<C> for TypedHandler<C, P, R> {
    fn call(&self, context: &mut C, params: Value) -> Result<Value> {
        let params = serde_json::from_value(params).map_err(|e| Error::invalid(e.to_string()))?;
        (self.execute)(context, params).into_outcome()
    }
}

struct Command<C> {
    descriptor: CommandDescriptor,
    handler: Box<dyn Handler<C>>,
}

/// Registration validates canonical names and aliases before installing anything.
pub struct CommandRegistry<C> {
    commands: BTreeMap<&'static str, Command<C>>,
    names: BTreeMap<&'static str, &'static str>,
}

impl<C> Default for CommandRegistry<C> {
    fn default() -> Self {
        Self {
            commands: BTreeMap::new(),
            names: BTreeMap::new(),
        }
    }
}

impl<C: 'static> CommandRegistry<C> {
    /// # Errors
    /// Returns invalid input for incomplete descriptors or conflict for reused names.
    pub fn register<P: DeserializeOwned + 'static, R: CommandOutput + 'static>(
        &mut self,
        descriptor: CommandDescriptor,
        execute: fn(&mut C, P) -> R,
    ) -> Result<()> {
        let mut unique = BTreeSet::new();
        for name in once(descriptor.method).chain(descriptor.aliases.iter().copied()) {
            if name.is_empty()
                || name.starts_with("rpc.")
                || descriptor.description.is_empty()
                || !descriptor.params.is_object()
            {
                return Err(Error::invalid("invalid agent command descriptor"));
            }
            if self.names.contains_key(name) || !unique.insert(name) {
                return Err(Error::new(
                    ErrorCode::Conflict,
                    format!("duplicate agent command {name}"),
                ));
            }
        }
        for name in unique {
            self.names.insert(name, descriptor.method);
        }
        self.commands.insert(
            descriptor.method,
            Command {
                descriptor,
                handler: Box::new(TypedHandler { execute }),
            },
        );
        Ok(())
    }

    #[must_use]
    pub fn descriptors(&self) -> Vec<CommandDescriptor> {
        self.commands
            .values()
            .map(|c| c.descriptor.clone())
            .collect()
    }

    #[must_use]
    pub fn descriptor(&self, name: &str) -> Option<&CommandDescriptor> {
        self.names
            .get(name)
            .and_then(|canonical| self.commands.get(canonical))
            .map(|c| &c.descriptor)
    }

    /// Unknown names are distinguished from errors produced by registered commands.
    /// # Errors
    /// Returns the command's parameter validation or execution error.
    pub fn dispatch(&self, context: &mut C, name: &str, params: Value) -> Option<Result<Value>> {
        self.names
            .get(name)
            .and_then(|canonical| self.commands.get(canonical))
            .map(|command| command.handler.call(context, params))
    }
}

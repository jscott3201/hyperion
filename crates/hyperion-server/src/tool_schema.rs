//! Bounded, transport-neutral tool declaration normalization and validation.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use serde_json::{Map, Value};

use crate::tool_call::ToolCall;

const MAX_DECLARATIONS: usize = 128;
const MAX_DECLARATION_BYTES: usize = 512 * 1024;
const MAX_SCHEMA_NODES: usize = 4_096;
const MAX_DEPTH: usize = 48;
const MAX_ARGUMENT_NODES: usize = 4_096;
const MAX_ARGUMENT_BYTES: usize = 64 * 1024;

const RESERVED_CONTROLS: &[&str] = &[
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<mask>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|think|>",
    "<|\"|>",
    "<|image>",
    "<image|>",
    "<|image|>",
    "<|audio>",
    "<audio|>",
    "<|audio|>",
    "<|video|>",
    "<|tool_call|>",
];

/// Whether the model may emit a tool call for this request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolMode {
    /// Tool declarations are retained for validation but not rendered.
    None,
    /// The model may select any declared tool.
    Auto,
}

/// Borrowed OpenAI-compatible tool controls.
pub struct OpenAiToolsInput<'a> {
    pub tools: Option<&'a Value>,
    pub tool_choice: Option<&'a Value>,
    pub parallel_tool_calls: Option<&'a Value>,
}

/// Borrowed Anthropic-compatible tool controls.
pub struct AnthropicToolsInput<'a> {
    pub tools: Option<&'a Value>,
    pub tool_choice: Option<&'a Value>,
}

/// A deterministic client-side declaration error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolSchemaError(String);

impl fmt::Display for ToolSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ToolSchemaError {}

/// A deterministic model-output validation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolValidationError(String);

impl fmt::Display for ToolValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ToolValidationError {}

#[derive(Debug)]
enum Schema {
    Object {
        properties: Vec<(String, Schema)>,
        required: Vec<String>,
        additional_properties: bool,
    },
    Array(Box<Schema>),
    String {
        values: Option<Vec<String>>,
    },
    Number,
    Integer,
    Boolean,
    Null,
    Nullable(Box<Schema>),
}

/// An immutable set of renderer-safe declarations and private validators.
pub struct ToolRegistry {
    mode: ToolMode,
    render_tools: Vec<Value>,
    validators: HashMap<String, Schema>,
}

impl ToolRegistry {
    pub fn from_openai(input: OpenAiToolsInput<'_>) -> Result<Self, ToolSchemaError> {
        validate_openai_parallel(input.parallel_tool_calls)?;
        let tools = tools_array(input.tools, "tools")?;
        let mode = parse_openai_choice(input.tool_choice, tools.len())?;
        Self::compile_openai(tools, mode)
    }

    pub fn from_anthropic(input: AnthropicToolsInput<'_>) -> Result<Self, ToolSchemaError> {
        let tools = tools_array(input.tools, "tools")?;
        let mode = parse_anthropic_choice(input.tool_choice, tools.len())?;
        Self::compile_anthropic(tools, mode)
    }

    #[must_use]
    pub fn mode(&self) -> ToolMode {
        self.mode
    }

    #[must_use]
    pub fn render_tools(&self) -> &[Value] {
        if self.mode == ToolMode::Auto {
            &self.render_tools
        } else {
            &[]
        }
    }

    pub fn validate_generated(&self, call: &ToolCall) -> Result<(), ToolValidationError> {
        let schema = self.validators.get(&call.name).ok_or_else(|| {
            ToolValidationError(format!("tool call $.name: unknown tool {:?}", call.name))
        })?;
        validate_argument_budget(&call.arguments)?;
        schema.validate(&call.arguments, "$.arguments")
    }

    fn compile_openai(tools: &[Value], mode: ToolMode) -> Result<Self, ToolSchemaError> {
        compile_tools(tools, mode, Provider::OpenAi)
    }

    fn compile_anthropic(tools: &[Value], mode: ToolMode) -> Result<Self, ToolSchemaError> {
        compile_tools(tools, mode, Provider::Anthropic)
    }
}

#[derive(Clone, Copy)]
enum Provider {
    OpenAi,
    Anthropic,
}

fn error(path: &str, message: impl fmt::Display) -> ToolSchemaError {
    ToolSchemaError(format!("tool schema {path}: {message}"))
}

fn validation_error(path: &str, message: impl fmt::Display) -> ToolValidationError {
    ToolValidationError(format!("tool call {path}: {message}"))
}

fn tools_array<'a>(tools: Option<&'a Value>, path: &str) -> Result<&'a [Value], ToolSchemaError> {
    match tools {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(values)) if values.len() <= MAX_DECLARATIONS => Ok(values),
        Some(Value::Array(values)) => Err(error(
            path,
            format_args!(
                "has {} declarations; maximum is {MAX_DECLARATIONS}",
                values.len()
            ),
        )),
        Some(_) => Err(error(path, "must be an array")),
    }
}

fn default_mode(tool_count: usize) -> ToolMode {
    if tool_count == 0 {
        ToolMode::None
    } else {
        ToolMode::Auto
    }
}

fn parse_openai_choice(
    choice: Option<&Value>,
    tool_count: usize,
) -> Result<ToolMode, ToolSchemaError> {
    match choice {
        None => Ok(default_mode(tool_count)),
        Some(Value::String(value)) if value == "auto" => Ok(ToolMode::Auto),
        Some(Value::String(value)) if value == "none" => Ok(ToolMode::None),
        Some(_) => Err(error("tool_choice", "must be \"auto\" or \"none\"")),
    }
}

fn parse_anthropic_choice(
    choice: Option<&Value>,
    tool_count: usize,
) -> Result<ToolMode, ToolSchemaError> {
    let Some(choice) = choice else {
        return Ok(default_mode(tool_count));
    };
    let object = expect_object(choice, "tool_choice")?;
    reject_unknown_fields(
        object,
        &["type", "disable_parallel_tool_use"],
        "tool_choice",
    )?;
    if let Some(value) = object.get("disable_parallel_tool_use") {
        match value {
            Value::Bool(false) => {}
            Value::Bool(true) => {
                return Err(error(
                    "tool_choice.disable_parallel_tool_use",
                    "true is unsupported",
                ));
            }
            _ => {
                return Err(error(
                    "tool_choice.disable_parallel_tool_use",
                    "must be a boolean",
                ));
            }
        }
    }
    match object.get("type") {
        Some(Value::String(value)) if value == "auto" => Ok(ToolMode::Auto),
        Some(Value::String(value)) if value == "none" => Ok(ToolMode::None),
        Some(_) => Err(error("tool_choice.type", "must be \"auto\" or \"none\"")),
        None => Err(error("tool_choice.type", "is required")),
    }
}

fn validate_openai_parallel(value: Option<&Value>) -> Result<(), ToolSchemaError> {
    match value {
        None | Some(Value::Bool(true)) => Ok(()),
        Some(Value::Bool(false)) => Err(error("parallel_tool_calls", "false is unsupported")),
        Some(_) => Err(error("parallel_tool_calls", "must be a boolean")),
    }
}

fn compile_tools(
    tools: &[Value],
    mode: ToolMode,
    provider: Provider,
) -> Result<ToolRegistry, ToolSchemaError> {
    let mut declaration_bytes = 0;
    for (index, tool) in tools.iter().enumerate() {
        inspect_strings_and_keys(
            tool,
            &format!("tools[{index}]"),
            &mut declaration_bytes,
            MAX_DECLARATION_BYTES,
            false,
            0,
        )?;
    }

    let mut schema_nodes = 0;
    let mut render_tools = Vec::with_capacity(tools.len());
    let mut validators = HashMap::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        let path = format!("tools[{index}]");
        let (name, description, schema_value) = parse_declaration(tool, &path, provider)?;
        let (name_path, schema_path) = match provider {
            Provider::OpenAi => (
                format!("{path}.function.name"),
                format!("{path}.function.parameters"),
            ),
            Provider::Anthropic => (format!("{path}.name"), format!("{path}.input_schema")),
        };
        validate_tool_name(name, &name_path)?;
        if validators.contains_key(name) {
            return Err(error(&name_path, "duplicates an earlier tool name"));
        }
        let compiled = compile_schema(schema_value, &schema_path, 0, &mut schema_nodes, true)?;
        let rendered = renderer_declaration(name, description, &compiled.rendered);
        validators.insert(name.to_owned(), compiled.validator);
        render_tools.push(rendered);
    }
    Ok(ToolRegistry {
        mode,
        render_tools,
        validators,
    })
}

struct CompiledSchema {
    rendered: Value,
    validator: Schema,
}

fn parse_declaration<'a>(
    value: &'a Value,
    path: &str,
    provider: Provider,
) -> Result<(&'a str, &'a str, &'a Value), ToolSchemaError> {
    match provider {
        Provider::OpenAi => {
            let outer = expect_object(value, path)?;
            reject_unknown_fields(outer, &["type", "function"], path)?;
            match outer.get("type") {
                Some(Value::String(kind)) if kind == "function" => {}
                Some(_) => return Err(error(&format!("{path}.type"), "must be \"function\"")),
                None => return Err(error(&format!("{path}.type"), "is required")),
            }
            let function_path = format!("{path}.function");
            let function = expect_object(
                outer
                    .get("function")
                    .ok_or_else(|| error(&function_path, "is required"))?,
                &function_path,
            )?;
            reject_unknown_fields(
                function,
                &["name", "description", "parameters", "strict"],
                &function_path,
            )?;
            validate_strict(function.get("strict"), &format!("{function_path}.strict"))?;
            let name = required_string(function, "name", &function_path)?;
            let description =
                optional_string(function, "description", &function_path)?.unwrap_or("");
            static EMPTY_SCHEMA: std::sync::LazyLock<Value> = std::sync::LazyLock::new(
                || serde_json::json!({"type":"object","properties":{},"required":[],"additionalProperties":false}),
            );
            let schema = function.get("parameters").unwrap_or(&EMPTY_SCHEMA);
            Ok((name, description, schema))
        }
        Provider::Anthropic => {
            let object = expect_object(value, path)?;
            reject_unknown_fields(
                object,
                &["name", "description", "input_schema", "strict"],
                path,
            )?;
            validate_strict(object.get("strict"), &format!("{path}.strict"))?;
            let name = required_string(object, "name", path)?;
            let description = optional_string(object, "description", path)?.unwrap_or("");
            let schema = object
                .get("input_schema")
                .ok_or_else(|| error(&format!("{path}.input_schema"), "is required"))?;
            Ok((name, description, schema))
        }
    }
}

fn renderer_declaration(name: &str, description: &str, schema: &Value) -> Value {
    let mut function = Map::new();
    function.insert("name".into(), Value::String(name.to_owned()));
    function.insert("description".into(), Value::String(description.to_owned()));
    function.insert("parameters".into(), schema.clone());
    let mut declaration = Map::new();
    declaration.insert("type".into(), Value::String("function".into()));
    declaration.insert("function".into(), Value::Object(function));
    Value::Object(declaration)
}

fn validate_strict(value: Option<&Value>, path: &str) -> Result<(), ToolSchemaError> {
    match value {
        None | Some(Value::Bool(false)) => Ok(()),
        Some(Value::Bool(true)) => Err(error(path, "true is unsupported")),
        Some(_) => Err(error(path, "must be a boolean")),
    }
}

fn compile_schema(
    value: &Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
    root: bool,
) -> Result<CompiledSchema, ToolSchemaError> {
    if depth > MAX_DEPTH {
        return Err(error(
            path,
            format_args!("exceeds maximum depth {MAX_DEPTH}"),
        ));
    }
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| error(path, "schema node count overflow"))?;
    if *nodes > MAX_SCHEMA_NODES {
        return Err(error(
            path,
            format_args!("exceeds maximum of {MAX_SCHEMA_NODES} schema nodes"),
        ));
    }
    let object = expect_object(value, path)?;
    reject_unknown_fields(
        object,
        &[
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "nullable",
            "description",
            "title",
            "$comment",
            "default",
            "examples",
        ],
        path,
    )?;
    let (kind, union_nullable) = parse_type(object.get("type"), &format!("{path}.type"))?;
    validate_annotations(object, path)?;
    let nullable =
        parse_nullable(object.get("nullable"), &format!("{path}.nullable"))? || union_nullable;
    if root && (kind != "object" || nullable) {
        return Err(error(path, "root schema must be a non-nullable object"));
    }
    validate_keywords_for_type(object, kind, path)?;

    let mut rendered = Map::new();
    if !root && let Some(description) = object.get("description") {
        rendered.insert("description".into(), description.clone());
    }
    rendered.insert("type".into(), Value::String(kind.to_owned()));
    if nullable {
        rendered.insert("nullable".into(), Value::Bool(true));
    }

    let validator = match kind {
        "object" => compile_object(object, path, depth, nodes, &mut rendered)?,
        "array" => {
            let item_path = format!("{path}.items");
            let items = object
                .get("items")
                .ok_or_else(|| error(&item_path, "is required for array schemas"))?;
            let compiled = compile_schema(items, &item_path, depth + 1, nodes, false)?;
            rendered.insert("items".into(), compiled.rendered);
            Schema::Array(Box::new(compiled.validator))
        }
        "string" => {
            let values = parse_string_enum(object.get("enum"), &format!("{path}.enum"))?;
            if let Some(values) = &values {
                rendered.insert(
                    "enum".into(),
                    Value::Array(values.iter().cloned().map(Value::String).collect()),
                );
            }
            Schema::String { values }
        }
        "number" => Schema::Number,
        "integer" => Schema::Integer,
        "boolean" => Schema::Boolean,
        "null" => Schema::Null,
        _ => unreachable!("parse_type only returns supported types"),
    };
    Ok(CompiledSchema {
        rendered: Value::Object(rendered),
        validator: if nullable {
            Schema::Nullable(Box::new(validator))
        } else {
            validator
        },
    })
}

fn compile_object(
    object: &Map<String, Value>,
    path: &str,
    depth: usize,
    nodes: &mut usize,
    rendered: &mut Map<String, Value>,
) -> Result<Schema, ToolSchemaError> {
    let properties_path = format!("{path}.properties");
    let empty_properties = Map::new();
    let properties = match object.get("properties") {
        None => &empty_properties,
        Some(Value::Object(properties)) => properties,
        Some(_) => return Err(error(&properties_path, "must be an object")),
    };
    let mut compiled_properties = Vec::with_capacity(properties.len());
    let mut rendered_properties = Map::new();
    for (name, schema) in properties {
        validate_raw_key(name, &format!("{properties_path}.{name}"))?;
        let property_path = format!("{properties_path}.{name}");
        let compiled = compile_schema(schema, &property_path, depth + 1, nodes, false)?;
        rendered_properties.insert(name.clone(), compiled.rendered);
        compiled_properties.push((name.clone(), compiled.validator));
    }
    let required = parse_required(
        object.get("required"),
        properties,
        &format!("{path}.required"),
    )?;
    let additional_properties = match object.get("additionalProperties") {
        None => true,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(error(
                &format!("{path}.additionalProperties"),
                "must be a boolean",
            ));
        }
    };
    rendered.insert("properties".into(), Value::Object(rendered_properties));
    rendered.insert(
        "required".into(),
        Value::Array(required.iter().cloned().map(Value::String).collect()),
    );
    Ok(Schema::Object {
        properties: compiled_properties,
        required,
        additional_properties,
    })
}

fn parse_type<'a>(
    value: Option<&'a Value>,
    path: &str,
) -> Result<(&'a str, bool), ToolSchemaError> {
    match value {
        Some(Value::String(kind)) if supported_type(kind) => Ok((kind, false)),
        Some(Value::Array(types)) if types.len() == 2 => {
            let mut concrete = None;
            let mut saw_null = false;
            for value in types {
                match value {
                    Value::String(kind) if kind == "null" && !saw_null => saw_null = true,
                    Value::String(kind)
                        if supported_type(kind) && kind != "null" && concrete.is_none() =>
                    {
                        concrete = Some(kind.as_str());
                    }
                    _ => {
                        return Err(error(
                            path,
                            "type union must contain one concrete type and null",
                        ));
                    }
                }
            }
            concrete
                .filter(|_| saw_null)
                .map(|kind| (kind, true))
                .ok_or_else(|| error(path, "type union must contain one concrete type and null"))
        }
        Some(Value::String(_)) => Err(error(path, "contains an unsupported type")),
        Some(_) => Err(error(
            path,
            "must be a supported type string or nullable pair",
        )),
        None => Err(error(path, "is required")),
    }
}

fn supported_type(kind: &str) -> bool {
    matches!(
        kind,
        "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
    )
}

fn parse_nullable(value: Option<&Value>, path: &str) -> Result<bool, ToolSchemaError> {
    match value {
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(error(path, "must be a boolean")),
    }
}

fn validate_annotations(object: &Map<String, Value>, path: &str) -> Result<(), ToolSchemaError> {
    for field in ["description", "title", "$comment"] {
        if object.get(field).is_some_and(|value| !value.is_string()) {
            return Err(error(&format!("{path}.{field}"), "must be a string"));
        }
    }
    if object
        .get("examples")
        .is_some_and(|value| !value.is_array())
    {
        return Err(error(&format!("{path}.examples"), "must be an array"));
    }
    Ok(())
}

fn validate_keywords_for_type(
    object: &Map<String, Value>,
    kind: &str,
    path: &str,
) -> Result<(), ToolSchemaError> {
    for keyword in ["properties", "required", "additionalProperties"] {
        if object.contains_key(keyword) && kind != "object" {
            return Err(error(
                &format!("{path}.{keyword}"),
                "is only valid for object schemas",
            ));
        }
    }
    if object.contains_key("items") && kind != "array" {
        return Err(error(
            &format!("{path}.items"),
            "is only valid for array schemas",
        ));
    }
    if object.contains_key("enum") && kind != "string" {
        return Err(error(
            &format!("{path}.enum"),
            "is only valid for string schemas",
        ));
    }
    Ok(())
}

fn parse_string_enum(
    value: Option<&Value>,
    path: &str,
) -> Result<Option<Vec<String>>, ToolSchemaError> {
    let Some(value) = value else { return Ok(None) };
    let Value::Array(values) = value else {
        return Err(error(path, "must be an array of strings"));
    };
    let mut output = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let Value::String(value) = value else {
            return Err(error(&format!("{path}[{index}]"), "must be a string"));
        };
        output.push(value.clone());
    }
    Ok(Some(output))
}

fn parse_required(
    value: Option<&Value>,
    properties: &Map<String, Value>,
    path: &str,
) -> Result<Vec<String>, ToolSchemaError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Value::Array(values) = value else {
        return Err(error(path, "must be an array of strings"));
    };
    let mut required = Vec::with_capacity(values.len());
    let mut seen = HashSet::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let Value::String(name) = value else {
            return Err(error(&format!("{path}[{index}]"), "must be a string"));
        };
        validate_raw_key(name, &format!("{path}[{index}]"))?;
        if !properties.contains_key(name) {
            return Err(error(
                &format!("{path}[{index}]"),
                "does not name a property",
            ));
        }
        if !seen.insert(name.as_str()) {
            return Err(error(
                &format!("{path}[{index}]"),
                "duplicates an earlier required name",
            ));
        }
        required.push(name.clone());
    }
    Ok(required)
}

impl Schema {
    fn validate(&self, value: &Value, path: &str) -> Result<(), ToolValidationError> {
        match self {
            Self::Nullable(schema) if value.is_null() => Ok(()),
            Self::Nullable(schema) => schema.validate(value, path),
            Self::Object {
                properties,
                required,
                additional_properties,
            } => {
                let Value::Object(object) = value else {
                    return Err(validation_error(path, "must be an object"));
                };
                for name in required {
                    if !object.contains_key(name) {
                        return Err(validation_error(&format!("{path}.{name}"), "is required"));
                    }
                }
                for (name, child) in object {
                    if let Some((_, schema)) =
                        properties.iter().find(|(candidate, _)| candidate == name)
                    {
                        schema.validate(child, &format!("{path}.{name}"))?;
                    } else if !additional_properties {
                        return Err(validation_error(
                            &format!("{path}.{name}"),
                            "additional property is not allowed",
                        ));
                    }
                }
                Ok(())
            }
            Self::Array(items) => {
                let Value::Array(values) = value else {
                    return Err(validation_error(path, "must be an array"));
                };
                for (index, value) in values.iter().enumerate() {
                    items.validate(value, &format!("{path}[{index}]"))?;
                }
                Ok(())
            }
            Self::String { values } => {
                let Value::String(value) = value else {
                    return Err(validation_error(path, "must be a string"));
                };
                if values
                    .as_ref()
                    .is_some_and(|values| !values.contains(value))
                {
                    return Err(validation_error(path, "is not an allowed enum value"));
                }
                Ok(())
            }
            Self::Number if value.is_number() => Ok(()),
            Self::Number => Err(validation_error(path, "must be a number")),
            Self::Integer if value.as_i64().is_some() || value.as_u64().is_some() => Ok(()),
            Self::Integer
                if value
                    .as_f64()
                    .is_some_and(|number| number.is_finite() && number.fract() == 0.0) =>
            {
                Ok(())
            }
            Self::Integer => Err(validation_error(path, "must be an integer")),
            Self::Boolean if value.is_boolean() => Ok(()),
            Self::Boolean => Err(validation_error(path, "must be a boolean")),
            Self::Null if value.is_null() => Ok(()),
            Self::Null => Err(validation_error(path, "must be null")),
        }
    }
}

fn validate_argument_budget(value: &Value) -> Result<(), ToolValidationError> {
    if !value.is_object() {
        return Err(validation_error("$.arguments", "must be an object"));
    }
    let mut nodes = 0;
    let mut bytes = 0;
    inspect_arguments(value, "$.arguments", 0, &mut nodes, &mut bytes)
}

fn inspect_arguments(
    value: &Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
    bytes: &mut usize,
) -> Result<(), ToolValidationError> {
    if depth > MAX_DEPTH {
        return Err(validation_error(
            path,
            format_args!("exceeds maximum depth {MAX_DEPTH}"),
        ));
    }
    *nodes = nodes.saturating_add(1);
    if *nodes > MAX_ARGUMENT_NODES {
        return Err(validation_error(
            path,
            format_args!("exceeds maximum of {MAX_ARGUMENT_NODES} nodes"),
        ));
    }
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                validate_argument_key(key, &format!("{path}.{key}"))?;
                add_argument_bytes(bytes, key.len(), path)?;
                inspect_arguments(value, &format!("{path}.{key}"), depth + 1, nodes, bytes)?;
            }
        }
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                inspect_arguments(value, &format!("{path}[{index}]"), depth + 1, nodes, bytes)?;
            }
        }
        Value::String(value) => {
            reject_reserved(value, path).map_err(|error| {
                ToolValidationError(error.0.replace("tool schema", "tool call"))
            })?;
            add_argument_bytes(bytes, value.len(), path)?;
        }
        _ => {}
    }
    Ok(())
}

fn add_argument_bytes(
    bytes: &mut usize,
    amount: usize,
    path: &str,
) -> Result<(), ToolValidationError> {
    *bytes = bytes.saturating_add(amount);
    if *bytes > MAX_ARGUMENT_BYTES {
        return Err(validation_error(
            path,
            format_args!("exceeds maximum of {MAX_ARGUMENT_BYTES} key/string bytes"),
        ));
    }
    Ok(())
}

fn inspect_strings_and_keys(
    value: &Value,
    path: &str,
    bytes: &mut usize,
    maximum: usize,
    property_key: bool,
    depth: usize,
) -> Result<(), ToolSchemaError> {
    const MAX_INSPECTION_DEPTH: usize = MAX_DEPTH * 2 + 8;
    if depth > MAX_INSPECTION_DEPTH {
        return Err(error(
            path,
            format_args!("exceeds maximum inspection depth {MAX_INSPECTION_DEPTH}"),
        ));
    }
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                reject_reserved(key, &format!("{path}.{key}"))?;
                if property_key {
                    validate_raw_key(key, &format!("{path}.{key}"))?;
                }
                *bytes = bytes.saturating_add(key.len());
                if *bytes > maximum {
                    return Err(error(
                        path,
                        format_args!("exceeds maximum of {maximum} key/string bytes"),
                    ));
                }
                let is_property_key = key == "properties";
                inspect_strings_and_keys(
                    value,
                    &format!("{path}.{key}"),
                    bytes,
                    maximum,
                    is_property_key,
                    depth + 1,
                )?;
            }
        }
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                inspect_strings_and_keys(
                    value,
                    &format!("{path}[{index}]"),
                    bytes,
                    maximum,
                    false,
                    depth + 1,
                )?;
            }
        }
        Value::String(value) => {
            reject_reserved(value, path)?;
            *bytes = bytes.saturating_add(value.len());
            if *bytes > maximum {
                return Err(error(
                    path,
                    format_args!("exceeds maximum of {maximum} key/string bytes"),
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_tool_name(name: &str, path: &str) -> Result<(), ToolSchemaError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(error(path, "must match ASCII [A-Za-z0-9_-]{1,64}"));
    }
    reject_reserved(name, path)
}

fn validate_raw_key(key: &str, path: &str) -> Result<(), ToolSchemaError> {
    if key.is_empty()
        || key.chars().any(|character| {
            character.is_control()
                || character.is_whitespace()
                || matches!(
                    character,
                    '"' | '<' | '>' | '{' | '}' | '[' | ']' | ',' | ':'
                )
        })
    {
        return Err(error(path, "is unsafe for raw template emission"));
    }
    reject_reserved(key, path)
}

fn validate_argument_key(key: &str, path: &str) -> Result<(), ToolValidationError> {
    validate_raw_key(key, path)
        .map_err(|error| ToolValidationError(error.0.replace("tool schema", "tool call")))
}

fn reject_reserved(value: &str, path: &str) -> Result<(), ToolSchemaError> {
    if let Some(control) = RESERVED_CONTROLS
        .iter()
        .find(|control| value.contains(**control))
    {
        return Err(error(
            path,
            format_args!("contains reserved control token {control:?}"),
        ));
    }
    Ok(())
}

fn expect_object<'a>(
    value: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, ToolSchemaError> {
    value
        .as_object()
        .ok_or_else(|| error(path, "must be an object"))
}

fn reject_unknown_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    path: &str,
) -> Result<(), ToolSchemaError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(error(&format!("{path}.{field}"), "unknown field"));
    }
    Ok(())
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<&'a str, ToolSchemaError> {
    match object.get(field) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(error(&format!("{path}.{field}"), "must be a string")),
        None => Err(error(&format!("{path}.{field}"), "is required")),
    }
}

fn optional_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<Option<&'a str>, ToolSchemaError> {
    match object.get(field) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(error(&format!("{path}.{field}"), "must be a string")),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Number, Value, json};

    use super::*;

    fn openai_tool(name: &str, schema: Value) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "tool description",
                "parameters": schema,
            }
        })
    }

    fn openai(tools: &Value) -> Result<ToolRegistry, ToolSchemaError> {
        ToolRegistry::from_openai(OpenAiToolsInput {
            tools: Some(tools),
            tool_choice: None,
            parallel_tool_calls: None,
        })
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments,
            raw: "<|tool_call>call:example{}<tool_call|>".into(),
            repaired: true,
        }
    }

    fn closed_schema() -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false,
        })
    }

    #[test]
    fn providers_normalize_to_identical_renderer_declarations() {
        let schema = json!({
            "type": "object",
            "description": "root is omitted",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "nested survives",
                    "title": "omitted",
                    "$comment": "omitted",
                    "default": "all",
                    "examples": ["all"]
                }
            },
            "required": ["query"],
            "additionalProperties": false
        });
        let openai_tools = json!([{
            "type": "function",
            "function": {
                "name": "search",
                "description": "find things",
                "parameters": schema.clone(),
                "strict": false
            }
        }]);
        let anthropic_tools = json!([{
            "name": "search",
            "description": "find things",
            "input_schema": schema,
            "strict": false
        }]);

        let openai = openai(&openai_tools).unwrap();
        let anthropic = ToolRegistry::from_anthropic(AnthropicToolsInput {
            tools: Some(&anthropic_tools),
            tool_choice: None,
        })
        .unwrap();

        assert_eq!(openai.render_tools(), anthropic.render_tools());
        assert_eq!(
            openai.render_tools(),
            &[json!({
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "find things",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "description": "nested survives",
                                "type": "string"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })]
        );
    }

    #[test]
    fn mode_defaults_and_explicit_choices_are_behavioral() {
        let empty = json!([]);
        let tools = json!([openai_tool("ping", closed_schema())]);
        let default_empty = openai(&empty).unwrap();
        let default_tools = openai(&tools).unwrap();
        assert_eq!(default_empty.mode(), ToolMode::None);
        assert!(default_empty.render_tools().is_empty());
        assert_eq!(default_tools.mode(), ToolMode::Auto);
        assert_eq!(default_tools.render_tools().len(), 1);

        let none = Value::String("none".into());
        let hidden = ToolRegistry::from_openai(OpenAiToolsInput {
            tools: Some(&tools),
            tool_choice: Some(&none),
            parallel_tool_calls: Some(&Value::Bool(true)),
        })
        .unwrap();
        assert_eq!(hidden.mode(), ToolMode::None);
        assert!(hidden.render_tools().is_empty());
        hidden.validate_generated(&call("ping", json!({}))).unwrap();

        for (kind, expected) in [("auto", ToolMode::Auto), ("none", ToolMode::None)] {
            let choice = json!({"type": kind, "disable_parallel_tool_use": false});
            let registry = ToolRegistry::from_anthropic(AnthropicToolsInput {
                tools: None,
                tool_choice: Some(&choice),
            })
            .unwrap();
            assert_eq!(registry.mode(), expected);
        }
    }

    #[test]
    fn nested_projection_and_validator_cover_the_supported_subset() {
        let schema = json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "count": {"type": "integer"},
                "note": {"type": ["null", "string"]},
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["mode", "count", "rows"],
            "additionalProperties": false
        });
        let tools = json!([openai_tool("run", schema)]);
        let registry = openai(&tools).unwrap();
        let parameters = &registry.render_tools()[0]["function"]["parameters"];
        assert_eq!(parameters["properties"]["note"]["type"], "string");
        assert_eq!(parameters["properties"]["note"]["nullable"], true);
        assert!(parameters.get("additionalProperties").is_none());

        let one_point_zero = Value::Number(Number::from_f64(1.0).unwrap());
        registry
            .validate_generated(&call(
                "run",
                json!({
                    "mode": "fast",
                    "count": one_point_zero,
                    "note": null,
                    "rows": [{"name": "first"}]
                }),
            ))
            .unwrap();

        for bad in [
            json!({"count": 1, "rows": []}),
            json!({"mode": "fast", "count": "1", "rows": []}),
            json!({"mode": "other", "count": 1, "rows": []}),
            json!({"mode": "fast", "count": 1, "rows": [{"name": 7}]}),
            json!({"mode": "fast", "count": 1, "rows": [{"name": "x", "extra": true}]}),
            json!({"mode": "fast", "count": 1, "rows": [], "extra": true}),
        ] {
            assert!(registry.validate_generated(&call("run", bad)).is_err());
        }
    }

    #[test]
    fn missing_parameters_is_a_closed_empty_object() {
        let tools = json!([{"type":"function","function":{"name":"ping"}}]);
        let registry = openai(&tools).unwrap();
        assert_eq!(registry.render_tools()[0]["function"]["description"], "");
        registry
            .validate_generated(&call("ping", json!({})))
            .unwrap();
        assert!(
            registry
                .validate_generated(&call("ping", json!({"extra": true})))
                .is_err()
        );
    }

    #[test]
    fn scalar_types_validate_without_coercion() {
        let registry = openai(&json!([openai_tool(
            "scalars",
            json!({
                "type":"object",
                "properties": {
                    "number":{"type":"number"},
                    "boolean":{"type":"boolean"},
                    "nothing":{"type":"null"}
                },
                "required":["number","boolean","nothing"],
                "additionalProperties":false
            })
        )]))
        .unwrap();
        registry
            .validate_generated(&call(
                "scalars",
                json!({"number":1.5,"boolean":true,"nothing":null}),
            ))
            .unwrap();
        assert!(
            registry
                .validate_generated(&call(
                    "scalars",
                    json!({"number":"1.5","boolean":1,"nothing":false}),
                ))
                .is_err()
        );
    }

    #[test]
    fn duplicate_names_and_required_entries_fail() {
        let duplicate_tools = json!([
            openai_tool("same", closed_schema()),
            openai_tool("same", closed_schema())
        ]);
        assert!(openai(&duplicate_tools).is_err());

        let duplicate_required = json!([openai_tool(
            "bad",
            json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x", "x"]
            })
        )]);
        assert!(openai(&duplicate_required).is_err());
    }

    #[test]
    fn strict_choice_and_parallel_controls_fail_closed() {
        for strict in [Value::Bool(true), Value::String("false".into())] {
            let tools = json!([{
                "type": "function",
                "function": {"name": "bad", "strict": strict}
            }]);
            assert!(openai(&tools).is_err());
        }

        let tools = json!([openai_tool("ping", closed_schema())]);
        for choice in [
            json!("required"),
            json!({"type":"function"}),
            json!(7),
            Value::Null,
        ] {
            assert!(
                ToolRegistry::from_openai(OpenAiToolsInput {
                    tools: Some(&tools),
                    tool_choice: Some(&choice),
                    parallel_tool_calls: None,
                })
                .is_err()
            );
        }
        for parallel in [json!(false), json!("true"), Value::Null] {
            assert!(
                ToolRegistry::from_openai(OpenAiToolsInput {
                    tools: Some(&tools),
                    tool_choice: None,
                    parallel_tool_calls: Some(&parallel),
                })
                .is_err()
            );
        }
        for choice in [
            json!({"type":"any"}),
            json!({"type":"tool","name":"ping"}),
            json!({"type":"auto","disable_parallel_tool_use":true}),
            json!({"type":"auto","unknown":false}),
            Value::Null,
        ] {
            assert!(
                ToolRegistry::from_anthropic(AnthropicToolsInput {
                    tools: None,
                    tool_choice: Some(&choice),
                })
                .is_err()
            );
        }
    }

    #[test]
    fn unknown_fields_and_unsupported_schema_constructs_fail() {
        let bad_tools = [
            json!([{"type":"function","function":{"name":"x"},"unknown":1}]),
            json!([{"type":"function","function":{"name":"x","unknown":1}}]),
            json!([{"name":"x","input_schema":{"type":"object"},"unknown":1}]),
        ];
        assert!(openai(&bad_tools[0]).is_err());
        assert!(openai(&bad_tools[1]).is_err());
        assert!(
            ToolRegistry::from_anthropic(AnthropicToolsInput {
                tools: Some(&bad_tools[2]),
                tool_choice: None,
            })
            .is_err()
        );

        let schemas = [
            json!({"type":"object","allOf":[]}),
            json!({"type":"object","additionalProperties":{"type":"string"}}),
            json!({"type":"array"}),
            json!({"type":"array","items":[]}),
            json!({"type":"string","enum":[1]}),
            json!({"type":["string","number"]}),
            json!({"type":["string","null","number"]}),
            json!({"type":"object","description":7}),
            json!({"type":"object","examples":"bad"}),
        ];
        for schema in schemas {
            let tools = json!([openai_tool("bad", schema)]);
            assert!(openai(&tools).is_err(), "schema should fail: {tools}");
        }
    }

    #[test]
    fn declaration_schema_and_argument_budgets_fail_closed() {
        let too_many = Value::Array(
            (0..=MAX_DECLARATIONS)
                .map(|index| openai_tool(&format!("t{index}"), closed_schema()))
                .collect(),
        );
        assert!(openai(&too_many).is_err());

        let huge_description = "x".repeat(MAX_DECLARATION_BYTES + 1);
        let huge = json!([{
            "type":"function",
            "function":{"name":"huge","description":huge_description}
        }]);
        assert!(openai(&huge).is_err());

        let properties: Map<String, Value> = (0..MAX_SCHEMA_NODES)
            .map(|index| (format!("p{index}"), json!({"type":"string"})))
            .collect();
        let too_many_nodes = json!([openai_tool(
            "nodes",
            json!({"type":"object","properties":properties})
        )]);
        assert!(openai(&too_many_nodes).is_err());

        let mut nested = json!({"type":"string"});
        for _ in 0..MAX_DEPTH {
            nested = json!({"type":"array","items":nested});
        }
        let too_deep = json!([openai_tool(
            "deep",
            json!({"type":"object","properties":{"value":nested}})
        )]);
        assert!(openai(&too_deep).is_err());

        let permissive = json!([openai_tool(
            "args",
            json!({"type":"object","additionalProperties":true})
        )]);
        let registry = openai(&permissive).unwrap();
        let oversized = "x".repeat(MAX_ARGUMENT_BYTES + 1);
        assert!(
            registry
                .validate_generated(&call("args", json!({"value":oversized})))
                .is_err()
        );
        let many_values: Vec<Value> = (0..MAX_ARGUMENT_NODES).map(|_| Value::Null).collect();
        assert!(
            registry
                .validate_generated(&call("args", json!({"values":many_values})))
                .is_err()
        );

        let mut deep_arguments = Value::Null;
        for _ in 0..=MAX_DEPTH {
            deep_arguments = Value::Array(vec![deep_arguments]);
        }
        assert!(
            registry
                .validate_generated(&call("args", json!({"value":deep_arguments})))
                .is_err()
        );
    }

    #[test]
    fn all_reserved_controls_fail_in_every_string_location() {
        for control in RESERVED_CONTROLS {
            let name = json!([openai_tool(&format!("x{control}"), closed_schema())]);
            assert!(openai(&name).is_err(), "tool name accepted {control:?}");

            let description = json!([{
                "type":"function",
                "function":{"name":"x","description":control}
            }]);
            assert!(
                openai(&description).is_err(),
                "description accepted {control:?}"
            );

            let schema_string = json!([openai_tool(
                "x",
                json!({
                    "type":"object",
                    "properties":{"value":{"type":"string","description":control}}
                })
            )]);
            assert!(
                openai(&schema_string).is_err(),
                "schema string accepted {control:?}"
            );

            let property = json!([openai_tool(
                "x",
                json!({"type":"object","properties":{control.to_string():{"type":"string"}}})
            )]);
            assert!(
                openai(&property).is_err(),
                "schema key accepted {control:?}"
            );

            let registry = openai(&json!([openai_tool(
                "x",
                json!({"type":"object","additionalProperties":true})
            )]))
            .unwrap();
            assert!(
                registry
                    .validate_generated(&call("x", json!({"value":control})))
                    .is_err(),
                "argument string accepted {control:?}"
            );
            assert!(
                registry
                    .validate_generated(&call("x", json!({control.to_string():"value"})))
                    .is_err(),
                "argument key accepted {control:?}"
            );
        }
    }

    #[test]
    fn raw_unsafe_keys_fail_without_reserved_controls() {
        for key in ["", "bad key", "bad:key", "bad,key", "bad\u{a0}key"] {
            let tools = json!([openai_tool(
                "x",
                json!({"type":"object","properties":{key.to_string():{"type":"string"}}})
            )]);
            assert!(openai(&tools).is_err(), "schema key accepted {key:?}");

            let registry = openai(&json!([openai_tool(
                "x",
                json!({"type":"object","additionalProperties":true})
            )]))
            .unwrap();
            assert!(
                registry
                    .validate_generated(&call("x", json!({key.to_string():true})))
                    .is_err(),
                "argument key accepted {key:?}"
            );
        }
    }

    #[test]
    fn generated_validation_never_mutates_the_borrowed_call() {
        let registry = openai(&json!([openai_tool(
            "known",
            json!({
                "type":"object",
                "properties":{"value":{"type":"string"}},
                "required":["value"],
                "additionalProperties":false
            })
        )]))
        .unwrap();

        let successful = ToolCall {
            name: "known".into(),
            arguments: json!({"value":"ok"}),
            raw: "exact raw success".into(),
            repaired: true,
        };
        let successful_before = successful.clone();
        registry.validate_generated(&successful).unwrap();
        assert_eq!(successful, successful_before);

        let failed = ToolCall {
            name: "known".into(),
            arguments: json!({"value":7}),
            raw: "exact raw failure".into(),
            repaired: false,
        };
        let failed_before = failed.clone();
        assert!(registry.validate_generated(&failed).is_err());
        assert_eq!(failed, failed_before);

        let unknown = call("unknown", json!({}));
        let unknown_before = unknown.clone();
        assert!(registry.validate_generated(&unknown).is_err());
        assert_eq!(unknown, unknown_before);
    }
}

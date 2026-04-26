pub mod analyzer;
pub mod embedder;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use self::analyzer::AnalyzerDefinition;
use self::embedder::EmbedderDefinition;

use crate::lexical::core::field::{
    BooleanOption, BytesOption, DateTimeOption, FloatOption, Geo3dOption, GeoOption, IntegerOption,
    TextOption,
};
use crate::vector::core::field::{FlatOption, HnswOption, IvfOption};

/// Policy for fields that are not declared in the schema.
///
/// Applied when a document is ingested with field names that do not appear in
/// [`Schema::fields`]. The default is [`DynamicFieldPolicy::Dynamic`], which
/// mirrors the "schema-less onboarding" design goal: users can start indexing
/// immediately without defining a schema upfront.
///
/// # Variants
///
/// - [`Strict`](Self::Strict): Unknown fields cause the ingest to fail. Use
///   when you want to enforce an exact schema contract.
/// - [`Dynamic`](Self::Dynamic) (default): Unknown fields are accepted; their
///   type is inferred from the value and a new field definition is added to
///   the schema automatically.
/// - [`Ignore`](Self::Ignore): Unknown fields are silently dropped. Use when
///   you want to ingest partially-structured data without rejecting it but
///   also without expanding the schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicFieldPolicy {
    /// Fail the ingest when any field is not declared in the schema.
    Strict,
    /// Infer the type of unknown fields and add them to the schema.
    #[default]
    Dynamic,
    /// Silently drop unknown fields.
    Ignore,
}

impl std::str::FromStr for DynamicFieldPolicy {
    type Err = crate::error::LaurusError;

    /// Parse a policy name (case-insensitive).
    ///
    /// Accepted values: `"strict"`, `"dynamic"`, `"ignore"`. This is the
    /// canonical policy parser used by all language bindings so the accepted
    /// spelling is identical across Python, Node.js, WASM, Ruby, and PHP.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::invalid_argument`] for any
    /// unrecognised value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "strict" => Ok(DynamicFieldPolicy::Strict),
            "dynamic" => Ok(DynamicFieldPolicy::Dynamic),
            "ignore" => Ok(DynamicFieldPolicy::Ignore),
            other => Err(crate::error::LaurusError::invalid_argument(format!(
                "unknown dynamic field policy '{other}' \
                 (expected 'strict', 'dynamic', or 'ignore')"
            ))),
        }
    }
}

/// Name of the automatically-injected external document ID field.
///
/// This is the sole field name with a `_` prefix that the engine accepts from
/// user code; all other `_`-prefixed names are rejected by
/// [`validate_field_name`].
pub const RESERVED_ID_FIELD: &str = "_id";

/// Returns `true` if `name` is a reserved field name that user code is
/// allowed to reference explicitly (currently only [`RESERVED_ID_FIELD`]).
///
/// # Arguments
///
/// * `name` - The field name to check.
pub fn is_allowed_reserved_field(name: &str) -> bool {
    name == RESERVED_ID_FIELD
}

/// Validates that a user-supplied field name does not collide with the
/// engine's reserved namespace.
///
/// Field names whose first character is `_` are reserved for the engine
/// (e.g. [`RESERVED_ID_FIELD`]) and cannot be declared by users. The only
/// exception is the allow-listed names returned by
/// [`is_allowed_reserved_field`].
///
/// # Arguments
///
/// * `name` - The field name to validate.
///
/// # Errors
///
/// Returns [`crate::error::LaurusError::invalid_argument`] if the name starts
/// with `_` and is not in the allow-list.
pub fn validate_field_name(name: &str) -> crate::error::Result<()> {
    if name.starts_with('_') && !is_allowed_reserved_field(name) {
        return Err(crate::error::LaurusError::invalid_argument(format!(
            "Field name '{name}' is reserved: names starting with '_' are \
             reserved for system fields (allowed: '{RESERVED_ID_FIELD}')"
        )));
    }
    Ok(())
}

/// Schema for the unified engine.
///
/// Declares what fields exist, their index types (lexical or vector),
/// and optional custom analyzer definitions. Custom analyzers are
/// referenced by name from [`TextOption::analyzer`].
///
/// The schema also carries a [`DynamicFieldPolicy`] that controls how
/// undeclared fields are handled during document ingestion. The default is
/// [`DynamicFieldPolicy::Dynamic`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Custom analyzer definitions, keyed by name.
    /// These can be referenced from text field `analyzer` settings.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub analyzers: HashMap<String, AnalyzerDefinition>,
    /// Embedder definitions, keyed by name.
    /// These can be referenced from vector field `embedder` settings.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub embedders: HashMap<String, EmbedderDefinition>,
    /// Options for each field.
    pub fields: HashMap<String, FieldOption>,
    /// Default fields for search.
    #[serde(default)]
    pub default_fields: Vec<String>,
    /// Policy for fields not declared in [`fields`](Self::fields).
    /// Defaults to [`DynamicFieldPolicy::Dynamic`].
    #[serde(default)]
    pub dynamic_field_policy: DynamicFieldPolicy,
}

impl Schema {
    pub fn new() -> Self {
        Self {
            analyzers: HashMap::new(),
            embedders: HashMap::new(),
            fields: HashMap::new(),
            default_fields: Vec::new(),
            dynamic_field_policy: DynamicFieldPolicy::default(),
        }
    }

    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for a single field in the unified schema.
///
/// Each variant directly represents a concrete field type.
/// For hybrid search, define separate fields for vector and lexical indexing.
///
/// Serializes using serde's externally tagged representation:
/// ```json
/// { "Text": { "indexed": true, "stored": true, "term_vectors": false } }
/// { "Hnsw": { "dimension": 384, "distance": "Cosine" } }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldOption {
    /// Text field options (lexical search).
    Text(TextOption),
    /// Integer field options.
    Integer(IntegerOption),
    /// Float field options.
    Float(FloatOption),
    /// Boolean field options.
    Boolean(BooleanOption),
    /// DateTime field options.
    DateTime(DateTimeOption),
    /// 2D geo field options.
    Geo(GeoOption),
    /// 3D ECEF geo field options.
    Geo3d(Geo3dOption),
    /// Bytes field options.
    Bytes(BytesOption),
    /// HNSW vector index options.
    Hnsw(HnswOption),
    /// Flat vector index options.
    Flat(FlatOption),
    /// IVF vector index options.
    Ivf(IvfOption),
}

impl FieldOption {
    /// Returns true if this is a vector field.
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Hnsw(_) | Self::Flat(_) | Self::Ivf(_))
    }

    /// Returns true if this is a lexical field.
    pub fn is_lexical(&self) -> bool {
        matches!(
            self,
            Self::Text(_)
                | Self::Integer(_)
                | Self::Float(_)
                | Self::Boolean(_)
                | Self::DateTime(_)
                | Self::Geo(_)
                | Self::Geo3d(_)
                | Self::Bytes(_)
        )
    }

    /// Converts to the vector-subsystem's `FieldOption` if this is a vector field.
    pub fn to_vector(&self) -> Option<crate::vector::core::field::FieldOption> {
        match self {
            Self::Hnsw(o) => Some(crate::vector::core::field::FieldOption::Hnsw(o.clone())),
            Self::Flat(o) => Some(crate::vector::core::field::FieldOption::Flat(o.clone())),
            Self::Ivf(o) => Some(crate::vector::core::field::FieldOption::Ivf(o.clone())),
            _ => None,
        }
    }

    /// Returns the embedder name if this is a vector field with an embedder configured.
    pub fn embedder_name(&self) -> Option<&str> {
        match self {
            Self::Hnsw(o) => o.embedder.as_deref(),
            Self::Flat(o) => o.embedder.as_deref(),
            Self::Ivf(o) => o.embedder.as_deref(),
            _ => None,
        }
    }

    /// Converts to the lexical-subsystem's `FieldOption` if this is a lexical field.
    pub fn to_lexical(&self) -> Option<crate::lexical::core::field::FieldOption> {
        match self {
            Self::Text(o) => Some(crate::lexical::core::field::FieldOption::Text(o.clone())),
            Self::Integer(o) => Some(crate::lexical::core::field::FieldOption::Integer(o.clone())),
            Self::Float(o) => Some(crate::lexical::core::field::FieldOption::Float(o.clone())),
            Self::Boolean(o) => Some(crate::lexical::core::field::FieldOption::Boolean(o.clone())),
            Self::DateTime(o) => Some(crate::lexical::core::field::FieldOption::DateTime(
                o.clone(),
            )),
            Self::Geo(o) => Some(crate::lexical::core::field::FieldOption::Geo(o.clone())),
            Self::Geo3d(o) => Some(crate::lexical::core::field::FieldOption::Geo3d(o.clone())),
            Self::Bytes(o) => Some(crate::lexical::core::field::FieldOption::Bytes(o.clone())),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct SchemaBuilder {
    analyzers: HashMap<String, AnalyzerDefinition>,
    embedders: HashMap<String, EmbedderDefinition>,
    fields: HashMap<String, FieldOption>,
    default_fields: Vec<String>,
    dynamic_field_policy: DynamicFieldPolicy,
}

impl SchemaBuilder {
    pub fn add_field(mut self, name: impl Into<String>, option: FieldOption) -> Self {
        let name = name.into();
        self.fields.insert(name, option);
        self
    }

    pub fn add_text_field(self, name: impl Into<String>, option: impl Into<TextOption>) -> Self {
        self.add_field(name, FieldOption::Text(option.into()))
    }

    pub fn add_integer_field(
        self,
        name: impl Into<String>,
        option: impl Into<IntegerOption>,
    ) -> Self {
        self.add_field(name, FieldOption::Integer(option.into()))
    }

    pub fn add_float_field(self, name: impl Into<String>, option: impl Into<FloatOption>) -> Self {
        self.add_field(name, FieldOption::Float(option.into()))
    }

    pub fn add_boolean_field(
        self,
        name: impl Into<String>,
        option: impl Into<BooleanOption>,
    ) -> Self {
        self.add_field(name, FieldOption::Boolean(option.into()))
    }

    pub fn add_datetime_field(
        self,
        name: impl Into<String>,
        option: impl Into<DateTimeOption>,
    ) -> Self {
        self.add_field(name, FieldOption::DateTime(option.into()))
    }

    pub fn add_geo_field(self, name: impl Into<String>, option: impl Into<GeoOption>) -> Self {
        self.add_field(name, FieldOption::Geo(option.into()))
    }

    /// Add a 3D ECEF geo field, indexed in a 3D BKD tree for sphere /
    /// k-NN queries (queries themselves arrive with #300–#302).
    pub fn add_geo3d_field(self, name: impl Into<String>, option: impl Into<Geo3dOption>) -> Self {
        self.add_field(name, FieldOption::Geo3d(option.into()))
    }

    pub fn add_bytes_field(self, name: impl Into<String>, option: impl Into<BytesOption>) -> Self {
        self.add_field(name, FieldOption::Bytes(option.into()))
    }

    pub fn add_hnsw_field(self, name: impl Into<String>, option: impl Into<HnswOption>) -> Self {
        self.add_field(name, FieldOption::Hnsw(option.into()))
    }

    pub fn add_flat_field(self, name: impl Into<String>, option: impl Into<FlatOption>) -> Self {
        self.add_field(name, FieldOption::Flat(option.into()))
    }

    pub fn add_ivf_field(self, name: impl Into<String>, option: impl Into<IvfOption>) -> Self {
        self.add_field(name, FieldOption::Ivf(option.into()))
    }

    pub fn add_default_field(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.default_fields.push(name);
        self
    }

    /// Add a custom analyzer definition to the schema.
    ///
    /// # Arguments
    ///
    /// * `name` - The analyzer name (referenced from `TextOption::analyzer`).
    /// * `definition` - The analyzer definition.
    pub fn add_analyzer(mut self, name: impl Into<String>, definition: AnalyzerDefinition) -> Self {
        self.analyzers.insert(name.into(), definition);
        self
    }

    /// Add an embedder definition to the schema.
    ///
    /// # Arguments
    ///
    /// * `name` - The embedder name (referenced from vector field `embedder`).
    /// * `definition` - The embedder definition.
    pub fn add_embedder(mut self, name: impl Into<String>, definition: EmbedderDefinition) -> Self {
        self.embedders.insert(name.into(), definition);
        self
    }

    /// Sets the policy for fields not declared in the schema.
    ///
    /// See [`DynamicFieldPolicy`] for the available options. When not set,
    /// the default is [`DynamicFieldPolicy::Dynamic`].
    ///
    /// # Arguments
    ///
    /// * `policy` - The dynamic field policy to apply during ingestion.
    pub fn dynamic_field_policy(mut self, policy: DynamicFieldPolicy) -> Self {
        self.dynamic_field_policy = policy;
        self
    }

    /// Build the schema, validating reserved field names.
    ///
    /// # Errors
    ///
    /// Returns an error if any field name starts with `_` and is not in the
    /// reserved allow-list (see [`validate_field_name`]).
    pub fn try_build(self) -> crate::error::Result<Schema> {
        for name in self.fields.keys() {
            validate_field_name(name)?;
        }
        Ok(Schema {
            analyzers: self.analyzers,
            embedders: self.embedders,
            fields: self.fields,
            default_fields: self.default_fields,
            dynamic_field_policy: self.dynamic_field_policy,
        })
    }

    /// Build the schema.
    ///
    /// # Panics
    ///
    /// Panics if any field name collides with a reserved name. Use
    /// [`try_build`](Self::try_build) for a fallible variant.
    pub fn build(self) -> Schema {
        self.try_build()
            .expect("SchemaBuilder::build: field name validation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::field::TextOption;

    #[test]
    fn default_dynamic_field_policy_is_dynamic() {
        assert_eq!(DynamicFieldPolicy::default(), DynamicFieldPolicy::Dynamic);
    }

    #[test]
    fn schema_new_uses_default_policy() {
        let schema = Schema::new();
        assert_eq!(schema.dynamic_field_policy, DynamicFieldPolicy::Dynamic);
    }

    #[test]
    fn schema_builder_sets_policy() {
        let schema = Schema::builder()
            .dynamic_field_policy(DynamicFieldPolicy::Strict)
            .build();
        assert_eq!(schema.dynamic_field_policy, DynamicFieldPolicy::Strict);
    }

    #[test]
    fn validate_field_name_accepts_regular_name() {
        assert!(validate_field_name("title").is_ok());
        assert!(validate_field_name("year_2024").is_ok());
        assert!(validate_field_name("a").is_ok());
    }

    #[test]
    fn validate_field_name_accepts_id() {
        assert!(validate_field_name(RESERVED_ID_FIELD).is_ok());
    }

    #[test]
    fn validate_field_name_rejects_underscore_prefix() {
        let err = validate_field_name("_score").unwrap_err();
        assert!(
            err.to_string().contains("reserved"),
            "unexpected error: {err}"
        );
        assert!(validate_field_name("_custom").is_err());
        assert!(validate_field_name("__foo").is_err());
    }

    #[test]
    fn schema_builder_try_build_rejects_reserved_name() {
        let result = Schema::builder()
            .add_field("_bad", FieldOption::Text(TextOption::default()))
            .try_build();
        assert!(result.is_err());
    }

    #[test]
    fn schema_builder_try_build_accepts_regular_names() {
        let result = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .try_build();
        assert!(result.is_ok());
    }

    #[test]
    fn schema_builder_add_geo3d_field_round_trips() {
        let schema = Schema::builder()
            .add_geo3d_field("position", Geo3dOption::default())
            .build();
        let opt = schema.fields.get("position").expect("field declared");
        match opt {
            FieldOption::Geo3d(g3d) => {
                assert!(g3d.indexed);
                assert!(g3d.stored);
            }
            other => panic!("expected FieldOption::Geo3d, got {other:?}"),
        }

        // The engine schema -> lexical schema bridge must preserve Geo3d.
        let lexical = opt.to_lexical().expect("Geo3d is a lexical field");
        assert!(matches!(
            lexical,
            crate::lexical::core::field::FieldOption::Geo3d(_)
        ));
        assert!(opt.is_lexical());
        assert!(!opt.is_vector());
    }

    #[test]
    fn dynamic_field_policy_serde_round_trip() {
        for policy in [
            DynamicFieldPolicy::Strict,
            DynamicFieldPolicy::Dynamic,
            DynamicFieldPolicy::Ignore,
        ] {
            let json = serde_json::to_string(&policy).unwrap();
            let back: DynamicFieldPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, back);
        }
    }
}

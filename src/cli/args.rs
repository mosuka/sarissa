//! Command line argument parsing for Sarissa CLI using clap.

use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Sarissa - A fast, featureful full-text search engine
#[derive(Parser, Debug, Clone)]
#[command(name = "sarissa")]
#[command(about = "A fast, featureful full-text search engine for Rust")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "Sarissa Contributors")]
#[command(long_about = None)]
pub struct SarissaArgs {
    /// Verbosity level (0=quiet, 1=normal, 2=verbose, 3=debug)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (overrides verbose)
    #[arg(short, long)]
    pub quiet: bool,

    /// Output format
    #[arg(short = 'f', long = "format", default_value = "human")]
    pub output_format: OutputFormat,

    /// Pretty-print JSON output
    #[arg(long)]
    pub pretty: bool,

    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Command,
}

impl SarissaArgs {
    /// Get the effective verbosity level
    pub fn verbosity(&self) -> u8 {
        if self.quiet {
            0
        } else {
            match self.verbose {
                0 => 1, // Default to normal
                n => n,
            }
        }
    }
}

/// Available CLI commands
#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Create a new search index
    #[command(name = "create-index")]
    CreateIndex(CreateIndexArgs),

    /// Add documents to an index
    #[command(name = "add-document")]
    AddDocument(AddDocumentArgs),

    /// Search an index
    Search(SearchArgs),

    /// Optimize an index
    Optimize(OptimizeArgs),

    /// Show index statistics
    Stats(StatsArgs),

    /// Run benchmarks
    Benchmark(BenchmarkArgs),

    /// Validate index integrity
    Validate(ValidateArgs),

    /// List all indices in a directory
    List(ListArgs),
}

/// Arguments for creating an index
#[derive(Parser, Debug, Clone)]
pub struct CreateIndexArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Schema definition file path (JSON)
    #[arg(short, long, value_name = "SCHEMA_FILE")]
    pub schema_file: Option<PathBuf>,

    /// Overwrite existing index
    #[arg(short, long)]
    pub force: bool,

    /// Create parent directories if they don't exist
    #[arg(long)]
    pub create_dirs: bool,
}

/// Arguments for adding documents
#[derive(Parser, Debug, Clone)]
pub struct AddDocumentArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Document file path (JSON or JSONL)
    #[arg(value_name = "DOCUMENT_FILE")]
    pub document_file: PathBuf,

    /// Batch size for bulk operations
    #[arg(short, long, default_value = "1000")]
    pub batch_size: usize,

    /// Don't commit after adding documents
    #[arg(long)]
    pub no_commit: bool,

    /// Show progress during indexing
    #[arg(long)]
    pub progress: bool,
}

/// Arguments for searching
#[derive(Parser, Debug, Clone)]
pub struct SearchArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Query string
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Field to search in (default: all fields)
    #[arg(long)]
    pub field: Option<String>,

    /// Maximum number of results to return
    #[arg(short, long, default_value = "10")]
    pub limit: usize,

    /// Offset for pagination
    #[arg(short, long, default_value = "0")]
    pub offset: usize,

    /// Include highlights in results
    #[arg(long)]
    pub highlight: bool,

    /// Fields to highlight (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub highlight_fields: Vec<String>,

    /// Include facets in results
    #[arg(long)]
    pub facets: bool,

    /// Facet fields to collect (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub facet_fields: Vec<String>,

    /// Search mode
    #[arg(short = 'm', long, default_value = "term")]
    pub mode: SearchMode,

    /// Enable spell correction
    #[arg(long)]
    pub spell_correction: bool,

    /// Disable spell correction
    #[arg(long, conflicts_with = "spell_correction")]
    pub no_spell_correction: bool,

    /// Show "Did you mean?" suggestions
    #[arg(long)]
    pub show_did_you_mean: bool,

    /// Automatically apply spelling corrections
    #[arg(long)]
    pub auto_correct: bool,

    /// Enable parallel search processing
    #[arg(long)]
    pub parallel: bool,

    /// Minimum score threshold for results
    #[arg(long)]
    pub min_score: Option<f32>,
}

/// Search modes available in CLI
#[derive(ValueEnum, Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Basic term search
    Term,
    /// Phrase search
    Phrase,
    /// Boolean search
    Boolean,
    /// Wildcard search
    Wildcard,
    /// Range search
    Range,
    /// Similarity search
    Similarity,
    /// Vector similarity search
    Vector,
}

/// Arguments for index optimization
#[derive(Parser, Debug, Clone)]
pub struct OptimizeArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Maximum number of segments after optimization
    #[arg(long)]
    pub max_segments: Option<usize>,

    /// Force optimization even if not needed
    #[arg(short, long)]
    pub force: bool,

    /// Show progress during optimization
    #[arg(long)]
    pub progress: bool,
}

/// Arguments for index statistics
#[derive(Parser, Debug, Clone)]
pub struct StatsArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Include detailed field statistics
    #[arg(short, long)]
    pub detailed: bool,

    /// Include index health information
    #[arg(long)]
    pub health: bool,

    /// Include storage usage breakdown
    #[arg(long)]
    pub storage: bool,
}

/// Arguments for benchmarking
#[derive(Parser, Debug, Clone)]
pub struct BenchmarkArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Benchmark configuration file
    #[arg(short, long)]
    pub config_file: Option<PathBuf>,

    /// Number of iterations
    #[arg(short, long, default_value = "100")]
    pub iterations: usize,

    /// Number of threads to use
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Output file for results
    #[arg(short, long)]
    pub output_file: Option<PathBuf>,

    /// Benchmark mode
    #[arg(short = 'm', long, default_value = "search")]
    pub mode: BenchmarkMode,

    /// Warmup iterations before benchmarking
    #[arg(long, default_value = "10")]
    pub warmup: usize,
}

/// Benchmark modes
#[derive(ValueEnum, Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkMode {
    /// Benchmark search operations
    Search,
    /// Benchmark indexing operations
    Indexing,
    /// Benchmark both search and indexing
    All,
}

/// Arguments for index validation
#[derive(Parser, Debug, Clone)]
pub struct ValidateArgs {
    /// Path to the index directory
    #[arg(value_name = "INDEX_PATH")]
    pub index_path: PathBuf,

    /// Fix issues found during validation
    #[arg(long)]
    pub fix: bool,

    /// Verbose validation output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Arguments for listing indices
#[derive(Parser, Debug, Clone)]
pub struct ListArgs {
    /// Directory to search for indices
    #[arg(value_name = "DIRECTORY", default_value = ".")]
    pub directory: PathBuf,

    /// Show detailed information
    #[arg(short, long)]
    pub long: bool,

    /// Include hidden indices
    #[arg(short = 'a', long)]
    pub all: bool,
}

/// Output formats for CLI
#[derive(ValueEnum, Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Human-readable output
    Human,
    /// JSON output
    Json,
    /// CSV output (for some commands)
    Csv,
    /// Table format
    Table,
    /// YAML output
    Yaml,
}

impl SearchArgs {
    /// Check if spell correction is enabled
    pub fn spell_correction_enabled(&self) -> bool {
        self.spell_correction && !self.no_spell_correction
    }
}

impl AddDocumentArgs {
    /// Check if commit should be performed
    pub fn should_commit(&self) -> bool {
        !self.no_commit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_basic_search_command() {
        let args = SarissaArgs::try_parse_from([
            "sarissa",
            "search",
            "/path/to/index",
            "test query",
            "--limit",
            "20",
            "--highlight",
        ])
        .unwrap();

        if let Command::Search(search_args) = args.command {
            assert_eq!(search_args.index_path, PathBuf::from("/path/to/index"));
            assert_eq!(search_args.query, "test query");
            assert_eq!(search_args.limit, 20);
            assert!(search_args.highlight);
        } else {
            panic!("Expected Search command");
        }
    }

    #[test]
    fn test_create_index_command() {
        let args = SarissaArgs::try_parse_from([
            "sarissa",
            "create-index",
            "/path/to/index",
            "--schema-file",
            "schema.json",
            "--force",
        ])
        .unwrap();

        if let Command::CreateIndex(create_args) = args.command {
            assert_eq!(create_args.index_path, PathBuf::from("/path/to/index"));
            assert_eq!(create_args.schema_file, Some(PathBuf::from("schema.json")));
            assert!(create_args.force);
        } else {
            panic!("Expected CreateIndex command");
        }
    }

    #[test]
    fn test_benchmark_command() {
        let args = SarissaArgs::try_parse_from([
            "sarissa",
            "benchmark",
            "/path/to/index",
            "--iterations",
            "500",
            "--mode",
            "all",
            "--threads",
            "4",
        ])
        .unwrap();

        if let Command::Benchmark(benchmark_args) = args.command {
            assert_eq!(benchmark_args.index_path, PathBuf::from("/path/to/index"));
            assert_eq!(benchmark_args.iterations, 500);
            assert!(matches!(benchmark_args.mode, BenchmarkMode::All));
            assert_eq!(benchmark_args.threads, Some(4));
        } else {
            panic!("Expected Benchmark command");
        }
    }

    #[test]
    fn test_verbosity_levels() {
        // Default verbosity
        let args = SarissaArgs::try_parse_from(["sarissa", "list"]).unwrap();
        assert_eq!(args.verbosity(), 1);

        // Verbose flag
        let args = SarissaArgs::try_parse_from(["sarissa", "-v", "list"]).unwrap();
        assert_eq!(args.verbosity(), 1);

        // Multiple verbose flags
        let args = SarissaArgs::try_parse_from(["sarissa", "-vv", "list"]).unwrap();
        assert_eq!(args.verbosity(), 2);

        // Quiet flag
        let args = SarissaArgs::try_parse_from(["sarissa", "--quiet", "list"]).unwrap();
        assert_eq!(args.verbosity(), 0);
    }

    #[test]
    fn test_output_format() {
        let args = SarissaArgs::try_parse_from(["sarissa", "--format", "json", "list"]).unwrap();
        assert!(matches!(args.output_format, OutputFormat::Json));
    }

    #[test]
    fn test_search_modes() {
        let args = SarissaArgs::try_parse_from([
            "sarissa",
            "search",
            "/path/to/index",
            "query",
            "--mode",
            "boolean",
        ])
        .unwrap();

        if let Command::Search(search_args) = args.command {
            assert!(matches!(search_args.mode, SearchMode::Boolean));
        } else {
            panic!("Expected Search command");
        }
    }

    #[test]
    fn test_spell_correction_flags() {
        // Spell correction enabled
        let args = SarissaArgs::try_parse_from([
            "sarissa",
            "search",
            "/path/to/index",
            "query",
            "--spell-correction",
        ])
        .unwrap();

        if let Command::Search(search_args) = args.command {
            assert!(search_args.spell_correction_enabled());
        }

        // Spell correction disabled
        let args = SarissaArgs::try_parse_from([
            "sarissa",
            "search",
            "/path/to/index",
            "query",
            "--no-spell-correction",
        ])
        .unwrap();

        if let Command::Search(search_args) = args.command {
            assert!(!search_args.spell_correction_enabled());
        }
    }
}

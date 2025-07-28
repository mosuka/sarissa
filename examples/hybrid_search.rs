//! Hybrid Search example - demonstrates combining keyword and vector search.

use sarissa::embeding::{EmbeddingConfig, EmbeddingMethod};
use sarissa::hybrid_search::{HybridSearchConfig, HybridSearchEngine, ScoreNormalization};
use sarissa::index::index::IndexConfig;
use sarissa::prelude::*;
use sarissa::query::TermQuery;
use sarissa::schema::{IdField, TextField};
use sarissa::search::{SearchEngine, SearchRequest};
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Hybrid Search Example - Combining Keyword and Vector Search ===\n");

    // Create a temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create a schema for traditional keyword search
    let mut schema = Schema::new()?;
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field("body", Box::new(TextField::new().indexed(true)))?;
    schema.add_field("author", Box::new(IdField::new()))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;

    // Create a keyword search engine
    let mut keyword_engine =
        SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Create hybrid search configuration
    let hybrid_config = HybridSearchConfig {
        keyword_weight: 0.6, // 60% weight for keyword search
        vector_weight: 0.4,  // 40% weight for vector search
        min_keyword_score: 0.0,
        min_vector_similarity: 0.1,
        max_results: 10,
        require_both: false, // Accept results from either search method
        normalization: ScoreNormalization::MinMax,
        embedding_config: EmbeddingConfig {
            dimension: 128,
            method: EmbeddingMethod::TfIdf,
            normalize: true,
            min_term_freq: 2,
            max_vocab_size: 10000,
            parallel: true,
        },
        ..Default::default()
    };

    // Create hybrid search engine
    let mut hybrid_engine = HybridSearchEngine::new(hybrid_config.clone())?;

    println!("=== Adding Documents ===");

    // Sample documents with both text content and simulated embeddings
    let documents = vec![
        Document::builder()
            .add_text("title", "Rust Programming Language")
            .add_text(
                "body",
                "Rust is a systems programming language focused on safety, speed, and concurrency. It provides zero-cost abstractions and memory safety without garbage collection.",
            )
            .add_text("author", "Steve Klabnik")
            .add_text("category", "programming")
            .build(),
        Document::builder()
            .add_text("title", "Python Machine Learning")
            .add_text(
                "body",
                "Python is widely used for machine learning and data science. Libraries like scikit-learn, TensorFlow, and PyTorch make it easy to build ML models.",
            )
            .add_text("author", "Sebastian Raschka")
            .add_text("category", "data-science")
            .build(),
        Document::builder()
            .add_text("title", "JavaScript Web Development")
            .add_text(
                "body",
                "JavaScript is the language of the web, enabling interactive user interfaces and server-side development with Node.js.",
            )
            .add_text("author", "Douglas Crockford")
            .add_text("category", "web-development")
            .build(),
        Document::builder()
            .add_text("title", "Deep Learning Fundamentals")
            .add_text(
                "body",
                "Deep learning uses neural networks with multiple layers to learn complex patterns in data. It's revolutionizing AI applications.",
            )
            .add_text("author", "Ian Goodfellow")
            .add_text("category", "artificial-intelligence")
            .build(),
        Document::builder()
            .add_text("title", "Rust Web Frameworks")
            .add_text(
                "body",
                "Rust offers several web frameworks like Actix-web, Warp, and Rocket for building high-performance web applications.",
            )
            .add_text("author", "Programming Community")
            .add_text("category", "web-development")
            .build(),
    ];

    // Train the embedder with sample text FIRST
    println!("Training embedder...");
    let training_texts = vec![
        "rust programming language systems software development safety performance concurrency",
        "python machine learning artificial intelligence data science algorithms tensorflow",
        "javascript web development frontend backend node frameworks react",
        "neural networks deep learning fundamentals algorithms patterns data",
        "programming languages rust python javascript development frameworks web systems",
        "machine learning deep learning artificial intelligence data science python",
        "web development javascript frameworks frontend backend programming",
        "rust web frameworks actix rocket warp high performance systems",
        "python data science machine learning libraries scikit pandas numpy",
    ];
    hybrid_engine.train_embedder(&training_texts).await?;

    // Add documents to keyword search engine
    println!("Adding {} documents to keyword index...", documents.len());
    keyword_engine.add_documents(documents.clone())?;

    // Add documents to hybrid search engine (after embedder is trained)
    println!("Adding documents to hybrid search engine...");
    for (doc_id, doc) in documents.iter().enumerate() {
        let mut fields = HashMap::new();

        // Extract text fields from document
        if let Some(title_field) = doc.get_field("title") {
            if let Some(title) = title_field.as_text() {
                fields.insert("title".to_string(), title.to_string());
            }
        }
        if let Some(body_field) = doc.get_field("body") {
            if let Some(body) = body_field.as_text() {
                fields.insert("body".to_string(), body.to_string());
            }
        }
        if let Some(author_field) = doc.get_field("author") {
            if let Some(author) = author_field.as_text() {
                fields.insert("author".to_string(), author.to_string());
            }
        }
        if let Some(category_field) = doc.get_field("category") {
            if let Some(category) = category_field.as_text() {
                fields.insert("category".to_string(), category.to_string());
            }
        }

        hybrid_engine.add_document(doc_id as u64, fields).await?;
    }

    println!("Embedder training completed!\n");

    // Display hybrid search engine statistics
    let stats = hybrid_engine.stats().await;
    println!("=== Hybrid Search Engine Statistics ===");
    println!("Total documents: {}", stats.total_documents);
    println!("Embedder trained: {}", stats.embedder_trained);
    println!("Embedding dimension: {}", stats.embedding_dimension);
    println!();

    println!("=== Hybrid Search Examples ===\n");

    // Example 1: Search for "Rust programming"
    println!("1. Hybrid search for 'Rust programming':");
    let keyword_query = Box::new(TermQuery::new("body", "Rust"));
    let results = hybrid_engine
        .search("Rust programming", &keyword_engine, keyword_query)
        .await?;

    println!("   Found {} results", results.len());
    println!("   Keyword matches: {}", results.keyword_matches);
    println!("   Vector matches: {}", results.vector_matches);
    println!("   Query time: {} ms", results.query_time_ms);

    for (i, result) in results.results.iter().take(3).enumerate() {
        println!(
            "   {}. Doc ID: {}, Hybrid Score: {:.4}",
            i + 1,
            result.doc_id,
            result.hybrid_score
        );
        if let Some(keyword_score) = result.keyword_score {
            println!("      Keyword Score: {:.4}", keyword_score);
        } else {
            println!("      Keyword Score: None");
        }
        if let Some(vector_similarity) = result.vector_similarity {
            println!("      Vector Similarity: {:.4}", vector_similarity);
        } else {
            println!("      Vector Similarity: None");
        }
        if let Some(document) = &result.document {
            if let Some(title) = document.get("title") {
                println!("      Title: {}", title);
            }
        }
        println!();
    }

    // Example 2: Search for "machine learning"
    println!("2. Hybrid search for 'machine learning':");
    let keyword_query = Box::new(TermQuery::new("body", "machine"));
    let results = hybrid_engine
        .search(
            "machine learning algorithms",
            &keyword_engine,
            keyword_query,
        )
        .await?;

    println!("   Found {} results", results.len());
    println!("   Keyword matches: {}", results.keyword_matches);
    println!("   Vector matches: {}", results.vector_matches);

    for (i, result) in results.results.iter().take(3).enumerate() {
        println!(
            "   {}. Doc ID: {}, Hybrid Score: {:.4}",
            i + 1,
            result.doc_id,
            result.hybrid_score
        );
        if let Some(keyword_score) = result.keyword_score {
            println!("      Keyword Score: {:.4}", keyword_score);
        } else {
            println!("      Keyword Score: None");
        }
        if let Some(vector_similarity) = result.vector_similarity {
            println!("      Vector Similarity: {:.4}", vector_similarity);
        } else {
            println!("      Vector Similarity: None");
        }
        if let Some(document) = &result.document {
            if let Some(title) = document.get("title") {
                println!("      Title: {}", title);
            }
        }
    }
    println!();

    // Example 3: Search for "web development"
    println!("3. Hybrid search for 'web development':");
    let keyword_query = Box::new(TermQuery::new("category", "web-development"));
    let results = hybrid_engine
        .search("web frontend javascript", &keyword_engine, keyword_query)
        .await?;

    println!("   Found {} results", results.len());
    println!("   Keyword matches: {}", results.keyword_matches);
    println!("   Vector matches: {}", results.vector_matches);

    for (i, result) in results.results.iter().take(3).enumerate() {
        println!(
            "   {}. Doc ID: {}, Hybrid Score: {:.4}",
            i + 1,
            result.doc_id,
            result.hybrid_score
        );
        if let Some(keyword_score) = result.keyword_score {
            println!("      Keyword Score: {:.4}", keyword_score);
        } else {
            println!("      Keyword Score: None");
        }
        if let Some(vector_similarity) = result.vector_similarity {
            println!("      Vector Similarity: {:.4}", vector_similarity);
        } else {
            println!("      Vector Similarity: None");
        }
        if let Some(document) = &result.document {
            if let Some(title) = document.get("title") {
                println!("      Title: {}", title);
            }
            if let Some(category) = document.get("category") {
                println!("      Category: {}", category);
            }
        }
    }
    println!();

    // Example 4: Demonstrate different normalization strategies
    println!("4. Testing different score normalization strategies:");

    let normalization_strategies = vec![
        ("None", ScoreNormalization::None),
        ("MinMax", ScoreNormalization::MinMax),
        ("ZScore", ScoreNormalization::ZScore),
        ("Rank", ScoreNormalization::Rank),
    ];

    for (name, strategy) in normalization_strategies {
        let mut config = HybridSearchConfig::default();
        config.normalization = strategy;
        config.max_results = 3;

        let _test_engine = HybridSearchEngine::new(config)?;
        println!("   {} normalization:", name);

        // Note: In a real implementation, we would need to re-add documents and train
        // For this example, we'll just show the configuration
        println!("      Strategy: {:?}", strategy);
    }
    println!();

    // Example 5: Configuration showcase
    println!("5. Hybrid search configuration options:");
    println!("   Keyword weight: {}", hybrid_config.keyword_weight);
    println!("   Vector weight: {}", hybrid_config.vector_weight);
    println!("   Min keyword score: {}", hybrid_config.min_keyword_score);
    println!(
        "   Min vector similarity: {}",
        hybrid_config.min_vector_similarity
    );
    println!("   Max results: {}", hybrid_config.max_results);
    println!("   Require both: {}", hybrid_config.require_both);
    println!("   Normalization: {:?}", hybrid_config.normalization);
    println!();

    // Example 6: Keyword-only search comparison
    println!("6. Pure keyword search comparison:");
    let keyword_query = Box::new(TermQuery::new("body", "programming"));
    let keyword_request = SearchRequest::new(keyword_query).load_documents(true);
    let keyword_results = keyword_engine.search_mut(keyword_request)?;

    println!(
        "   Pure keyword search found {} results",
        keyword_results.total_hits
    );
    for (i, hit) in keyword_results.hits.iter().take(3).enumerate() {
        println!(
            "   {}. Score: {:.4}, Doc ID: {}",
            i + 1,
            hit.score,
            hit.doc_id
        );
        if let Some(doc) = &hit.document {
            if let Some(field_value) = doc.get_field("title") {
                if let Some(title) = field_value.as_text() {
                    println!("      Title: {}", title);
                }
            }
        }
    }
    println!();

    // Clean up
    hybrid_engine.clear().await?;
    keyword_engine.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hybrid_search_example() {
        let result = main().await;
        assert!(result.is_ok());
    }
}

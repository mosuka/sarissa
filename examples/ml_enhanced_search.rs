//! Machine Learning Enhanced Search Example
//!
//! This example demonstrates how to use the ML module to enhance search quality
//! with learning-to-rank and query expansion features.

use sarissa::index::index::IndexConfig;
use sarissa::ml::query_expansion::{QueryExpansion, QueryExpansionConfig};
use sarissa::ml::ranking::{LearningToRank, ModelType, RankingConfig};
use sarissa::ml::{
    FeedbackSignal, FeedbackType, MLConfig, MLContext, SearchHistoryItem, UserSession,
};
use sarissa::prelude::*;
use sarissa::query::{BooleanQuery, TermQuery};
use sarissa::schema::{IdField, TextField};
use sarissa::search::{SearchEngine, SearchRequest};
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Machine Learning Enhanced Search Example ===\n");

    // Create temporary directory for the index
    let temp_dir = TempDir::new().unwrap();
    println!("Creating index in: {:?}", temp_dir.path());

    // Create schema
    let mut schema = Schema::new();
    schema.add_field(
        "title",
        Box::new(TextField::new().stored(true).indexed(true)),
    )?;
    schema.add_field(
        "body",
        Box::new(TextField::new().indexed(true).stored(true)),
    )?;
    schema.add_field("author", Box::new(IdField::new()))?;
    schema.add_field("category", Box::new(TextField::new().indexed(true)))?;

    // Create search engine
    let mut engine = SearchEngine::create_in_dir(temp_dir.path(), schema, IndexConfig::default())?;

    // Configure ML features
    let ml_config = MLConfig {
        enabled: true,
        models_directory: temp_dir.path().join("models").to_str().unwrap().to_string(),
        ranking: RankingConfig {
            enabled: true,
            model_type: ModelType::GBDT,
            online_learning: true,
            ..Default::default()
        },
        query_expansion: QueryExpansionConfig {
            enabled: true,
            max_expansions: 5,
            enable_synonyms: true,
            enable_semantic: true,
            ..Default::default()
        },
        ..Default::default()
    };

    println!("=== Adding Documents ===");

    // Sample documents about various programming topics
    let documents = vec![
        Document::builder()
            .add_text("title", "Introduction to Rust Programming")
            .add_text("body", "Rust is a systems programming language that focuses on safety, speed, and concurrency. It achieves memory safety without using garbage collection.")
            .add_text("author", "John Doe")
            .add_text("category", "programming")
            .build(),
        Document::builder()
            .add_text("title", "Machine Learning with Python")
            .add_text("body", "Python is the most popular language for machine learning and artificial intelligence. Libraries like TensorFlow and PyTorch make it easy to build models.")
            .add_text("author", "Jane Smith")
            .add_text("category", "data-science")
            .build(),
        Document::builder()
            .add_text("title", "Deep Learning Fundamentals")
            .add_text("body", "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn from data. It powers many AI applications today.")
            .add_text("author", "Alice Johnson")
            .add_text("category", "artificial-intelligence")
            .build(),
        Document::builder()
            .add_text("title", "Web Development with JavaScript")
            .add_text("body", "JavaScript is essential for modern web development. From frontend frameworks like React to backend with Node.js, JavaScript is everywhere.")
            .add_text("author", "Bob Wilson")
            .add_text("category", "web-development")
            .build(),
        Document::builder()
            .add_text("title", "Data Structures and Algorithms")
            .add_text("body", "Understanding data structures and algorithms is fundamental for software engineering. Topics include arrays, trees, graphs, and sorting algorithms.")
            .add_text("author", "Charlie Brown")
            .add_text("category", "computer-science")
            .build(),
        Document::builder()
            .add_text("title", "Natural Language Processing")
            .add_text("body", "NLP enables computers to understand human language. Applications include sentiment analysis, machine translation, and chatbots.")
            .add_text("author", "David Lee")
            .add_text("category", "artificial-intelligence")
            .build(),
    ];

    // Add documents
    println!("Adding {} documents to index...", documents.len());
    engine.add_documents(documents)?;

    // Create ML context with user session and search history
    let ml_context = MLContext {
        user_session: Some(UserSession {
            session_id: "example-session-123".to_string(),
            user_id: Some("user-456".to_string()),
            start_time: chrono::Utc::now(),
            user_agent: Some("Mozilla/5.0".to_string()),
            ip_address: "127.0.0.1".to_string(),
        }),
        search_history: vec![
            SearchHistoryItem {
                query: "machine learning".to_string(),
                clicked_documents: vec!["1".to_string(), "2".to_string()],
                dwell_times: HashMap::new(),
                timestamp: chrono::Utc::now() - chrono::Duration::hours(1),
                result_count: 3,
            },
            SearchHistoryItem {
                query: "python programming".to_string(),
                clicked_documents: vec!["1".to_string()],
                dwell_times: HashMap::new(),
                timestamp: chrono::Utc::now() - chrono::Duration::hours(2),
                result_count: 2,
            },
        ],
        user_preferences: HashMap::from([
            ("data-science".to_string(), 0.8),
            ("artificial-intelligence".to_string(), 0.7),
            ("programming".to_string(), 0.6),
        ]),
        timestamp: chrono::Utc::now(),
    };

    println!("\n=== Query Expansion Example ===");

    // Initialize query expander
    let query_expander = QueryExpansion::new(ml_config.query_expansion.clone())?;

    // Note: In a real implementation, synonyms would be loaded from a dictionary file
    // For this example, we'll proceed without adding synonyms manually

    // Expand a query
    let original_query = "ML python";
    println!("Original query: '{}'", original_query);

    let expanded_query = query_expander.expand_query(original_query, &ml_context)?;
    println!("Expanded query intent: {:?}", expanded_query.intent);
    println!("Expansion confidence: {:.2}", expanded_query.confidence);
    println!("Expanded terms: {:?}", expanded_query.expanded_terms);

    // Search with different terms to show diversity
    let search_terms = vec!["machine", "learning", "python", "artificial"];
    let mut bool_query = BooleanQuery::new();
    for term in search_terms {
        bool_query.add_should(Box::new(TermQuery::new("body", term)));
    }

    let mut search_request = SearchRequest::new(Box::new(bool_query));
    search_request.config.load_documents = true;
    search_request.config.max_docs = 5;

    let results = engine.search_mut(search_request)?;

    println!("\nSearch results with query expansion:");
    for (i, hit) in results.hits.iter().enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title_field) = doc.get_field("title") {
                if let Some(title) = title_field.as_text() {
                    println!("  {}. {} (score: {:.4})", i + 1, title, hit.score);
                }
            }
        }
    }

    println!("\n=== Learning to Rank Example ===");

    // Initialize LTR system
    let ltr_system = LearningToRank::new(ml_config.ranking.clone())?;

    // Simulate user feedback
    let feedback_signals = vec![
        FeedbackSignal {
            query: "machine learning".to_string(),
            document_id: "1".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 0.8,
            timestamp: chrono::Utc::now(),
        },
        FeedbackSignal {
            query: "machine learning".to_string(),
            document_id: "2".to_string(),
            feedback_type: FeedbackType::DwellTime(std::time::Duration::from_secs(45)),
            relevance_score: 0.9,
            timestamp: chrono::Utc::now(),
        },
        FeedbackSignal {
            query: "python programming".to_string(),
            document_id: "1".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 0.7,
            timestamp: chrono::Utc::now(),
        },
        FeedbackSignal {
            query: "deep learning".to_string(),
            document_id: "2".to_string(),
            feedback_type: FeedbackType::Click,
            relevance_score: 0.95,
            timestamp: chrono::Utc::now(),
        },
    ];

    // Process feedback for LTR system
    println!(
        "Processing {} feedback signals for learning...",
        feedback_signals.len()
    );
    for signal in feedback_signals {
        ltr_system.process_feedback(signal)?;
    }

    // Search and re-rank results
    let query = Box::new(TermQuery::new("body", "machine"));
    let mut search_request = SearchRequest::new(query);
    search_request.config.load_documents = true;
    search_request.config.max_docs = 10;

    let search_results = engine.search_mut(search_request)?;

    println!("\nOriginal search results:");
    for (i, hit) in search_results.hits.iter().take(3).enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title_field) = doc.get_field("title") {
                if let Some(title) = title_field.as_text() {
                    println!("  {}. {} (score: {:.4})", i + 1, title, hit.score);
                }
            }
        }
    }

    // Re-rank using ML model
    println!("\nApplying learning-to-rank...");

    // Extract documents from search results
    let documents: Vec<Document> = search_results
        .hits
        .iter()
        .filter_map(|hit| hit.document.clone())
        .collect();

    // Create reranking context
    let reranking_context = sarissa::ml::ranking::RerankingContext {
        vector_similarities: HashMap::new(),
        semantic_distances: HashMap::new(),
        user_context_score: Some(0.8),
        additional_features: HashMap::new(),
    };

    let reranked_results = ltr_system.rerank_results(
        "machine learning",
        &search_results,
        &documents,
        &reranking_context,
    )?;

    println!("Re-ranked results:");
    for (i, hit) in reranked_results.hits.iter().take(3).enumerate() {
        if let Some(doc) = &hit.document {
            if let Some(title_field) = doc.get_field("title") {
                if let Some(title) = title_field.as_text() {
                    println!("  {}. {} (score: {:.4})", i + 1, title, hit.score);
                }
            }
        }
    }

    // Compare ranking differences
    println!("\nRanking comparison:");
    let original_scores: Vec<f32> = search_results
        .hits
        .iter()
        .take(3)
        .map(|hit| hit.score)
        .collect();
    let reranked_scores: Vec<f32> = reranked_results
        .hits
        .iter()
        .take(3)
        .map(|hit| hit.score)
        .collect();

    println!("Original scores: {:?}", original_scores);
    println!("Re-ranked scores: {:?}", reranked_scores);

    if original_scores != reranked_scores {
        println!("✓ Learning-to-rank successfully modified the document scores!");
    } else {
        println!(
            "⚠ Learning-to-rank did not change the scores (model may need more training data)"
        );
    }

    println!("\n=== ML Context Features ===");

    // Demonstrate ML context usage
    println!(
        "User session ID: {}",
        ml_context.user_session.as_ref().unwrap().session_id
    );
    println!(
        "Search history entries: {}",
        ml_context.search_history.len()
    );
    println!("User preferences:");
    for (category, weight) in &ml_context.user_preferences {
        println!("  - {}: {:.2}", category, weight);
    }

    // Feature extraction example
    println!("\n=== Feature Extraction Example ===");

    let feature_extractor = sarissa::ml::features::FeatureExtractor::new();
    let _feature_context = sarissa::ml::features::FeatureContext {
        document_id: "doc1".to_string(),
        vector_similarity: Some(0.85),
        semantic_distance: Some(0.3),
        user_context_score: Some(0.75),
        timestamp: chrono::Utc::now(),
    };

    // Extract features from multiple documents to verify diversity
    for (i, doc) in documents.iter().take(3).enumerate() {
        let feature_context = sarissa::ml::features::FeatureContext {
            document_id: format!("doc{}", i + 1),
            vector_similarity: Some(0.7 + i as f64 * 0.1), // Vary by document
            semantic_distance: Some(0.3 - i as f64 * 0.05),
            user_context_score: Some(0.75 - i as f64 * 0.1),
            timestamp: chrono::Utc::now(),
        };

        match feature_extractor.extract_features("machine learning", doc, &feature_context) {
            Ok(features) => {
                let title = doc
                    .get_field("title")
                    .and_then(|f| f.as_text())
                    .unwrap_or("Unknown");
                println!("Features for Document {}: '{}'", i + 1, title);
                println!("  - BM25 Score: {:.4}", features.bm25_score);
                println!("  - TF-IDF Score: {:.4}", features.tf_idf_score);
                println!(
                    "  - Query Term Coverage: {:.4}",
                    features.query_term_coverage
                );
                println!("  - Vector Similarity: {:.4}", features.vector_similarity);
                println!("  - Document Length: {}", features.document_length);
                println!("  - Click Through Rate: {:.4}", features.click_through_rate);
                println!(
                    "  - Document Popularity: {:.4}",
                    features.document_popularity
                );
            }
            Err(e) => {
                println!("Feature extraction failed for document {}: {}", i + 1, e);
            }
        }
        println!(); // Empty line for readability
    }

    // Clean up
    engine.close()?;

    println!("\n=== Example Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ml_enhanced_search() {
        let result = main().await;
        assert!(result.is_ok());
    }
}

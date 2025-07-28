//! Example demonstrating the PipelineAnalyzer
//!
//! This example shows how to create a custom analysis pipeline by combining
//! different tokenizers and filters.

use sarissa::analysis::{
    Analyzer, LowercaseFilter, PipelineAnalyzer, RegexTokenizer, StopFilter, Tokenizer,
};
use std::sync::Arc;

fn main() -> sarissa::error::Result<()> {
    // Create a regex tokenizer
    let tokenizer = Arc::new(RegexTokenizer::new()?);

    // Create a pipeline analyzer with multiple filters
    let analyzer = PipelineAnalyzer::new(tokenizer.clone())
        .add_filter(Arc::new(LowercaseFilter::new()))
        .add_filter(Arc::new(StopFilter::from_words(vec![
            "the", "and", "a", "an",
        ])))
        .with_name("custom_pipeline");

    // Sample text to analyze
    let text = "The Quick Brown Fox and the Lazy Dog";

    println!("Analyzing text: \"{}\"", text);
    println!(
        "Using pipeline: {} tokenizer -> lowercase filter -> stop filter",
        tokenizer.name()
    );
    println!();

    // Analyze the text
    let tokens: Vec<_> = analyzer.analyze(text)?.collect();

    // Display the results
    println!("Tokens produced:");
    for (i, token) in tokens.iter().enumerate() {
        println!(
            "  [{}] \"{}\" (offset: {}-{}, position: {})",
            i, token.text, token.start_offset, token.end_offset, token.position
        );
    }

    println!();
    println!("Total tokens: {}", tokens.len());

    // Create another pipeline without stop words
    println!("\n--- Pipeline without stop words ---");
    let analyzer_no_stop =
        PipelineAnalyzer::new(tokenizer).add_filter(Arc::new(LowercaseFilter::new()));

    let tokens_no_stop: Vec<_> = analyzer_no_stop.analyze(text)?.collect();

    println!("Tokens produced:");
    for (i, token) in tokens_no_stop.iter().enumerate() {
        println!(
            "  [{}] \"{}\" (offset: {}-{}, position: {})",
            i, token.text, token.start_offset, token.end_offset, token.position
        );
    }

    println!();
    println!("Total tokens: {}", tokens_no_stop.len());

    Ok(())
}

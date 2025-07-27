//! Example demonstrating filter pipelines and chaining multiple filters.

use sarissa::analysis::token_filter::{
    BoostFilter, Filter, LimitFilter, LowercaseFilter, RemoveEmptyFilter, StopFilter, StemFilter,
    StripFilter,
};
use sarissa::analysis::tokenizer::{RegexTokenizer, Tokenizer, WhitespaceTokenizer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== Filter Pipeline Examples ===\n");

    // Example 1: Basic text processing pipeline
    println!("1. Basic Text Processing Pipeline:");
    let text = "  The QUICK brown foxes are RUNNING quickly through the forest.  ";
    basic_pipeline(text)?;

    // Example 2: Search engine pipeline
    println!("\n2. Search Engine Pipeline:");
    let search_text = "The developers are programming APPLICATIONS using advanced algorithms";
    search_engine_pipeline(search_text)?;

    // Example 3: Social media content pipeline
    println!("\n3. Social Media Content Pipeline:");
    let social_text = "LOL this is AMAZING!!! Check out #rustlang programming   ";
    social_media_pipeline(social_text)?;

    // Example 4: Academic text processing
    println!("\n4. Academic Text Processing:");
    let academic_text = "The experimental methodology demonstrates significant improvements";
    academic_pipeline(academic_text)?;

    // Example 5: Comparing different pipeline configurations
    println!("\n5. Pipeline Configuration Comparison:");
    let comparison_text = "Programming languages are continuously EVOLVING with new features";
    compare_pipelines(comparison_text)?;

    // Example 6: Custom pipeline with boost
    println!("\n6. Custom Pipeline with Boosting:");
    let boost_text = "important critical significant normal regular basic";
    boosted_pipeline(boost_text)?;

    // Example 7: Minimal processing pipeline
    println!("\n7. Minimal Processing Pipeline:");
    let minimal_text = "Rust is fast and safe";
    minimal_pipeline(minimal_text)?;

    // Example 8: Maximum processing pipeline
    println!("\n8. Maximum Processing Pipeline:");
    let max_text = "   The RESEARCHERS are DEVELOPING innovative SOLUTIONS   ";
    maximum_pipeline(max_text)?;

    println!("\n=== Pipeline Strategies ===\n");
    
    println!("Basic Pipeline (Web Search):");
    println!("  1. Tokenize → 2. Lowercase → 3. Stop Words → 4. Stem");
    println!("  Use case: General web search, basic text processing");
    
    println!("\nSearch Engine Pipeline:");
    println!("  1. Tokenize → 2. Strip → 3. Lowercase → 4. Stop Words → 5. Stem → 6. Limit");
    println!("  Use case: Search engines, document retrieval");
    
    println!("\nSocial Media Pipeline:");
    println!("  1. Tokenize → 2. Lowercase → 3. Remove Empty → 4. Custom Stop Words");
    println!("  Use case: Social media analysis, sentiment analysis");
    
    println!("\nAcademic Pipeline:");
    println!("  1. Tokenize → 2. Strip → 3. Lowercase → 4. Stem (Porter) → 5. Remove Empty");
    println!("  Use case: Academic research, scientific texts");
    
    println!("\nBoosted Pipeline:");
    println!("  1. Tokenize → 2. Lowercase → 3. Boost → 4. Limit");
    println!("  Use case: Relevance ranking, weighted search");

    Ok(())
}

fn basic_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    // Step 1: Tokenize
    let tokenizer = WhitespaceTokenizer::new();
    let mut tokens = tokenizer.tokenize(text)?;
    println!("After tokenization: {:?}", 
        tokens.collect::<Vec<_>>().iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Step 2: Lowercase
    tokens = tokenizer.tokenize(text)?;
    let lowercase_filter = LowercaseFilter::new();
    tokens = lowercase_filter.filter(tokens)?;
    println!("After lowercase: {:?}", 
        tokens.collect::<Vec<_>>().iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Step 3: Stop words removal
    tokens = tokenizer.tokenize(text)?;
    tokens = lowercase_filter.filter(tokens)?;
    let stop_filter = StopFilter::new();
    tokens = stop_filter.filter(tokens)?;
    println!("After stop words: {:?}", 
        tokens.collect::<Vec<_>>().iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Step 4: Stemming
    tokens = tokenizer.tokenize(text)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = stop_filter.filter(tokens)?;
    let stem_filter = StemFilter::new();
    tokens = stem_filter.filter(tokens)?;
    let final_tokens: Vec<_> = tokens.collect();
    println!("After stemming: {:?}", 
        final_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    Ok(())
}

fn search_engine_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    let tokenizer = RegexTokenizer::new()?;
    let mut tokens = tokenizer.tokenize(text)?;
    
    // Chain multiple filters
    let strip_filter = StripFilter::new();
    let lowercase_filter = LowercaseFilter::new();
    let stop_filter = StopFilter::new();
    let stem_filter = StemFilter::new();
    let limit_filter = LimitFilter::new(10);
    
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = stop_filter.filter(tokens)?;
    tokens = stem_filter.filter(tokens)?;
    tokens = limit_filter.filter(tokens)?;
    
    let final_tokens: Vec<_> = tokens.collect();
    println!("Final result: {:?}", 
        final_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    println!("Token count: {}", final_tokens.len());
    
    Ok(())
}

fn social_media_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    let tokenizer = WhitespaceTokenizer::new();
    let mut tokens = tokenizer.tokenize(text)?;
    
    // Social media specific processing
    let lowercase_filter = LowercaseFilter::new();
    let remove_empty_filter = RemoveEmptyFilter::new();
    let custom_stop_filter = StopFilter::from_words(vec![
        "lol".to_string(), "omg".to_string(), "wow".to_string()
    ]);
    
    tokens = lowercase_filter.filter(tokens)?;
    tokens = remove_empty_filter.filter(tokens)?;
    tokens = custom_stop_filter.filter(tokens)?;
    
    let final_tokens: Vec<_> = tokens.collect();
    println!("Final result: {:?}", 
        final_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    Ok(())
}

fn academic_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    let tokenizer = RegexTokenizer::new()?;
    let mut tokens = tokenizer.tokenize(text)?;
    
    // Academic text processing
    let strip_filter = StripFilter::new();
    let lowercase_filter = LowercaseFilter::new();
    let stem_filter = StemFilter::new(); // Porter stemmer for precision
    let remove_empty_filter = RemoveEmptyFilter::new();
    
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = stem_filter.filter(tokens)?;
    tokens = remove_empty_filter.filter(tokens)?;
    
    let final_tokens: Vec<_> = tokens.collect();
    println!("Final result: {:?}", 
        final_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    Ok(())
}

fn compare_pipelines(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    // Pipeline A: Minimal processing
    println!("\nPipeline A (Minimal):");
    let tokenizer = WhitespaceTokenizer::new();
    let mut tokens_a = tokenizer.tokenize(text)?;
    let lowercase_filter = LowercaseFilter::new();
    tokens_a = lowercase_filter.filter(tokens_a)?;
    let result_a: Vec<_> = tokens_a.collect();
    println!("Result: {:?}", result_a.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Pipeline B: Standard processing
    println!("\nPipeline B (Standard):");
    let mut tokens_b = tokenizer.tokenize(text)?;
    let stop_filter = StopFilter::new();
    let stem_filter = StemFilter::simple(); // Simple stemmer
    tokens_b = lowercase_filter.filter(tokens_b)?;
    tokens_b = stop_filter.filter(tokens_b)?;
    tokens_b = stem_filter.filter(tokens_b)?;
    let result_b: Vec<_> = tokens_b.collect();
    println!("Result: {:?}", result_b.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    // Pipeline C: Aggressive processing
    println!("\nPipeline C (Aggressive):");
    let mut tokens_c = tokenizer.tokenize(text)?;
    let porter_stem_filter = StemFilter::new(); // Porter stemmer
    let limit_filter = LimitFilter::new(5);
    tokens_c = lowercase_filter.filter(tokens_c)?;
    tokens_c = stop_filter.filter(tokens_c)?;
    tokens_c = porter_stem_filter.filter(tokens_c)?;
    tokens_c = limit_filter.filter(tokens_c)?;
    let result_c: Vec<_> = tokens_c.collect();
    println!("Result: {:?}", result_c.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    Ok(())
}

fn boosted_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    let tokenizer = WhitespaceTokenizer::new();
    let mut tokens = tokenizer.tokenize(text)?;
    
    // Apply boost to important terms
    let lowercase_filter = LowercaseFilter::new();
    let boost_filter = BoostFilter::new(2.0);
    let limit_filter = LimitFilter::new(4);
    
    tokens = lowercase_filter.filter(tokens)?;
    tokens = boost_filter.filter(tokens)?;
    tokens = limit_filter.filter(tokens)?;
    
    let final_tokens: Vec<_> = tokens.collect();
    println!("Final result with boost:");
    for (i, token) in final_tokens.iter().enumerate() {
        println!("  {}: '{}' (boost: {:.2})", i, token.text, token.boost);
    }
    
    Ok(())
}

fn minimal_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize(text)?;
    let final_tokens: Vec<_> = tokens.collect();
    println!("Final result (tokenization only): {:?}", 
        final_tokens.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    Ok(())
}

fn maximum_pipeline(text: &str) -> Result<()> {
    println!("Input: {:?}", text);
    
    let tokenizer = RegexTokenizer::new()?;
    let mut tokens = tokenizer.tokenize(text)?;
    
    // Apply all filters in sequence
    let strip_filter = StripFilter::new();
    let lowercase_filter = LowercaseFilter::new();
    let remove_empty_filter = RemoveEmptyFilter::new();
    let stop_filter = StopFilter::new();
    let stem_filter = StemFilter::new();
    let boost_filter = BoostFilter::new(1.5);
    let limit_filter = LimitFilter::new(8);
    
    println!("Processing steps:");
    
    tokens = strip_filter.filter(tokens)?;
    let step1: Vec<_> = tokens.collect();
    println!("  1. Strip: {:?}", step1.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    tokens = tokenizer.tokenize(text)?;
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    let step2: Vec<_> = tokens.collect();
    println!("  2. Lowercase: {:?}", step2.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    tokens = tokenizer.tokenize(text)?;
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = remove_empty_filter.filter(tokens)?;
    let step3: Vec<_> = tokens.collect();
    println!("  3. Remove empty: {:?}", step3.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    tokens = tokenizer.tokenize(text)?;
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = remove_empty_filter.filter(tokens)?;
    tokens = stop_filter.filter(tokens)?;
    let step4: Vec<_> = tokens.collect();
    println!("  4. Stop words: {:?}", step4.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    tokens = tokenizer.tokenize(text)?;
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = remove_empty_filter.filter(tokens)?;
    tokens = stop_filter.filter(tokens)?;
    tokens = stem_filter.filter(tokens)?;
    let step5: Vec<_> = tokens.collect();
    println!("  5. Stemming: {:?}", step5.iter().map(|t| &t.text).collect::<Vec<_>>());
    
    tokens = tokenizer.tokenize(text)?;
    tokens = strip_filter.filter(tokens)?;
    tokens = lowercase_filter.filter(tokens)?;
    tokens = remove_empty_filter.filter(tokens)?;
    tokens = stop_filter.filter(tokens)?;
    tokens = stem_filter.filter(tokens)?;
    tokens = boost_filter.filter(tokens)?;
    tokens = limit_filter.filter(tokens)?;
    let final_tokens: Vec<_> = tokens.collect();
    
    println!("  6. Final result:");
    for (i, token) in final_tokens.iter().enumerate() {
        println!("    {}: '{}' (boost: {:.2})", i, token.text, token.boost);
    }
    
    Ok(())
}
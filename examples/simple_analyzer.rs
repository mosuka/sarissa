//! Example demonstrating the SimpleAnalyzer
//!
//! This example shows how to use a simple analyzer that only performs
//! tokenization without any filtering.

use sarissa::analysis::{Analyzer, RegexTokenizer, SimpleAnalyzer, WhitespaceTokenizer};
use std::sync::Arc;

fn main() -> sarissa::error::Result<()> {
    println!("=== SimpleAnalyzer Example ===\n");

    // Sample text to analyze
    let text = "Hello, World! This is a SIMPLE test-case with 123 numbers.";

    // Example 1: Simple analyzer with regex tokenizer
    println!("1. Simple analyzer with RegexTokenizer:");
    let regex_tokenizer = Arc::new(RegexTokenizer::new()?);
    let simple_regex = SimpleAnalyzer::new(regex_tokenizer);

    let tokens: Vec<_> = simple_regex.analyze(text)?.collect();
    
    println!("   Input: \"{}\"", text);
    println!("   Tokens:");
    for token in &tokens {
        println!("     - \"{}\" (position: {})", token.text, token.position);
    }
    println!("   Total tokens: {}\n", tokens.len());

    // Example 2: Simple analyzer with whitespace tokenizer
    println!("2. Simple analyzer with WhitespaceTokenizer:");
    let whitespace_tokenizer = Arc::new(WhitespaceTokenizer::new());
    let simple_whitespace = SimpleAnalyzer::new(whitespace_tokenizer);

    let tokens: Vec<_> = simple_whitespace.analyze(text)?.collect();
    
    println!("   Input: \"{}\"", text);
    println!("   Tokens:");
    for token in &tokens {
        println!("     - \"{}\" (position: {})", token.text, token.position);
    }
    println!("   Total tokens: {}\n", tokens.len());

    // Example 3: Demonstrating that SimpleAnalyzer preserves case
    println!("3. Case preservation:");
    let case_text = "UPPERCASE lowercase MixedCase";
    let tokens: Vec<_> = simple_regex.analyze(case_text)?.collect();
    
    println!("   Input: \"{}\"", case_text);
    println!("   Tokens:");
    for token in &tokens {
        println!("     - \"{}\"", token.text);
    }

    Ok(())
}
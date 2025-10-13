//! Simple example demonstrating synonym expansion with SynonymGraphFilter.
//!
//! This example shows how to:
//! - Create a synonym dictionary
//! - Apply synonym expansion to text
//! - Observe how tokens are expanded with synonyms

use sarissa::analysis::token_filter::{Filter, SynonymDictionary, SynonymGraphFilter};
use sarissa::analysis::tokenizer::{Tokenizer, WhitespaceTokenizer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== Synonym Search Example ===\n");

    // 1. Create a synonym dictionary
    println!("1. Creating synonym dictionary...");
    let mut dict = SynonymDictionary::new(None)?;

    // Add synonym groups
    dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);

    dict.add_synonym_group(vec![
        "ai".to_string(),
        "artificial intelligence".to_string(),
    ]);

    println!("   ✓ Added synonyms:");
    println!("     - 'ml' ↔ 'machine learning'");
    println!("     - 'ai' ↔ 'artificial intelligence'\n");

    // 2. Process text with synonyms
    println!("2. Processing text: \"ml and ai tutorial\"\n");

    let tokenizer = WhitespaceTokenizer;
    let tokens = tokenizer.tokenize("ml and ai tutorial")?;

    let synonym_filter = SynonymGraphFilter::new(dict, true);
    let result_tokens = synonym_filter.filter(tokens)?;

    println!("   Resulting tokens:");
    for (i, token) in result_tokens.enumerate() {
        println!(
            "     [{}] text='{}' position={} pos_inc={} pos_len={}",
            i, token.text, token.position, token.position_increment, token.position_length
        );
    }

    println!("\n=== Explanation ===");
    println!(
        "• position_increment=0 means the token is at the same position as the previous token"
    );
    println!("• position_length>1 means the token spans multiple positions (multi-word synonym)");
    println!("• 'ml' is expanded to include 'machine learning' as a synonym");
    println!("• 'ai' is expanded to include 'artificial intelligence' as a synonym");

    Ok(())
}

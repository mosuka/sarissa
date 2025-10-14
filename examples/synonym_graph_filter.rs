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
    println!("=== SynonymGraphFilter Usage Example ===\n");

    // 1. Create a synonym dictionary
    println!("Step 1: Creating synonym dictionary");
    let mut dict = SynonymDictionary::new(None)?;

    // Add synonym groups (bidirectional)
    dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    dict.add_synonym_group(vec![
        "ai".to_string(),
        "artificial intelligence".to_string(),
    ]);

    println!("  ✓ Added synonyms:");
    println!("    - 'ml' ↔ 'machine learning'");
    println!("    - 'ai' ↔ 'artificial intelligence'\n");

    // 2. Create the filter
    println!("Step 2: Creating SynonymGraphFilter");
    let tokenizer = WhitespaceTokenizer;
    let synonym_filter = SynonymGraphFilter::new(dict, true); // keep_original=true
    println!("  ✓ Filter created (keeping original tokens)\n");

    // 3. Apply filter to text
    println!("Step 3: Applying filter to text");
    let input_text = "ml tutorial";
    println!("  Input: \"{}\"\n", input_text);

    let tokens = tokenizer.tokenize(input_text)?;
    let result_tokens = synonym_filter.filter(tokens)?;

    // 4. Display results
    println!("  Output tokens:");
    for (i, token) in result_tokens.enumerate() {
        println!(
            "    [{}] '{}' (position={}, pos_inc={}, pos_len={})",
            i, token.text, token.position, token.position_increment, token.position_length
        );
    }

    println!("\n  Explanation:");
    println!("    - 'ml' is the original token");
    println!("    - 'machine' and 'learning' are synonym expansions");
    println!("    - 'machine' has pos_len=2, spanning both 'machine' and 'learning' positions");
    println!("    - This allows phrase queries to match correctly\n");

    Ok(())
}

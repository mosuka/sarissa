//! Example demonstrating the usage of UnicodeWordTokenizer.

use sarissa::analysis::tokenizer::{Tokenizer, UnicodeWordTokenizer};
use sarissa::error::Result;

fn main() -> Result<()> {
    println!("=== UnicodeWordTokenizer Examples ===\n");

    let tokenizer = UnicodeWordTokenizer::new();

    // Example 1: Basic English text
    println!("1. Basic English text:");
    demonstrate_tokenizer(&tokenizer, "Hello, world! How are you?")?;

    // Example 2: Mixed languages
    println!("\n2. Mixed languages:");
    demonstrate_tokenizer(&tokenizer, "Hello 世界 Мир mundo العالم")?;

    // Example 3: French with accents
    println!("\n3. French with accents:");
    demonstrate_tokenizer(&tokenizer, "café, naïve, résumé, français")?;

    // Example 4: German with umlauts
    println!("\n4. German with umlauts:");
    demonstrate_tokenizer(&tokenizer, "Müller, Größe, Weiß, Straße")?;

    // Example 5: Japanese (Hiragana, Katakana, Kanji)
    println!("\n5. Japanese text:");
    demonstrate_tokenizer(&tokenizer, "こんにちは、世界！カタカナ and ひらがな")?;

    // Example 6: Arabic text
    println!("\n6. Arabic text:");
    demonstrate_tokenizer(&tokenizer, "مرحبا بالعالم، كيف حالك؟")?;

    // Example 7: Russian Cyrillic
    println!("\n7. Russian Cyrillic:");
    demonstrate_tokenizer(&tokenizer, "Привет, мир! Как дела?")?;

    // Example 8: Mixed with numbers and punctuation
    println!("\n8. Mixed with numbers and punctuation:");
    demonstrate_tokenizer(
        &tokenizer,
        "Price: $123.45, Количество: 10件, Tél: +33-1-23-45-67-89",
    )?;

    // Example 9: Programming code
    println!("\n9. Programming code:");
    demonstrate_tokenizer(
        &tokenizer,
        "fn main() { let x = 42; println!(\"Hello, {}\", x); }",
    )?;

    // Example 10: Emoji and symbols
    println!("\n10. Emoji and symbols:");
    demonstrate_tokenizer(&tokenizer, "Rust 🦀 is fast! ⚡ Love it ❤️ 100% 👍")?;

    // Example 11: Hyphenated words
    println!("\n11. Hyphenated words:");
    demonstrate_tokenizer(&tokenizer, "state-of-the-art, twenty-one, self-explanatory")?;

    // Example 12: Contractions
    println!("\n12. Contractions:");
    demonstrate_tokenizer(&tokenizer, "don't, can't, won't, it's, we're")?;

    // Example 13: URLs and emails
    println!("\n13. URLs and emails:");
    demonstrate_tokenizer(
        &tokenizer,
        "Visit https://example.com or email user@domain.com",
    )?;

    // Example 14: Chinese text
    println!("\n14. Chinese text:");
    demonstrate_tokenizer(&tokenizer, "你好世界！这是一个测试。")?;

    // Example 15: Mixed scripts and numbers
    println!("\n15. Mixed scripts and numbers:");
    demonstrate_tokenizer(&tokenizer, "Rust 2024年 версия 1.75 в наличии")?;

    println!("\n=== Word Boundary Analysis ===\n");

    // Demonstrate word boundary detection
    word_boundary_demo(&tokenizer)?;

    Ok(())
}

fn demonstrate_tokenizer(tokenizer: &dyn Tokenizer, text: &str) -> Result<()> {
    println!("Input: \"{text}\"");

    let tokens: Vec<_> = tokenizer.tokenize(text)?.collect();

    if tokens.is_empty() {
        println!("No tokens found.");
    } else {
        println!("Tokens ({}):", tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            // Show character count for Unicode awareness
            let char_count = token.text.chars().count();
            let byte_count = token.text.len();
            println!(
                "  {}: \"{}\" (pos: {}, offset: {}..{}, chars: {}, bytes: {})",
                i,
                token.text,
                token.position,
                token.start_offset,
                token.end_offset,
                char_count,
                byte_count
            );
        }
    }

    Ok(())
}

fn word_boundary_demo(tokenizer: &UnicodeWordTokenizer) -> Result<()> {
    println!("16. Word boundary analysis:");

    let examples = vec![
        ("English", "Hello-world, it's great!"),
        ("Spaces", "word1   word2\t\tword3"),
        ("Punctuation", "word1,word2;word3.word4"),
        ("Numbers", "abc123def456ghi"),
        ("Mixed", "café123مرحبا456世界"),
    ];

    for (name, text) in examples {
        println!("\n{name}: \"{text}\"");
        let tokens: Vec<_> = tokenizer.tokenize(text)?.collect();
        let words: Vec<String> = tokens.iter().map(|t| t.text.to_string()).collect();
        println!("  Words: {words:?}");
    }

    Ok(())
}

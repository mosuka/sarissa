//! Integration tests for advanced query functionality.

use sarissa::analysis::*;
use sarissa::error::Result;
use sarissa::query::*;

#[test]
fn test_phrase_query_creation() -> Result<()> {
    let phrase_query = PhraseQuery::from_phrase("content", "quick brown");

    assert_eq!(phrase_query.field(), "content");
    assert_eq!(phrase_query.terms(), &["quick", "brown"]);
    assert_eq!(phrase_query.slop(), 0);
    assert_eq!(phrase_query.boost(), 1.0);

    Ok(())
}

#[test]
fn test_phrase_query_with_slop() -> Result<()> {
    let phrase_query = PhraseQuery::from_phrase("content", "brown fox").with_slop(2);

    assert_eq!(phrase_query.slop(), 2);

    Ok(())
}

#[test]
fn test_wildcard_query_creation() -> Result<()> {
    let wildcard_query = WildcardQuery::new("title", "qu*")?;

    assert_eq!(wildcard_query.field(), "title");
    assert_eq!(wildcard_query.pattern(), "qu*");
    assert_eq!(wildcard_query.boost(), 1.0);

    Ok(())
}

#[test]
fn test_wildcard_pattern_matching() -> Result<()> {
    // Test simple wildcard
    let query = WildcardQuery::new("field", "hello*")?;
    assert!(query.matches("hello"));
    assert!(query.matches("helloworld"));
    assert!(!query.matches("hell"));

    // Test question mark
    let query = WildcardQuery::new("field", "h?llo")?;
    assert!(query.matches("hello"));
    assert!(query.matches("hallo"));
    assert!(!query.matches("hllo"));
    assert!(!query.matches("heello"));

    Ok(())
}

#[test]
fn test_range_query_creation() -> Result<()> {
    let range_query = RangeQuery::new("category", Some("a".to_string()), Some("m".to_string()));

    assert_eq!(range_query.field(), "category");
    assert_eq!(*range_query.lower_bound(), Bound::Included("a".to_string()));
    assert_eq!(*range_query.upper_bound(), Bound::Included("m".to_string()));
    assert_eq!(range_query.boost(), 1.0);

    Ok(())
}

#[test]
fn test_range_query_contains() -> Result<()> {
    let query = RangeQuery::new(
        "field",
        Some("apple".to_string()),
        Some("zebra".to_string()),
    );

    assert!(query.contains("apple"));
    assert!(query.contains("hello"));
    assert!(query.contains("zebra"));
    assert!(!query.contains("aardvark"));
    assert!(!query.contains("zoo"));

    Ok(())
}

#[test]
fn test_range_query_greater_than() -> Result<()> {
    let query = RangeQuery::greater_than("field", "hello".to_string());

    assert!(!query.contains("hello"));
    assert!(query.contains("world"));
    assert!(!query.contains("apple"));

    Ok(())
}

#[test]
fn test_query_boost_modification() -> Result<()> {
    let mut phrase_query = PhraseQuery::from_phrase("content", "test phrase");
    phrase_query.set_boost(2.5);
    assert_eq!(phrase_query.boost(), 2.5);

    let mut wildcard_query = WildcardQuery::new("field", "test*")?;
    wildcard_query.set_boost(1.5);
    assert_eq!(wildcard_query.boost(), 1.5);

    let mut range_query = RangeQuery::new("field", None, None);
    range_query.set_boost(3.0);
    assert_eq!(range_query.boost(), 3.0);

    Ok(())
}

#[test]
fn test_stemming_functionality() -> Result<()> {
    let porter_stemmer = PorterStemmer::new();

    assert_eq!(porter_stemmer.stem("running"), "run");
    assert_eq!(porter_stemmer.stem("flies"), "fli");
    assert_eq!(porter_stemmer.stem("agreed"), "agre");

    let simple_stemmer = SimpleStemmer::new();
    assert_eq!(simple_stemmer.stem("running"), "runn");
    assert_eq!(simple_stemmer.stem("beautiful"), "beauti");

    Ok(())
}

#[test]
fn test_stem_filter() -> Result<()> {
    let filter = StemFilter::new();
    let tokens = vec![
        Token::new("running", 0),
        Token::new("flies", 1),
        Token::new("test", 2).stop(),
    ];
    let token_stream = Box::new(tokens.into_iter());

    let result: Vec<Token> = filter.filter(token_stream)?.collect();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].text, "run");
    assert_eq!(result[1].text, "fli");
    assert_eq!(result[2].text, "test"); // Stopped tokens are not processed
    assert!(result[2].is_stopped());

    Ok(())
}

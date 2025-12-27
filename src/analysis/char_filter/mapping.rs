use aho_corasick::AhoCorasick;
use std::collections::HashMap;

use super::{CharFilter, Transformation};

pub struct MappingCharFilter {
    ac: AhoCorasick,
    replacements: Vec<String>,
}

impl MappingCharFilter {
    pub fn new(mapping: HashMap<String, String>) -> crate::error::Result<Self> {
        let mut keys = Vec::new();
        let mut replacements = Vec::new();

        // Sort keys to ensure deterministic order if needed, though AhoCorasick handles ordering options.
        // We iterate mapping. AhoCorasick uses parallel arrays for construction.
        for (k, v) in mapping {
            keys.push(k);
            replacements.push(v);
        }

        let ac = AhoCorasick::new(&keys)
            .map_err(|e| crate::error::SarissaError::Anyhow(anyhow::Error::from(e)))?;

        Ok(Self { ac, replacements })
    }
}

impl CharFilter for MappingCharFilter {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>) {
        let mut output = String::with_capacity(input.len());
        let mut transformations = Vec::new();

        let mut last_match_end = 0;

        // Find all non-overlapping matches
        for m in self.ac.find_iter(input) {
            let match_start = m.start();
            let match_end = m.end();
            let pattern_index = m.pattern();
            let replacement = &self.replacements[pattern_index.as_usize()];

            // transform: last_match_end..match_start is unchanged
            output.push_str(&input[last_match_end..match_start]);

            let new_start = output.len();
            output.push_str(replacement);
            let new_end = output.len();

            // Record transformation
            transformations.push(Transformation::new(
                match_start,
                match_end,
                new_start,
                new_end,
            ));

            last_match_end = match_end;
        }

        output.push_str(&input[last_match_end..]);

        (output, transformations)
    }

    fn name(&self) -> &'static str {
        "mapping"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_char_filter() {
        let mut mapping = HashMap::new();
        mapping.insert("ph".to_string(), "f".to_string());
        mapping.insert("qu".to_string(), "k".to_string());

        let filter = MappingCharFilter::new(mapping).unwrap();
        let input = "phone queue";
        let (output, trans) = filter.filter(input);

        assert_eq!(output, "fone keue");
        // "ph" -> "f", "qu" -> "k"
        assert_eq!(trans.len(), 2);

        assert_eq!(trans[0].original_start, 0); // ph
        assert_eq!(trans[0].original_end, 2);
        assert_eq!(trans[0].new_start, 0); // f
        assert_eq!(trans[0].new_end, 1);

        // "one " unchanged. original[2..6] ("one ") -> new[1..5] ("one ")
        // Next match "qu" at original 6..8. new starts at 5.

        assert_eq!(trans[1].original_start, 6); // qu
        assert_eq!(trans[1].original_end, 8);
        assert_eq!(trans[1].new_start, 5); // k
        assert_eq!(trans[1].new_end, 6);
    }
}

use unicode_normalization::UnicodeNormalization;

use super::{CharFilter, Transformation};

/// Supported Unicode normalization forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationForm {
    NFC,
    NFD,
    NFKC,
    NFKD,
}

/// A char filter that performs Unicode normalization.
pub struct UnicodeNormalizationCharFilter {
    form: NormalizationForm,
}

impl UnicodeNormalizationCharFilter {
    pub fn new(form: NormalizationForm) -> Self {
        Self { form }
    }
}

impl CharFilter for UnicodeNormalizationCharFilter {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>) {
        let normalized: String = match self.form {
            NormalizationForm::NFC => input.nfc().collect(),
            NormalizationForm::NFD => input.nfd().collect(),
            NormalizationForm::NFKC => input.nfkc().collect(),
            NormalizationForm::NFKD => input.nfkd().collect(),
        };

        // TODO: Implement proper transformation tracking for Unicode normalization.
        // Since Unicode normalization can change length and character counts non-linearly,
        // precise mapping requires iterating and comparing.
        // For now, we return empty transformations, implying identity or "whole string changed".
        // PipelineAnalyzer logic needs to handle empty transformations gracefully (identity).
        (normalized, Vec::new())
    }

    fn name(&self) -> &'static str {
        "unicode_normalization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfc_normalization() {
        let filter = UnicodeNormalizationCharFilter::new(NormalizationForm::NFC);
        // "Amélie" where 'é' is composed (U+00E9)
        let input = "Am\u{00e9}lie";
        let (output, _) = filter.filter(input);
        assert_eq!(output, "Amélie");

        // "Amélie" where 'é' is decomposed (U+0065 U+0301)
        let input_decomposed = "Am\u{0065}\u{0301}lie";
        let (output, _) = filter.filter(input_decomposed);
        // Should be normalized to composed form
        assert_eq!(output, "Am\u{00e9}lie");
    }

    #[test]
    fn test_nfkc_normalization() {
        let filter = UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC);
        // Fullwidth "Ａ" to halfwidth "A"
        let input = "\u{ff21}";
        let (output, _) = filter.filter(input);
        assert_eq!(output, "A");
    }
}

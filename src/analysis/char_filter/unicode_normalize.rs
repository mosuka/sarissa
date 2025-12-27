use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

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
        if input.is_empty() {
            return (String::new(), Vec::new());
        }

        let mut normalized_text = String::with_capacity(input.len());
        let mut transformations = Vec::new();
        let mut input_offset = 0;
        let mut output_offset = 0;

        for grapheme in input.graphemes(true) {
            let normalized_grapheme: String = match self.form {
                NormalizationForm::NFC => grapheme.nfc().collect(),
                NormalizationForm::NFD => grapheme.nfd().collect(),
                NormalizationForm::NFKC => grapheme.nfkc().collect(),
                NormalizationForm::NFKD => grapheme.nfkd().collect(),
            };

            let input_len = grapheme.len();
            let output_len = normalized_grapheme.len();

            if normalized_grapheme != grapheme {
                transformations.push(Transformation {
                    original_start: input_offset,
                    original_end: input_offset + input_len,
                    new_start: output_offset,
                    new_end: output_offset + output_len,
                });
            }

            normalized_text.push_str(&normalized_grapheme);
            input_offset += input_len;
            output_offset += output_len;
        }

        (normalized_text, transformations)
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
    fn test_nfkc_offset_correction() {
        let filter = UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC);
        // "㌂" (U+3302) -> "アンペア" (4 chars).
        // input len: 3 bytes (Japanese char)
        // output len: 12 bytes (4 Japanese chars)
        let input = "㌂";
        let (output, transformations) = filter.filter(input);
        assert_eq!(output, "アンペア");
        assert_eq!(transformations.len(), 1);
        assert_eq!(transformations[0].original_start, 0);
        assert_eq!(transformations[0].original_end, 3);
        assert_eq!(transformations[0].new_start, 0);
        assert_eq!(transformations[0].new_end, 12);
    }

    #[test]
    fn test_nfkc_shrinking_offset() {
        let filter = UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC);
        // "ＳＯＮＹ" (Fullwidth Latin Capital Letters)
        // Each char is 3 bytes in UTF-8. Total 12 bytes.
        // Normalized to "SONY" (4 chars, 1 byte each). Total 4 bytes.
        let input = "ＳＯＮＹ";
        let (output, transformations) = filter.filter(input);
        assert_eq!(output, "SONY");
        assert_eq!(transformations.len(), 4);

        // Check first char 'Ｓ' -> 'S'
        assert_eq!(transformations[0].original_start, 0);
        assert_eq!(transformations[0].original_end, 3);
        assert_eq!(transformations[0].new_start, 0);
        assert_eq!(transformations[0].new_end, 1);

        // Check last char 'Ｙ' -> 'Y'
        // Original offset: 9..12
        // New offset: 3..4
        assert_eq!(transformations[3].original_start, 9);
        assert_eq!(transformations[3].original_end, 12);
        assert_eq!(transformations[3].new_start, 3);
        assert_eq!(transformations[3].new_end, 4);
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

use std::collections::BTreeMap;

use super::{CharFilter, Transformation};

const KANJI_ITERATION_MARK: char = '々';
const HIRAGANA_ITERATION_MARK: char = 'ゝ';
const HIRAGANA_DAKUON_ITERATION_MARK: char = 'ゞ';
const KATAKANA_ITERATION_MARK: char = 'ヽ';
const KATAKANA_DAKUON_ITERATION_MARK: char = 'ヾ';

fn hiragana_add_dakuon(c: &char) -> char {
    let codepoint = *c as u32;
    match codepoint {
        0x304b..=0x3062 if codepoint % 2 == 1 => unsafe { char::from_u32_unchecked(codepoint + 1) },
        0x3064..=0x3069 if codepoint % 2 == 0 => unsafe { char::from_u32_unchecked(codepoint + 1) },
        0x306f..=0x307d if codepoint % 3 == 0 => unsafe { char::from_u32_unchecked(codepoint + 1) },
        _ => *c,
    }
}

fn hiragana_remove_dakuon(c: &char) -> char {
    let codepoint = *c as u32;
    match codepoint {
        0x304b..=0x3062 if codepoint % 2 == 0 => unsafe { char::from_u32_unchecked(codepoint - 1) },
        0x3064..=0x3069 if codepoint % 2 == 1 => unsafe { char::from_u32_unchecked(codepoint - 1) },
        0x306f..=0x307d if codepoint % 3 == 1 => unsafe { char::from_u32_unchecked(codepoint - 1) },
        _ => *c,
    }
}

fn katakana_add_dakuon(c: &char) -> char {
    let codepoint = *c as u32;
    match codepoint {
        0x30ab..=0x30c2 if codepoint % 2 == 1 => unsafe { char::from_u32_unchecked(codepoint + 1) },
        0x30c4..=0x30c9 if codepoint % 2 == 0 => unsafe { char::from_u32_unchecked(codepoint + 1) },
        0x30cf..=0x30dd if codepoint % 3 == 0 => unsafe { char::from_u32_unchecked(codepoint + 1) },
        _ => *c,
    }
}

fn katakana_remove_dakuon(c: &char) -> char {
    let codepoint = *c as u32;
    match codepoint {
        0x30ab..=0x30c2 if codepoint % 2 == 0 => unsafe { char::from_u32_unchecked(codepoint - 1) },
        0x30c4..=0x30c9 if codepoint % 2 == 1 => unsafe { char::from_u32_unchecked(codepoint - 1) },
        0x30cf..=0x30dd if codepoint % 3 == 1 => unsafe { char::from_u32_unchecked(codepoint - 1) },
        _ => *c,
    }
}

pub struct JapaneseIterationMarkCharFilter {
    normalize_kanji: bool,
    normalize_kana: bool,
}

impl JapaneseIterationMarkCharFilter {
    pub fn new(normalize_kanji: bool, normalize_kana: bool) -> Self {
        Self {
            normalize_kanji,
            normalize_kana,
        }
    }

    fn normalize(&self, iter_marks: &BTreeMap<usize, &char>, text_chars: &[char]) -> String {
        let mut normalized_str = String::new();

        let first_iter_mark_pos = iter_marks.keys().next().copied().unwrap_or(0);

        // The logic in Lindera seems to calculate pos_diff based on first mark key.
        // However, here we just need to find the character preceding the iteration mark sequence.
        // If marks are at indices `i, i+1, ...`, the preceding char is at `i-1`.
        // Lindera logic seems a bit specific to how they handled iteration.
        // Let's simplified but robust logic:
        // Identify the "source" character index relative to the run of iteration marks.

        // If the first mark is at 0, there is no previous char, so it stays as mark (fail-safe).

        // But wait, iteration marks repeat the PREVIOUS character.
        // If we have "佐々木", '々' is at 1. Previous is '佐' at 0.
        // If we have "時々刻々", first '々' at 1 repeats 0. Second '々' at 3 repeats 2.
        // The helper `normalize` function in Lindera takes a BATCH of iteration marks?
        // Ah, looking at Lindera code:
        // `iter_marks` seems to be a batch of contiguous or related marks?
        // No, `iter_marks` acts on the whole text scan loop.
        // Let's look at `apply` in Lindera. It collects marks until a non-mark char is found, then normalizes that batch.

        // Replicating Lindera behavior:
        // It calculates `pos_diff` which seems to assume the previous char(s) are at `first_iter_mark_pos - pos_diff`?
        // Lindera: `let pos_diff = if first_iter_mark_pos < iter_marks.len() { first_iter_mark_pos } else { iter_marks.len() };`
        // If we have "佐々", mark at 1. len=1. pos_diff = 1. index = 1 - 1 = 0. Correct.
        // If we have "代々木", mark at 1. len=1. pos_diff=1. index=0. Correct.
        // If we have "馬鹿々々しい", marks at 2, 3. len=2.
        //  - mark at 2: index = 2 - 2 = 0 ("馬") ??
        //  - mark at 3: index = 3 - 2 = 1 ("鹿") ??
        // This implies "馬鹿" repeats as "々々". Correct.

        let pos_diff = if first_iter_mark_pos < iter_marks.len() {
            first_iter_mark_pos
        } else {
            iter_marks.len()
        };

        for (iter_mark_pos, iter_mark) in iter_marks.iter() {
            let iter_mark_index = *iter_mark_pos - pos_diff;
            match *(*iter_mark) {
                KANJI_ITERATION_MARK if self.normalize_kanji => {
                    let replacement = text_chars.get(iter_mark_index).unwrap_or(iter_mark);
                    normalized_str.push(*replacement);
                }
                HIRAGANA_ITERATION_MARK if self.normalize_kana => {
                    let replacement = text_chars.get(iter_mark_index).unwrap_or(iter_mark);
                    normalized_str.push(hiragana_remove_dakuon(replacement));
                }
                HIRAGANA_DAKUON_ITERATION_MARK if self.normalize_kana => {
                    let replacement = text_chars.get(iter_mark_index).unwrap_or(iter_mark);
                    let replacement = hiragana_add_dakuon(replacement);
                    normalized_str.push(replacement);
                }
                KATAKANA_ITERATION_MARK if self.normalize_kana => {
                    let replacement = text_chars.get(iter_mark_index).unwrap_or(iter_mark);
                    normalized_str.push(katakana_remove_dakuon(replacement));
                }
                KATAKANA_DAKUON_ITERATION_MARK if self.normalize_kana => {
                    let replacement = text_chars.get(iter_mark_index).unwrap_or(iter_mark);
                    let replacement = katakana_add_dakuon(replacement);
                    normalized_str.push(replacement);
                }
                _ => {
                    normalized_str.push(**iter_mark);
                }
            }
        }

        normalized_str
    }
}

impl CharFilter for JapaneseIterationMarkCharFilter {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>) {
        let mut output = String::with_capacity(input.len());
        let mut transformations = Vec::new();

        let text_chars: Vec<char> = input.chars().collect();
        // Map from character index to byte offset
        let mut char_indices: Vec<usize> = input.char_indices().map(|(i, _)| i).collect();
        char_indices.push(input.len()); // End sentinel

        let mut iter_marks = BTreeMap::new();

        for (i, &c) in text_chars.iter().enumerate() {
            let is_mark = match c {
                KANJI_ITERATION_MARK => self.normalize_kanji,
                HIRAGANA_ITERATION_MARK
                | HIRAGANA_DAKUON_ITERATION_MARK
                | KATAKANA_ITERATION_MARK
                | KATAKANA_DAKUON_ITERATION_MARK => self.normalize_kana,
                _ => false,
            };

            if is_mark {
                iter_marks.insert(i, &text_chars[i]);
            } else {
                if !iter_marks.is_empty() {
                    let normalized = self.normalize(&iter_marks, &text_chars);

                    // Calculate original byte range
                    let start_char_idx = *iter_marks.keys().next().unwrap();
                    let end_char_idx = *iter_marks.keys().last().unwrap();
                    let original_start_byte = char_indices[start_char_idx];
                    let original_end_byte = char_indices[end_char_idx + 1];

                    // Append normalized string
                    let new_start_byte = output.len();
                    output.push_str(&normalized);
                    let new_end_byte = output.len();

                    transformations.push(Transformation::new(
                        original_start_byte,
                        original_end_byte,
                        new_start_byte,
                        new_end_byte,
                    ));

                    iter_marks.clear();
                }
                output.push(c);
            }
        }

        // Handle trailing marks
        if !iter_marks.is_empty() {
            let normalized = self.normalize(&iter_marks, &text_chars);

            let start_char_idx = *iter_marks.keys().next().unwrap();
            let end_char_idx = *iter_marks.keys().last().unwrap();
            let original_start_byte = char_indices[start_char_idx];
            let original_end_byte = char_indices[end_char_idx + 1];

            let new_start_byte = output.len();
            output.push_str(&normalized);
            let new_end_byte = output.len();

            transformations.push(Transformation::new(
                original_start_byte,
                original_end_byte,
                new_start_byte,
                new_end_byte,
            ));
        }

        (output, transformations)
    }

    fn name(&self) -> &'static str {
        "japanese_iteration_mark"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_japanese_iteration_mark_kanji() {
        let filter = JapaneseIterationMarkCharFilter::new(true, true);
        let input = "佐々木";
        let (output, trans) = filter.filter(input);
        assert_eq!(output, "佐佐木");
        assert_eq!(trans.len(), 1);
        // "々" (3 bytes) at index 3 (after "佐") replaced by "佐" (3 bytes)
        // Original: "佐" [0..3], "々" [3..6], "木" [6..9]
        // New: "佐" [0..3], "佐" [3..6], "木" [6..9]
        // Transformation: orig[3..6] -> new[3..6]
        assert_eq!(trans[0].original_start, 3);
        assert_eq!(trans[0].original_end, 6);
    }

    #[test]
    fn test_japanese_iteration_mark_multi() {
        let filter = JapaneseIterationMarkCharFilter::new(true, true);
        let input = "馬鹿々々しい";
        let (output, _) = filter.filter(input);
        assert_eq!(output, "馬鹿馬鹿しい");
    }

    #[test]
    fn test_japanese_iteration_mark_kana() {
        let filter = JapaneseIterationMarkCharFilter::new(true, true);
        let input = "いすゞ";
        let (output, _) = filter.filter(input);
        assert_eq!(output, "いすず");
    }
}

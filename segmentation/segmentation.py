from typing import List, Tuple

from razdel import sentenize
from transformers import PreTrainedTokenizer


def split_text_by_sentences(text: str) -> List[Tuple[int, int]]:
    sentence_bounds = []
    for sent in sentenize(text):
        sent_start = sent.start
        sent_end = sent_start + len(sent.text)
        sentence_bounds.append((sent_start, sent_end))
    return sentence_bounds


def segment_long_text(text: str, tokenizer: PreTrainedTokenizer, max_text_len: int) -> List[Tuple[int, int]]:
    prepared_text = text.strip()
    found_idx = text.find(prepared_text)
    if found_idx < 0:
        raise RuntimeError(f'The text cannot be segmented! {text}')
    if len(prepared_text) == 0:
        return []
    tokens = tokenizer.tokenize(text.replace('\r', ' ').replace('\n', ' '))
    if len(tokens) <= max_text_len:
        return [(found_idx, found_idx + len(prepared_text))]
    sentence_bounds = split_text_by_sentences(text)
    number_of_sentences = len(sentence_bounds)
    if number_of_sentences <= 2:
        return sentence_bounds
    middle_idx = max(number_of_sentences // 2 - 1, 0)
    bounds_of_left_text = (sentence_bounds[0][0], sentence_bounds[middle_idx][1])
    bounds_of_right_text = (sentence_bounds[middle_idx + 1][0], sentence_bounds[number_of_sentences - 1][1])
    left_segments = segment_long_text(
        text[bounds_of_left_text[0]:bounds_of_left_text[1]],
        tokenizer, max_text_len
    )
    right_segments = segment_long_text(
        text[bounds_of_right_text[0]:bounds_of_right_text[1]],
        tokenizer, max_text_len
    )
    united_segments = [
        (segment_start + bounds_of_left_text[0], segment_end + bounds_of_left_text[0])
        for segment_start, segment_end in left_segments
    ]
    united_segments += [
        (segment_start + bounds_of_right_text[0], segment_end + bounds_of_right_text[0])
        for segment_start, segment_end in right_segments
    ]
    return united_segments

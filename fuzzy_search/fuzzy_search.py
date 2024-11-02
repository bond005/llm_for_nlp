from typing import List, Tuple, Union

import numpy as np
from nltk import wordpunct_tokenize


def tokenize_text(s: str) -> List[Tuple[int, int, str]]:
    tokens = wordpunct_tokenize(s)
    res = []
    start_pos = 0
    for it in tokens:
        found_idx = s[start_pos:].find(it)
        if found_idx < 0:
            raise ValueError(f'The text cannot be tokenized! {s} != {tokens}')
        res.append((found_idx + start_pos, found_idx + start_pos + len(it), it))
        start_pos = found_idx + start_pos + len(it)
    return res


def levenshtein(seq1: str, seq2: str) -> int:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]),
                    int(matrix[x, y - 1]) + 1
                )
            else:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]) + 1,
                    int(matrix[x, y - 1]) + 1
                )
    return int(matrix[size_x - 1, size_y - 1])


def find_substring(full_string: str, substring: str) -> Union[Tuple[int, int], None]:
    tokens_of_full_string = tokenize_text(full_string)
    tokens_of_substring = tokenize_text(substring)
    full_string_len = len(tokens_of_full_string)
    substring_len = len(tokens_of_substring)
    if (substring_len == 0) or (full_string_len == 0):
        return None
    if substring_len > full_string_len:
        return None
    if substring_len == full_string_len:
        return tokens_of_full_string[0][0], tokens_of_full_string[-1][1]
    best_res = (tokens_of_full_string[0][0], tokens_of_full_string[substring_len - 1][1])
    best_dist = levenshtein(
        ''.join([it1[2] for it1 in tokens_of_substring]),
        ''.join([it2[2] for it2 in tokens_of_full_string[0:substring_len]])
    )
    for idx in range(1, full_string_len - substring_len):
        cur_dist = levenshtein(
            ''.join([it1[2] for it1 in tokens_of_substring]),
            ''.join([it2[2] for it2 in tokens_of_full_string[idx:(substring_len + idx)]])
        )
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_res = (tokens_of_full_string[idx][0], tokens_of_full_string[idx + substring_len - 1][1])
    return best_res

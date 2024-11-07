import copy
import gc
from typing import List
import warnings

import torch
from transformers import PreTrainedTokenizer, GenerationMixin, GenerationConfig


def generate_answer(questions: List[str], tokenizer: PreTrainedTokenizer, config: GenerationConfig,
                    model: GenerationMixin, max_new_tokens: int, device: str) -> List[str]:
    nonempty_questions = []
    new_config = copy.copy(config)
    new_config.max_new_tokens = max_new_tokens
    for cur in questions:
        if len(cur.strip()) > 0:
            nonempty_questions.append(cur)
    if len(nonempty_questions) == 0:
        return ['' for _ in range(len(questions))]
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    x = tokenizer(nonempty_questions, return_tensors='pt', padding=True).to(device)
    out = model.generate(**x, generation_config=new_config)
    out = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(x.input_ids, out)
    ]
    del x
    answers_for_nonempty_texts = [
        tokenizer.decode(cur, skip_special_tokens=True).strip().replace('\r\n', '\n') for cur in out
    ]
    del out
    gc.collect()
    torch.cuda.empty_cache()
    united_answers = []
    idx = 0
    for cur in questions:
        if len(cur.strip()) > 0:
            united_answers.append(answers_for_nonempty_texts[idx])
            idx += 1
        else:
            united_answers.append('')
    return united_answers

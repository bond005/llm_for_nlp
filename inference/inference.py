import gc
from typing import List
import warnings

import torch
from transformers import PreTrainedTokenizer, GenerationMixin, GenerationConfig


def generate_answer(questions: List[str], tokenizer: PreTrainedTokenizer, config: GenerationConfig,
                    model: GenerationMixin, device: str) -> List[str]:
    nonempty_questions = []
    for cur in questions:
        if len(cur.strip()) > 0:
            nonempty_questions.append(cur)
    if len(nonempty_questions) == 0:
        return ['' for _ in range(len(questions))]
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    x = tokenizer(nonempty_questions, return_tensors='pt', padding=True).to(device)
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            out = model.generate(**x, generation_config=config)
    except:
        warnings.warn('The PyTorch scaled dot product attention is not supported.')
        out = model.generate(**x, generation_config=config)
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

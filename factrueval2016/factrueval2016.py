import copy
import math
from typing import List, Tuple

from transformers import PreTrainedTokenizer, GenerationMixin, GenerationConfig

from segmentation.segmentation import segment_long_text
from inference.inference import generate_answer
from instructions.instructions import prepare_prompt_for_ner
from fuzzy_search.fuzzy_search import find_substring, levenshtein


NO_ENTITIES: str = 'в этом тексте нет именованных сущностей такого типа'
CHAR_ERROR_RATE_THRESHOLD: float = 0.5


def calculate_entity_bounds(source_text: str, entities: List[str]) -> List[Tuple[int, int]]:
    start_pos = 0
    entity_bounds = []
    for cur_entity in entities:
        matched = find_substring(source_text[start_pos:], cur_entity)
        if matched is not None:
            new_bounds = (matched[0][0] + start_pos, matched[0][1] + start_pos)
            cer = levenshtein(cur_entity, source_text[new_bounds[0]:new_bounds[1]]) / len(cur_entity)
            if cer < CHAR_ERROR_RATE_THRESHOLD:
                entity_bounds.append(new_bounds)
                start_pos = new_bounds[1]
    return sorted(entity_bounds, key=lambda it: (it[0], it[1] - it[0]))


def recognize_named_entities(document: str,
                             tokenizer: PreTrainedTokenizer, model: GenerationMixin, config: GenerationConfig,
                             max_text_len: int = 1024, batch_size: int = 1,
                             device: str = 'cuda:0') -> List[Tuple[str, int, int]]:
    if len(document.strip()) == 0:
        return []
    paragraphs_of_document = segment_long_text(document, tokenizer, max_text_len)
    number_of_tokens_in_negative_answer = len(tokenizer.tokenize(NO_ENTITIES))
    paragraph_texts = []
    for paragraph_start, paragraph_end in paragraphs_of_document:
        paragraph_text = document[paragraph_start:paragraph_end]
        paragraph_text = paragraph_text.replace('\r', ' ').replace('\n', ' ')
        paragraph_texts.append(paragraph_text)
        del paragraph_text
    copied_config = copy.copy(config)
    copied_config.do_sample = False
    copied_config.min_new_tokens = 1
    copied_config.repetition_penalty = 1.0
    recognized_entities = []
    n_batches = math.ceil(len(paragraph_texts) / batch_size)
    entity_classes = {'org': 'Организация', 'loc': 'Местоположение', 'per': 'Человек'}
    for k in sorted(list(entity_classes.keys())):
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(len(paragraph_texts), batch_start + batch_size)
            max_text_len = 0
            questions = []
            for cur_text in paragraph_texts[batch_start:batch_end]:
                questions.append(
                    tokenizer.apply_chat_template(
                        prepare_prompt_for_ner(entity_classes[k], cur_text),
                        tokenize=False, add_generation_prompt=True
                    )
                )
                cur_text_len = len(tokenizer.tokenize(cur_text))
                if cur_text_len > max_text_len:
                    max_text_len = cur_text_len
            answers = generate_answer(
                questions=questions,
                tokenizer=tokenizer,
                config=copied_config,
                model=model,
                max_new_tokens=max(max_text_len, number_of_tokens_in_negative_answer),
                device=device
            )
            del questions
            for sample_idx in range(batch_start, batch_end):
                entities = list(filter(
                    lambda it2: len(it2) > 0,
                    map(
                        lambda it1: it1.strip(),
                        answers[sample_idx - batch_start].split('\n')
                    )
                ))
                if len(entities) > 0:
                    entity_bounds = calculate_entity_bounds(paragraph_texts[sample_idx], entities)
                    for entity_start, entity_end in entity_bounds:
                        recognized_entities.append(
                            (
                                k,
                                entity_start + paragraphs_of_document[sample_idx][0],
                                entity_end - entity_start
                            )
                        )
                    del entity_bounds
                del entities
    recognized_entities.sort(key=lambda it: (it[1], it[2], it[0]))
    del copied_config
    return recognized_entities

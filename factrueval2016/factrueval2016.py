import copy
import math
from typing import List, Tuple

from razdel import tokenize
from transformers import PreTrainedTokenizer, GenerationMixin, GenerationConfig

from segmentation.segmentation import segment_long_text
from inference.inference import generate_answer
from instructions.instructions import prepare_prompt_for_ner


NO_ENTITIES: str = 'в этом тексте нет именованных сущностей такого типа'


def find_subphrase(full_phrase: List[str], subphrase: List[str]) -> int:
    n = len(subphrase)
    subphrase_ = ' '.join(subphrase).lower()
    if n > len(full_phrase):
        return -1
    if n == len(full_phrase):
        if subphrase_ == ' '.join(full_phrase).lower():
            return 0
        return -1
    found_idx = -1
    for idx in range(len(full_phrase) - n + 1):
        if subphrase_ == ' '.join(full_phrase[idx:(idx + n)]).lower():
            found_idx = idx
            break
    return found_idx


def match_entities_to_tokens(tokens: List[str], prepared_entities: List[List[str]],
                             entity_bounds: List[Tuple[int, int]],
                             penalty: int, start_pos: int = 0) -> List[Tuple[List[Tuple[int, int]], int]]:
    if len(prepared_entities) == 0:
        return [([], 0)]
    if (len(tokens) == 0) and (len(prepared_entities) > 0):
        return [(entity_bounds, penalty + len(prepared_entities))]
    new_penalty = penalty
    found_token_idx = find_subphrase(tokens, prepared_entities[0])
    if found_token_idx >= 0:
        entity_bounds_ = entity_bounds + [(
            found_token_idx + start_pos,
            found_token_idx + len(prepared_entities[0]) + start_pos
        )]
        if len(prepared_entities) > 1:
            res = match_entities_to_tokens(tokens[(found_token_idx + len(prepared_entities[0])):],
                                           prepared_entities[1:], entity_bounds_,
                                           new_penalty, found_token_idx + len(prepared_entities[0]) + start_pos)
            res += match_entities_to_tokens(tokens[found_token_idx:], prepared_entities[1:], entity_bounds_,
                                            new_penalty, found_token_idx + start_pos)
        else:
            res = [(entity_bounds_, new_penalty)]
    else:
        new_penalty += 1
        if len(prepared_entities) > 1:
            res = match_entities_to_tokens(tokens, prepared_entities[1:], entity_bounds, new_penalty, start_pos)
        else:
            res = [(entity_bounds, new_penalty)]
    return res


def check_nested(bounds: List[Tuple[int, int]]) -> bool:
    if len(bounds) < 2:
        return False
    ok = False
    for idx in range(1, len(bounds)):
        if bounds[idx - 1][1] > bounds[idx][0]:
            ok = True
            break
    return ok


def calculate_entity_bounds(source_text: str, entities: List[str], nested: bool = False) -> List[Tuple[int, int]]:
    tokens_of_text = [it.text for it in tokenize(source_text)]
    if ' '.join(tokens_of_text).lower() == NO_ENTITIES:
        return []
    token_bounds = []
    start_pos = 0
    for cur_token in tokens_of_text:
        found_char_idx = source_text[start_pos:].find(cur_token)
        if found_char_idx < 0:
            err_msg = f'The token {cur_token} is not found in the text {source_text}'
            raise RuntimeError(err_msg)
        token_bounds.append((found_char_idx + start_pos, found_char_idx + start_pos + len(cur_token)))
        start_pos = found_char_idx + start_pos + len(cur_token)
    prepared_entities = []
    for cur_entity in entities:
        postprocessed_entity = cur_entity.strip()
        prepared_entities.append([it.text for it in tokenize(postprocessed_entity)])
    variants_of_entity_bounds = match_entities_to_tokens(tokens_of_text, prepared_entities, [], 0)
    variants_of_entity_bounds.sort(key=lambda it: (it[1], abs(len(entities) - len(it[0]))))
    variants_of_entity_bounds = list(filter(lambda it: len(it[0]) > 0, variants_of_entity_bounds))
    variants_of_entity_bounds_ = [
        (
            sorted(list(set(cur[0])), key=lambda x: (x[0], x[1])),
            cur[1]
        )
        for cur in variants_of_entity_bounds
    ]
    del variants_of_entity_bounds
    entity_bounds = []
    if len(variants_of_entity_bounds_) > 0:
        if not nested:
            variants_of_entity_bounds_ = list(filter(lambda it: not check_nested(it[0]), variants_of_entity_bounds_))
        for entity_token_start, entity_token_end in variants_of_entity_bounds_[0][0]:
            entity_bounds.append((
                token_bounds[entity_token_start][0],
                token_bounds[entity_token_end - 1][1]
            ))
    return entity_bounds


def recognize_named_entities(document: str,
                             tokenizer: PreTrainedTokenizer, model: GenerationMixin, config: GenerationConfig,
                             max_text_len: int = 1024, batch_size: int = 1,
                             device: str = 'cuda:0') -> List[Tuple[str, int, int]]:
    if len(document.strip()) == 0:
        return []
    paragraphs_of_document = segment_long_text(document, tokenizer, max_text_len)
    number_of_tokens_in_negative_answer = len(tokenizer.tokenize(NO_ENTITIES))
    paragraph_texts = []
    max_new_tokens = 0
    for paragraph_start, paragraph_end in paragraphs_of_document:
        paragraph_text = document[paragraph_start:paragraph_end]
        paragraph_text = paragraph_text.replace('\r', ' ').replace('\n', ' ')
        if len(paragraph_text.strip()) > 0:
            max_new_tokens = max(
                max_new_tokens,
                max(
                    number_of_tokens_in_negative_answer,
                    len(tokenizer.tokenize(paragraph_text))
                ) + 3
            )
        paragraph_texts.append(paragraph_text)
        del paragraph_text
    if max_new_tokens == 0:
        return []
    copied_config = copy.deepcopy(config)
    copied_config.do_sample = False
    copied_config.num_beams = 3
    copied_config.max_new_tokens = max_new_tokens
    copied_config.min_new_tokens = 1
    recognized_entities = []
    n_batches = math.ceil(len(paragraph_texts) / batch_size)
    entity_classes = {'org': 'Организация', 'loc': 'Местоположение', 'per': 'Человек'}
    for k in sorted(list(entity_classes.keys())):
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(len(paragraph_texts), batch_start + batch_size)
            questions = [
                tokenizer.apply_chat_template(
                    prepare_prompt_for_ner(entity_classes[k], cur_text),
                    tokenize=False, add_generation_prompt=True
                )
                for cur_text in paragraph_texts[batch_start:batch_end]
            ]
            answers = generate_answer(questions, tokenizer, copied_config, model, device)
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

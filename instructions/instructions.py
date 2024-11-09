import codecs
import csv
import random
from typing import Dict, List, Optional


def prepare_prompt_for_ner(entity_class: str, input_text: str) -> List[Dict[str, str]]:
    prepared_entity_class = entity_class.strip()
    if len(prepared_entity_class) == 0:
        raise ValueError(f'The specified entity class is empty!')
    system_prompt = (f'Найди, пожалуйста, все именованные сущности типа \"{" ".join(prepared_entity_class.split())}\" '
                     f'в следующем тексте и выпиши список таких сущностей.')
    user_prompt = input_text.strip()
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]


def prepare_prompt_for_summarization(input_text: str) -> List[Dict[str, str]]:
    system_prompt = 'Выполни саммаризацию и выдели, пожалуйста, основную мысль следующего текста.'
    user_prompt = input_text.strip()
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]


def prepare_prompt_for_asr_correction(input_text: str,
                                      additional_prompt: Optional[str] = None,
                                      few_shots: Optional[List[List[Dict[str, str]]]] = None) -> List[Dict[str, str]]:
    system_prompt = 'Исправь, пожалуйста, ошибки распознавания речи в следующем тексте.'
    if additional_prompt is not None:
        system_prompt += (' ' + ' '.join(additional_prompt.strip().split()))
    system_prompt = system_prompt.strip()
    user_prompt = input_text.strip()
    res = [{'role': 'system', 'content': system_prompt}]
    if few_shots is not None:
        if len(few_shots) > 0:
            for val in few_shots:
                res += val
    res.append({'role': 'user', 'content': user_prompt})
    return res


def prepare_prompt_for_toxicity_detection(input_text: str) -> List[Dict[str, str]]:
    system_prompt = ('Подскажи, пожалуйста, является ли токсичным (неприятным для какой-то группы людей, '
                     'нарушающим принципы этики) следующий текст?')
    user_prompt = input_text.strip()
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]


def prepare_prompt_for_detoxification(input_text: str) -> List[Dict[str, str]]:
    system_prompt = ('Перепиши, пожалуйста, следующий текст так, чтобы он перестал быть токсичным '
                     '(неприятным для какой-то группы людей, нарушающим принципы этики).')
    user_prompt = input_text.strip()
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]


def load_fewshot(csv_fname: str) -> List[List[Dict[str, str]]]:
    with codecs.open(csv_fname, mode='r', encoding='utf-8') as fp:
        data_reader = csv.reader(fp)
        rows = list(filter(lambda it: len(it) > 0, data_reader))
    if len(rows) == 0:
        raise IOError(f'The "{csv_fname}" is empty.')
    true_header = ['QUERY', 'REFERENCE']
    if rows[0] != true_header:
        raise IOError(f'The "{csv_fname}" is empty. The expected header is {true_header}, the got one is {rows[0]}.')
    samples = []
    for it in rows[1:]:
        new_sample = [{'role': 'user', 'content': it[0]}, {'role': 'assistant', 'content': it[1]}]
        samples.append(new_sample)
        del new_sample
    if len(samples) == 0:
        raise IOError(f'The "{csv_fname}" is empty.')
    return samples

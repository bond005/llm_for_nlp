from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import sys
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from tqdm import tqdm


text_generation_logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
RUSSIAN_LETTERS: Set[str] = {'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р',
                             'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
                             'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р',
                             'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'}
ENGLISH_LETTERS: Set[str] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                             'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                             'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'}
DIGITS: Set[str] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
PUNCTUATION: Set[str] = {'.', ',', '"', '\'', ':', ';', '-', '+', '=', '*', '\\', '/', '!', '&', '?', '(', ')',
                         '{', '}', '[', ']', '<', '>', '~', '`', '#', '$', '№', '%', '^', '@', '…', '—', '|',
                         '‰', '‱', '—', '∙', '◦', '⁍', '⁌', '⁃', '‣', '•', '−', '‐', '—', '–', '−', '‐',
                         '«', '»', '§', '_', chr(0x305), chr(0x304), chr(0x302), chr(0x301), chr(0x300), '←', '→',
                         '↑', '↓', '⇈', '↮', '↚', '↛', '↔', '⇒', '⇐', '⇔', '↦', '⇸', '↠', '⇾', '⤀', '↣', '↪',
                         '⤔', '⤖', '⤗', '⇋', '◌⃗', '◌⃑', '↯', '×', '°', '³', '²', chr(0x00B1), '©',
                         '₽', '“', '„', 'ν', 'і', 'η', 'λ', 'ο', 'σ', 'χ', '’', '∧', '、', '。', '一', '“', '”',
                         '∗', '√', 'γ', 'μ', 'π', 'τ', 'ψ', 'ω', 'ψ', 'δ', 'β', 'θ', '≥', 'α', '‒', '®', '±', '∓',
                         'Ƒ', 'ƒ', 'Ɵ', 'Ǿ', 'ǿ', 'Ʌ', 'Ɇ', 'ɇ', 'Ǝ', 'Ɛ', 'ƞ', 'Ʃ', 'ʃ', 'Θ', '≤'}
SPACES: Set[str] = {' ', '\n', '\r', '\t', chr(160), chr(8199), chr(8239), chr(8288)}
RANDOM_SEED: int = 42


def load_data(fname: str) -> List[Dict[str, Union[str, List[Tuple[str, str]]]]]:
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        samples = list(map(
            lambda it3: json.loads(it3),
            filter(
                lambda it2: len(it2) > 0,
                map(
                    lambda it1: it1.strip(),
                    fp.readlines()
                )
            )
        ))
    if len(samples) == 0:
        raise IOError(f'The file "{fname}" is empty!')
    prepared_samples = []
    for idx, val in enumerate(samples):
        err_msg = f'"{fname}": sample {idx} is wrong!'
        if not isinstance(val, dict):
            raise ValueError(err_msg + f' Expected {type({"a": "b"})}, got {type(val)}.')
        if 'system' not in val:
            raise ValueError(err_msg + f' The "system" key is not found. Existing keys are: {list(val.keys())}')
        if 'query' not in val:
            raise ValueError(err_msg + f' The "query" key is not found. Existing keys are: {list(val.keys())}')
        if 'history' not in val:
            raise ValueError(err_msg + f' The "history" key is not found. Existing keys are: {list(val.keys())}')
        if not isinstance(val['history'], list):
            raise ValueError(err_msg + f' The "history" has an incorrect type!')
        history = []
        for it in val['history']:
            if not isinstance(it, list):
                raise ValueError(err_msg + f' The "history" has an incorrect type!')
            if len(it) != 2:
                raise ValueError(err_msg + f' The "history" has an incorrect type!')
            if (not isinstance(it[0], str)) or (not isinstance(it[1], str)):
                raise ValueError(err_msg + f' The "history" has an incorrect type!')
            history.append((it[0], it[1]))
        prepared_samples.append({
            'system': val['system'],
            'query': val['query'],
            'history': history
        })
    return prepared_samples


def instruction_to_text(instruction: Dict[str, Union[str, List[Tuple[str, str]]]],
                        tokenizer: PreTrainedTokenizer) -> str:
    messages = [{'role': 'system', 'content': instruction['system']}]
    for answer, question in instruction['history']:
        messages += [
            {
                'role': 'user',
                'content': answer
            },
            {
                'role': 'assistant',
                'content': question
            }
        ]
    messages.append({'role': 'user', 'content': instruction['query']})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    del messages
    return text


def is_correct(s: str) -> bool:
    set_of_characters = set(' '.join(s.strip().split()).lower())
    unknown_characters = set_of_characters - (SPACES | RUSSIAN_LETTERS | ENGLISH_LETTERS | PUNCTUATION | DIGITS)
    if len(unknown_characters) > 0:
        warn_msg = f'There are unknown characters in the text. They are: {sorted(list(unknown_characters))}.'
        text_generation_logger.warning(warn_msg)
    return len(unknown_characters) == 0


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        text_generation_logger.error(err_msg)
        raise ValueError(err_msg)
    torch.cuda.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='llm_name', type=str, required=True,
                        help='The input name of tested LLM')
    parser.add_argument('-i', '--input', dest='input_file', type=str, required=True,
                        help='The input JSONL file name with instructions.')
    parser.add_argument('-o', '--output', dest='output_file', type=str, required=True,
                        help='The output JSONL file name.')
    parser.add_argument('-r', '--repeats', dest='repeats_number', type=int, required=False, default=1,
                        help='Repeats of each instruction.')
    parser.add_argument('-t', '--temperature', dest='temperature', type=float, required=False,
                        default=None, help='Temperature of generation.')
    parser.add_argument('--maxlen', dest='maxlen', type=int, required=False,
                        default=None, help='The maximal length of generated answer.')
    args = parser.parse_args()

    output_fname = os.path.normpath(args.output_file)
    if os.path.isdir(output_fname):
        err_msg = f'"{output_fname}" is a directory!'
        text_generation_logger.error(err_msg)
        raise IOError(err_msg)
    elif not os.path.isfile(output_fname):
        base_dir = os.path.dirname(output_fname)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                err_msg = f'The directory "{base_dir}" does not exist!'
                text_generation_logger.error(err_msg)
                raise IOError(err_msg)

    try:
        input_instructions = load_data(args.input_file)
    except Exception as err:
        text_generation_logger.error(str(err))
        raise
    info_msg = f'There are {len(input_instructions)} instructions are loaded from "{args.input_file}".'
    text_generation_logger.info(info_msg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        generation = GenerationConfig.from_pretrained(args.llm_name)
    except Exception as err:
        text_generation_logger.error(str(err))
        raise
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    if not generation.do_sample:
        generation.do_sample = True
    if args.temperature is not None:
        generation.temperature = args.temperature
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_name,
            torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',
            device_map='cuda:0'
        )
    except:
        text_generation_logger.warning('The Scaled-Dot Product Attention is not used.')
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_name,
                torch_dtype=torch.bfloat16,
                device_map='cuda:0'
            )
        except Exception as err:
            text_generation_logger.error(str(err))
            raise
    text_generation_logger.info(f'The LLM is loaded from "{args.llm_name}".')

    max_length = 0 if (generation.max_new_tokens is None) else generation.max_new_tokens
    for cur_instruction in input_instructions:
        if 'history' in cur_instruction:
            if len(cur_instruction['history']) > 0:
                for question, answer in cur_instruction['history']:
                    n_tokens = max(
                        len(tokenizer.tokenize(question, add_special_tokens=True)),
                        len(tokenizer.tokenize(answer, add_special_tokens=True))
                    )
                    if n_tokens > max_length:
                        max_length = n_tokens
        n_tokens = max(
            len(tokenizer.tokenize(cur_instruction['system'], add_special_tokens=True)),
            len(tokenizer.tokenize(cur_instruction['query'], add_special_tokens=True))
        )
        if n_tokens > max_length:
            max_length = n_tokens
    max_length = max(20, round(1.5 * max_length))
    if args.maxlen is not None:
        if args.maxlen > max_length:
            max_length = args.maxlen
    text_generation_logger.info(f'Maximal number of new tokens is {max_length}.')
    generation.max_new_tokens = max_length

    n_success = 0
    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore', buffering=0) as fp:
        for cur_instruction in tqdm(input_instructions):
            x = tokenizer(
                [instruction_to_text(cur_instruction, tokenizer)],
                return_tensors='pt',
                padding=True
            ).to(model.device)
            set_of_answers = set()
            for _ in range(args.repeats_number):
                out = model.generate(**x, generation_config=generation)
                out = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(x.input_ids, out)
                ]
                answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()
                del out
                if (len(answer) > 0) and is_correct(answer):
                    if ' '.join(answer.split()).strip().lower() not in set_of_answers:
                        new_sample = {
                            'system': cur_instruction['system'],
                            'query': cur_instruction['query'],
                            'response': answer,
                            'history': cur_instruction['history']
                        }
                        fp.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                        del new_sample
                        n_success += 1
                        set_of_answers.add(' '.join(answer.split()).strip().lower())
            del x, set_of_answers
    info_msg = f'{n_success} answers from {len(input_instructions) * args.repeats_number} are successfully generated.'
    text_generation_logger.info(info_msg)


if __name__ == '__main__':
    text_generation_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    text_generation_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('text_generation.log')
    file_handler.setFormatter(formatter)
    text_generation_logger.addHandler(file_handler)
    main()

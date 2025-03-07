from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import signal
import sys
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm import EngineArgs, LLMEngine
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
                         '₽', '“', '„', 'ν', 'і', 'η', 'λ', 'ο', 'σ', 'χ', '’', '∧', '、', '。', '一', '“', '”'}
SPACES: Set[str] = {' ', '\n', '\r', '\t', chr(160), chr(8199), chr(8239), chr(8288)}
RANDOM_SEED: int = 42


def handle_exit(signal, frame):
    if torch.distributed.is_initialized():
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
    sys.exit(0)


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

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='llm_name', type=str, required=True,
                        help='The input name of tested LLM')
    parser.add_argument('-i', '--input', dest='input_file', type=str, required=True,
                        help='The input JSONL file name with instructions.')
    parser.add_argument('-o', '--output', dest='output_file', type=str, required=True,
                        help='The output JSONL file name.')
    parser.add_argument('-t', '--temperature', dest='temperature', type=float, required=False,
                        default=None, help='Temperature of generation.')
    parser.add_argument('--max_in_len', dest='max_input_len', type=int, required=True,
                        help='The maximal length of input query.')
    parser.add_argument('--max_out_len', dest='max_output_len', type=int, required=True,
                        help='The maximal length of generated answer.')
    parser.add_argument('--minibatch', dest='minibatch', type=int, required=False,
                        default=None, help='The mini-batch size.')
    args = parser.parse_args()

    if args.minibatch is None:
        minibatch_size = 1
    else:
        minibatch_size = args.minibatch
        if minibatch_size < 1:
            err_msg = f'The mini-batch is too small! Expected a positive integer, got {args.minibatch}.'
            text_generation_logger.error(err_msg)
            raise ValueError(err_msg)

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

    generation.max_new_tokens = args.max_output_len
    sampling_params = SamplingParams(
        temperature=generation.temperature,
        top_p=generation.top_p,
        repetition_penalty=generation.repetition_penalty,
        max_tokens=generation.max_new_tokens
    )

    max_model_len = args.max_output_len + args.max_input_len
    try:
        model = LLM(model=args.llm_name, gpu_memory_utilization=0.95, max_model_len=max_model_len)
    except Exception as err:
        text_generation_logger.error(str(err))
        if torch.distributed.is_initialized():
            if hasattr(LLMEngine, 'shutdown'):
                LLMEngine.shutdown()
            torch.distributed.destroy_process_group()
            torch.cuda.empty_cache()
        raise
    text_generation_logger.info(f'The LLM is loaded from "{args.llm_name}".')

    n_success = 0
    batch_of_texts = []
    input_lengths = []
    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore', buffering=0) as fp:
        for cur_instruction in tqdm(input_instructions):
            textualized_instruction = instruction_to_text(cur_instruction, tokenizer)
            n_tokens = len(tokenizer.tokenize(textualized_instruction, add_special_tokens=True))
            input_lengths.append(n_tokens)
            if n_tokens > args.max_input_len:
                continue
            batch_of_texts.append({
                'system': cur_instruction['system'],
                'query': cur_instruction['query'],
                'history': cur_instruction['history'],
                'input': textualized_instruction
            })
            if len(batch_of_texts) >= minibatch_size:
                try:
                    outputs = model.generate([it['input'] for it in batch_of_texts], sampling_params, use_tqdm=False)
                except BaseException as err:
                    text_generation_logger.error(str(err))
                    if torch.distributed.is_initialized():
                        if hasattr(LLMEngine, 'shutdown'):
                            LLMEngine.shutdown()
                        torch.distributed.destroy_process_group()
                        torch.cuda.empty_cache()
                    raise
                for idx, val in enumerate(outputs):
                    answer = val.outputs[0].text.strip()
                    if (len(answer) > 0) and is_correct(answer):
                        new_sample = {
                            'system': batch_of_texts[idx]['system'],
                            'query': batch_of_texts[idx]['query'],
                            'response': answer,
                            'history': batch_of_texts[idx]['history']
                        }
                        fp.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                        del new_sample
                        n_success += 1
                del outputs
                del batch_of_texts
                batch_of_texts = []
    if len(batch_of_texts) > 0:
        try:
            outputs = model.generate([it['input'] for it in batch_of_texts], sampling_params)
        except BaseException as err:
            text_generation_logger.error(str(err))
            if torch.distributed.is_initialized():
                if hasattr(LLMEngine, 'shutdown'):
                    LLMEngine.shutdown()
                torch.distributed.destroy_process_group()
                torch.cuda.empty_cache()
            raise
        for idx, val in enumerate(outputs):
            answer = val.outputs[0].text.strip()
            if (len(answer) > 0) and is_correct(answer):
                new_sample = {
                    'system': batch_of_texts[idx]['system'],
                    'query': batch_of_texts[idx]['query'],
                    'response': answer,
                    'history': batch_of_texts[idx]['history']
                }
                fp.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                del new_sample
                n_success += 1
    input_lengths.sort()
    info_msg = (f'Lengths of input sequences: minimal = {input_lengths[0]}, maximal = {input_lengths[-1]}, '
                f'median = {input_lengths[(len(input_lengths) - 1) // 2]}, '
                f'mean = {round(sum(input_lengths) / len(input_lengths))}.')
    text_generation_logger.info(info_msg)
    info_msg = f'{n_success} answers from {len(input_instructions)} are successfully generated.'
    text_generation_logger.info(info_msg)
    is_distributed_initialized = torch.distributed.is_initialized()
    text_generation_logger.info(f'torch.distributed.is_initialized() = {is_distributed_initialized}')
    if is_distributed_initialized:
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()


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

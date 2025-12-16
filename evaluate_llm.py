from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import signal
import sys
from typing import Dict, List, Tuple, Union

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import evaluate
from nltk import wordpunct_tokenize
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm import EngineArgs, LLMEngine
import torch
from tqdm import tqdm


llm_eval_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def handle_exit(signal, frame):
    if torch.distributed.is_initialized():
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
    sys.exit(0)


def load_data(fname: str) -> List[Dict[str, Union[str, List[Tuple[str, str]]]]]:
    samples = []
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        for line_idx, cur_line in enumerate(fp.readlines()):
            if len(cur_line.strip()) > 0:
                try:
                    new_sample = json.loads(cur_line.strip())
                except Exception as err:
                    llm_eval_logger.warning(f'{fname}: line {line_idx} is bad!\n{str(err)}')
                    new_sample = None
                if new_sample is not None:
                    samples.append(new_sample)
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
        if 'response' not in val:
            raise ValueError(err_msg + f' The "response" key is not found. Existing keys are: {list(val.keys())}')
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
            'response': val['response'],
            'history': history
        })
    return prepared_samples


def split_text_into_words(source_text) -> List[Tuple[int, int]]:
    words = wordpunct_tokenize(source_text)
    start_pos = 0
    word_boundaries = []
    for cur_word in words:
        found_idx = source_text[start_pos:].find(cur_word)
        if found_idx < 0:
            err_msg = f'The text cannot be tokenized, because word {cur_word} is not found in the text!\n{source_text}'
            raise RuntimeError(err_msg)
        word_boundaries.append((found_idx + start_pos, found_idx + start_pos + len(cur_word)))
        start_pos = found_idx + start_pos + len(cur_word)
    return word_boundaries


def instruction_to_text(instruction: Dict[str, Union[str, List[Tuple[str, str]]]],
                        tokenizer: PreTrainedTokenizer, max_tokens: int) -> str:
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
    user_query = instruction['query']
    word_boundaries_from_query = split_text_into_words(user_query)
    text = tokenizer.apply_chat_template(
        messages + [{'role': 'user', 'content': user_query.strip()}],
        tokenize=False,
        add_generation_prompt=True
    )
    num_tokens = len(tokenizer.tokenize(text, add_special_tokens=True))
    if num_tokens > max_tokens:
        warn_msg = (f'The instruction is too long and will be reduced!\n'
                    f'{json.dumps(instruction, ensure_ascii=False, indent=4)}')
        llm_eval_logger.warning(warn_msg)
    while len(tokenizer.tokenize(text, add_special_tokens=True)) > max_tokens:
        if len(word_boundaries_from_query) == 0:
            break
        word_boundaries_from_query = word_boundaries_from_query[:-1]
        text_end = word_boundaries_from_query[-1][1]
        text = tokenizer.apply_chat_template(
            messages + [{'role': 'user', 'content': user_query[:text_end].strip()}],
            tokenize=False,
            add_generation_prompt=True
        )
    del messages
    if len(word_boundaries_from_query) == 0:
        err_msg = f'The instruction is too long!\n{json.dumps(instruction, ensure_ascii=False, indent=4)}'
        llm_eval_logger.error(err_msg)
        raise RuntimeError(err_msg)
    return text


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        llm_eval_logger.error(err_msg)
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
                        help='The output JSON file name with report.')
    parser.add_argument('-t', '--temperature', dest='temperature', type=float, required=False,
                        default=None, help='Temperature of generation.')
    parser.add_argument('--max_in_len', dest='max_input_len', type=int, required=True,
                        help='The maximal length of input query.')
    parser.add_argument('--max_out_len', dest='max_output_len', type=int, required=True,
                        help='The maximal length of generated answer.')
    parser.add_argument('--gpu_mem_util', dest='gpu_mem_util', type=float, required=False, default=0.9,
                        help='How much of the xPU’s VRAM is pre-allocated for the KV cache.')
    parser.add_argument('--max_num_batched_tokens', dest='max_num_batched_tokens', type=int,
                        required=False, default=None,
                        help='The maximal number of batched tokens per iteration with vLLM.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_file)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        llm_eval_logger.error(err_msg)
        raise IOError(err_msg)

    output_fname = os.path.normpath(args.output_file)
    if not os.path.isfile(output_fname):
        base_dir = os.path.dirname(output_fname)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                err_msg = f'The directory "{base_dir}" does not exist!'
                llm_eval_logger.error(err_msg)
                raise IOError(err_msg)

    try:
        chrf = evaluate.load('chrf')
    except Exception as err:
        llm_eval_logger.error(str(err))
        raise

    try:
        input_instructions = load_data(input_fname)
    except Exception as err:
        llm_eval_logger.error(str(err))
        raise
    info_msg = f'There are {len(input_instructions)} instructions are loaded from "{args.input_file}".'
    llm_eval_logger.info(info_msg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        generation = GenerationConfig.from_pretrained(args.llm_name)
    except Exception as err:
        llm_eval_logger.error(str(err))
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
    llm_eval_logger.info(f'max_model_len = {max_model_len}.')
    if args.max_num_batched_tokens is None:
        max_num_batched_tokens = 512
        while max_num_batched_tokens < min(max_model_len * 2, 32768):
            max_num_batched_tokens *= 2
    else:
        max_num_batched_tokens = args.max_num_batched_tokens
    llm_eval_logger.info(f'max_num_batched_tokens = {max_num_batched_tokens}.')
    try:
        model = LLM(
            model=args.llm_name,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            seed=RANDOM_SEED,
        )
    except Exception as err:
        llm_eval_logger.error(str(err))
        if torch.distributed.is_initialized():
            if hasattr(LLMEngine, 'shutdown'):
                LLMEngine.shutdown()
            torch.distributed.destroy_process_group()
            torch.cuda.empty_cache()
        raise
    llm_eval_logger.info(f'The LLM is loaded from "{args.llm_name}".')

    input_prompts = []
    input_lengths = []
    predictions = []
    references = []
    for cur_instruction in tqdm(input_instructions, desc='tokenize input instructions'):
        textualized_instruction = instruction_to_text(cur_instruction, tokenizer, args.max_input_len)
        n_tokens = len(tokenizer.tokenize(textualized_instruction, add_special_tokens=True))
        input_lengths.append(n_tokens)
        references.append([cur_instruction['response']])
        input_prompts.append(textualized_instruction)
    input_lengths.sort()
    info_msg = (f'Lengths of input sequences: minimal = {input_lengths[0]}, maximal = {input_lengths[-1]}, '
                f'median = {input_lengths[(len(input_lengths) - 1) // 2]}, '
                f'mean = {round(sum(input_lengths) / len(input_lengths))}.')
    llm_eval_logger.info(info_msg)

    outputs = model.generate(input_prompts, sampling_params)
    for val in outputs:
        answer = val.outputs[0].text.strip()
        predictions.append(answer)
    if torch.distributed.is_initialized():
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()

    total_quality = chrf.compute(
        predictions=predictions,
        references=references
    )['score']
    info_msg = f'ChrF++ of LLM {args.llm_name} on dataset {args.input_file} is {round(total_quality, 4)}.'
    llm_eval_logger.info(info_msg)
    detailed_results = []
    for test_idx in range(len(predictions)):
        new_score = chrf.compute(
            predictions=predictions[test_idx:(test_idx + 1)],
            references=references[test_idx:(test_idx + 1)]
        )['score']
        new_res = {
            'question': input_instructions[test_idx]['query'],
            'reference': references[test_idx][0],
            'prediction': predictions[test_idx],
            'ChrF++': round(new_score, 4)
        }
        detailed_results.append(new_res)
        del new_res
    detailed_results.sort(key=lambda it: -it['ChrF++'])
    report = {
        'total': {
            'model': args.llm_name,
            'test_set': {
                'name': args.input_file,
                'tokens_per_seq': {
                    'min': input_lengths[0],
                    'median': input_lengths[(len(input_lengths) - 1) // 2],
                    'mean': round(sum(input_lengths) / len(input_lengths)),
                    'max': input_lengths[-1]
                }
            },
            'ChrF++': round(total_quality, 4)
        },
        'detailed': detailed_results
    }
    with codecs.open(output_fname, mode='w', encoding='utf-8') as fp:
        json.dump(obj=report, fp=fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    llm_eval_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    llm_eval_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('llm_eval.log')
    file_handler.setFormatter(formatter)
    llm_eval_logger.addHandler(file_handler)
    main()

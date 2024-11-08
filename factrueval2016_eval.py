from argparse import ArgumentParser
import codecs
import logging
import os
import random
import sys
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from factrueval2016.factrueval2016 import recognize_named_entities


factrueval_logger = logging.getLogger(__name__)


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        factrueval_logger.error(err_msg)
        raise ValueError(err_msg)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The input name of tested LLM.')
    parser.add_argument('-o', '--output', dest='output_submission_dir', type=str, required=True,
                        help='The output directory for the submission.')
    parser.add_argument('-i', '--input', dest='input_data_name', type=str, required=True,
                        help='The FactRuEval-2016 directory with test data.')
    parser.add_argument('--dtype', dest='dtype', required=False, default='float32', type=str,
                        choices=['float32', 'float16', 'bfloat16', 'bf16', 'fp16', 'fp32'],
                        help='The PyTorch tensor type for inference.')
    parser.add_argument('--minibatch', dest='minibatch_size', required=False, default=1, type=int,
                        help='The mini-batch size.')
    parser.add_argument('--max_len', dest='max_len', required=False, default=1024, type=int,
                        help='The maximal tokens number in the recognized text.')
    args = parser.parse_args()

    submission_dirname: str = os.path.normpath(args.output_submission_dir)
    if not os.path.isdir(submission_dirname):
        basedir = os.path.dirname(submission_dirname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory {basedir} does not exist!'
                factrueval_logger.error(err_msg)
                raise IOError(err_msg)
        os.mkdir(submission_dirname)

    dataset_dirname = os.path.normpath(args.input_data_name)
    if not os.path.isdir(dataset_dirname):
        err_msg = f'The directory {dataset_dirname} does not exist!'
        factrueval_logger.error(err_msg)
        raise IOError(err_msg)

    model_name = os.path.normpath(args.model_name)
    if not os.path.isdir(model_name):
        err_msg = f'The directory {model_name} does not exist!'
        factrueval_logger.error(err_msg)
        raise IOError(err_msg)

    all_text_files = list(filter(
        lambda it: it.lower().endswith('.txt') and it.lower().startswith('book_'),
        os.listdir(dataset_dirname)
    ))
    if len(all_text_files) == 0:
        err_msg = f'The directory "{dataset_dirname}" does not contain any text file!'
        factrueval_logger.error(err_msg)
        raise IOError(err_msg)
    base_names: List[str] = []
    source_texts: List[str] = []
    for cur_fname in tqdm(all_text_files):
        full_fname = os.path.join(dataset_dirname, cur_fname)
        point_pos = cur_fname.rfind('.')
        if point_pos >= 0:
            new_base_name = cur_fname[:point_pos].strip()
        else:
            new_base_name = cur_fname
        if len(new_base_name) == 0:
            err_msg = f'The name "{new_base_name}" is wrong!'
            factrueval_logger.error(err_msg)
            raise RuntimeError(err_msg)
        base_names.append(new_base_name)
        with open(full_fname, 'rb') as fp:
            full_text = fp.read().decode('utf-8').replace('\r', '')
        if len(full_text.strip()) == 0:
            err_msg = f'The file "{full_fname}" is empty!'
            factrueval_logger.error(err_msg)
            raise IOError(err_msg)
        source_texts.append(full_text)
    info_msg = f'There are {len(source_texts)} texts in the "{dataset_dirname}".'
    factrueval_logger.info(info_msg)
    text_lengths = sorted([len(it) for it in source_texts])
    info_msg = (f'The minimal text length is {text_lengths[0]}, '
                f'the mean text length is {round(sum(text_lengths) / len(source_texts))}, '
                f'the maximal text length is {text_lengths[-1]}.')
    factrueval_logger.info(info_msg)

    device = 'cuda:0'
    try:
        if args.dtype in {'float16', 'fp16'}:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        elif args.dtype in {'bfloat16', 'bf16'}:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    except Exception as err:
        factrueval_logger.error(str(err))
        raise
    model.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generation_config = GenerationConfig.from_pretrained(model_name)
    except Exception as err:
        factrueval_logger.error(str(err))
        raise
    tokenizer.padding_side = 'left'
    factrueval_logger.info(f'The instruct model "{os.path.basename(model_name)}" is loaded.')
    for base_fname, source_text in tqdm(zip(base_names, source_texts), total=len(source_texts)):
        try:
            recognition_results = recognize_named_entities(
                source_text, tokenizer, model, generation_config,
                max_text_len=args.max_len, device=device, batch_size=args.minibatch_size
            )
        except Exception as err:
            factrueval_logger.error(str(err))
            raise
        full_fname = os.path.join(submission_dirname, base_fname + '.task1')
        with codecs.open(filename=full_fname, mode='w', encoding='utf-8') as fp:
            for res in recognition_results:
                fp.write(f'{res[0]} {res[1]} {res[2]}\n')
    factrueval_logger.info('Named entities in all texts are recognized.')


if __name__ == '__main__':
    factrueval_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    factrueval_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('factrueval2016_recognition.log')
    file_handler.setFormatter(formatter)
    factrueval_logger.addHandler(file_handler)
    main()

from argparse import ArgumentParser
import codecs
import json
import logging
import math
import random
import os
import signal
import sys

logging.getLogger('vllm').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
os.environ['VLLM_NO_PROGRESS_BAR'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from nltk import wordpunct_tokenize
import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, LLMEngine

relation_extraction_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42
PROBABILITY_THRESHOLD: float = 0.77


def handle_exit(signal, frame):
    if torch.distributed.is_initialized():
        if hasattr(LLMEngine, 'shutdown'):
            LLMEngine.shutdown()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
    sys.exit(0)


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        relation_extraction_logger.error(err_msg)
        raise ValueError(err_msg)
    torch.cuda.manual_seed(RANDOM_SEED)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='llm_name', type=str, required=True,
                        help='The input name of tested LLM')
    parser.add_argument('-i', '--input', dest='input_file', type=str, required=True,
                        help='The input JSON file name with parsed knowledge database without relations.')
    parser.add_argument('-o', '--output', dest='output_file', type=str, required=True,
                        help='The input JSON file name with parsed knowledge database with extracted relations.')
    parser.add_argument('--max_in_len', dest='max_input_len', type=int, required=True,
                        help='The maximal length of input query.')
    parser.add_argument('--max_out_len', dest='max_output_len', type=int, required=True,
                        help='The maximal length of generated answer.')
    parser.add_argument('--num_seqs', dest='num_seqs', type=int, required=False,
                        default=256, help='The maximum number of sequences that the vllm library can parallelize.')
    args = parser.parse_args()

    max_num_seqs = args.num_seqs
    if max_num_seqs < 1:
        err_msg = f'The maximum number of sequences is too small! Expected a positive integer, got {args.num_seqs}.'
        relation_extraction_logger.error(err_msg)
        raise ValueError(err_msg)

    output_fname = os.path.normpath(args.output_file)
    if os.path.isdir(output_fname):
        err_msg = f'"{output_fname}" is a directory!'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    elif not os.path.isfile(output_fname):
        base_dir = os.path.dirname(output_fname)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                err_msg = f'The directory "{base_dir}" does not exist!'
                relation_extraction_logger.error(err_msg)
                raise IOError(err_msg)

    input_fname = os.path.normpath(args.input_file)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        knowledge_samples = json.load(fp)
    relation_extraction_logger.info(f'There are {len(knowledge_samples)} samples in the knowledge database.')
    if not isinstance(knowledge_samples, list):
        err_msg = f'The file "{input_fname}" contains a wrong knowledge.'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    if not all(map(lambda it: isinstance(it, dict), knowledge_samples)):
        err_msg = f'The file "{input_fname}" contains a wrong knowledge.'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    if not all(map(lambda it: {'url', 'name', 'content', 'parsed'} <= set(it.keys()), knowledge_samples)):
        err_msg = f'The file "{input_fname}" contains a wrong knowledge.'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    if not all(map(lambda it: isinstance(it['parsed'], dict) and isinstance(it['content'], str), knowledge_samples)):
        err_msg = f'The file "{input_fname}" contains a wrong knowledge.'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)
    if not any(map(lambda it: 'normalized_entities' in it['parsed'], knowledge_samples)):
        err_msg = f'The file "{input_fname}" contains a wrong knowledge.'
        relation_extraction_logger.error(err_msg)
        raise IOError(err_msg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    except Exception as err:
        relation_extraction_logger.error(str(err))
        raise
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=100,
                                     max_tokens=args.max_output_len, logprobs=5)
    max_model_len = args.max_output_len + args.max_input_len
    try:
        model = LLM(
            model=args.llm_name,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs
        )
    except Exception as err:
        relation_extraction_logger.error(str(err))
        raise
    relation_extraction_logger.info(f'The specialized LLM for relation extraction is loaded from "{args.llm_name}".')

    system_prompt = 'Вы - эксперт в области анализа текстов и извлечения семантической информации из них.'
    prompt_for_relation_extraction = ('Напишите, что означает отношение между двумя именованными сущностями в тексте, '
                                      'то есть раскройте смысл этого отношения относительно текста (либо напишите '
                                      'прочерк, если между двумя именованными сущностями отсутствует отношение).\n\n'
                                      'Первая именованная сущность: {first_normalized_entity}\n\n'
                                      'Вторая именованная сущность: {second_normalized_entity}\n\n'
                                      'Текст: {source_text}\n\nСмысл отношения между двумя именованными сущностями: ')
    prompt_for_relation_keywords = ('Внимательно изучите следующий текст.\n\n```text\n{full_text}\n```\n\n'
                                    'В данном тексте между двумя именованными сущностями `{first_entity}` и '
                                    '`{second_entity}` установлено отношение, описываемое как:\n\n'
                                    '```text\n{relation_description}\n```\n\nВыпишите, пожалуйста, одно или несколько '
                                    'ключевых слов, которые кратко описывают общую природу, концепцию или тему этого '
                                    'отношения между двумя именованными сущностями `{first_entity}` и '
                                    '`{second_entity}` (если таких ключевых слов больше одного, то они должны быть '
                                    'разделены запятой).\n\nКлючевые слова: ')

    for sample_idx in trange(len(knowledge_samples)):
        full_text = knowledge_samples[sample_idx]['content']
        if 'relations' in knowledge_samples[sample_idx]['parsed']:
            del knowledge_samples[sample_idx]['parsed']['relations']
        if 'normalized_entities' in knowledge_samples[sample_idx]['parsed']:
            entities_dict = knowledge_samples[sample_idx]['parsed']['normalized_entities']
        else:
            entities_dict = dict()
        if len(entities_dict) > 0:
            relations_dict = dict()
            entities_list = sorted(list(entities_dict.keys()))
            messages = []
            for first_entity in entities_list:
                first_entity_ = ' '.join(first_entity.split()).strip()
                for second_entity in entities_list:
                    second_entity_ = ' '.join(second_entity.split()).strip()
                    if first_entity_.lower() != second_entity_.lower():
                        user_prompt = prompt_for_relation_extraction.format(
                            first_normalized_entity=first_entity_,
                            second_normalized_entity=second_entity_,
                            source_text=full_text
                        )
                        messages.append(tokenizer.apply_chat_template(
                            [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': user_prompt}
                            ],
                            tokenize=False,
                            add_generation_prompt=True
                        ))
                        del user_prompt
            outputs = model.generate(messages, sampling_params, use_tqdm=False)
            assert len(outputs) == len(messages)
            del messages
            output_idx = 0
            for first_entity in entities_list:
                first_entity_ = ' '.join(first_entity.split()).strip()
                for second_entity in entities_list:
                    second_entity_ = ' '.join(second_entity.split()).strip()
                    if first_entity_.lower() != second_entity_.lower():
                        relation_description = ' '.join(outputs[output_idx].outputs[0].text.strip().split()).strip()
                        relation_logprob = 0.0
                        num_tokens = 0
                        for logprob_dict in outputs[output_idx].outputs[0].logprobs:
                            token_id = list(logprob_dict.keys())[0]
                            relation_logprob += float(logprob_dict[token_id].logprob)
                            num_tokens += 1
                        relation_logprob /= num_tokens
                        relation_prob = math.exp(relation_logprob)
                        output_idx += 1
                        tokens_of_description = list(filter(
                            lambda tok: tok.isalpha(),
                            wordpunct_tokenize(relation_description.lower())
                        ))
                        if (len(tokens_of_description) > 0) and (relation_prob >= PROBABILITY_THRESHOLD):
                            filtered_description = ' '.join(tokens_of_description)
                            is_correct_relation = ((filtered_description != 'нет') and
                                                   (filtered_description != 'no') and
                                                   (filtered_description.find(' не содержит') < 0) and
                                                   (filtered_description.find('нет отношен') < 0) and
                                                   (filtered_description.find('no relation') < 0) and
                                                   (filtered_description.find('нет связ') < 0))
                            if is_correct_relation:
                                if first_entity not in relations_dict:
                                    relations_dict[first_entity] = dict()
                                if second_entity not in relations_dict[first_entity]:
                                    relations_dict[first_entity][second_entity] = {
                                        'description': relation_description,
                                        'probability': relation_prob
                                    }
            if len(relations_dict) > 0:
                messages = []
                for first_entity in sorted(list(relations_dict.keys())):
                    for second_entity in sorted(list(relations_dict[first_entity].keys())):
                        user_prompt = prompt_for_relation_keywords.format(
                            full_text=full_text,
                            first_entity=first_entity,
                            second_entity=second_entity,
                            relation_description=relations_dict[first_entity][second_entity]['description']
                        )
                        messages.append(tokenizer.apply_chat_template(
                            [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': user_prompt}
                            ],
                            tokenize=False,
                            add_generation_prompt=True
                        ))
                        del user_prompt
                outputs = model.generate(messages, sampling_params, use_tqdm=False)
                assert len(outputs) == len(messages)
                del messages
                output_idx = 0
                for first_entity in sorted(list(relations_dict.keys())):
                    for second_entity in sorted(list(relations_dict[first_entity].keys())):
                        relation_keywords = ' '.join(outputs[output_idx].outputs[0].text.strip().split()).strip()
                        output_idx += 1
                        keyword_tokens = list(filter(
                            lambda tok: tok.isalpha(),
                            wordpunct_tokenize(relation_keywords.lower())
                        ))
                        if len(keyword_tokens) > 0:
                            relations_dict[first_entity][second_entity]['keywords'] = relation_keywords
                        else:
                            del relations_dict[first_entity][second_entity]
                    if len(relations_dict[first_entity]) == 0:
                        del relations_dict[first_entity]
                if len(relations_dict) > 0:
                    knowledge_samples[sample_idx]['parsed']['relations'] = relations_dict
            del relations_dict, entities_list
        del entities_dict

    with codecs.open(output_fname, mode='w', encoding='utf-8') as fp:
        json.dump(fp=fp, obj=knowledge_samples, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    relation_extraction_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    relation_extraction_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('relation_extraction.log')
    file_handler.setFormatter(formatter)
    relation_extraction_logger.addHandler(file_handler)
    main()

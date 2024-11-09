from argparse import ArgumentParser
import codecs
import csv
import gc
import logging
import os
import random
import sys

from bert_score import BERTScorer
from datasets import load_dataset
from datasets.features import Audio
from nltk import wordpunct_tokenize
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from instructions.instructions import prepare_prompt_for_asr_correction, load_fewshot
from inference.inference import generate_answer
from fuzzy_search.fuzzy_search import levenshtein


asr_logger = logging.getLogger(__name__)
SAMPLING_FREQUENCY: int = 16_000


def calculate_cer(predicted: str, reference: str) -> float:
    if len(reference.strip()) == 0:
        if len(predicted.strip()) == 0:
            cer = 0.0
        else:
            cer = float(len(reference))
    else:
        dist = levenshtein(predicted, reference)
        cer = dist / float(len(reference))
    return cer


def remove_punctuation_and_capitalization(s: str) -> str:
    return ' '.join(list(filter(lambda it: it.isalnum(), wordpunct_tokenize(s.lower())))).strip()


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='llm_name', type=str, required=True,
                        help='The input name of tested LLM')
    parser.add_argument('-a', '--acoustic', dest='acoustic_model_name', type=str, required=True,
                        help='The input name of acoustic model.')
    parser.add_argument('-o', '--output', dest='output_report', type=str, required=True,
                        help='The output report name (CSV file).')
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, required=True,
                        help='The test set name.')
    parser.add_argument('--dtype', dest='dtype', required=False, default='float32', type=str,
                        choices=['float32', 'float16', 'bfloat16', 'bf16', 'fp16', 'fp32'],
                        help='The PyTorch tensor type for inference.')
    parser.add_argument('--eval_bert_name', dest='eval_bert_name', type=str, required=False,
                        default='ai-forever/ru-en-RoSBERTa',
                        help='Path to the BERT-like encoder for BERT score calculation.')
    parser.add_argument('--eval_bert_layer', dest='eval_bert_layer', type=int, required=False,
                        default=20, help='The hidden layer of the BERT-like encoder for BERT score calculation.')
    parser.add_argument('--prompt', dest='additional_prompt', type=str, required=False, default=None,
                        help='Additional prompt for ASR correction with LLM.')
    parser.add_argument('--fewshot', dest='fewshot', type=str, required=False, default=None,
                        help='Additional CSV file with samples for few-shot prompting.')
    args = parser.parse_args()

    report_fname: str = os.path.normpath(args.output_report)
    if not os.path.isfile(report_fname):
        basedir = os.path.dirname(report_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory {basedir} does not exist!'
                asr_logger.error(err_msg)
                raise IOError(err_msg)

    if args.fewshot is None:
        fewshot_samples = None
    else:
        fewshot_fname = os.path.normpath(args.fewshot)
        if not os.path.isfile(fewshot_fname):
            err_msg = f'The file {fewshot_fname} does not exist!'
            asr_logger.error(err_msg)
            raise IOError(err_msg)
        fewshot_samples = load_fewshot(fewshot_fname)
        asr_logger.info(f'There are {len(fewshot_samples)} few-shot samples in the "{fewshot_fname}".')

    device = 'cuda:0'
    try:
        if args.dtype in {'float16', 'fp16'}:
            transcriber = pipeline(
                'automatic-speech-recognition', model=args.acoustic_model_name,
                chunk_length_s=10, stride_length_s=(4, 2), torch_dtype=torch.float16, device=device
            )
        elif args.dtype in {'bfloat16', 'bf16'}:
            transcriber = pipeline(
                'automatic-speech-recognition', model=args.acoustic_model_name,
                chunk_length_s=10, stride_length_s=(4, 2), torch_dtype=torch.bfloat16, device=device
            )
        else:
            transcriber = pipeline(
                'automatic-speech-recognition', model=args.acoustic_model_name,
                chunk_length_s=10, stride_length_s=(4, 2), torch_dtype=torch.float32, device=device
            )
    except Exception as err:
        asr_logger.error(str(err))
        raise
    asr_logger.info(f'The ASR model "{os.path.basename(args.acoustic_model_name)}" is loaded.')

    try:
        testset = load_dataset(args.dataset_name, split='test')
    except Exception as err:
        asr_logger.warning(str(err))
        try:
            testset = load_dataset(args.dataset_name, split='validation')
        except Exception as err:
            asr_logger.error(str(err))
            raise
    if testset.features['audio'].sampling_rate != SAMPLING_FREQUENCY:
        testset = testset.cast_column('audio', Audio(sampling_rate=SAMPLING_FREQUENCY))
    info_msg = f'The test set "{os.path.basename(args.dataset_name)}" is loaded. There are {len(testset)} samples.'
    asr_logger.info(info_msg)

    try:
        true_annotations = [str(it) for it in testset['transcription']]
    except Exception as err:
        asr_logger.warning(str(err))
        try:
            true_annotations = [str(it) for it in testset['sentence']]
        except:
            asr_logger.error(str(err))
            raise
    try:
        sounds = [(it['array'] if isinstance(it['array'], np.ndarray) else np.array(it['array'], dtype=np.float32))
                  for it in testset['audio']]
    except Exception as err:
        asr_logger.error(str(err))
        raise
    del testset
    asr_results = [transcriber(cur_cound)['text'] for cur_cound in tqdm(sounds)]
    del transcriber
    gc.collect()
    torch.cuda.empty_cache()
    asr_logger.info(f'All sounds are recognized with ASR model.')

    try:
        if args.dtype in {'float16', 'fp16'}:
            llm = AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=torch.float16).to(device)
        elif args.dtype in {'bfloat16', 'bf16'}:
            llm = AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=torch.bfloat16).to(device)
        else:
            llm = AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=torch.float32).to(device)
    except Exception as err:
        asr_logger.error(str(err))
        raise
    llm.eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        generation_config = GenerationConfig.from_pretrained(args.llm_name)
    except Exception as err:
        asr_logger.error(str(err))
        raise
    generation_config.do_sample = False
    generation_config.min_new_tokens = 1
    tokenizer.padding_side = 'left'
    asr_logger.info(f'The large language model "{os.path.basename(args.llm_name)}" is loaded.')

    corrected_results = []
    for cur in tqdm(asr_results):
        if len(cur.strip()) == 0:
            corrected_results.append('')
        else:
            question = tokenizer.apply_chat_template(
                prepare_prompt_for_asr_correction(
                    input_text=cur,
                    additional_prompt=args.additional_prompt,
                    few_shots=fewshot_samples
                ),
                tokenize=False, add_generation_prompt=True
            )
            cur_text_len = len(tokenizer.tokenize(cur))
            max_text_len = max(10, round(1.5 * cur_text_len))
            answer = generate_answer(
                questions=[question],
                tokenizer=tokenizer,
                config=generation_config,
                model=llm,
                max_new_tokens=max_text_len,
                device=device
            )[0]
            corrected_results.append(' '.join(answer.split()).strip())
    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    asr_logger.info(f'All recognition results are corrected with LLM.')

    try:
        scorer = BERTScorer(
            model_type=args.eval_bert_name,
            num_layers=args.eval_bert_layer,
            rescale_with_baseline=False,
            device=device,
            batch_size=1,
            idf=False
        )
    except Exception as err:
        asr_logger.error(str(err))
        raise

    with codecs.open(report_fname, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow([
            'Reference', 'Recognized', 'Recognized and corrected',
            'Source CER', 'Source F1',
            'CER after correction', 'F1 after correction',
            'CER after correction (with punctuation)', 'F1 after correction (with punctuation)'
        ])
        for reference, corrected, source in zip(true_annotations, corrected_results, asr_results):
            normalized_source = remove_punctuation_and_capitalization(source)
            normalized_reference = remove_punctuation_and_capitalization(reference)
            normalized_corrected = remove_punctuation_and_capitalization(corrected)
            source_cer = calculate_cer(predicted=normalized_source, reference=normalized_reference)
            _, _, source_F1 = scorer.score(cands=[normalized_source], refs=[normalized_reference])
            cer_after_correction = calculate_cer(
                predicted=normalized_corrected,
                reference=normalized_reference
            )
            _, _, F1_after_correction = scorer.score(cands=[normalized_corrected], refs=[normalized_reference])
            cer_after_correction_with_punct = calculate_cer(
                predicted=corrected,
                reference=reference
            )
            _, _, F1_after_correction_with_punct = scorer.score(cands=[corrected], refs=[reference])
            data_writer.writerow([
                reference, source, corrected,
                '{0:.6f}'.format(source_cer), '{0:.6f}'.format(float(source_F1[0])),
                '{0:.6f}'.format(cer_after_correction), '{0:.6f}'.format(float(F1_after_correction[0])),
                '{0:.6f}'.format(cer_after_correction_with_punct),
                '{0:.6f}'.format(float(F1_after_correction_with_punct[0]))
            ])


if __name__ == '__main__':
    asr_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    asr_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('podlodka_speech_recognition.log')
    file_handler.setFormatter(formatter)
    asr_logger.addHandler(file_handler)
    main()

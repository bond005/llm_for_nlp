from argparse import ArgumentParser
import codecs
import csv
import gc
import logging
import os
import random
import sys

from datasets import load_dataset
from datasets.features import Audio
from nltk import wordpunct_tokenize
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

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
    parser.add_argument('-m', '--model', dest='acoustic_model_name', type=str, required=True,
                        help='The input name of acoustic model.')
    parser.add_argument('-o', '--output', dest='output_fname', type=str, required=True,
                        help='The output data name (CSV file) with few-shot samples.')
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, required=True,
                        help='The data set name.')
    parser.add_argument('-s', '--split', dest='dataset_split', type=str, required=True,
                        help='The data set split (train, validation or test).')
    parser.add_argument('-n', '--number', dest='fewshot_number', type=int, required=True,
                        help='The number of the few-shot samples.')
    args = parser.parse_args()

    fewshot_data_fname: str = os.path.normpath(args.output_fname)
    if not os.path.isfile(fewshot_data_fname):
        basedir = os.path.dirname(fewshot_data_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory {basedir} does not exist!'
                asr_logger.error(err_msg)
                raise IOError(err_msg)

    device = 'cuda:0'
    try:
        transcriber = pipeline(
            'automatic-speech-recognition', model=args.acoustic_model_name,
            chunk_length_s=10, stride_length_s=(4, 2), torch_dtype=torch.float16, device=device
        )
    except Exception as err:
        asr_logger.error(str(err))
        raise
    asr_logger.info(f'The ASR model "{os.path.basename(args.acoustic_model_name)}" is loaded.')

    try:
        speech_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as err:
        asr_logger.error(str(err))
        raise
    if speech_dataset.features['audio'].sampling_rate != SAMPLING_FREQUENCY:
        speech_dataset = speech_dataset.cast_column('audio', Audio(sampling_rate=SAMPLING_FREQUENCY))
    info_msg = (f'The speech dataset "{os.path.basename(args.dataset_name)}" is loaded. '
                f'There are {len(speech_dataset)} samples.')
    asr_logger.info(info_msg)
    if args.fewshot_number > len(speech_dataset):
        err_msg = 'The number of the few-shot samples is larger then the dataset size!'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)

    try:
        true_annotations = [str(it) for it in speech_dataset['transcription']]
    except Exception as err:
        asr_logger.warning(str(err))
        try:
            true_annotations = [str(it) for it in speech_dataset['sentence']]
        except:
            asr_logger.error(str(err))
            raise
    try:
        sounds = [(it['array'] if isinstance(it['array'], np.ndarray) else np.array(it['array'], dtype=np.float32))
                  for it in speech_dataset['audio']]
    except Exception as err:
        asr_logger.error(str(err))
        raise
    del speech_dataset
    asr_results = [transcriber(cur_cound)['text'] for cur_cound in tqdm(sounds)]
    del transcriber
    gc.collect()
    torch.cuda.empty_cache()
    asr_logger.info(f'All sounds are recognized with ASR model.')

    pairs = list(filter(
        lambda it2: (len(remove_punctuation_and_capitalization(it2[0])) > 0) and
                    (len(remove_punctuation_and_capitalization(it2[0])) > 0),
        map(
            lambda it1: (
                ' '.join(it1[0].split()).strip(),
                ' '.join(it1[1].split()).strip()
            ),
            zip(true_annotations, asr_results)
        )
    ))
    asr_logger.info(f'There are {len(pairs)} nonempty pairs "annotated-recognized" from {len(asr_results)}.')
    if args.fewshot_number > len(pairs):
        err_msg = 'The number of the few-shot samples is larger then the nonempty samples number!'
        asr_logger.error(err_msg)
        raise ValueError(err_msg)

    if args.fewshot_number > 1:
        character_error_rates = []
        cer_sum = 0.0
        for reference, recognized in pairs:
            normalized_result = remove_punctuation_and_capitalization(recognized)
            normalized_reference = remove_punctuation_and_capitalization(reference)
            cer = calculate_cer(predicted=normalized_result, reference=normalized_reference)
            character_error_rates.append(cer)
            cer_sum += cer
        sample_indices = np.array(list(range(len(pairs))), dtype=np.int32)
        selection_probabilities = np.maximum(
            0.001, np.array(character_error_rates, dtype=np.float64)
        )
        selection_probabilities /= np.sum(selection_probabilities)
        selected_indices = np.random.choice(a=sample_indices, size=args.fewshot_number // 2,
                                            replace=False, p=selection_probabilities).tolist()
        del selection_probabilities
        selection_probabilities = np.maximum(
            0.001, 1.0 - np.array(character_error_rates, dtype=np.float64)
        )
        selection_probabilities /= np.sum(selection_probabilities)
        selected_indices += np.random.choice(a=sample_indices, size=(args.fewshot_number - args.fewshot_number // 2),
                                             replace=False, p=selection_probabilities).tolist()
        del selection_probabilities
        selected_indices = list(set(map(lambda val: int(val), selected_indices)))
        random.shuffle(selected_indices)
    else:
        selected_indices = [random.choice(list(range(len(pairs))))]
    with codecs.open(fewshot_data_fname, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['QUERY', 'REFERENCE'])
        for sample_index in selected_indices:
            data_writer.writerow([
                ' '.join(pairs[sample_index][1].split()).strip(),
                ' '.join(pairs[sample_index][0].split()).strip()
            ])
    asr_logger.info(f'Few-shot prompts are successfully prepared and saved into "{fewshot_data_fname}".')


if __name__ == '__main__':
    asr_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    asr_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fewshot_for_speech_recognition.log')
    file_handler.setFormatter(formatter)
    asr_logger.addHandler(file_handler)
    main()

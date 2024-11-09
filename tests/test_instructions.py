import os
import sys
import unittest

try:
    from instructions.instructions import prepare_prompt_for_asr_correction, load_fewshot
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from instructions.instructions import prepare_prompt_for_asr_correction, load_fewshot


class TestInstructions(unittest.TestCase):
    def test_prepare_prompt_for_asr_correction_01(self):
        asr_result = 'уласны в москве интерне только в большом году что лепровели'
        true_prompt = [
            {
                'role': 'system',
                'content': 'Исправь, пожалуйста, ошибки распознавания речи в следующем тексте.'
            },
            {
                'role': 'user',
                'content': 'уласны в москве интерне только в большом году что лепровели'
            }
        ]
        calculated_prompt = prepare_prompt_for_asr_correction(asr_result)
        self.assertIsInstance(calculated_prompt, list)
        self.assertEqual(len(calculated_prompt), len(true_prompt))
        for idx, val in enumerate(calculated_prompt):
            self.assertIsInstance(val, dict)
            for k in true_prompt[idx]:
                self.assertIn(k, val)
                self.assertEqual(val[k], true_prompt[idx][k])

    def test_prepare_prompt_for_asr_correction_02(self):
        asr_result = 'уласны в москве интерне только в большом году что лепровели'
        true_prompt = [
            {
                'role': 'system',
                'content': 'Исправь, пожалуйста, ошибки распознавания речи в следующем тексте. '
                           'Учитывай, что беседа идёт про интернет и всё, что с ним связано.'
            },
            {
                'role': 'user',
                'content': 'уласны в москве интерне только в большом году что лепровели'
            }
        ]
        calculated_prompt = prepare_prompt_for_asr_correction(
            asr_result,
            additional_prompt='Учитывай, что беседа идёт про интернет и всё, что с ним связано.'
        )
        self.assertIsInstance(calculated_prompt, list)
        self.assertEqual(len(calculated_prompt), len(true_prompt))
        for idx, val in enumerate(calculated_prompt):
            self.assertIsInstance(val, dict)
            for k in true_prompt[idx]:
                self.assertIn(k, val)
                self.assertEqual(val[k], true_prompt[idx][k])

    def test_prepare_prompt_for_asr_correction_03(self):
        asr_result = 'уласны в москве интерне только в большом году что лепровели'
        few_shots = [
            [
                {
                    'role': 'user',
                    'content': 'нейро сети эта харошо'
                },
                {
                    'role': 'assistant',
                    'content': 'Нейросети - это хорошо.'
                }
            ],
            [
                {
                    'role': 'user',
                    'content': 'марози сонцы день чадесны'
                },
                {
                    'role': 'assistant',
                    'content': 'Мороз и солнце; день чудесный!'
                }
            ]
        ]
        true_prompt = [
            {
                'role': 'system',
                'content': 'Исправь, пожалуйста, ошибки распознавания речи в следующем тексте.'
            },
            {
                'role': 'user',
                'content': 'нейро сети эта харошо'
            },
            {
                'role': 'assistant',
                'content': 'Нейросети - это хорошо.'
            },
            {
                'role': 'user',
                'content': 'марози сонцы день чадесны'
            },
            {
                'role': 'assistant',
                'content': 'Мороз и солнце; день чудесный!'
            },
            {
                'role': 'user',
                'content': 'уласны в москве интерне только в большом году что лепровели'
            }
        ]
        calculated_prompt = prepare_prompt_for_asr_correction(asr_result, few_shots=few_shots)
        self.assertIsInstance(calculated_prompt, list)
        self.assertEqual(len(calculated_prompt), len(true_prompt))
        for idx, val in enumerate(calculated_prompt):
            self.assertIsInstance(val, dict)
            for k in true_prompt[idx]:
                self.assertIn(k, val)
                self.assertEqual(val[k], true_prompt[idx][k])

    def test_load_fewshot(self):
        fname = os.path.join(os.path.dirname(__file__), 'test_data', 'few_shot_prompts.csv')
        true_few_shots = [
            [
                {
                    'role': 'user',
                    'content': 'нейро сети эта харошо'
                },
                {
                    'role': 'assistant',
                    'content': 'Нейросети - это хорошо.'
                }
            ],
            [
                {
                    'role': 'user',
                    'content': 'марози сонцы день чадесны'
                },
                {
                    'role': 'assistant',
                    'content': 'Мороз и солнце; день чудесный!'
                }
            ]
        ]
        loaded_few_shots = load_fewshot(fname)
        self.assertIsInstance(loaded_few_shots, list)
        self.assertEqual(len(loaded_few_shots), len(true_few_shots))
        for idx in range(len(true_few_shots)):
            self.assertIsInstance(loaded_few_shots[idx], list)
            self.assertEqual(len(loaded_few_shots[idx]), len(true_few_shots[idx]))
            for loaded_val, true_val in zip(loaded_few_shots[idx], true_few_shots[idx]):
                self.assertIsInstance(loaded_val, dict)
                self.assertEqual(set(loaded_val.keys()), set(true_val.keys()))
                for k in true_val:
                    self.assertEqual(loaded_val[k], true_val[k])


if __name__ == '__main__':
    unittest.main(verbosity=2)

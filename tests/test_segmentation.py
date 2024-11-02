import os
import sys
import unittest

from transformers import AutoTokenizer

try:
    from segmentation.segmentation import segment_long_text, split_text_by_sentences
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from segmentation.segmentation import segment_long_text, split_text_by_sentences


class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'model_with_tokenizer')
        cls.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def test_split_text_by_sentences_01(self):
        s = ' Мама мыла раму. Папа мыл синхрофазотрон.  Саша смотрел мультики.'
        true_bounds = [
            (1, 16),
            (17, 41),
            (43, 65)
        ]
        calculated_bounds = split_text_by_sentences(s)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(calculated_bounds), len(true_bounds))
        err_msg = f'TRUE: {true_bounds}      CALCULATED: {calculated_bounds}'
        for idx, val in enumerate(calculated_bounds):
            self.assertIsInstance(val, tuple, msg=err_msg)
            self.assertEqual(len(val), 2, msg=err_msg)
            self.assertIsInstance(val[0], int, msg=err_msg)
            self.assertIsInstance(val[1], int, msg=err_msg)
            self.assertEqual(val, true_bounds[idx], msg=err_msg)

    def test_segment_long_text_01(self):
        s = ' Мама мыла раму. Папа мыл синхрофазотрон.  Саша смотрел мультики. Студенты делали лабы.'
        max_text_len = 1024
        true_bounds = [(1, 87)]
        calculated_bounds = segment_long_text(s, self.tokenizer, max_text_len)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(calculated_bounds), len(true_bounds))
        err_msg = f'TRUE: {true_bounds}      CALCULATED: {calculated_bounds}'
        for idx, val in enumerate(calculated_bounds):
            self.assertIsInstance(val, tuple, msg=err_msg)
            self.assertEqual(len(val), 2, msg=err_msg)
            self.assertIsInstance(val[0], int, msg=err_msg)
            self.assertIsInstance(val[1], int, msg=err_msg)
            self.assertEqual(val, true_bounds[idx], msg=err_msg)

    def test_segment_long_text_02(self):
        s = '  '
        max_text_len = 1024
        calculated_bounds = segment_long_text(s, self.tokenizer, max_text_len)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(calculated_bounds), 0)

    def test_segment_long_text_03(self):
        s = ' Мама мыла раму. Папа мыл синхрофазотрон.  Саша смотрел мультики. Студенты делали лабы.'
        max_text_len = 30
        true_bounds = [(1, 41), (43, 87)]
        calculated_bounds = segment_long_text(s, self.tokenizer, max_text_len)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(calculated_bounds), len(true_bounds))
        err_msg = f'TRUE: {true_bounds}      CALCULATED: {calculated_bounds}'
        for idx, val in enumerate(calculated_bounds):
            self.assertIsInstance(val, tuple, msg=err_msg)
            self.assertEqual(len(val), 2, msg=err_msg)
            self.assertIsInstance(val[0], int, msg=err_msg)
            self.assertIsInstance(val[1], int, msg=err_msg)
            self.assertEqual(val, true_bounds[idx], msg=err_msg)

    def test_segment_long_text_04(self):
        s = ' Мама мыла раму. Папа мыл синхрофазотрон.  Саша смотрел мультики. Студенты делали лабы.'
        max_text_len = 10
        true_bounds = [(1, 16), (17, 41), (43, 65), (66, 87)]
        calculated_bounds = segment_long_text(s, self.tokenizer, max_text_len)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(calculated_bounds), len(true_bounds))
        err_msg = f'TRUE: {true_bounds}      CALCULATED: {calculated_bounds}'
        for idx, val in enumerate(calculated_bounds):
            self.assertIsInstance(val, tuple, msg=err_msg)
            self.assertEqual(len(val), 2, msg=err_msg)
            self.assertIsInstance(val[0], int, msg=err_msg)
            self.assertIsInstance(val[1], int, msg=err_msg)
            self.assertEqual(val, true_bounds[idx], msg=err_msg)

    def test_segment_long_text_05(self):
        s = ' Мама мыла раму. Папа мыл синхрофазотрон.  Саша смотрел мультики. Студенты делали лабы.'
        max_text_len = 4
        true_bounds = [(1, 16), (17, 41), (43, 65), (66, 87)]
        calculated_bounds = segment_long_text(s, self.tokenizer, max_text_len)
        self.assertIsInstance(calculated_bounds, list)
        self.assertEqual(len(calculated_bounds), len(true_bounds))
        err_msg = f'TRUE: {true_bounds}      CALCULATED: {calculated_bounds}'
        for idx, val in enumerate(calculated_bounds):
            self.assertIsInstance(val, tuple, msg=err_msg)
            self.assertEqual(len(val), 2, msg=err_msg)
            self.assertIsInstance(val[0], int, msg=err_msg)
            self.assertIsInstance(val[1], int, msg=err_msg)
            self.assertEqual(val, true_bounds[idx], msg=err_msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)

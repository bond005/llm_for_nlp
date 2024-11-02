import os
import sys
import unittest

try:
    from fuzzy_search.fuzzy_search import find_substring
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from fuzzy_search.fuzzy_search import find_substring


class TestFuzzySearch(unittest.TestCase):
    def test_find_substring_01(self):
        full_string = 'Мама мыла раму.'
        substring = 'мыла'
        true_res = (5, 9)
        calculated_res = find_substring(full_string, substring)
        self.assertIsInstance(calculated_res, tuple)
        self.assertEqual(len(calculated_res), 2)
        self.assertIsInstance(calculated_res[0], int)
        self.assertIsInstance(calculated_res[1], int)
        self.assertEqual(calculated_res, true_res)

    def test_find_substring_02(self):
        full_string = 'Мама мыла раму.'
        substring = 'мала'
        true_res = (5, 9)
        calculated_res = find_substring(full_string, substring)
        self.assertIsInstance(calculated_res, tuple)
        self.assertEqual(len(calculated_res), 2)
        self.assertIsInstance(calculated_res[0], int)
        self.assertIsInstance(calculated_res[1], int)
        self.assertEqual(calculated_res, true_res)

    def test_find_substring_03(self):
        full_string = 'Мама мыла раму.'
        substring = 'мsла'
        true_res = (5, 9)
        calculated_res = find_substring(full_string, substring)
        self.assertIsInstance(calculated_res, tuple)
        self.assertEqual(len(calculated_res), 2)
        self.assertIsInstance(calculated_res[0], int)
        self.assertIsInstance(calculated_res[1], int)
        self.assertEqual(calculated_res, true_res)

    def test_find_substring_04(self):
        full_string = 'Мама мыла раму.'
        substring = 'Папа мыл синхрофазотрон.'
        true_res = (0, 15)
        calculated_res = find_substring(full_string, substring)
        self.assertIsInstance(calculated_res, tuple)
        self.assertEqual(len(calculated_res), 2)
        self.assertIsInstance(calculated_res[0], int)
        self.assertIsInstance(calculated_res[1], int)
        self.assertEqual(calculated_res, true_res)

    def test_find_substring_05(self):
        full_string = 'Мама мыла раму.'
        substring = 'Папа мыл синхрофазотрон. Вот такой у нас пара!'
        calculated_res = find_substring(full_string, substring)
        self.assertIsNone(calculated_res)

    def test_find_substring_06(self):
        full_string = 'Мама мыла раму.'
        substring = ''
        calculated_res = find_substring(full_string, substring)
        self.assertIsNone(calculated_res)

    def test_find_substring_07(self):
        full_string = ''
        substring = 'Папа мыл синхрофазотрон.'
        calculated_res = find_substring(full_string, substring)
        self.assertIsNone(calculated_res)


if __name__ == '__main__':
    unittest.main(verbosity=2)

import os
import sys
import unittest

try:
    from factrueval2016.factrueval2016 import calculate_entity_bounds
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from factrueval2016.factrueval2016 import calculate_entity_bounds


class TestFactRuEval(unittest.TestCase):
    def test_calculate_entity_bounds_01(self):
        s = ('Новосибирский государственный университет расположен в Академгородке - самом живописном '
             'месте Новосибирска.')
        entities = ['Новосибирский государственный университет', 'Академгородке', 'Новосибирска']
        true_bounds = [(0, 41), (55, 68), (94, 106)]
        calc_bounds = calculate_entity_bounds(s, entities)
        self.assertIsInstance(calc_bounds, list)
        err_msg = f'{calc_bounds} != {true_bounds}'
        self.assertEqual(len(calc_bounds), len(true_bounds), msg=err_msg)
        for idx, val in enumerate(true_bounds):
            self.assertIsInstance(calc_bounds[idx], tuple)
            self.assertEqual(len(calc_bounds[idx]), 2)
            self.assertIsInstance(calc_bounds[idx][0], int)
            self.assertIsInstance(calc_bounds[idx][1], int)
            self.assertEqual(calc_bounds[idx], val, msg=err_msg)

    def test_calculate_entity_bounds_02(self):
        s = ('Новосибирский государственный университет расположен в Академгородке - самом живописном '
             'месте Новосибирска.')
        entities = ['Новосибирск государственый университет', 'Академгородке', 'Новосибирск']
        true_bounds = [(0, 41), (55, 68), (94, 106)]
        calc_bounds = calculate_entity_bounds(s, entities)
        self.assertIsInstance(calc_bounds, list)
        err_msg = f'{calc_bounds} != {true_bounds}'
        self.assertEqual(len(calc_bounds), len(true_bounds), msg=err_msg)
        for idx, val in enumerate(true_bounds):
            self.assertIsInstance(calc_bounds[idx], tuple)
            self.assertEqual(len(calc_bounds[idx]), 2)
            self.assertIsInstance(calc_bounds[idx][0], int)
            self.assertIsInstance(calc_bounds[idx][1], int)
            self.assertEqual(calc_bounds[idx], val, msg=err_msg)

    def test_calculate_entity_bounds_03(self):
        s = ('Новосибирский государственный университет расположен в Академгородке - самом живописном '
             'месте Новосибирска.')
        entities = []
        calc_bounds = calculate_entity_bounds(s, entities)
        self.assertIsInstance(calc_bounds, list)
        self.assertEqual(len(calc_bounds), 0)

    def test_calculate_entity_bounds_04(self):
        s = ('Новосибирский государственный университет расположен в Академгородке - самом живописном '
             'месте Новосибирска.')
        entities = ['Новосибирск государственый университет', 'тру-ля-ля', 'Академгородке', 'Новосибисрк']
        true_bounds = [(0, 41), (55, 68), (94, 106)]
        calc_bounds = calculate_entity_bounds(s, entities)
        self.assertIsInstance(calc_bounds, list)
        err_msg = f'{calc_bounds} != {true_bounds}'
        self.assertEqual(len(calc_bounds), len(true_bounds), msg=err_msg)
        for idx, val in enumerate(true_bounds):
            self.assertIsInstance(calc_bounds[idx], tuple)
            self.assertEqual(len(calc_bounds[idx]), 2)
            self.assertIsInstance(calc_bounds[idx][0], int)
            self.assertIsInstance(calc_bounds[idx][1], int)
            self.assertEqual(calc_bounds[idx], val, msg=err_msg)

    def test_calculate_entity_bounds_05(self):
        s = ('Новосибирский государственный университет расположен в Академгородке - самом живописном '
             'месте Новосибирска.')
        entities = ['Новосибирск государственый университет', 'Академгородке', 'Новосибирск', 'Новосибирск',
                    'Новосибирск', 'Новосибирск', 'Новосибирск']
        true_bounds = [(0, 41), (55, 68), (94, 106)]
        calc_bounds = calculate_entity_bounds(s, entities)
        self.assertIsInstance(calc_bounds, list)
        err_msg = f'{calc_bounds} != {true_bounds}'
        self.assertEqual(len(calc_bounds), len(true_bounds), msg=err_msg)
        for idx, val in enumerate(true_bounds):
            self.assertIsInstance(calc_bounds[idx], tuple)
            self.assertEqual(len(calc_bounds[idx]), 2)
            self.assertIsInstance(calc_bounds[idx][0], int)
            self.assertIsInstance(calc_bounds[idx][1], int)
            self.assertEqual(calc_bounds[idx], val, msg=err_msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)

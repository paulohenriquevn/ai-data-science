import unittest
import pandas as pd
import numpy as np
from src.analyzers.missing_values import MissingValuesAnalyzer, MissingValuesProblem, MissingValuesSolution


class TestMissingValuesAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = MissingValuesAnalyzer()
        
    def test_no_missing_values(self):
        """Testa análise em DataFrame sem valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['problem'], MissingValuesProblem.NO_MISSING_VALUES.name)
        self.assertIsNone(result[0]['solution'])
        self.assertEqual(result[0]['column'], [])
        
    def test_less_than_5_percent_missing(self):
        """Testa análise em DataFrame com menos de 5% de valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2, 3, 4, None, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'col2': ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['column'], 'col1')
        self.assertEqual(result[0]['problem'], MissingValuesProblem.LESS_5.name)
        self.assertEqual(result[0]['solution'], MissingValuesSolution.IMPUTATION_MEDIA)
        
    def test_between_5_and_30_percent_missing(self):
        """Testa análise em DataFrame com entre 5% e 30% de valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, None, None, None]
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['column'], 'col1')
        self.assertEqual(result[0]['problem'], MissingValuesProblem.BETWEEN_5_30.name)
        self.assertEqual(result[0]['solution'], MissingValuesSolution.IMPUTATION_MEDIANA)
        
    def test_greater_than_30_percent_missing(self):
        """Testa análise em DataFrame com mais de 30% de valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, None, None, None, None]
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['column'], 'col1')
        self.assertEqual(result[0]['problem'], MissingValuesProblem.GREATER_30.name)
        self.assertEqual(result[0]['solution'], MissingValuesSolution.REMOVE_COLUMN)
        
    def test_multiple_columns_with_missing_values(self):
        """Testa análise em DataFrame com múltiplas colunas com valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5, 1, 2, 3, 4, 5],
            'col2': [None, 2, 3, None, 5, 1, 2, 3, 4, 5 ],
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 2)
        
        # Verificar se as colunas estão nos resultados (a ordem pode variar)
        columns = [r['column'] for r in result]
        self.assertIn('col1', columns)
        self.assertIn('col2', columns)
        
        # Verificar as soluções para cada coluna
        for r in result:
            if r['column'] == 'col1':
                self.assertEqual(r['problem'], MissingValuesProblem.BETWEEN_5_30.name)
                self.assertEqual(r['solution'], MissingValuesSolution.IMPUTATION_MEDIANA)
            elif r['column'] == 'col2':
                self.assertEqual(r['problem'], MissingValuesProblem.BETWEEN_5_30.name)
                self.assertEqual(r['solution'], MissingValuesSolution.IMPUTATION_MEDIANA)
                
    def test_edge_case_exactly_5_percent(self):
        """Testa caso de borda com exatamente 5% de valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['column'], 'col1')
        self.assertEqual(result[0]['problem'], MissingValuesProblem.BETWEEN_5_30.name)
        self.assertEqual(result[0]['solution'], MissingValuesSolution.IMPUTATION_MEDIANA)
        
    def test_edge_case_exactly_30_percent(self):
        """Testa caso de borda com exatamente 30% de valores ausentes."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, None, None, None]
        })
        
        result = self.analyzer.analyze(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['column'], 'col1')
        self.assertEqual(result[0]['problem'], MissingValuesProblem.BETWEEN_5_30.name)
        self.assertEqual(result[0]['solution'], MissingValuesSolution.IMPUTATION_MEDIANA)


if __name__ == '__main__':
    unittest.main()
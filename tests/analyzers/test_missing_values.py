import unittest
import pandas as pd
import numpy as np
from src.analyzers.missing_values import MissingValuesAnalyzer, MissingValuesProblemType, MissingValuesSolution

class TestMissingValuesAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.analyzer = MissingValuesAnalyzer()
        
    def test_no_missing_values(self):
        """Testa o caso de nenhum valor ausente"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        results = self.analyzer.analyze(df)
        
        # Verifica se há resultados para ambas as colunas
        self.assertEqual(len(results), 2)
        
        # Verifica se ambas as colunas são classificadas como sem problemas
        for result in results:
            self.assertEqual(result['problem'], MissingValuesProblemType.NENHUM.name)
            self.assertEqual(result['description'], MissingValuesProblemType.NENHUM.value)
            self.assertIn(MissingValuesSolution.NENHUMA_ACAO_NECESSARIA.name, result['actions'])
            self.assertEqual(result['statistics']['missing_count'], 0)
            self.assertEqual(result['statistics']['missing_percent'], 0)
            self.assertEqual(result['suggestion'], MissingValuesSolution.NENHUMA_ACAO_NECESSARIA)
    
    def test_few_missing_values(self):
        """Testa o caso de poucos valores ausentes (<5%)"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, None, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'col2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
        })
        
        results = self.analyzer.analyze(df)
        
        # Encontra o resultado para col1
        col1_result = next(r for r in results if r['column'] == 'col1')
        
        # Verifica se col1 é classificada como tendo poucos valores ausentes
        self.assertEqual(col1_result['problem'], MissingValuesProblemType.POUCO.name)
        self.assertEqual(col1_result['description'], MissingValuesProblemType.POUCO.value)
        self.assertEqual(col1_result['statistics']['missing_count'], 1)
        self.assertEqual(col1_result['statistics']['missing_percent'], 5)
        self.assertIn(MissingValuesSolution.IMPUTE_MEDIA.name, col1_result['actions'])
        self.assertEqual(col1_result['suggestion'], MissingValuesSolution.IMPUTE_MEDIA)
        
        # Verifica se col2 é classificada como sem problemas
        col2_result = next(r for r in results if r['column'] == 'col2')
        self.assertEqual(col2_result['problem'], MissingValuesProblemType.NENHUM.name)
    
    def test_many_missing_values(self):
        """Testa o caso de muitos valores ausentes (>20%)"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, None, None, None, None, 8, 9, 10]
        })
        
        results = self.analyzer.analyze(df)
        col1_result = results[0]
        
        # Verifica se col1 é classificada como tendo muitos valores ausentes
        self.assertEqual(col1_result['problem'], MissingValuesProblemType.MUITO.name)
        self.assertEqual(col1_result['description'], MissingValuesProblemType.MUITO.value)
        self.assertEqual(col1_result['statistics']['missing_count'], 4)
        self.assertEqual(col1_result['statistics']['missing_percent'], 40)
        self.assertIn(MissingValuesSolution.MICE_IMPUTER.name, col1_result['actions'])
        self.assertEqual(col1_result['suggestion'], MissingValuesSolution.MICE_IMPUTER)
    
    def test_target_variable_missing(self):
        """Testa o caso de valores ausentes na variável alvo"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [1, 0, None, 1, 0]
        })
        
        results = self.analyzer.analyze(df)
        target_result = next(r for r in results if r['column'] == 'target')
        
        # Verifica se a coluna target é classificada corretamente
        self.assertEqual(target_result['problem'], MissingValuesProblemType.ALVO_MISSING.name)
        self.assertEqual(target_result['description'], MissingValuesProblemType.ALVO_MISSING.value)
        self.assertIn(MissingValuesSolution.EXCLUIR_LINHAS.name, target_result['actions'])
        self.assertEqual(target_result['suggestion'], MissingValuesSolution.EXCLUIR_LINHAS)
    
    def test_categorical_column_suggestion(self):
        """Testa se colunas categóricas recebem sugestões apropriadas"""
        df = pd.DataFrame({
            'cat_col': ['a', 'b', 'c', None, 'e', None, 'g', 'h', 'i', 'j']
        })
        
        results = self.analyzer.analyze(df)
        cat_result = results[0]
        
        # Verifica se a coluna categórica recebe a sugestão de categoria 'Desconhecido'
        self.assertIn(MissingValuesSolution.CATEGORIA_DESCONHECIDA.name, cat_result['actions'])
        self.assertEqual(cat_result['suggestion'], MissingValuesSolution.CATEGORIA_DESCONHECIDA)

if __name__ == '__main__':
    unittest.main() 
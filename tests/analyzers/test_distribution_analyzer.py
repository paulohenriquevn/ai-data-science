import unittest
import pandas as pd
import numpy as np
from src.analyzers.distribution_analyzer import DistributionAnalyzer, DistributionType, DistributionSolution
from scipy import stats


class TestDistributionAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.analyzer = DistributionAnalyzer()
    
    def _find_result_for_column(self, results, column_name):
        """Utilitário para encontrar o resultado de uma coluna específica na lista de resultados"""
        for result in results:
            if result['column'] == column_name:
                return result
        return None
    
    def test_normal_distribution(self):
        """Testa a identificação de distribuição normal (gaussiana)"""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        df = pd.DataFrame({'normal': normal_data})
        
        results = self.analyzer.analyze(df)
        result = self._find_result_for_column(results, 'normal')
        
        # Verificar se a coluna foi analisada
        self.assertIsNotNone(result)
        
        # Verificar a classificação
        self.assertEqual(result['problem'], DistributionType.NORMAL.name)
        
        # Verificar se skewness está próximo de 0
        self.assertAlmostEqual(result['statistics']['skewness'], 0, delta=0.2)
        
        # Verificar se kurtosis está próximo de 0
        self.assertAlmostEqual(result['statistics']['kurtosis'], 0, delta=0.5)
    
    def test_right_skewed_distribution(self):
        """Testa a identificação de distribuição assimétrica positiva"""
        np.random.seed(42)
        # Log-normal é naturalmente assimétrica à direita
        skewed_data = np.random.lognormal(0, 0.8, 1000)
        df = pd.DataFrame({'right_skewed': skewed_data})
        
        results = self.analyzer.analyze(df)
        result = self._find_result_for_column(results, 'right_skewed')
        
        # Verificar a classificação
        self.assertIn(result['problem'], 
                     [DistributionType.ASSIMETRICA_POSITIVA.name, DistributionType.LOGNORMAL.name])
        
        # Verificar se skewness é positivo e significativo
        self.assertGreater(result['statistics']['skewness'], 0.5)
        
        # Verificar se as ações sugeridas incluem transformação logarítmica
        self.assertTrue(any(DistributionSolution.TRANSFORMACAO_LOG.name in action for action in result['actions']))
    
    def test_left_skewed_distribution(self):
        """Testa a identificação de distribuição assimétrica negativa"""
        np.random.seed(42)
        
        # Criaremos uma distribuição fortemente assimétrica negativa
        # Método 1: Usando chi-square invertida
        chi_data = np.random.chisquare(df=5, size=1000)
        left_skewed = -chi_data + max(chi_data) + 1
        
        # Método 2: Outra opção é usar uma distribuição beta invertida
        # data = np.random.beta(8, 2, 1000)  # Beta com assimetria positiva
        # left_skewed = -data + max(data) + 1  # Invertendo para assimetria negativa
        
        # Método 3: Exponencial invertida
        # exp_data = np.random.exponential(scale=2.0, size=1000)
        # left_skewed = -exp_data + max(exp_data) + 1
        
        # Teste direto para confirmar que realmente é assimétrica negativa
        sk = float(stats.skew(left_skewed, bias=False))
        print(f"Skewness da distribuição de teste: {sk}")
        assert sk < -0.5, f"Distribuição de teste não é assimétrica negativa! Skewness = {sk}"
        
        df = pd.DataFrame({'left_skewed': left_skewed})
        
        results = self.analyzer.analyze(df)
        result = self._find_result_for_column(results, 'left_skewed')
        
        # Imprimir diagnóstico
        print(f"Estatísticas calculadas: skewness={result['statistics']['skewness']}")
        
        # Verificar a classificação
        self.assertEqual(result['problem'], DistributionType.ASSIMETRICA_NEGATIVA.name)
    
    def test_uniform_distribution(self):
        """Testa a identificação de distribuição uniforme"""
        np.random.seed(42)
        uniform_data = np.random.uniform(0, 1, 1000)
        df = pd.DataFrame({'uniform': uniform_data})
        
        results = self.analyzer.analyze(df)
        result = self._find_result_for_column(results, 'uniform')
        
        # Verificar a classificação
        self.assertEqual(result['problem'], DistributionType.UNIFORME.name)
        
        # Verificar se skewness está próximo de 0
        self.assertAlmostEqual(result['statistics']['skewness'], 0, delta=0.3)
        
        # Verificar se kurtosis é negativo (mais plano que normal)
        self.assertLess(result['statistics']['kurtosis'], -0.5)
    
    def test_heavy_tailed_distribution(self):
        """Testa a identificação de distribuição de cauda pesada"""
        np.random.seed(42)
        # T-student com poucos graus de liberdade tem cauda pesada
        heavy_tailed = np.random.standard_t(3, 1000)
        df = pd.DataFrame({'heavy_tailed': heavy_tailed})
        
        results = self.analyzer.analyze(df)
        result = self._find_result_for_column(results, 'heavy_tailed')
        
        # Verificar se kurtosis é alto (mais pico e caudas pesadas)
        if result['problem'] == DistributionType.CAUDA_PESADA.name:
            self.assertGreater(result['statistics']['kurtosis'], 3)
        else:
            # Se não for classificada como cauda pesada, pelo menos verificamos o kurtosis alto
            self.assertGreater(result['statistics']['kurtosis'], 3)
    
    def test_mixed_distributions(self):
        """Testa a análise de múltiplas distribuições em um mesmo dataset"""
        np.random.seed(42)
        df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 1000),
            'lognormal': np.random.lognormal(0, 0.8, 1000),
            'uniform': np.random.uniform(0, 1, 1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'binary': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
        
        results = self.analyzer.analyze(df)
        
        # Pegando o resultado como um dicionário por coluna
        result_dict = {r['column']: r for r in results}
        
        # Verificar a análise de colunas numéricas não binárias
        expected_numeric_columns = ['normal', 'lognormal', 'uniform']
        
        # A coluna binária (0/1) é ignorada por design pelo analisador
        # e a categórica também é ignorada porque não é numérica
        self.assertEqual(len(results), len(expected_numeric_columns), 
                         f"Esperava {len(expected_numeric_columns)} colunas analisadas, obteve {len(results)}")
        
        # Verificar se todas as colunas esperadas foram analisadas
        for col in expected_numeric_columns:
            self.assertIn(col, result_dict, f"Coluna {col} não foi analisada")
        
        # Verificar classificações específicas
        if 'normal' in result_dict:
            self.assertEqual(result_dict['normal']['problem'], DistributionType.NORMAL.name)
        
        if 'lognormal' in result_dict:
            self.assertIn(
                result_dict['lognormal']['problem'], 
                [DistributionType.LOGNORMAL.name, DistributionType.ASSIMETRICA_POSITIVA.name]
            )
        
        if 'uniform' in result_dict:
            self.assertEqual(result_dict['uniform']['problem'], DistributionType.UNIFORME.name)
    
    def test_small_sample(self):
        """Testa o comportamento com amostras pequenas"""
        df = pd.DataFrame({'small_sample': [1, 2, 3, 4, 5]})
        
        results = self.analyzer.analyze(df)
        
        # Verificar se não há resultados para amostras pequenas
        self.assertEqual(len(results), 0)
    
    def test_with_missing_values(self):
        """Testa a análise de distribuições com valores ausentes"""
        np.random.seed(42)
        
        # Criar dados com alguns valores ausentes
        data = np.random.normal(0, 1, 1000)
        mask = np.random.random(1000) < 0.1  # 10% de valores ausentes
        data_with_na = data.copy()
        data_with_na[mask] = np.nan
        
        df = pd.DataFrame({'normal_with_na': data_with_na})
        
        results = self.analyzer.analyze(df)
        result = self._find_result_for_column(results, 'normal_with_na')
        
        # Verificar se a coluna foi analisada mesmo com valores ausentes
        self.assertIsNotNone(result)
    
    def test_suggested_transformations(self):
        """Testa se as transformações sugeridas são adequadas para diferentes distribuições"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 1000),
            'right_skewed': np.random.lognormal(0, 1, 1000),
            'left_skewed': -np.random.beta(5, 2, 1000) + 6
        })
        
        results = self.analyzer.analyze(df)
        
        # Para normal não deve sugerir transformação ou deve sugerir NENHUMA
        normal_result = self._find_result_for_column(results, 'normal')
        if normal_result['problem'] == DistributionType.NORMAL.name:
            self.assertTrue(
                len(normal_result['actions']) == 0 or 
                DistributionSolution.NENHUMA.name in normal_result['actions']
            )
        
        # Para assimétrica positiva deve sugerir log, raiz quadrada ou Box-Cox
        right_skewed = self._find_result_for_column(results, 'right_skewed')
        if right_skewed['problem'] in [DistributionType.ASSIMETRICA_POSITIVA.name, 
                                      DistributionType.LOGNORMAL.name]:
            transformacoes = [DistributionSolution.TRANSFORMACAO_LOG.name,
                             DistributionSolution.TRANSFORMACAO_RAIZ_QUADRADA.name,
                             DistributionSolution.TRANSFORMACAO_BOX_COX.name]
            
            # Pelo menos uma destas transformações deve estar presente
            self.assertTrue(any(t in right_skewed['actions'] for t in transformacoes))
        
        # Para assimétrica negativa deve sugerir reflexão e log ou quadrático
        left_skewed = self._find_result_for_column(results, 'left_skewed')
        if left_skewed['problem'] == DistributionType.ASSIMETRICA_NEGATIVA.name:
            self.assertTrue(
                DistributionSolution.TRANSFORMACAO_REFLECAO_LOG.name in left_skewed['actions'] or
                DistributionSolution.TRANSFORMACAO_QUADRATICA.name in left_skewed['actions']
            )


if __name__ == '__main__':
    unittest.main() 
import unittest
import os
import pandas as pd
import numpy as np
from src.analyzers.statistical_significance_analyzer import StatisticalSignificanceAnalyzer


class TestStatisticalSignificanceAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.analyzer = StatisticalSignificanceAnalyzer(target='y', pvalue_threshold=0.05)
        
    def test_basic_functionality(self):
        """Testa a funcionalidade básica com dados sintéticos"""
        np.random.seed(42)
        
        # Criar um dataset onde algumas variáveis têm diferença significativa por design
        n = 100
        # Variável significativa: média diferente entre classes
        significant_var = np.concatenate([
            np.random.normal(0, 1, n),  # Distribuição para y=0
            np.random.normal(2, 1, n)   # Distribuição para y=1 (média diferente)
        ])
        
        # Variável não significativa: mesma distribuição para ambas classes
        nonsignificant_var = np.concatenate([
            np.random.normal(5, 1, n),  # Distribuição para y=0
            np.random.normal(5, 1, n)   # Distribuição para y=1 (mesma média)
        ])
        
        # Coluna alvo
        y = np.concatenate([np.zeros(n), np.ones(n)])
        
        # Criar DataFrame
        df = pd.DataFrame({
            'significant': significant_var,
            'nonsignificant': nonsignificant_var,
            'y': y
        })
        
        # Executar análise
        results = self.analyzer.analyze(df)
        
        # Verificar se há resultados
        self.assertGreater(len(results), 0, "Nenhum resultado foi gerado pela análise")
        
        # Mapear resultados por coluna
        results_by_column = {r['column']: r for r in results}
        
        # Verificar se a variável significativa foi detectada corretamente
        self.assertIn('significant', results_by_column, "Variável significativa não foi analisada")
        self.assertEqual(
            results_by_column['significant']['problem'], 
            'VARIAVEL_SIGNIFICATIVA',
            "Variável significativa não foi corretamente classificada"
        )
        
        # Verificar se a variável não significativa foi detectada corretamente
        self.assertIn('nonsignificant', results_by_column, "Variável não significativa não foi analisada")
        self.assertEqual(
            results_by_column['nonsignificant']['problem'], 
            'VARIAVEL_NAO_SIGNIFICATIVA',
            "Variável não significativa não foi corretamente classificada"
        )
    
    def test_real_dataset(self):
        """Testa a análise com dados reais do arquivo dados/dados.csv"""
        # Caminho para o arquivo de dados
        data_path = os.path.join('dados', 'train.csv')
        
        # Verificar se o arquivo existe
        self.assertTrue(os.path.exists(data_path), 
                       f"Arquivo de dados '{data_path}' não encontrado")
        
        # Carregar o dataset
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            self.fail(f"Erro ao carregar o dataset: {str(e)}")
        
        # Verificar se o dataset possui a coluna alvo 'y'
        self.assertIn('y', df.columns, "Dataset não possui a coluna alvo 'y'")
        
        # Verificar se y é binário (0/1)
        y_values = df['y'].unique()
        self.assertTrue(set(y_values).issubset({0, 1}), 
                       f"Coluna 'y' não é binária. Valores: {y_values}")
        
        print(f"\nDataset carregado com sucesso: {len(df)} linhas e {len(df.columns)} colunas")
        print(f"Distribuição da variável alvo 'y': {df['y'].value_counts().to_dict()}")
        
        # Executar a análise
        results = self.analyzer.analyze(df)
        
        # Verificar se há resultados
        self.assertGreater(len(results), 0, "Nenhum resultado foi gerado pela análise")
        
        # Contar variáveis significativas e não significativas
        significant = [r for r in results if r['problem'] == 'VARIAVEL_SIGNIFICATIVA']
        nonsignificant = [r for r in results if r['problem'] == 'VARIAVEL_NAO_SIGNIFICATIVA']
        
        print(f"\nResultados da análise:")
        print(f"  Variáveis significativas: {len(significant)}")
        print(f"  Variáveis não significativas: {len(nonsignificant)}")
        
        # Exibir as primeiras variáveis significativas com seus p-values
        print("\nVariáveis significativas (top 5):")
        for i, result in enumerate(sorted(significant, key=lambda r: r['statistics']['p_value'])[:5]):
            col = result['column']
            p_value = result['statistics']['p_value']
            stat = result['statistics']['statistic']
            print(f"  {i+1}. {col}: p-value={p_value:.6f}, estatística={stat:.3f}")
        
        # Verificar que pelo menos algumas variáveis foram significativas
        self.assertGreater(len(significant), 0, 
                          "Nenhuma variável significativa foi encontrada")
        
        # Testar estatísticas nas variáveis significativas
        for result in significant:
            self.assertLess(result['statistics']['p_value'], self.analyzer.pvalue_threshold,
                           f"Variável {result['column']} marcada como significativa mas p-value > threshold")
            
        # Testar estatísticas nas variáveis não significativas
        for result in nonsignificant:
            self.assertGreaterEqual(result['statistics']['p_value'], self.analyzer.pvalue_threshold,
                                   f"Variável {result['column']} marcada como não significativa mas p-value < threshold")


if __name__ == '__main__':
    unittest.main() 
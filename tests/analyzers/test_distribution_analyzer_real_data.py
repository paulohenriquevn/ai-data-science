import unittest
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from src.analyzers.distribution_analyzer import DistributionAnalyzer, DistributionType, DistributionSolution


class TestDistributionAnalyzerRealData(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.analyzer = DistributionAnalyzer()
        # Caminho para o arquivo de dados
        self.data_path = os.path.join('dados', 'train.csv')
    
    def _find_result_for_column(self, results, column_name):
        """Utilitário para encontrar o resultado de uma coluna específica na lista de resultados"""
        for result in results:
            if result['column'] == column_name:
                return result
        return None
    
    def test_real_dataset_analysis(self):
        """Testa o analisador em um dataset real"""
        # Verifica se o arquivo existe
        self.assertTrue(os.path.exists(self.data_path), 
                       f"Arquivo de dados '{self.data_path}' não encontrado")
        
        # Carrega o dataset
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            self.fail(f"Erro ao carregar o dataset: {str(e)}")
        
        # Verifica se o dataset não está vazio
        self.assertGreater(len(df), 0, "O dataset está vazio")
        self.assertGreater(len(df.columns), 0, "O dataset não possui colunas")
        
        print(f"\nDataset carregado com sucesso: {len(df)} linhas e {len(df.columns)} colunas")
        
        # Informações básicas sobre o dataset
        print("\nInformações do dataset:")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        print(f"Colunas numéricas ({len(numeric_columns)}): {numeric_columns}")
        
        # Verifica se existem colunas numéricas para analisar
        self.assertGreater(len(numeric_columns), 0, "O dataset não possui colunas numéricas para análise")
        
        # Executa a análise
        results = self.analyzer.analyze(df)
        
        # Verifica se a análise produziu resultados
        self.assertGreater(len(results), 0, "A análise não produziu resultados")
        
        # Exibe resumo dos resultados
        print("\nResumo das distribuições detectadas:")
        distribution_counts = {}
        
        for result in results:
            dist_type = result['problem']
            if dist_type not in distribution_counts:
                distribution_counts[dist_type] = 0
            distribution_counts[dist_type] += 1
        
        for dist_type, count in distribution_counts.items():
            print(f"  {dist_type}: {count} colunas")
        
        # Exibe detalhes para cada tipo de distribuição
        print("\nDetalhes das distribuições por tipo:")
        for dist_type in distribution_counts.keys():
            print(f"\n{dist_type}:")
            for result in results:
                if result['problem'] == dist_type:
                    col = result['column']
                    skewness = result['statistics']['skewness']
                    kurtosis = result['statistics']['kurtosis']
                    actions = ', '.join(result['actions']) if result['actions'] else "Nenhuma"
                    
                    print(f"  {col}: skewness={skewness:.2f}, kurtosis={kurtosis:.2f}")
                    print(f"    Ação sugerida: {result['solution']}")
                    print(f"    Ações sugeridas: {actions}")
                    
                    # Limita a 3 exemplos por tipo
                    if sum(1 for r in results if r['problem'] == dist_type and r['column'] != col) >= 3:
                        remaining = sum(1 for r in results if r['problem'] == dist_type) - 3
                        if remaining > 0:
                            print(f"    ... e mais {remaining} colunas")
                        break
        
        # Teste básico para verificar consistência das ações recomendadas
        for result in results:
            dist_type = result['problem']
            actions = result['actions']
            
            # Verificações específicas baseadas no tipo de distribuição
            if dist_type == DistributionType.NORMAL.name:
                # Distribuições normais não devem ter transformações ou devem ter NENHUMA
                self.assertTrue(
                    len(actions) == 0 or DistributionSolution.NENHUMA.name in actions,
                    f"Distribuição normal ({result['column']}) não deveria ter ações transformativas"
                )
            
            elif dist_type == DistributionType.ASSIMETRICA_POSITIVA.name:
                # Assimetria positiva deve ter pelo menos uma transformação apropriada
                transformacoes = [
                    DistributionSolution.TRANSFORMACAO_LOG.name,
                    DistributionSolution.TRANSFORMACAO_RAIZ_QUADRADA.name,
                    DistributionSolution.TRANSFORMACAO_BOX_COX.name
                ]
                self.assertTrue(
                    any(t in actions for t in transformacoes),
                    f"Distribuição assimétrica positiva ({result['column']}) não tem transformações apropriadas"
                )


if __name__ == '__main__':
    unittest.main() 
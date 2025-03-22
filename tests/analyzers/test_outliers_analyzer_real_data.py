import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.analyzers.outliers_analyzer import OutlierAnalyzer, OutliersProblem, OutliersSolution


class TestOutlierAnalyzerRealData(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.analyzer = OutlierAnalyzer(threshold=0.05)  # 5% de outliers como limiar
        # Caminho para o arquivo de dados
        self.data_path = os.path.join('dados', 'train.csv')
    
    def _find_result_for_column(self, results, column_name):
        """Utilitário para encontrar o resultado de uma coluna específica na lista de resultados"""
        for result in results:
            if result['column'] == column_name:
                return result
        return None
    
    def test_real_dataset_analysis(self):
        """Testa o analisador de outliers em um dataset real"""
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
        print(f"Colunas numéricas ({len(numeric_columns)}): {numeric_columns[:10]}...")
        
        # Verifica se existem colunas numéricas para analisar
        self.assertGreater(len(numeric_columns), 0, "O dataset não possui colunas numéricas para análise")
        
        # Executa a análise
        results = self.analyzer.analyze(df)
        
        # Verifica se a análise produziu resultados
        print(f"\nResultados da análise: {len(results)} colunas com outliers significativos")
        
        # Se não houver outliers significativos, não falha o teste
        if len(results) == 0:
            print("Nenhuma coluna com proporção significativa de outliers foi encontrada")
            return
        
        # Exibe resumo dos resultados
        for i, result in enumerate(sorted(results, 
                                         key=lambda r: r['statistics']['outlier_ratio'], 
                                         reverse=True)):
            col = result['column']
            ratio = result['statistics']['outlier_ratio']
            num = result['statistics']['num_outliers']
            q1 = result['statistics']['q1']
            q3 = result['statistics']['q3']
            iqr = result['statistics']['iqr']
            
            print(f"\n{i+1}. Coluna: {col}")
            print(f"   Proporção de outliers: {ratio:.2%} ({num} valores)")
            print(f"   Q1: {q1:.3f}, Q3: {q3:.3f}, IQR: {iqr:.3f}")
            print(f"   Ações sugeridas: {', '.join(result['actions'])}")
            
            # Limita a 5 colunas no output
            if i >= 4:
                remaining = len(results) - 5
                if remaining > 0:
                    print(f"\n... e mais {remaining} colunas com outliers significativos")
                break
        
        # Gráfico para a coluna com mais outliers (opcional)
        if len(results) > 0:
            # Pega a coluna com maior proporção de outliers
            top_outlier_col = sorted(results, 
                                    key=lambda r: r['statistics']['outlier_ratio'], 
                                    reverse=True)[0]['column']
            
            # Verifica as estatísticas para diagnóstico
            result = self._find_result_for_column(results, top_outlier_col)
            q1 = result['statistics']['q1']
            q3 = result['statistics']['q3']
            iqr = result['statistics']['iqr']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            serie = df[top_outlier_col].dropna()
            outliers = serie[(serie < lower_bound) | (serie > upper_bound)]
            
            print(f"\nVerificação da coluna com mais outliers ({top_outlier_col}):")
            print(f"  Total de valores: {len(serie)}")
            print(f"  Valores outliers: {len(outliers)} ({len(outliers)/len(serie):.2%})")
            print(f"  Limites: [{lower_bound:.3f}, {upper_bound:.3f}]")
            
            # Verificações básicas
            self.assertEqual(
                len(outliers), 
                result['statistics']['num_outliers'],
                "Inconsistência na contagem de outliers"
            )
            
            self.assertAlmostEqual(
                len(outliers) / len(serie),
                result['statistics']['outlier_ratio'],
                places=4,
                msg="Inconsistência na proporção de outliers"
            )
            
            # Verifica se as ações sugeridas são coerentes
            self.assertIn('WINSORIZACAO', result['actions'], 
                         "Winsorização deve ser uma das ações sugeridas")
            
            self.assertIn('TRATAMENTO_OUTLIERS', result['solution'], 
                         "A solução deve envolver tratamento de outliers")


if __name__ == '__main__':
    unittest.main() 
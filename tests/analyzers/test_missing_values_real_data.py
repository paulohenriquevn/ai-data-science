import unittest
import os
import pandas as pd
import numpy as np
from src.analyzers.missing_values import MissingValuesAnalyzer, MissingValuesProblemType, MissingValuesSolution


class TestMissingValuesAnalyzerRealData(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.analyzer = MissingValuesAnalyzer()
        # Caminho para o arquivo de dados
        self.data_path = os.path.join('dados', 'train.csv')
        
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
        
        # Informações gerais sobre o dataset
        print("\nInformações do dataset:")
        print(f"Tipos de dados:\n{df.dtypes}")
        print(f"\nValores ausentes por coluna:")
        for col in df.columns:
            missing = df[col].isna().sum()
            missing_percent = (missing / len(df)) * 100
            print(f"  {col}: {missing} ({missing_percent:.2f}%)")
        
        # Executa a análise
        results = self.analyzer.analyze(df)
        
        # Verifica se a análise foi executada corretamente
        self.assertEqual(len(results), len(df.columns), 
                        "O número de resultados deve ser igual ao número de colunas")
        
        # Exibe os resultados da análise
        print("\nResultados da análise:")
        for result in results:
            col = result['column']
            problem = result['problem']
            suggestion = result['suggestion']
            print(f"  {col}: {problem} -> {suggestion}")
            
            # Verifica a consistência do resultado
            self.assertIn(problem, [p.name for p in MissingValuesProblemType], 
                         f"Tipo de problema inválido: {problem}")
            self.assertIn('statistics', result, 
                         "Resultado deve conter estatísticas sobre valores ausentes")
            
            # Se o problema é "NENHUM", não deveria haver valores ausentes
            if problem == MissingValuesProblemType.NENHUM.name:
                self.assertEqual(result['statistics']['missing_count'], 0, 
                               f"Coluna {col} marcada como sem ausentes, mas possui valores ausentes")
            
            # Se há valores ausentes, verificar se as ações sugeridas são adequadas
            if result['statistics']['missing_count'] > 0:
                self.assertGreater(len(result['actions']), 0, 
                                 f"Coluna {col} possui valores ausentes, mas não há ações sugeridas")
        
        # Análise mais aprofundada para colunas com problemas específicos
        print("\nAnálise detalhada por tipo de problema:")
        problem_counts = {}
        for result in results:
            problem = result['problem']
            if problem not in problem_counts:
                problem_counts[problem] = 0
            problem_counts[problem] += 1
        
        for problem, count in problem_counts.items():
            print(f"  {problem}: {count} colunas")
            
            # Para cada tipo de problema, exibir uma coluna exemplo
            for result in results:
                if result['problem'] == problem:
                    col = result['column']
                    missing_percent = result['statistics']['missing_percent']
                    print(f"    Exemplo: {col} ({missing_percent:.2f}% ausentes)")
                    break


if __name__ == '__main__':
    unittest.main() 
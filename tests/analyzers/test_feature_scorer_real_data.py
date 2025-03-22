import unittest
import os
import pandas as pd
import numpy as np
from src.analyzers.feature_scorer import FeatureScorer
from src.analyzers.missing_values import MissingValuesAnalyzer
from src.analyzers.distribution_analyzer import DistributionAnalyzer
from src.analyzers.outliers_analyzer import OutlierAnalyzer
from src.analyzers.statistical.statistical_significance_analyzer import StatisticalSignificanceAnalyzer
from src.analyzers.correlation_analyzer import CorrelationAnalyzer


class TestFeatureScorerRealData(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        self.data_path = os.path.join('dados', 'train.csv')
        
        # Instanciando os analisadores
        self.missing_analyzer = MissingValuesAnalyzer()
        self.distribution_analyzer = DistributionAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer(threshold=0.05)
        self.significance_analyzer = StatisticalSignificanceAnalyzer(target='y')
        self.correlation_analyzer = CorrelationAnalyzer(target='y')
        
        # Instanciando o pontuador de features
        self.feature_scorer = FeatureScorer()
    
    def test_real_dataset_analysis(self):
        """Testa o pontuador de features em um dataset real"""
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
        
        # Executando as análises individuais
        results_missing = self.missing_analyzer.analyze(df)
        results_distribution = self.distribution_analyzer.analyze(df)
        results_outliers = self.outlier_analyzer.analyze(df)
        results_significance = self.significance_analyzer.analyze(df)
        results_correlation = self.correlation_analyzer.analyze(df)
        
        # Combinando todos os resultados das análises
        all_analysis_results = [
            results_missing,
            results_distribution,
            results_outliers,
            results_significance,
            results_correlation
        ]
        
        # Exibindo um resumo das análises individuais:
        print("\nResumo das análises individuais:")
        print(f"  Análise de valores ausentes: {len(results_missing)} resultados")
        print(f"  Análise de distribuição: {len(results_distribution)} resultados")
        print(f"  Análise de outliers: {len(results_outliers)} resultados")
        print(f"  Análise de significância: {len(results_significance)} resultados")
        print(f"  Análise de correlação: {len(results_correlation)} resultados")
        
        # Executando o pontuador de features
        final_scores = self.feature_scorer.analyze(df, all_analysis_results)
        
        # Verificando se foram gerados resultados
        self.assertGreater(len(final_scores), 0, "Nenhum resultado de pontuação foi gerado")
        
        # Exibindo as variáveis com melhor pontuação
        print("\nVariáveis com melhor pontuação (Top 10):")
        for i, result in enumerate(sorted(final_scores, key=lambda x: x['score_total'], reverse=True)[:10]):
            col = result['column']
            score = result['score_total']
            selected = "Sim" if result['selecionar'] else "Não"
            justificativas = ", ".join(result['justificativas'])
            
            print(f"{i+1}. {col}: Pontuação={score}, Selecionada={selected}")
            print(f"   Justificativas: {justificativas}")
        
        # Contando variáveis selecionadas e rejeitadas
        selected_vars = [r for r in final_scores if r['selecionar']]
        rejected_vars = [r for r in final_scores if not r['selecionar']]
        
        print(f"\nResume de seleção:")
        print(f"  Variáveis selecionadas: {len(selected_vars)}")
        print(f"  Variáveis rejeitadas: {len(rejected_vars)}")
        
        # Verificando que temos pelo menos algumas variáveis selecionadas
        self.assertGreater(len(selected_vars), 0, "Nenhuma variável foi selecionada")
        
        # Verificando que variáveis selecionadas têm pontuação >= 3
        for var in selected_vars:
            self.assertGreaterEqual(var['score_total'], 3,
                                   f"Variável selecionada {var['column']} tem pontuação menor que 3: {var['score_total']}")
        
        # Verificando que variáveis rejeitadas têm pontuação < 3
        for var in rejected_vars:
            self.assertLess(var['score_total'], 3,
                           f"Variável rejeitada {var['column']} tem pontuação maior ou igual a 3: {var['score_total']}")
        
        # Verificando as variáveis mais bem pontuadas em detalhes
        if len(selected_vars) > 0:
            top_var = sorted(selected_vars, key=lambda x: x['score_total'], reverse=True)[0]
            print(f"\nAnálise detalhada da variável mais bem pontuada: {top_var['column']}")
            print(f"  Pontuação total: {top_var['score_total']}")
            
            # Verificando cada justificativa
            for justificativa in top_var['justificativas']:
                print(f"  - {justificativa}")
            
            # Verificando se a variável tem justificativas positivas suficientes
            positive_criteria = [
                'CORRELACAO_COM_TARGET',
                'CORRELACAO_FORTE_POSITIVA', 
                'CORRELACAO_FORTE_NEGATIVA',
                'VARIAVEL_SIGNIFICATIVA'
            ]
            
            has_positive_criteria = any(any(pc in j for pc in positive_criteria) for j in top_var['justificativas'])
            self.assertTrue(has_positive_criteria, 
                           f"A variável principal não tem nenhum critério positivo importante")


if __name__ == '__main__':
    unittest.main() 
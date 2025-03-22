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
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['a', 'b', 'c', 'd', 'e']
        })
        
        results = self.analyzer.analyze(df)
        
        self.assertEqual(len(results), 2)
        
        # Verifica se ambas as colunas são classificadas como sem problemas
        for result in results:
            self.assertEqual(result['problem'], MissingValuesProblemType.NENHUM.name)
            self.assertEqual(result['description'], MissingValuesProblemType.NENHUM.value)
            self.assertIn(MissingValuesSolution.NENHUMA_ACAO_NECESSARIA.name, result['actions'])
            self.assertEqual(result['statistics']['missing_count'], 0)
            self.assertEqual(result['statistics']['missing_percent'], 0)
            self.assertEqual(result['solution'], MissingValuesSolution.NENHUMA_ACAO_NECESSARIA)
            self.assertEqual(result['statistics']['missing_count'], 0)
    
    def test_few_missing_values(self):
        """Testa o caso de poucos valores ausentes (<5%)"""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, None] + [6] * 25,
            'categorical': ['a', 'b', 'c', 'd', 'e'] + ['f'] * 25
        })
        
        results = self.analyzer.analyze(df)
        
        numeric_result = next(r for r in results if r['column'] == 'numeric')
        self.assertEqual(numeric_result['problem'], MissingValuesProblemType.POUCO.name)
        self.assertIn(MissingValuesSolution.IMPUTE_MEDIA.name, numeric_result['actions'])
        
    def test_moderate_missing_values(self):
        """Testa o caso de quantidade moderada de valores ausentes (5-20%)"""
        # 2 em 20 = 10%
        df = pd.DataFrame({
            'numeric': [1, 2, 3, None, None] + [6] * 15,
        })
        
        results = self.analyzer.analyze(df)
        
        self.assertEqual(results[0]['problem'], MissingValuesProblemType.ESTRUTURAL.name)
        self.assertEqual(results[0]['statistics']['missing_percent'], 10)
        
    def test_many_missing_values(self):
        """Testa o caso de muitos valores ausentes (>20%)"""
        # 6 em 20 = 30%
        df = pd.DataFrame({
            'numeric': [1, 2, None, None, None, None, None, None] + [9] * 12,
        })
        
        results = self.analyzer.analyze(df)
        
        self.assertEqual(results[0]['problem'], MissingValuesProblemType.MUITO.name)
        self.assertEqual(results[0]['statistics']['missing_percent'], 30)
        self.assertEqual(results[0]['solution'], MissingValuesSolution.MICE_IMPUTER)
        
    def test_categorical_missing_values(self):
        """Testa como valores ausentes em variáveis categóricas são tratados"""
        # 4 em 20 = 20%
        df = pd.DataFrame({
            'categorical': ['a', 'b', None, None, None, None] + ['x'] * 14,
        })
        
        results = self.analyzer.analyze(df)
        
        self.assertEqual(results[0]['statistics']['missing_percent'], 20)
        self.assertEqual(results[0]['solution'], MissingValuesSolution.CATEGORIA_DESCONHECIDA)
        
    def test_target_variable_auto_detection(self):
        """Testa a detecção automática de variável alvo"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'variavel_target': [0, 1, None, 1, 0]  # Nome explícito como target
        })
        
        results = self.analyzer.analyze(df)
        
        target_result = next(r for r in results if r['column'] == 'variavel_target')
        self.assertEqual(target_result['problem'], MissingValuesProblemType.ALVO_MISSING.name)
        self.assertEqual(target_result['solution'], MissingValuesSolution.EXCLUIR_LINHAS)
        
    def test_target_variable_explicit(self):
        """Testa a especificação explícita de variável alvo"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, None, 5],
            'feature2': [10, 20, 30, 40, None], 
            'resultado': [0, 1, None, 1, 0]  # Não contém palavras-chave típicas
        })
        
        # Cria analisador com especificação explícita da variável alvo
        analyzer_with_target = MissingValuesAnalyzer(target_column='resultado')
        results = analyzer_with_target.analyze(df)
        
        resultado_result = next(r for r in results if r['column'] == 'resultado')
        self.assertEqual(resultado_result['problem'], MissingValuesProblemType.ALVO_MISSING.name)
        
        # O mesmo dataframe com analisador padrão não deve detectar 'resultado' como alvo
        standard_results = self.analyzer.analyze(df)
        resultado_standard = next(r for r in standard_results if r['column'] == 'resultado')
        self.assertNotEqual(resultado_standard['problem'], MissingValuesProblemType.ALVO_MISSING.name)
        
    def test_mar_pattern(self):
        """Testa detecção de Missing At Random (MAR)"""
        # Criamos dados onde ausência em y está correlacionada com valores de x
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        
        # Introduz ausência em y quando x > 0.5 (cerca de 30% dos dados)
        mask = x > 0.5
        y[mask] = np.nan
        
        # Nomeando as colunas para evitar detecção de target pelo nome
        df = pd.DataFrame({'valor_x': x, 'valor_y': y})
        
        results = self.analyzer.analyze(df)
        y_result = next(r for r in results if r['column'] == 'valor_y')
        
        # Com esse padrão, deveria identificar como relacionada a outras variáveis
        self.assertEqual(y_result['problem'], MissingValuesProblemType.RELACIONADA.name)
        self.assertEqual(y_result['solution'], MissingValuesSolution.MICE_IMPUTER)
        
    def test_mnar_pattern(self):
        """Testa detecção de Missing Not At Random (MNAR)"""
        # Criamos dados onde ausência está correlacionada com os próprios valores
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        
        # Introduz ausência em x para valores extremos (aproximadamente 10%)
        mask = (x > 1.5) | (x < -1.5)  # Aproximadamente valores outliers
        x_with_missing = x.copy()
        x_with_missing[mask] = np.nan
        
        df = pd.DataFrame({'valor_extremo': x_with_missing})
        
        results = self.analyzer.analyze(df)
        
        # Com esse padrão, deveria identificar como relacionada ao próprio valor
        self.assertEqual(results[0]['problem'], MissingValuesProblemType.RELACIONADA_AO_VALOR.name)
        self.assertEqual(results[0]['solution'], MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR)
        
    def test_systematic_pattern(self):
        """Testa detecção de padrões sistemáticos de ausência"""
        # Criamos padrão sistemático (ex: a cada 5 registros)
        data = []
        for i in range(100):
            if i % 5 == 0:
                data.append(np.nan)
            else:
                data.append(float(i))  # Convertendo para float para evitar problemas
                
        df = pd.DataFrame({'padrao_sistematico': data})  # Nome sem palavras-chave de alvo
        
        results = self.analyzer.analyze(df)
        
        # Com esse padrão periódico, deveria identificar como padrão sistemático
        self.assertEqual(results[0]['problem'], MissingValuesProblemType.PADRAO_SISTEMATICO.name)
        self.assertEqual(results[0]['solution'], MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR)
        
    def test_mixed_datatypes(self):
        """Testa mix de tipos de dados com diferentes padrões de ausência"""
        df = pd.DataFrame({
            'numeric_few': [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10% ausentes
            'numeric_many': [1.0, None, None, None, 5.0, 6.0, 7.0, 8.0, None, 10.0],  # 40% ausentes
            'categorical': ['a', 'b', None, 'd', 'e', None, 'g', 'h', 'i', 'j']  # 20% ausentes
        })
        
        results = self.analyzer.analyze(df)
        
        # Verifica se cada coluna recebeu o tratamento adequado
        num_few = next(r for r in results if r['column'] == 'numeric_few')
        num_many = next(r for r in results if r['column'] == 'numeric_many')
        cat = next(r for r in results if r['column'] == 'categorical')
        
        self.assertEqual(num_few['problem'], MissingValuesProblemType.ESTRUTURAL.name)
        self.assertEqual(num_many['problem'], MissingValuesProblemType.MUITO.name)
        self.assertEqual(cat['solution'], MissingValuesSolution.CATEGORIA_DESCONHECIDA)
        
    def test_edge_cases(self):
        """Testa casos extremos"""
        # 1. Coluna 100% ausente
        # 2. Exatamente 5% ausentes (fronteira)
        # 3. Exatamente 20% ausentes (fronteira)
        df = pd.DataFrame({
            'all_missing': [None] * 20,  # 100% ausentes
            'exact_5pct': [None] + [1.0] * 19,  # 5% ausentes
            'exact_20pct': [None] * 4 + [1.0] * 16  # 20% ausentes
        })
        
        results = self.analyzer.analyze(df)
        
        all_missing = next(r for r in results if r['column'] == 'all_missing')
        exact_5pct = next(r for r in results if r['column'] == 'exact_5pct')
        exact_20pct = next(r for r in results if r['column'] == 'exact_20pct')
        
        self.assertEqual(all_missing['problem'], MissingValuesProblemType.MUITO.name)
        self.assertEqual(all_missing['statistics']['missing_percent'], 100)
        
        self.assertEqual(exact_5pct['problem'], MissingValuesProblemType.POUCO.name)
        self.assertEqual(exact_5pct['statistics']['missing_percent'], 5)
        
        self.assertEqual(exact_20pct['problem'], MissingValuesProblemType.ESTRUTURAL.name)
        self.assertEqual(exact_20pct['statistics']['missing_percent'], 20)
        
    def test_realistic_dataset(self):
        """Testa com um conjunto de dados mais realista"""
        # Simulando uma base de dados de clientes
        np.random.seed(42)
        n = 1000
        
        # ID (sem valores ausentes)
        client_id = list(range(1, n+1))
        
        # Idade (poucos ausentes, aleatórios) - array de floats para permitir NaN
        age = np.random.randint(18, 80, n).astype(float)
        missing_indices = np.random.choice(n, int(n*0.03), replace=False)
        age[missing_indices] = np.nan
        
        # Renda (muitos ausentes, relacionados à idade - MAR)
        income = np.random.normal(3000, 1500, n)
        # Pessoas mais velhas tendem a não informar renda
        income[age > 60] = np.nan
        
        # Categoria de produto (alguns ausentes)
        categories = np.random.choice(['A', 'B', 'C', 'D'], n)
        missing_indices = np.random.choice(n, int(n*0.12), replace=False)
        categories = np.array(categories, dtype=object)
        categories[missing_indices] = None
        
        # Score de crédito (poucos extremos ausentes - MNAR)
        credit_score = np.random.normal(600, 100, n)
        # Scores extremos tendem a estar ausentes
        credit_score[(credit_score < 400) | (credit_score > 800)] = np.nan
        
        # Target: Churn (algumas ausências)
        churn = np.random.choice([0, 1], n).astype(float)
        missing_indices = np.random.choice(n, int(n*0.05), replace=False)
        churn[missing_indices] = np.nan
        
        df = pd.DataFrame({
            'client_id': client_id,
            'age': age,
            'income': income,
            'product_category': categories,
            'credit_score': credit_score,
            'churn_target': churn
        })
        
        analyzer_with_target = MissingValuesAnalyzer(target_column='churn_target')
        results = analyzer_with_target.analyze(df)
        
        # Verificações específicas para este conjunto de dados realista
        client_id_result = next(r for r in results if r['column'] == 'client_id')
        age_result = next(r for r in results if r['column'] == 'age')
        income_result = next(r for r in results if r['column'] == 'income')
        category_result = next(r for r in results if r['column'] == 'product_category')
        credit_score_result = next(r for r in results if r['column'] == 'credit_score')
        churn_result = next(r for r in results if r['column'] == 'churn_target')
        
        self.assertEqual(client_id_result['problem'], MissingValuesProblemType.NENHUM.name)
        self.assertEqual(age_result['problem'], MissingValuesProblemType.POUCO.name)
        self.assertIn(income_result['problem'], 
                     [MissingValuesProblemType.RELACIONADA.name, MissingValuesProblemType.MUITO.name])
        self.assertEqual(category_result['solution'], MissingValuesSolution.CATEGORIA_DESCONHECIDA)
        self.assertIn(credit_score_result['problem'], 
                     [MissingValuesProblemType.RELACIONADA_AO_VALOR.name, 
                      MissingValuesProblemType.POUCO.name])
        self.assertEqual(churn_result['problem'], MissingValuesProblemType.ALVO_MISSING.name)

if __name__ == '__main__':
    unittest.main() 
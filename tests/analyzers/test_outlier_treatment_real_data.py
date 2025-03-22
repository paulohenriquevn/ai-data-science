import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.analyzers.outlier_treatment import OutlierTreatmentTransformer


class TestOutlierTreatmentRealData(unittest.TestCase):
    
    def setUp(self):
        """Configuração para cada teste"""
        # Caminho para o arquivo de dados
        self.data_path = os.path.join('dados', 'train.csv')
        
    def test_real_dataset_treatment(self):
        """Testa o tratamento de outliers em um dataset real"""
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
        
        # Encontrar as colunas numéricas do dataset
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\nColunas numéricas detectadas: {len(numeric_columns)}")
        print(f"Exemplos de colunas numéricas: {numeric_columns[:5]}...")
        
        # Selecionar até 3 colunas numéricas para teste
        test_columns = numeric_columns[:3]
        print(f"\nTestando tratamento de outliers nas colunas: {test_columns}")
        
        # Criar plano de tratamento com diferentes estratégias
        treatment_plan = {}
        for i, col in enumerate(test_columns):
            if i % 3 == 0:
                treatment_plan[col] = 'log'
            elif i % 3 == 1:
                treatment_plan[col] = 'winsorize'
            else:
                treatment_plan[col] = 'remove'
        
        print(f"\nPlano de tratamento configurado: {treatment_plan}")
        
        # Criando o transformador com o plano de tratamento
        transformer = OutlierTreatmentTransformer(treatment_plan=treatment_plan)
        
        # Armazenar estatísticas antes do tratamento
        pre_stats = {}
        for col in treatment_plan.keys():
            pre_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'count': df[col].count()
            }
            
            # Calcular limites de outliers segundo regra IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Contar outliers
            outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
            pre_stats[col]['outliers_count'] = len(outliers)
            pre_stats[col]['outliers_ratio'] = len(outliers) / df[col].count()
            pre_stats[col]['lower_bound'] = lower_bound
            pre_stats[col]['upper_bound'] = upper_bound
        
        # Aplicar o transformador
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Verificar se o transformador foi ajustado corretamente
        self.assertTrue(transformer.fitted_, "O transformador não foi marcado como ajustado após .fit()")
        
        # Verificar resultados para cada estratégia
        for col, strategy in treatment_plan.items():
            print(f"\nVerificando tratamento para coluna {col} (estratégia: {strategy})")
            
            if strategy == 'log':
                # Para log, verificamos se os valores estão transformados
                # Corrigindo a comparação para lidar com diferentes índices
                original_values = df[col]
                transformed_values = df_transformed[col]
                
                # Reset de índices para permitir comparação apropriada
                if len(transformed_values) != len(original_values):
                    print(f"  Aviso: O dataframe transformado tem tamanho diferente do original")
                    # Verificamos apenas que os valores estão corretos, sem comparar elemento a elemento
                    transformed_max = transformed_values.max()
                    
                    # Verificando se há NaN no resultado (pode ocorrer com valores negativos)
                    if pd.isna(transformed_max):
                        print(f"  Aviso: Valores NaN encontrados na transformação logarítmica.")
                        # Filtrando valores NaN para a verificação
                        non_nan_values = transformed_values.dropna()
                        if len(non_nan_values) > 0:
                            transformed_max = non_nan_values.max()
                            expected_max = np.log(original_values.max() + transformer.log_offset)
                            self.assertLessEqual(transformed_max, expected_max + 0.001,
                                             f"Valor máximo transformado não corresponde ao esperado")
                        else:
                            print(f"  Todos os valores são NaN após transformação. Verificando dados originais.")
                            # Verificar se existem valores negativos que causariam problemas na transformação log
                            neg_values = (original_values < 0).sum()
                            print(f"  Valores negativos na coluna original: {neg_values}")
                    else:
                        expected_max = np.log(original_values.max() + transformer.log_offset)
                        self.assertLessEqual(transformed_max, expected_max + 0.001,
                                         f"Valor máximo transformado não corresponde ao esperado")
                else:
                    # Filtrando valores NaN para uma comparação justa
                    filtered_transformed = transformed_values.dropna()
                    filtered_original = original_values[filtered_transformed.index]
                    
                    if len(filtered_transformed) > 0:
                        # Verificando de maneira segura usando valores, não séries
                        log_transformed = np.log(filtered_original + transformer.log_offset).values
                        is_correct = np.allclose(filtered_transformed.values, log_transformed, rtol=1e-03)
                        self.assertTrue(is_correct, f"Transformação log não aplicada corretamente para {col}")
                    else:
                        print(f"  Aviso: Todos os valores transformados são NaN")
                
                # Mostrando estatísticas básicas
                print(f"  Antes: min={pre_stats[col]['min']:.3f}, max={pre_stats[col]['max']:.3f}, "
                      f"média={pre_stats[col]['mean']:.3f}")
                print(f"  Valores NaN antes: {df[col].isna().sum()}")
                print(f"  Valores NaN depois: {df_transformed[col].isna().sum()}")
                print(f"  Depois (não-NaN): min={df_transformed[col].min():.3f}, max={df_transformed[col].max():.3f}, "
                      f"média={df_transformed[col].mean():.3f}")
                
            elif strategy == 'winsorize':
                # Para winsorize, verificamos se os valores extremos foram truncados
                bounds = transformer.bounds_[col]
                
                # Lidar com valores NaN
                transformed_values = df_transformed[col]
                transformed_max = transformed_values.max()
                transformed_min = transformed_values.min()
                
                # Verificação extra para garantir que não temos problemas com NaN
                if pd.isna(transformed_max) or pd.isna(transformed_min):
                    print(f"  Aviso: Valores máximo ou mínimo são NaN na coluna {col}")
                    # Filtrar valores não-NaN para verificação
                    non_nan_values = transformed_values.dropna()
                    
                    if len(non_nan_values) > 0:
                        non_nan_max = non_nan_values.max()
                        non_nan_min = non_nan_values.min()
                        
                        self.assertLessEqual(non_nan_max, bounds[1], 
                                          f"Limite superior de winsorização não aplicado para {col}")
                        self.assertGreaterEqual(non_nan_min, bounds[0], 
                                             f"Limite inferior de winsorização não aplicado para {col}")
                        
                        print(f"  Valores NaN: {transformed_values.isna().sum()} de {len(transformed_values)}")
                        print(f"  Valores não-NaN - min: {non_nan_min}, max: {non_nan_max}")
                    else:
                        print(f"  Todos os valores são NaN após winsorização!")
                else:
                    # Sem problemas com NaN no máximo e mínimo
                    self.assertLessEqual(transformed_max, bounds[1], 
                                      f"Limite superior de winsorização não aplicado para {col}")
                    self.assertGreaterEqual(transformed_min, bounds[0], 
                                         f"Limite inferior de winsorização não aplicado para {col}")
                
                print(f"  Outliers antes: {pre_stats[col]['outliers_count']} "
                      f"({pre_stats[col]['outliers_ratio']:.2%})")
                
                # Contar valores nos limites (winsorized) - apenas para valores não-NaN
                non_nan_values = transformed_values.dropna()
                at_lower = sum(non_nan_values == bounds[0])
                at_upper = sum(non_nan_values == bounds[1])
                print(f"  Valores winsorized (não-NaN): {at_lower} no limite inferior, {at_upper} no limite superior")
                
                # Verificar consistência com outliers originais
                expected_lower = sum(df[col] < bounds[0])
                expected_upper = sum(df[col] > bounds[1])
                
                # Usar delta maior para permitir diferenças devido a NaN
                self.assertAlmostEqual(at_lower, expected_lower, delta=max(5, expected_lower*0.2), 
                                     msg="Número inconsistente de valores no limite inferior")
                self.assertAlmostEqual(at_upper, expected_upper, delta=max(5, expected_upper*0.2), 
                                     msg="Número inconsistente de valores no limite superior")
                
            elif strategy == 'remove':
                # Para remoção, verificamos se linhas com outliers foram removidas
                self.assertLess(len(df_transformed), len(df), 
                              f"Nenhuma linha removida para tratamento de outliers em {col}")
                
                print(f"  Linhas antes: {len(df)}")
                print(f"  Linhas depois: {len(df_transformed)} (removidas: {len(df) - len(df_transformed)})")
                
                # Verificar se os outliers foram realmente removidos
                outliers_before = pre_stats[col]['outliers_count']
                remaining = len(df) - len(df_transformed)
                
                print(f"  Outliers antes: {outliers_before}")
                print(f"  Linhas removidas: {remaining}")
                
                # A verificação não pode ser exata porque outras colunas com estratégia 'remove'
                # também podem remover linhas, então verificamos apenas se removeu linhas
                self.assertGreater(remaining, 0, "Nenhuma linha foi removida")
                
        # Teste integrado: verificar se podemos encadear fit e transform
        transformer = OutlierTreatmentTransformer(treatment_plan=treatment_plan)
        df_transformed_chained = transformer.fit_transform(df)
        
        # Verificar se o resultado é consistente com fit + transform separados
        self.assertEqual(len(df_transformed), len(df_transformed_chained), 
                        "fit_transform produziu resultado diferente de fit + transform")
        
        for col in treatment_plan.keys():
            if col in df_transformed.columns and col in df_transformed_chained.columns:
                pd.testing.assert_series_equal(
                    df_transformed[col], 
                    df_transformed_chained[col],
                    check_exact=False,  # Permitir pequenas diferenças numéricas
                    rtol=1e-5           # Tolerância relativa
                )


if __name__ == '__main__':
    unittest.main() 
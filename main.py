# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.analyzers.distribution_analyzer import DistributionAnalyzer
from src.analyzers.correlation.correlation_analyzer import CorrelationAnalyzer
from src.analyzers.outliers_analyzer import OutlierAnalyzer
from src.analyzers.feature_scorer import FeatureScorer
from src.analyzers.feature_engineering_plan_step import FeatureEngineeringPlanStep
from src.analyzers.feature_engineering_step import FeatureEngineeringStep

from src.analyzers.pca_plan_step import PCAPlanStep
from src.analyzers.pca_step import PCAStep
from src.analyzers.eda_pipeline import EDAPipeline
from src.utils import detect_and_replace_placeholders
import io
import sys
from datetime import datetime

# Statistical
from src.analyzers.statistical.statistical_significance_analyzer import StatisticalSignificanceAnalyzer

# Normalization
from src.analyzers.normalization.normalization_executor import NormalizationExecutor
from src.analyzers.normalization.normalization_plan import NormalizationPlan

# Balance
from src.analyzers.balance.balance_analyzer import BalanceAnalyzer
from src.analyzers.balance.balance_plan import BalancePlanStep
from src.analyzers.balance.balance_executor import BalanceExecutorStep

# Missing Values
from src.analyzers.missing_values.missing_values_analyzer import MissingValuesAnalyzer
from src.analyzers.missing_values.missing_values_plan import MissingValuesPlan
from src.analyzers.missing_values.missing_values_executor import MissingValuesExecutor

# Categorical
from src.analyzers.categorical.categorical_analyzer import CategoricalAnalyzer
from src.analyzers.categorical.categorical_plan import CategoricalPlan
from src.analyzers.categorical.categorical_executor import CategoricalExecutor


def print_section(title):
    """Imprime um cabeçalho de seção formatado"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def print_subsection(title):
    """Imprime um subcabeçalho formatado"""
    print("\n" + "-"*80)
    print(f" {title} ".center(80, "-"))
    print("-"*80)

def display_report(report, name="Relatório"):
    """Exibe um relatório de forma estruturada"""
    print_subsection(name)
    
    if isinstance(report, dict):
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"\n>> {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (list, tuple)) and len(subvalue) > 10:
                        print(f"  - {subkey}: {subvalue[:5]} ... (mais {len(subvalue)-5} itens)")
                    else:
                        print(f"  - {subkey}: {subvalue}")
            elif isinstance(value, (list, tuple)) and len(value) > 10:
                print(f">> {key}: {value[:5]} ... (mais {len(value)-5} itens)")
            else:
                print(f">> {key}: {value}")
    else:
        print(report)

def main():
    # Iniciar captura de saída para arquivo
    original_stdout = sys.stdout
    output_capture = io.StringIO()
    sys.stdout = output_capture
    
    print_section("ANÁLISE EXPLORATÓRIA DE DADOS")
    
    # 1. Carregamento e pré-processamento dos dados
    print_section("1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
    df = pd.read_csv('dados/train.csv')
    # Substituir valores faltantes codificados
    df.replace(-999, pd.NA, inplace=True)
    
    print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("\nPrimeiras 5 linhas do dataset:")
    print(df.head())
    
    print("\nInformações do dataset:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())
    
    print("\nEstatísticas descritivas:")
    df = detect_and_replace_placeholders(df)
    print(df.describe().to_string())
    
    # Verificar e identificar a coluna target
    target_col = 'y' if 'y' in df.columns else 'target' if 'target' in df.columns else 'Churn' if 'Churn' in df.columns else None
    if target_col:
        print(f"\nColuna target identificada: '{target_col}'")
    else:
        print("\nNenhuma coluna target padrão (y, target, Churn) encontrada")
    
    # 2. Executar a pipeline de análise exploratória
    print_section("2. PIPELINE DE ANÁLISE EXPLORATÓRIA DE DADOS")
    
    # Configurar a pipeline com todos os analisadores necessários
    eda_pipeline = EDAPipeline(
        missing_analyzer=MissingValuesAnalyzer(),
        distribution_analyzer=DistributionAnalyzer(),
        correlation_analyzer=CorrelationAnalyzer(target=target_col if target_col else 'y'),
        significance_analyzer=StatisticalSignificanceAnalyzer(),
        outlier_analyzer=OutlierAnalyzer(),
        feature_plan_generator=FeatureEngineeringPlanStep,
        feature_engineer=FeatureEngineeringStep,
        normalization_plan_generator=NormalizationPlan,
        normalizer=NormalizationExecutor,
        pca_plan_generator=PCAPlanStep,
        pca_executor=PCAStep,
        feature_scorer=FeatureScorer()
    )
    
    # Executar a pipeline completa
    print("Executando pipeline de análise completa...")
    eda_pipeline.run(df)
    
    # 3. Exibir resultados da análise
    
    # 3.1 Resultados dos análisadores individuais
    print_section("3. RESULTADOS DAS ANÁLISES")
    
    print_subsection("Análise de Valores Ausentes")
    display_report(eda_pipeline.reports['missing'], "Relatório de Valores Ausentes")
    
    print_subsection("Análise de Distribuição")
    display_report(eda_pipeline.reports['distribution'], "Relatório de Distribuição")
    
    print_subsection("Análise de Correlação")
    display_report(eda_pipeline.reports['correlation'], "Relatório de Correlação")
    
    print_subsection("Análise de Significância Estatística")
    display_report(eda_pipeline.reports['significance'], "Relatório de Significância Estatística")
    
    print_subsection("Análise de Outliers")
    display_report(eda_pipeline.reports['outliers'], "Relatório de Outliers")
    
    # 3.3 Resultados das transformações
    print_section("4. RESULTADOS DAS TRANSFORMAÇÕES")
    
    print_subsection("Dataset Transformado")
    if eda_pipeline.df_transformed is not None:
        print(f"Dimensões do dataset transformado: {eda_pipeline.df_transformed.shape}")
        print(eda_pipeline.df_transformed.head())
    else:
        print("Dataset transformado não disponível.")
    
    print_subsection("Dataset PCA")
    if eda_pipeline.df_pca is not None:
        print(f"Dimensões do dataset PCA: {eda_pipeline.df_pca.shape}")
        print(eda_pipeline.df_pca.head())
    else:
        print("PCA não foi aplicado ou não retornou resultados.")
    
    # 3.4 Pontuação de Features
    print_section("5. SCORING DE FEATURES")
    if 'feature_score' in eda_pipeline.reports and eda_pipeline.reports['feature_score']:
        feature_report = eda_pipeline.reports['feature_score']
        
        if isinstance(feature_report, list) and len(feature_report) > 0:
            print(f"Número de variáveis analisadas: {len(feature_report)}")
            
            # Exibir as top 10 features com maior pontuação
            sorted_features = sorted(feature_report, key=lambda x: x.get('score_total', 0) if isinstance(x, dict) else 0, reverse=True)
            for i, feature_info in enumerate(sorted_features[:10], 1):
                if isinstance(feature_info, dict):
                    col = feature_info.get('column', 'Desconhecido')
                    score = feature_info.get('score_total', 0)
                    selected = feature_info.get('selecionar', False)
                    justifications = feature_info.get('justificativas', [])
                    
                    print(f"\n{i}. {col} (Score: {score})")
                    print(f"   Recomendação: {'Selecionar' if selected else 'Não selecionar'}")
                    if justifications:
                        print(f"   Justificativas: {', '.join(justifications[:3])}...")
                else:
                    print(f"\n{i}. Formato de feature desconhecido: {feature_info}")
            
            # Tentar criar visualização de scores
            try:
                # Visualizar scores em um gráfico
                plt.figure(figsize=(12, 8))
                
                feature_data = []
                for f in sorted_features:
                    if isinstance(f, dict):
                        feature_data.append({
                            'Feature': f.get('column', 'Unknown'),
                            'Score': f.get('score_total', 0),
                            'Selected': f.get('selecionar', False)
                        })
                
                if feature_data:
                    feature_df = pd.DataFrame(feature_data)
                    
                    # Filtrar para mostrar apenas as 15 principais features
                    top_features = feature_df.head(15)
                    
                    # Criar gráfico de barras colorido por seleção
                    sns.barplot(x='Score', y='Feature', 
                                hue='Selected', data=top_features,
                                palette={True: 'green', False: 'red'})
                    
                    plt.title('Top 15 Features por Importância')
                    plt.legend(title='Selecionada')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png')
                    print("\nGráfico de importância de features salvo como 'feature_importance.png'")
                    
                    # Exibir lista das features selecionadas
                    selected_features = [f.get('column') for f in sorted_features if isinstance(f, dict) and f.get('selecionar', False)]
                    if selected_features:
                        print_subsection("Features Recomendadas para Seleção")
                        for i, feature in enumerate(selected_features, 1):
                            print(f"{i}. {feature}")
                    else:
                        print("\nNenhuma feature foi selecionada pelo algoritmo.")
            except Exception as e:
                print(f"\nErro ao criar visualização de features: {str(e)}")
        else:
            print("O formato do relatório de features não é o esperado.")
    else:
        print("Pontuação de features não disponível ou não foi executada.")
    
    # 4. Visualizações adicionais
    print_section("6. VISUALIZAÇÕES")
    
    # 4.1 Gráfico de valores ausentes
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Valores Ausentes no Dataset')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png')
    print("\nGráfico de valores ausentes salvo como 'missing_values_heatmap.png'")
    
    # 4.2 Matriz de correlação
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("\nMatriz de correlação salva como 'correlation_matrix.png'")
    
    print_section("ANÁLISE CONCLUÍDA")
    print("Todos os relatórios e visualizações foram gerados com sucesso.")
    print("Verifique os arquivos de imagem salvos no diretório atual para visualizações.")
    
    print_section("ANÁLISE DE VALORES AUSENTES")   
    print_subsection("Análise de Valores Ausentes")
    analyzer = MissingValuesAnalyzer()
    missing_report = analyzer.analyze(df)
    display_report(missing_report, "Relatório de Valores Ausentes")
    
    print_subsection("Plano de Valores Ausentes")
    plan = MissingValuesPlan(missing_report).generate()
    display_report(plan, "Plano de Valores Ausentes")
    
    print_subsection("Executar Plano de Valores Ausentes")
    df = MissingValuesExecutor(plan).execute(df)
    display_report(df, "Dataset com Valores Ausentes Executados")
    
    print_section("ANÁLISE DE VALORES AUSENTES CONCLUÍDA")   
    print_subsection("Análise de Valores Ausentes Final")
    analyzer_final = MissingValuesAnalyzer()
    missing_report_final = analyzer_final.analyze(df)
    display_report(missing_report_final, "Relatório de Valores Ausentes Final")
    
    print_section("ANÁLISE DE CATEGORIA")   
    print_subsection("Análise de Categoria")
    analyzer = CategoricalAnalyzer()
    categorical_report = analyzer.analyze(df)
    display_report(categorical_report, "Relatório de Categoria")    
    
    print_subsection("Plano de Categoria")
    plan = CategoricalPlan(categorical_report).generate()
    display_report(plan, "Plano de Categoria")
    
    print_subsection("Executar Plano de Categoria")
    df = CategoricalExecutor(plan).execute(df)
    display_report(df, "Dataset com Categoria Executado")
    
    print_section("ANÁLISE DE BALANCEAMENTO")   
    print_subsection("Análise de Balanceamento")
    analyzer = BalanceAnalyzer(target_column="y")
    balance_report = analyzer.analyze(df)
    display_report(balance_report, "Relatório de Balanceamento") 

    print_subsection("Plano de Balanceamento")
    balance_plan = BalancePlanStep(balance_report).generate()
    display_report(balance_plan, "Plano de Balanceamento")
    
    print_subsection("Executar Balanceamento")
    executor = BalanceExecutorStep(strategy=balance_plan["strategy"], target_column="y")
    balanced_df = executor.execute(df)
    display_report(balanced_df, "Dataset Balanceado")
    
    print_section("ANÁLISE DE BALANCEAMENTO CONCLUÍDA")   
    print_subsection("Análise de Balanceamento Final")
    analyzer_final = BalanceAnalyzer(target_column="y")
    balance_report_final = analyzer_final.analyze(balanced_df)
    display_report(balance_report_final, "Relatório de Balanceamento Final") 
    
    # Salvar datasets processados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar dataset transformado
    if eda_pipeline.df_transformed is not None:
        transformed_filename = f"dataset_transformado_{timestamp}.csv"
        eda_pipeline.df_transformed.to_csv(transformed_filename, index=False)
        print(f"\nDataset transformado salvo como: {transformed_filename}")
    
    # Criar arquivo de saída com timestamp
    output_file = f"analise_dataset_{timestamp}.txt"
    
    # Restaurar stdout e salvar a saída capturada no arquivo
    sys.stdout = original_stdout
    
    # Salvar conteúdo capturado no arquivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_capture.getvalue())
    
    print(f"Análise concluída com sucesso! Relatório salvo em: {output_file}")

if __name__ == "__main__":
    main()

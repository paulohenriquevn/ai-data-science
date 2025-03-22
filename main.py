# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.analyzers.missing_values import MissingValuesAnalyzer
from src.analyzers.distribution_analyzer import DistributionAnalyzer
from src.analyzers.correlation_analyzer import CorrelationAnalyzer
from src.analyzers.statistical_significance_analyzer import StatisticalSignificanceAnalyzer
from src.analyzers.outliers_analyzer import OutlierAnalyzer
from src.analyzers.outlier_treatment import OutlierTreatmentTransformer
from src.analyzers.feature_scorer import FeatureScorer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import sys
from datetime import datetime
from src.utils import detect_and_replace_placeholders
from src.analyzers.feature_engineering_plan_step import FeatureEngineeringPlanStep
from src.analyzers.feature_engineering_step import FeatureEngineeringStep
from src.analyzers.normalization_step import NormalizationStep
from src.analyzers.normalization_plan_step import NormalizationPlanStep
from src.analyzers.pca_plan_step import PCAPlanStep
from src.analyzers.pca_step import PCAStep
from src.analyzers.eda_pipeline import EDAPipeline

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
    
    # 1. Carregamento e limpeza inicial dos dados
    print_section("1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
    df = pd.read_csv('dados/train.csv')  # Alterado para o arquivo train.csv
    # Supondo que valores ausentes estejam representados por -999
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
        
    
    # Convertendo colunas booleanas para string para evitar erros de processamento
    bool_cols = df.select_dtypes(include=['bool']).columns
    df_for_analysis = df.copy()
    for col in bool_cols:
        df_for_analysis[col] = df_for_analysis[col].astype(str)
        
    # 0. Análise exploratória de dados
    print_section("0. PIPELINE DE ANÁLISE EXPLORATÓRIA DE DADOS")
    eda_pipeline = EDAPipeline(
        missing_analyzer=MissingValuesAnalyzer(),
        distribution_analyzer=DistributionAnalyzer(),
        correlation_analyzer=CorrelationAnalyzer(),
        significance_analyzer=StatisticalSignificanceAnalyzer(),
        outlier_analyzer=OutlierAnalyzer(),
        feature_plan_generator=FeatureEngineeringPlanStep,
        feature_engineer=FeatureEngineeringStep,
        normalization_plan_generator=NormalizationPlanStep,
        normalizer=NormalizationStep,
        pca_plan_generator=PCAPlanStep,
        pca_executor=PCAStep,
        feature_scorer=FeatureScorer()
    )
    eda_pipeline.run(df)
    df_final = eda_pipeline.df_transformed
    df_pca = eda_pipeline.df_pca
    print_subsection("Dataset final")
    print(df_final.head())
    print_subsection("Dataset PCA")
    print(df_pca.head())
    
    
    
    
    # Lista para armazenar os resultados de todas as análises
    todas_analises = []
    
    # 2. Análise e tratamento de valores ausentes
    print_section("2. ANÁLISE DE VALORES AUSENTES")
    mv_analyzer = MissingValuesAnalyzer()
    mv_report = mv_analyzer.analyze(df_for_analysis)
    todas_analises.append(mv_report)
    display_report(mv_report, "Relatório de Valores Ausentes")
    
    # Exibir estatísticas de valores ausentes por coluna
    print_subsection("Porcentagem de Valores Ausentes por Coluna")
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_pct)
    
    # Visualizar gráfico de valores ausentes
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Valores Ausentes no Dataset')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png')
    print("\nGráfico de valores ausentes salvo como 'missing_values_heatmap.png'")
    
    # 3. Análise de distribuição
    print_section("3. ANÁLISE DE DISTRIBUIÇÃO")
    dist_analyzer = DistributionAnalyzer()
    dist_report = dist_analyzer.analyze(df_for_analysis)
    todas_analises.append(dist_report)
    display_report(dist_report, "Relatório de Distribuição das Variáveis")
    
    # Plotar distribuições de algumas colunas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns[:5]  # Primeiras 5 colunas numéricas
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribuição: {col}')
    plt.tight_layout()
    plt.savefig('distributions.png')
    print("\nGráficos de distribuição salvos como 'distributions.png'")
    
    # 4. Análise de correlação
    print_section("4. ANÁLISE DE CORRELAÇÃO")
    # Usar a coluna target identificada anteriormente ou 'y' como padrão
    corr_analyzer = CorrelationAnalyzer(target=target_col if target_col else 'y')
    corr_report = corr_analyzer.analyze(df_for_analysis)
    todas_analises.append(corr_report)
    display_report(corr_report, "Relatório de Correlação")
    
    # Visualizar matriz de correlação
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
    
    # 5. Análise de significância estatística
    print_section("5. ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA")
    stat_analyzer = StatisticalSignificanceAnalyzer()
    stat_report = stat_analyzer.analyze(df_for_analysis)
    todas_analises.append(stat_report)
    display_report(stat_report, "Relatório de Significância Estatística")
    
    # 6. Análise e tratamento de outliers
    print_section("6. ANÁLISE E TRATAMENTO DE OUTLIERS")
    outlier_analyzer = OutlierAnalyzer()
    outlier_report = outlier_analyzer.analyze(df_for_analysis)
    todas_analises.append(outlier_report)
    display_report(outlier_report, "Relatório de Outliers")
    
    # Visualizar boxplots para detectar outliers
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[col].dropna())
        plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig('outliers_boxplot.png')
    print("\nBoxplots para outliers salvos como 'outliers_boxplot.png'")
    
    # -------------------------------------------------------------------------------
    # TRATAMENTO AUTOMÁTICO DE OUTLIERS APÓS ANÁLISE
    # -------------------------------------------------------------------------------

    print_subsection("Tratamento Automático de Outliers com base na análise")

    # Gerar plano de tratamento com base no relatório
    treatment_plan = {}
    for item in outlier_report:
        col = item.get("column")
        stats = item.get("statistics", {})
        skew = stats.get("skewness", 0)
        ratio = stats.get("outlier_ratio", 0)

        if skew is not None and skew > 1.5:
            treatment_plan[col] = "log"
        elif ratio >= 0.1:
            treatment_plan[col] = "winsorize"
        else:
            treatment_plan[col] = "remove"

    # Aplicar tratamento com transformer
    outlier_transformer = OutlierTreatmentTransformer(treatment_plan=treatment_plan)
    outlier_transformer.fit(df_for_analysis)
    df_for_analysis = outlier_transformer.transform(df_for_analysis)

    print(f"{len(treatment_plan)} variáveis tratadas com estratégias específicas.")
    print(f"Plano aplicado: {treatment_plan}")
    
    # 6. Análise e tratamento de outliers após tratamento
    print_section("6. ANÁLISE E TRATAMENTO DE OUTLIERS APÓS TRATAMENTO")
    outlier_analyzer = OutlierAnalyzer()
    outlier_report = outlier_analyzer.analyze(df_for_analysis)
    todas_analises.append(outlier_report)
    display_report(outlier_report, "Relatório de Outliers após tratamento")
    
    # Visualizar boxplots para detectar outliers
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df_for_analysis[col].dropna())
        plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig('outliers_boxplot_tratado.png')
    print("\nBoxplots para outliers tratados salvos como 'outliers_boxplot_tratado.png'")
    
    # 6. Plano para engenharia de features
    print_section("6. PLANO PARA ENGENHARIA DE FEATURES")
    plan_generator = FeatureEngineeringPlanStep(
        distribution_report=dist_report,
        outlier_report=outlier_report,
        significance_report=stat_report,
        correlation_report=corr_report
    )
    feature_engineering_plan = plan_generator.generate_plan()
    display_report(feature_engineering_plan, "Plano de Engenharia de Features")

    # 7. Aplicar o plano de engenharia de features aos dados
    print_section("7. ENGENHARIA DE FEATURES")
    step = FeatureEngineeringStep(plan=feature_engineering_plan)
    df_enriched = step.transform(df_for_analysis)
    print_subsection("Dataset com engenharia de features")
    print(df_enriched.head())
    
    # 8. Plano para normalização
    print_section("8. PLANO PARA NORMALIZAÇÃO")
    normalization_plan = NormalizationPlanStep(distribution_report=dist_report, outlier_report=outlier_report)
    normalization_plan = normalization_plan.generate_plan()
    display_report(normalization_plan, "Plano de Normalização")

    # 9. Normalizar os dados
    print_section("9. NORMALIZAÇÃO DOS DADOS")
    normalizer = NormalizationStep(normalization_plan=normalization_plan)
    normalizer.fit(df_enriched)
    df_normalized = normalizer.transform(df_enriched)
    print_subsection("Dataset com normalização")
    print(df_normalized.head()) 
    
    # 10. Plano para PCA
    print_section("10. PLANO PARA PCA")
    pca_plan = PCAPlanStep(df=df_normalized, significance_report=stat_report, correlation_report=corr_report)
    pca_plan = pca_plan.generate_plan()
    display_report(pca_plan, "Plano de PCA")
    
    # 11. PCA
    print_section("11. PCA")
    pca_step = PCAStep(pca_plan=pca_plan)
    df_pca = pca_step.fit_transform(df_normalized)
    print_subsection("Dataset com PCA")
    print(df_pca.head())

    # 11. Feature scoring e seleção de características
    print_section("11. SCORING E SELEÇÃO DE FEATURES")
    if target_col:
        print(f"Utilizando '{target_col}' como coluna alvo para análise de features")
        
        # Instanciar o FeatureScorer
        scorer = FeatureScorer()
        
        # Passo todas as análises anteriores para o FeatureScorer
        print(f"Consolidando resultados de {len(todas_analises)} análises anteriores")
        feature_report = scorer.analyze(df_for_analysis, todas_analises)
        
        print_subsection("Resultados da Pontuação de Features")
        print(f"Número de variáveis analisadas: {len(feature_report)}")
        
        # Exibir as top 10 features com maior pontuação
        sorted_features = sorted(feature_report, key=lambda x: x.get('score_total', 0), reverse=True)
        for i, feature_info in enumerate(sorted_features[:10], 1):
            col = feature_info.get('column', 'Desconhecido')
            score = feature_info.get('score_total', 0)
            selected = feature_info.get('selecionar', False)
            justifications = feature_info.get('justificativas', [])
            
            print(f"\n{i}. {col} (Score: {score})")
            print(f"   Recomendação: {'Selecionar' if selected else 'Não selecionar'}")
            print(f"   Justificativas: {', '.join(justifications[:3])}...")
        
        # Visualizar scores em um gráfico
        if sorted_features:
            plt.figure(figsize=(12, 8))
            feature_df = pd.DataFrame([
                {
                    'Feature': f.get('column', 'Unknown'),
                    'Score': f.get('score_total', 0),
                    'Selected': f.get('selecionar', False)
                }
                for f in sorted_features
            ])
            
            # Filtrar para mostrar apenas as 15 principais features
            top_features = feature_df.head(15)
            
            # Criar gráfico de barras colorido por seleção
            bars = sns.barplot(x='Score', y='Feature', 
                               hue='Selected', data=top_features,
                               palette={True: 'green', False: 'red'})
            
            plt.title('Top 15 Features por Importância')
            plt.legend(title='Selecionada')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("\nGráfico de importância de features salvo como 'feature_importance.png'")
            
            # Exibir lista das features selecionadas
            selected_features = [f.get('column') for f in sorted_features if f.get('selecionar', False)]
            print_subsection("Features Recomendadas para Seleção")
            for i, feature in enumerate(selected_features, 1):
                print(f"{i}. {feature}")
    else:
        print("\nNenhuma coluna target encontrada para avaliação de features.")
    
    # 11. PCA para análise de componentes principais (opcional)
    print_section("11. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    numeric_df = df.select_dtypes(include=np.number).dropna()
    
    if len(numeric_df) > 0:
        # Normalizar dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Aplicar PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Mostrar variância explicada
        print_subsection("Variância Explicada pelos Componentes")
        var_ratio = pca.explained_variance_ratio_
        cum_var_ratio = np.cumsum(var_ratio)
        
        for i, (var, cum_var) in enumerate(zip(var_ratio, cum_var_ratio)):
            print(f"PC{i+1}: {var:.4f} ({cum_var:.4f} acumulado)")
            if cum_var > 0.9 and i > 2:
                print(f">> {i+1} componentes explicam mais de 90% da variância")
                break
        
        # Gráfico da variância explicada
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(var_ratio) + 1), var_ratio, alpha=0.5, label='Variância Individual')
        plt.step(range(1, len(cum_var_ratio) + 1), cum_var_ratio, where='mid', label='Variância Acumulada')
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% da Variância')
        plt.xlabel('Componentes Principais')
        plt.ylabel('Proporção da Variância Explicada')
        plt.legend(loc='best')
        plt.title('PCA - Análise de Componentes Principais')
        plt.tight_layout()
        plt.savefig('pca_variance.png')
        print("\nGráfico de PCA salvo como 'pca_variance.png'")
    else:
        print("Dados numéricos insuficientes para PCA.")
    
    print_section("ANÁLISE CONCLUÍDA")
    print("Todos os relatórios e visualizações foram gerados com sucesso.")
    print("Verifique os arquivos de imagem salvos no diretório atual para visualizações.")
    
    # Criar arquivo de saída com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"analise_dataset_{timestamp}.txt"
    
    # Restaurar stdout e salvar a saída capturada no arquivo
    sys.stdout = original_stdout
    
    # Salvar conteúdo capturado no arquivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_capture.getvalue())
    
    print(f"Análise concluída com sucesso! Relatório salvo em: {output_file}")

if __name__ == "__main__":
    main()

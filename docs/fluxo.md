#### 1. Detecção de outliers (`OutlierAnalyzer`) ✔️  
- Aplicação do método IQR para cada variável  
- Cálculo da proporção de outliers  
- Recomendações: winsorização, transformação, remoção
- Estratégias baseadas em skewness e outlier ratio


#### 2. Padronização das variáveis numéricas ⏳  
- Gerar plano com `NormalizationPlanStep`  
- Aplicar `StandardScaler` ou `RobustScaler` com `NormalizationStep`

#### 3. Engenharia de features ⏳  
- Aplicar transformações (log, sqrt, square)  
- Criar variáveis interativas e flags  
- Gerar plano com `FeatureEngineeringPlanStep` e aplicar com `FeatureEngineeringStep`

#### 7. Testes estatísticos (`StatisticalSignificanceAnalyzer`) ✔️  
- Aplicação de `t-test` ou `Mann-Whitney` entre `y=0` e `y=1`  
- Geração de p-values  
- Classificação como significativas ou não

#### 8. Aplicação de PCA para redução de dimensionalidade ⏳  
- Gerar plano com `PCAPlanStep`  
- Executar redução com `PCAExecutorStep`  
- Salvar curva de variância explicada

#### 9. Seleção final de variáveis com base no PCA + FeatureScorer ⏳  
- Reavaliar variáveis após transformação  
- Consolidar `df_pca` com scores

#### 10. Consolidação dos insights (`FeatureScorer`) ✔️  
- Agregação dos resultados anteriores  
- Atribuição de score por variável  
- Definição de variáveis recomendadas

#### 11. Construção de modelo baseline 🧠  
- Aplicar modelos simples (LogisticRegression, XGBoost, etc.)  
- Avaliar métricas iniciais (ROC, AUC, precisão)

---

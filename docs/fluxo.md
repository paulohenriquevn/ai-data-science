
### ✅ **Checklist de Atividades Concluídas**

#### 1. Carregamento e pré-processamento inicial do dataset ✔️  
- Leitura do arquivo e identificação do target  
- Conversão de tipos e limpeza inicial

#### 2. Identificação e categorização de dados ausentes (`MissingValuesAnalyzer`) ✔️  
- Cálculo de proporções ausentes por coluna  
- Classificação (poucos, muitos, MNAR/MAR)  
- Sugestões de tratamento (imputação, flags, exclusão)

#### 3. Detecção de tipos de distribuição (`DistributionAnalyzer`) ✔️  
- Análise da distribuição de cada variável numérica  
- Classificação: normal, assimétrica, multimodal, etc.  
- Visualização com histogramas

#### 4. Cálculo de skewness e kurtosis ✔️  
- Cálculo e interpretação dos valores  
- Indicação de transformações adequadas (log, sqrt, square)

#### 5. Análise de correlação (`CorrelationAnalyzer`) ✔️  
- Cálculo de correlação com a variável alvo e entre variáveis  
- Identificação de colinearidade

#### 6. Classificação da correlação por intensidade e sentido ✔️  
- Agrupamento por: fraca, moderada, forte  
- Direção: positiva, negativa  
- Recomendações para engenharia de features ou remoção

#### 7. Testes estatísticos (`StatisticalSignificanceAnalyzer`) ✔️  
- Aplicação de `t-test` ou `Mann-Whitney` entre `y=0` e `y=1`  
- Geração de p-values  
- Classificação como significativas ou não

#### 8. Detecção de outliers (`OutlierAnalyzer`) ✔️  
- Aplicação do método IQR para cada variável  
- Cálculo da proporção de outliers  
- Recomendações: winsorização, transformação, remoção

#### 9. Consolidação dos insights (`FeatureScorer`) ✔️  
- Agregação dos resultados anteriores  
- Atribuição de score por variável  
- Definição de variáveis recomendadas

#### 10. Definição do plano de continuidade ✔️  
- Montagem do plano de tratamento de outliers, feature engineering e PCA  
- Encapsulamento da lógica em classes modulares

#### 11. Remoção/Tratamento de outliers com base na análise anterior ⏳  
- Aplicar `OutlierTreatmentTransformer`  
- Estratégias baseadas em skewness e outlier ratio

#### 12. Engenharia de features ⏳  
- Aplicar transformações (log, sqrt, square)  
- Criar variáveis interativas e flags  
- Gerar plano com `FeatureEngineeringPlanStep` e aplicar com `FeatureEngineeringStep`

#### 13. Padronização das variáveis numéricas ⏳  
- Gerar plano com `NormalizationPlanStep`  
- Aplicar `StandardScaler` ou `RobustScaler` com `NormalizationStep`

#### 14. Aplicação de PCA para redução de dimensionalidade ⏳  
- Gerar plano com `PCAPlanStep`  
- Executar redução com `PCAExecutorStep`  
- Salvar curva de variância explicada

#### 15. Seleção final de variáveis com base no PCA + FeatureScorer ⏳  
- Reavaliar variáveis após transformação  
- Consolidar `df_pca` com scores

#### 16. Construção de modelo baseline 🧠  
- Aplicar modelos simples (LogisticRegression, XGBoost, etc.)  
- Avaliar métricas iniciais (ROC, AUC, precisão)

---

Se quiser, posso gerar esse checklist em Markdown pronto para documentação oficial ou até mesmo exportar como `.md` ou `.pdf`. Deseja isso?
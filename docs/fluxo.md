

#### 8. Seleção final de variáveis com base no PCA + FeatureScorer ⏳  
- Reavaliar variáveis após transformação  
- Consolidar `df_pca` com scores

#### 9. Testes estatísticos (`StatisticalSignificanceAnalyzer`) ✔️  
- Aplicação de `t-test` ou `Mann-Whitney` entre `y=0` e `y=1`  
- Geração de p-values  
- Classificação como significativas ou não

#### 10. Consolidação dos insights (`FeatureScorer`) ✔️  
- Agregação dos resultados anteriores  
- Atribuição de score por variável  
- Definição de variáveis recomendadas

#### 11. Construção de modelo baseline 🧠  
- Aplicar modelos simples (LogisticRegression, XGBoost, etc.)  
- Avaliar métricas iniciais (ROC, AUC, precisão)

---

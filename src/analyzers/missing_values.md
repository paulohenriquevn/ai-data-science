# Técnicas de Tratamento de Dados Ausentes por Cenário

| Cenário/Problema | Mecanismo | Técnicas Recomendadas | Técnicas a Evitar | Observações |
|------------------|-----------|------------------------|-------------------|-------------|
| **Poucos dados ausentes** (<5%) | MCAR | • Exclusão de linhas Imputação por média/mediana/moda KNN | • Técnicas complexas (overkill) | O impacto na análise é geralmente pequeno |
| **Muitos dados ausentes** (>20%) | Qualquer | • Exclusão de colunas (se a variável não for crítica) Imputação múltipla Modelos que toleram valores ausentes (ex: XGBoost) | • Exclusão de linhas Imputação simples | Considerar se a coluna deve ser mantida |
| **Dados ausentes em séries temporais** | MAR/MNAR | • Interpolação Forward/backward fill Modelos ARIMA com imputação | • Exclusão de linhas Imputação por média geral | Considerar a autocorrelação temporal |
| **Dados categóricos ausentes** | Qualquer | • Categoria "Desconhecido" Imputação por moda Modelos baseados em árvores | • Técnicas para dados numéricos | Considerar o significado semântico da ausência |
| **Ausência estrutural** (MCAR) | MCAR | • Exclusão de linhas Qualquer método de imputação simples | • Métodos complexos (desnecessários) | A aleatoriedade permite abordagens mais simples |
| **Ausência relacionada a outras variáveis** (MAR) | MAR | • Imputação por regressão KNN Imputação múltipla Indicadores binários + imputação | • Exclusão de linhas Imputação simples sem indicadores | Importante modelar a relação com outras variáveis |
| **Ausência relacionada ao próprio valor** (MNAR) | MNAR | • Modelos de seleção Análise de sensibilidade Indicadores binários + imputação Imputação múltipla com variáveis auxiliares | • Exclusão de linhas Imputação simples | O caso mais desafiador; pode exigir coleta adicional de dados |
| **Valores ausentes em variáveis preditoras** | Qualquer | • Imputação prévia à modelagem Algoritmos tolerantes a valores ausentes | • Exclusão se muitas linhas afetadas | Depende do algoritmo de modelagem usado |
| **Valores ausentes na variável alvo** | Qualquer | • Exclusão dessas linhas (para modelos supervisionados) Tratar como problema semi-supervisionado | • Imputação na variável alvo | A imputação da variável alvo pode introduzir viés substancial |
| **Padrões sistemáticos de ausência** | MAR/MNAR | • Análise do padrão de ausência Modelagem explícita do processo de ausência Indicadores binários | • Ignorar o padrão Exclusão simples | O padrão de ausência pode conter informação valiosa |
| **Outliers marcados como ausentes** | MNAR | • Detecção de outliers antes da imputação Métodos robustos (mediana, Huber) | • Média simples Regressão sem regularização | Tratar outliers separadamente da imputação |
| **Variáveis altamente correlacionadas com dados ausentes** | MAR | • Imputação por regressão usando correlações Métodos baseados em similaridade (KNN) | • Imputação independente por coluna | Aproveitar a estrutura de correlação dos dados |
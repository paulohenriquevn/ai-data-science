```mermaid
graph TD
    A[Início] --> B{{Existem valores ausentes?}}
    B -->|Sim| C[Calcular % de missing]
    C --> D{{< 5%?}}
    D -->|Sim| E[Imputar média/mediana/moda]
    D -->|Não| F{{5% a 30%?}}
    F -->|Sim| G[Imputação por KNN ou modelos preditivos]
    F -->|Não| H[Remover feature ou usar MICE]
    B -->|Não| I{{Existem outliers?}}
    I -->|Sim| J[Usar Z-Score/IQR]
    J --> K{{Outliers extremos?}}
    K -->|Sim| L[Winsorization ou remoção]
    K -->|Não| M[Robust Scaling ou transformações]
    I -->|Não| N{{Distribuição normal?}}
    N -->|Não| O[Aplicar transformações (log, Box-Cox)]
    N -->|Sim| P{{Correlações > 0.9?}}
    P -->|Sim| Q[Remover feature ou PCA]
    P -->|Não| R{{Classes desbalanceadas?}}
    R -->|Sim| S[SMOTE, Undersampling ou pesos]
    R -->|Não| T{{Variáveis categóricas?}}
    T -->|Sim| U[One-Hot ou Target Encoding]
    T -->|Não| V[Pré-Processamento]

    V --> W{{Dados corrompidos/duplicados?}}
    W -->|Sim| X[Corrigir tipos/remover duplicatas]
    W -->|Não| Y{{Dados de texto?}}
    Y -->|Sim| Z[Limpeza, Stemming, Embeddings]
    Y -->|Não| AA[Escalonar dados]
    AA --> BB{{Variáveis categóricas?}}
    BB -->|Sim| CC[One-Hot/Ordinal Encoding]
    BB -->|Não| DD[Engenharia de Features]

    DD --> EE{{Features temporais?}}
    EE -->|Sim| FF[Criar dia/mês/média móvel]
    EE -->|Não| GG{{Interações entre features?}}
    GG -->|Sim| HH[Features polinomiais]
    GG -->|Não| II{{Features contínuas?}}
    II -->|Sim| JJ[Binning/Discretização]
    II -->|Não| KK{{Alta dimensionalidade?}}
    KK -->|Sim| LL[PCA, LDA ou UMAP]
    KK -->|Não| MM[Modelagem]
```
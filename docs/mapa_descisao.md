# **Mapa de Decisão Completo para EDA, Pré-Processamento e Engenharia de Features**

Este documento organiza as técnicas das etapas de **Análise Exploratória de Dados (EDA)**, **Pré-Processamento** e **Engenharia de Features**, seguindo um fluxo de decisão estruturado.

---

## **1️⃣ Análise Exploratória de Dados (EDA)**

### **1.1. Valores Faltantes**
**Pergunta:** Existem valores ausentes nos dados?
- **Sim** → Calcular a porcentagem de valores ausentes por variável:
  - **< 5% de ausência** → Usar imputação simples (média/mediana/moda) ou prosseguir para modelagem.
  - **5% a 30% de ausência** → Analisar padrões de ausência com matrizes de missingness ou gráficos de calor.
  - **> 30% de ausência** → Avaliar remoção da variável ou uso de técnicas avançadas (MICE).
- **Não** → Prosseguir para a próxima etapa.

---

### **1.2. Análise de Outliers**
**Pergunta:** Existem valores extremos em variáveis numéricas?
- **Sim** → Escolher método de detecção:
  - **Z-Score > 3 ou IQR** → Identificar outliers extremos. Decidir entre remoção, Winsorization ou transformações.
  - **Impacto moderado** → Usar Robust Scaling ou normalização para mitigar efeitos.
- **Não** → Prosseguir para análise de distribuição.

---

### **1.3. Distribuição das Variáveis**
**Pergunta:** As variáveis seguem distribuição normal?
- **Não** → Aplicar teste de Shapiro-Wilk ou KS-Test:
  - **Assimetria** → Transformar com log, Box-Cox ou Yeo-Johnson.
  - **Multimodalidade** → Aplicar binning ou agrupamento por similaridade.
- **Sim** → Analisar correlações.

---

### **1.4. Correlações e Colinearidade**
**Pergunta:** Existem correlações fortes entre variáveis?
- **Sim** → Calcular matriz de correlação:
  - **Correlação > 0.9** → Remover uma das variáveis ou aplicar PCA.
  - **Correlação entre 0.7-0.9** → Avaliar seleção de features ou agregação.
- **Não** → Verificar balanceamento de classes (classificação).

---

### **1.5. Balanceamento de Classes (Classificação)**
**Pergunta:** As classes estão desbalanceadas?
- **Sim** → Calcular proporção de classes:
  - **Classe minoritária < 30%** → Aplicar SMOTE ou ajuste de pesos.
  - **Classe majoritária > 80%** → Usar undersampling ou ensembles balanceados.
- **Não** → Analisar variáveis categóricas.

---

### **1.6. Variáveis Categóricas**
**Pergunta:** Existem variáveis categóricas?
- **Sim** → Avaliar cardinalidade:
  - **< 10 categorias** → Preparar para One-Hot Encoding.
  - **≥ 10 categorias** → Considerar Target Encoding com validação cruzada ou Hashing Trick.
- **Não** → Finalizar EDA.

---

## **2️⃣ Pré-Processamento de Dados**

### **2.1. Dados Corrompidos ou Inconsistentes**
**Pergunta:** Há dados inválidos (ex: strings em colunas numéricas)?
- **Sim** → Corrigir tipos de dados ou remover registros inconsistentes.
- **Não** → Verificar duplicatas.

---

### **2.2. Dados Duplicados**
**Pergunta:** Existem registros duplicados?
- **Sim** → Remover duplicatas ou agregar valores.
- **Não** → Processar texto (se aplicável).

---

### **2.3. Processamento de Texto (NLP)**
**Pergunta:** Os dados incluem texto?
- **Sim** → Aplicar pipeline NLP:
  - **Remover stop words, HTML, ruído** → Limpeza básica.
  - **Stemming/Lemmatização** → Uniformizar termos.
  - **Correção ortográfica** → Padronizar texto.
- **Não** → Balancear classes (se necessário).

---

### **2.4. Normalização e Escalonamento**
**Pergunta:** As variáveis estão em escalas diferentes?
- **Sim** → Escolher método:
  - **Sem outliers** → MinMaxScaler ou StandardScaler.
  - **Com outliers** → RobustScaler.
- **Não** → Codificar variáveis categóricas.

---

### **2.5. Codificação de Variáveis Categóricas**
**Pergunta:** Há variáveis categóricas não codificadas?
- **Sim** → Escolher técnica:
  - **Ordinal** → Label Encoding.
  - **Nominal** → One-Hot Encoding ou Hashing Trick.
- **Não** → Finalizar pré-processamento.

---

## **3️⃣ Engenharia de Features**

### **3.1. Features Temporais**
**Pergunta:** Existem variáveis de data/hora?
- **Sim** → Extrair componentes temporais:
  - **Sazonalidade** → Criar dia da semana, mês, feriados.
  - **Tendências** → Médias móveis ou diferenciação.
- **Não** → Criar interações entre variáveis.

---

### **3.2. Interações entre Variáveis**
**Pergunta:** Relações não lineares entre variáveis são relevantes?
- **Sim** → Gerar features polinomiais ou multiplicativas.
- **Não** → Aplicar binning (se necessário).

---

### **3.3. Binning/Discretização**
**Pergunta:** Variáveis contínuas precisam ser categorizadas?
- **Sim** → Dividir em intervalos lógicos (ex: idade em faixas).
- **Não** → Reduzir dimensionalidade.

---

### **3.4. Redução de Dimensionalidade**
**Pergunta:** Existem muitas features?
- **Sim** → Escolher método:
  - **Supervisionado** → LDA ou Seleção por Importância (Random Forest).
  - **Não supervisionado** → PCA ou UMAP.
- **Não** → Criar embeddings (NLP).

---

### **3.5. Embeddings de Texto (NLP)**
**Pergunta:** Texto precisa ser convertido em vetores?
- **Sim** → Usar TF-IDF, Word2Vec ou BERT.
- **Não** → Finalizar engenharia de features.

---

## **4️⃣ Resumo do Fluxo**

1. **EDA**  
   - Tratar missing, outliers, distribuições, correlações e balanceamento.  
2. **Pré-Processamento**  
   - Limpar dados, normalizar, codificar categóricas e processar texto.  
3. **Engenharia de Features**  
   - Criar features temporais, interações, binning e reduzir dimensionalidade.  

## ✅ Prompt Unificado para Desenvolvimento por Etapas

Você é um engenheiro de software especialista em ciência de dados. Sua tarefa é **desenvolver um sistema completo** em **Python** com base nas etapas de Análise Exploratória de Dados (EDA), Pré-Processamento de Dados e Engenharia de Features, conforme os fluxos de decisão fornecidos.

---

## 🎯 Objetivo
Criar um sistema modular, testável e documentado que permita o processamento automático de datasets com base nas melhores práticas descritas nos documentos técnicos fornecidos (`fluxograma.md`, `mapa_decisao.md`, `guias.md`).

---

## 🔧 Etapas de Desenvolvimento (por partes)

### Etapa 1: Estrutura do Projeto
Siga a estrutura abaixo:

```
src/
├── eda_analyzer.py       # Classes: EDAAnalyzer, MissingValuesAnalyzer, OutlierDetector, etc.
├── data_preprocessor.py  # Classes: DataPreprocessor, DataTypeValidator, DuplicateCleaner, etc.
├── feature_engineer.py   # Classes: FeatureEngineer, TemporalFeatureGenerator, etc.
tests/
├── test_eda_analyzer.py
├── test_data_preprocessor.py
└── test_feature_engineer.py
```

Crie também os arquivos:
- `requirements.txt`
- `README.md` com diagrama de classes simplificado

---

## 🧠 Orientações Técnicas

### 1. Modularidade
- Cada domínio (EDA, pré-processamento, engenharia de features) deve estar em um módulo separado.
- Utilize o conteúdo dos fluxogramas e decisões técnicas como base para construir a lógica de cada classe.

### 2. Testabilidade
- Use `pytest` para criar testes unitários para cada classe.
- Cobertura mínima de 80% por módulo.
- Exemplo:
```python
def test_missing_values_analyzer():
    df = pd.DataFrame({'col1': [1, None, 3], 'col2': [None, None, 5]})
    analyzer = MissingValuesAnalyzer(threshold_high=0.3)
    result = analyzer.analyze(df)
    assert result.suggest_strategy() == "Imputation"
```

### 3. Licenças e Bibliotecas
Use apenas bibliotecas com licença MIT, BSD ou Apache:
- `pandas`, `numpy`, `scipy`
- `scikit-learn`, `seaborn`, `plotly`
- `pytest`, `mypy`

### 4. Documentação e Boas Práticas
- Adicione **docstrings** no padrão **Google Style**
- Inclua **type hints** em todos os métodos públicos
- Utilize **logging** com o módulo `logging`
- Crie exceções customizadas quando necessário
- No `README.md`, inclua um exemplo de uso e uma explicação dos módulos

---

## 🧩 Etapas Internas a Serem Implementadas

Use o conteúdo de `mapa_decisao.md`, `guias.md` e `fluxograma.md` para implementar:
1. **EDAAnalyzer**
   - Detecção de valores faltantes
   - Detecção de outliers (Z-Score, IQR)
   - Análise de normalidade
   - Correlações (> 0.9)
   - Balanceamento de classes
2. **DataPreprocessor**
   - Conversão de tipos
   - Remoção de duplicatas
   - Correção de dados corrompidos
   - Normalização e escalonamento
   - Codificação de variáveis categóricas
   - Pipeline NLP (opcional)
3. **FeatureEngineer**
   - Features temporais
   - Interações entre variáveis
   - Binning/discretização
   - Redução de dimensionalidade (PCA, LDA)
   - Embeddings de texto

---

## 🖥️ Exemplo de Uso Esperado
```python
from src.eda_analyzer import EDAAnalyzer
from src.data_preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer

df = load_some_dataframe()

eda = EDAAnalyzer()
eda_report = eda.analyze(df)

pre = DataPreprocessor()
df_clean = pre.pipeline(df)

eng = FeatureEngineer()
df_final = eng.transform(df_clean)
```

---

## 🚀 Comece agora pela **Etapa 1: Criar a estrutura inicial dos arquivos e a classe `EDAAnalyzer` com base nos valores faltantes**. Quando finalizar, prossiga para as próximas etapas.

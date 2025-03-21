## ✅ Prompt Unificado para Desenvolvimento por Etapas

Você é um engenheiro de software especialista em ciência de dados. Sua tarefa é **desenvolver um sistema completo** em **Python** com base nas etapas de Análise Exploratória de Dados (EDA).
Os documentos (`mapa_decisao.md`, `guias.md`) fornecem informações sobre os possiveis problemas e soluções para cada tipo de problema no dataset.

## 🔧 Etapas de Desenvolvimento (por partes)

### Etapa 1: Mapear os possiveis problemas e soluções para cada tipo de problema no dataset
### Etapa 2: Criar para todas as classes os enums de problemas e soluções
### Etapa 3: Implemente um metodo para retornar o resultado com as informações columns, problem, solution, description, action e choices.


Siga a estrutura abaixo:
```
class MissingValuesProblem(Enum):
    LESS_5 = "LESS_5"
    BETWEEN_5_30 = "BETWEEN_5_30"
    GREATER_30 = "GREATER_30"
    NO_MISSING_VALUES = "NO_MISSING_VALUES"
    
class MissingValuesSolution(Enum):
    IMPUTATION_MEDIA = "IMPUTATION_MEDIA"
    IMPUTATION_MEDIANA = "IMPUTATION_MEDIANA"
    REMOVE_COLUMN = "REMOVE_COLUMN"

class MissingValuesAnalyzer(AnalysisStep):
    def __init__(self):
        self.result = []

    def analyze(self, data: pd.DataFrame) -> dict:
        return [{
                'columns': [],
                'problem': MissingValuesProblem.BETWEEN_5_30.name,
                'solution': MissingValuesSolution.IMPUTATION_MEDIA.name,
                'description': 'A coluna possui valores ausentes entre 5% e 30%.',
                'action': 'Imputar a média dos valores.',
                'choices': [
                    {
                        'name': 'IMPUTATION_MEDIA',
                        'description': 'Imputar a média dos valores.'
                    },
                    {
                        'name': 'IMPUTATION_MEDIANA',
                        'description': 'Imputar a mediana dos valores.'
                    },
                    {
                        'name': 'REMOVE_COLUMN',
                        'description': 'Remover a coluna.'
                    }
                ]
            }]
```

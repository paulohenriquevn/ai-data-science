import pandas as pd


class FeatureInteractionAnalyzer:
    def __init__(self, target_column: str, correlation_gain_threshold: float = 0.05):
        self.target_column = target_column
        self.correlation_gain_threshold = correlation_gain_threshold

    def analyze(self, df: pd.DataFrame) -> list:
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna alvo '{self.target_column}' não encontrada no DataFrame.")

        results = []
        num_df = df.select_dtypes(include="number").drop(columns=[self.target_column], errors="ignore")
        y = df[self.target_column]

        # Filtra apenas colunas numéricas sem NA
        num_df = num_df.dropna(axis=1, how="any")

        cols = num_df.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                s1, s2 = num_df[col1], num_df[col2]

                # Calcular correlação individual com y
                corr1 = abs(s1.corr(y))
                corr2 = abs(s2.corr(y))

                # Geração de interação (produto)
                inter = s1 * s2
                corr_inter = abs(inter.corr(y))

                gain = corr_inter - max(corr1, corr2)
                if gain >= self.correlation_gain_threshold:
                    results.append({
                        'column': [col1, col2],
                        'problem': 'potential_interaction',
                        'problem_description': f'Ganho de correlação {gain:.3f} com y ao combinar {col1} * {col2}.',
                        'suggestion': 'CRIAR_INTERACAO_PRODUTO',
                        'actions': ['CRIAR_INTERACAO_PRODUTO', 'CRIAR_RAZAO', 'CRIAR_DIFERENCA'],
                        'statistics': {
                            'correlation_var1': corr1,
                            'correlation_var2': corr2,
                            'correlation_interaction': corr_inter,
                            'correlation_gain': gain
                        }
                    })

        return results

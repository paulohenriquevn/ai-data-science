import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class FeatureImportanceAnalyzer:
    def __init__(self, target_column: str, importance_threshold: float = 0.001, top_n: int = 30):
        self.target_column = target_column
        self.importance_threshold = importance_threshold
        self.top_n = top_n

    def analyze(self, df: pd.DataFrame) -> list:
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna alvo '{self.target_column}' não encontrada no DataFrame.")

        df = df.copy()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Codificação de variáveis categóricas e imputação básica
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        imputer = SimpleImputer(strategy="most_frequent")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Codificar y se necessário
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        # Treinar modelo RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_

        # Organizar importância
        ranked = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
        results = []

        for rank, (col, imp) in enumerate(ranked, 1):
            actions = []
            if rank > self.top_n:
                actions.append("LIMITE_TOP_N")
            if imp < self.importance_threshold:
                actions.append("REMOVER_VARIAVEL")

            if actions:
                results.append({
                    'column': col,
                    'problem': 'low_feature_importance',
                    'problem_description': f'Importância da variável é baixa ({imp:.4f}), rank {rank}.',
                    'suggestion': actions[0],
                    'actions': list(set(actions)),
                    'statistics': {
                        'importance': imp,
                        'rank': rank
                    }
                })

        return results

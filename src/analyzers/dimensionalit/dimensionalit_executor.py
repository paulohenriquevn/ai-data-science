from sklearn.decomposition import PCA
import pandas as pd


class DimensionalityExecutor:
    def __init__(self, apply_pca=True, pca_variance_threshold=0.95):
        self.apply_pca = apply_pca
        self.pca_variance_threshold = pca_variance_threshold
        self.pca_model = None

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()
        cols_to_remove = []

        # Etapa 1: Remoção por variância muito baixa
        for step in plan:
            col = step['column']
            action = step['suggestion']
            stats = step.get("parameters", {})
            if action == "VARIANCE_THRESHOLD" and col in df.columns:
                cols_to_remove.append(col)

        df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

        # Etapa 2: Aplicação de PCA se necessário
        pca_columns = [step['column'] for step in plan if step['suggestion'] == 'PCA' and step['column'] in df.columns]

        if self.apply_pca and len(pca_columns) >= 2:
            numeric_df = df[pca_columns].dropna()
            pca = PCA(n_components=self.pca_variance_threshold)
            pca_result = pca.fit_transform(numeric_df)
            pca_cols = [f'PCA_{i+1}' for i in range(pca_result.shape[1])]
            df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=numeric_df.index)

            # Substituir colunas originais pelos componentes principais
            df.drop(columns=pca_columns, inplace=True, errors='ignore')
            df = pd.concat([df, df_pca], axis=1)
            self.pca_model = pca

        return df


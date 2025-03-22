from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

class PCAStep:
    def __init__(self, pca_plan, output_path="pca_variance.png"):
        """
        Args:
            pca_plan: dict gerado pelo PCAPlanStep contendo:
                - pca_candidate_columns: list[str]
                - recommended_n_components: int
            output_path: caminho para salvar o gráfico de variância acumulada
        """
        self.pca_plan = pca_plan
        self.output_path = output_path
        self.pca = None

    def fit_transform(self, df):
        candidate_cols = self.pca_plan.get("pca_candidate_columns", [])
        n_components = self.pca_plan.get("recommended_n_components")

        if not candidate_cols or not n_components:
            print("[INFO] Nenhuma variável candidata ao PCA foi identificada no plano.")
            return pd.DataFrame(index=df.index)  # Retorna DataFrame vazio

        X = df[candidate_cols].dropna()
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)

        # Gerar DataFrame dos componentes
        df_pca = pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=X.index
        )

        # Regerar curva e salvar gráfico
        cumulative_variance = self.pca.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_variance, marker='o')
        plt.axhline(y=cumulative_variance[n_components-1], color='red', linestyle='--', label=f'{cumulative_variance[n_components-1]*100:.1f}%')
        plt.axvline(x=n_components-1, color='green', linestyle='--', label=f'{n_components} componentes')
        plt.title("Variância Acumulada - PCA")
        plt.xlabel("Número de Componentes")
        plt.ylabel("Variância Explicada Acumulada")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()

        return df_pca

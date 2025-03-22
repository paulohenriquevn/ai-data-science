import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def detect_and_replace_placeholders(
        df: pd.DataFrame,
        candidates=[-999, -1, 9999, 999, 0],
        min_freq: float = 0.01,
        skew_threshold: float = 1.5,
        kurt_threshold: float = 3.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Detecta valores que atuam como ausentes e substitui por np.nan usando heurísticas de frequência, IQR, skewness e kurtosis.

        Args:
            df (pd.DataFrame): DataFrame original
            candidates (list): Lista de valores candidatos
            min_freq (float): Frequência mínima para investigar o valor
            skew_threshold (float): Mínima skewness para considerar a distribuição distorcida
            kurt_threshold (float): Mínima kurtosis para considerar cauda pesada
            verbose (bool): Exibe relatório

        Returns:
            pd.DataFrame: Novo DataFrame com placeholders substituídos por np.nan
        """
        df = df.copy()
        placeholder_report = {}

        for val in candidates:
            affected_cols = []

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    series = df[col]
                    count_val = (series == val).sum()
                    perc_val = count_val / len(series)

                    if perc_val >= min_freq:
                        # Remover valor suspeito para análise
                        cleaned = series.replace(val, np.nan).dropna()

                        if len(cleaned) < 10:
                            continue  # Muito poucos dados para análise estatística

                        # Estatísticas da distribuição
                        q1 = cleaned.quantile(0.25)
                        q3 = cleaned.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        s = skew(cleaned)
                        k = kurtosis(cleaned)

                        # Heurísticas combinadas
                        is_extreme = val < lower_bound or val > upper_bound
                        is_distorted = abs(s) > skew_threshold or k > kurt_threshold

                        if is_extreme and is_distorted:
                            df[col] = df[col].replace(val, np.nan)
                            affected_cols.append(col)

            if affected_cols:
                placeholder_report[val] = affected_cols
                if verbose:
                    print(f"🔍 Substituído {val} por NaN nas colunas: {affected_cols}")

        if verbose and not placeholder_report:
            print("✅ Nenhum placeholder suspeito detectado.")

        return df

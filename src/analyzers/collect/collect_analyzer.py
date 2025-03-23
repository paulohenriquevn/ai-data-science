import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from src.analyzers.base.analysis_base import AnalysisStep


class CollectAnalyzer(AnalysisStep):
    def __init__(self):
        self.placeholders = [
            '-', 
            'NA', 
            'N/A', 
            'NULL', 
            'nan', 
            'None', 
            '?', 
            'missing', 
            'unknown', 
            'undefined', 
            'blank', 
            'empty', 
            'null', 
            'nan', 
            'none',
            'NENHUM',
            'NÃO APLICÁVEL',
        ]
        self.analysis_output = []

    def analyze(self, data: pd.DataFrame) -> list:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Os dados devem ser um DataFrame do pandas")

        if data.empty:
            raise ValueError("O DataFrame está vazio")

        for col in data.columns:
            series = data[col]

            # Valores nulos
            self._nulls_values(data, col, series)

            # Placeholders em strings
            self._placeholders_strings(col, series)

            # Conversões de tipo
            self._convert_types(col, series)

            # Cardinalidade baixa
            # self._low_cardinality(col, series)

        # Detectar placeholders numéricos com heurísticas
        self._detect_numeric_placeholders(data)

        return self.analysis_output

    def _low_cardinality(self, col, series):
        unique_count = series.nunique(dropna=True)
        if series.dtype == 'object' and 1 < unique_count <= 10:
            self.analysis_output.append({
                    'column': col,
                    'problem': 'low_cardinality',
                    'problem_description': f'A coluna "{col}" possui baixa cardinalidade ({unique_count} valores únicos).',
                    'solution': 'Avaliar para encoding categórico ou agrupamento.',
                    'actions': ['EVALUATE_ENCODING'],
                    'suggestion': 'EVALUATE_ENCODING',
                    'statistics': {
                        'unique_values': unique_count
                    }
                })

    def _convert_types(self, col, series):
        conversion_type = self._suggest_type_conversion(series)
        if conversion_type:
            self.analysis_output.append({
                    'column': col,
                    'problem': 'type_conversion_suggested',
                    'problem_description': f'A coluna "{col}" pode ser convertida para o tipo {conversion_type}.',
                    'solution': f'Converter a coluna para {conversion_type}.',
                    'actions': [f'CONVERT_TO_{conversion_type.upper()}'],
                    'suggestion': f'CONVERT_TO_{conversion_type.upper()}',
                    'statistics': {
                        'original_dtype': str(series.dtype)
                    }
                })

    def _placeholders_strings(self, col, series):
        if series.dtype == 'object':
            placeholders_found = {}
            series_str = series.astype(str).str.strip().str.lower()
            for ph in self.placeholders:
                count = (series_str == ph.lower()).sum()
                if count > 0:
                    placeholders_found[ph] = count
            if placeholders_found:
                total = sum(placeholders_found.values())
                self.analysis_output.append({
                        'column': col,
                        'problem': 'placeholder_values',
                        'problem_description': f'A coluna "{col}" contém {total} valores reservados.',
                        'solution': 'Substituir os placeholders por np.nan e tratar como valores nulos.',
                        'actions': ['REPLACE_PLACEHOLDERS_WITH_NAN'],
                        'suggestion': 'REPLACE_PLACEHOLDERS_WITH_NAN',
                        'statistics': {
                            'placeholder_counts': placeholders_found
                        }
                    })

    def _nulls_values(self, data, col, series):
        null_count = series.isnull().sum()
        missing_pct = null_count / len(data) * 100
        if null_count > 0:
            suggestion = 'FILL_NA' if missing_pct < 50 else 'DROP_NA'
            self.analysis_output.append({
                    'column': col,
                    'problem': 'missing_values',
                    'problem_description': f'A coluna "{col}" contém {null_count} valores nulos.',
                    'solution': 'Preencher ou remover valores nulos com base no contexto.',
                    'actions': ['FILL_NA', 'DROP_NA'],
                    'suggestion': suggestion,
                    'statistics': {
                        'missing_count': null_count,
                        'missing_percentage': missing_pct
                    }
                })

    def _detect_numeric_placeholders(self, df: pd.DataFrame,
                                     candidates=[-999, -1, 9999, 999, 0],
                                     min_freq: float = 0.01,
                                     skew_threshold: float = 1.5,
                                     kurt_threshold: float = 3.0):
        for val in candidates:
            for col in df.select_dtypes(include=[np.number]).columns:
                series = df[col]
                count_val = (series == val).sum()
                perc_val = count_val / len(series)

                if perc_val >= min_freq:
                    cleaned = series.replace(val, np.nan).dropna()
                    if len(cleaned) < 10:
                        continue

                    q1 = cleaned.quantile(0.25)
                    q3 = cleaned.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    s = skew(cleaned)
                    k = kurtosis(cleaned)

                    is_extreme = val < lower_bound or val > upper_bound
                    is_distorted = abs(s) > skew_threshold or k > kurt_threshold

                    if is_extreme and is_distorted:
                        self.analysis_output.append({
                            'column': col,
                            'problem': 'numeric_placeholder_detected',
                            'problem_description': f'A coluna "{col}" contém valores numéricos suspeitos de serem placeholders: [{val}].',
                            'solution': 'Substituir os valores por np.nan e tratar como valores ausentes.',
                            'actions': ['REPLACE_NUMERIC_PLACEHOLDERS_WITH_NAN'],
                            'suggestion': 'REPLACE_NUMERIC_PLACEHOLDERS_WITH_NAN',
                            'statistics': {
                                'suspected_placeholders': [val]
                            }
                        })

    def _suggest_type_conversion(self, series: pd.Series):
        if series.dtype != 'object':
            return None

        try:
            cleaned = series.dropna().astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '').str.strip()
            parsed = pd.to_numeric(cleaned, errors='coerce')
            if parsed.notna().sum() / len(cleaned) > 0.9:
                if (parsed % 1 == 0).all():
                    return 'integer'
                else:
                    return 'float'
        except:
            pass

        try:
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().sum() / len(series.dropna()) > 0.9:
                return 'datetime'
        except:
            pass

        lower_vals = series.dropna().astype(str).str.lower().unique().tolist()
        bool_sets = [
            {'true', 'false'}, {'yes', 'no'}, {'y', 'n'}, {'1', '0'}, {'sim', 'não'}, {'t', 'f'}
        ]
        for bool_set in bool_sets:
            if set(lower_vals).issubset(bool_set):
                return 'boolean'

        return None



import pandas as pd
import numpy as np
from src.analyzers.base.analysis_base import ExecutionStep


class CollectExecutor(ExecutionStep):
    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step['column']
            action = step['suggestion']
            params = step.get('parameters', {})

            if action == 'FILL_NA':
                df[col] = df[col].fillna(0)

            elif action == 'DROP_NA':
                df = df[df[col].notna()]

            elif action == 'REPLACE_PLACEHOLDERS_WITH_NAN':
                # Valores reservados já conhecidos
                reserved = ['-', 'NA', 'N/A', 'NULL', 'nan', 'None', '?', 'missing', 'unknown', 'undefined', '']
                df[col] = df[col].astype(str).str.strip().str.lower().replace(reserved, np.nan)

            elif action == 'REPLACE_NUMERIC_PLACEHOLDERS_WITH_NAN':
                placeholders = params.get('suspected_placeholders', [])
                for val in placeholders:
                    df[col] = df[col].replace(val, np.nan)

            elif action == 'CONVERT_TO_INTEGER':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

            elif action == 'CONVERT_TO_FLOAT':
                df[col] = pd.to_numeric(df[col], errors='coerce')

            elif action == 'CONVERT_TO_BOOLEAN':
                df[col] = df[col].astype(str).str.lower().map({
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    'y': True, 'n': False,
                    '1': True, '0': False,
                    'sim': True, 'não': False,
                    't': True, 'f': False
                }).astype('boolean')

            elif action == 'CONVERT_TO_DATETIME':
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df


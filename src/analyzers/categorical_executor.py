import pandas as pd
import numpy as np

class CategoricalExecutor:
    def __init__(self, plan: dict):
        self.plan = plan
        self.ordinal_maps = {}
        # Lista de tipos que devem ser convertidos
        self.categorical_types = ['object', 'category', 'string']

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols_aplicadas = []
        
        # Primeiro passo: aplicar o plano fornecido
        self._apply_plan(df, cols_aplicadas)
        
        # Segundo passo: verificar se ainda existem colunas categóricas e aplicar transformação automática
        remaining_categorical = df.select_dtypes(include=self.categorical_types).columns
        
        if len(remaining_categorical) > 0:
            print(f"⚠️ Encontradas {len(remaining_categorical)} colunas categóricas não especificadas no plano:")
            print(f"   {list(remaining_categorical)}")
            
            # Criar plano automático para as colunas restantes (ONE_HOT para colunas com poucos valores, ORDINAL para muitas)
            auto_plan = {}
            for col in remaining_categorical:
                unique_values = df[col].nunique()
                if unique_values <= 10:  # Poucas categorias -> ONE_HOT
                    auto_plan[col] = "ONE_HOT"
                else:  # Muitas categorias -> ORDINAL
                    auto_plan[col] = "ORDINAL"
            
            print(f"🔄 Aplicando transformação automática às colunas restantes:")
            for col, strategy in auto_plan.items():
                print(f"   - {col}: {strategy}")
            
            # Aplicar o plano automático
            self._apply_transformations(df, auto_plan, cols_aplicadas)
        
        # Verificação final: tentar garantir que não há mais colunas categóricas
        final_categorical = df.select_dtypes(include=self.categorical_types).columns
        
        if len(final_categorical) == 0:
            print("✅ Todas as variáveis categóricas foram transformadas com sucesso.")
        else:
            print(f"⚠️ Ainda restam {len(final_categorical)} variáveis categóricas não transformadas:")
            print(f"   {list(final_categorical)}")
            print("🔄 Aplicando tratamento de último recurso para colunas restantes...")
            
            # Último recurso: converter colunas problemáticas para numérico de forma agressiva
            for col in final_categorical:
                try:
                    # Tentativa 1: Converter para números onde possível, NaN onde não for possível
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Substituir NaN por -999 como marcador
                    df[col] = df[col].fillna(-999)
                    print(f"   ✓ Coluna '{col}' convertida para numérico")
                except Exception as e:
                    # Se falhar completamente, remover a coluna
                    print(f"   ✗ Coluna '{col}' não pôde ser convertida e será removida: {str(e)}")
                    df = df.drop(columns=[col])
            
            # Verificação final após último recurso
            final_check = df.select_dtypes(include=self.categorical_types).columns
            if len(final_check) == 0:
                print("✅ Todas as variáveis categóricas foram tratadas após ações de último recurso.")
            else:
                print(f"⚠️ Removendo {len(final_check)} colunas que não puderam ser transformadas: {list(final_check)}")
                df = df.drop(columns=final_check)
        
        return df
    
    def _apply_plan(self, df, cols_aplicadas):
        """Aplica o plano inicial às colunas especificadas"""
        self._apply_transformations(df, self.plan, cols_aplicadas)
    
    def _apply_transformations(self, df, plan, cols_aplicadas):
        """Aplica transformações específicas às colunas de acordo com o plano"""
        for col, strategy in plan.items():
            if col not in df.columns:
                print(f"⚠️ Coluna '{col}' não encontrada no dataset. Pulando.")
                continue

            try:
                if strategy == "ORDINAL":
                    # Converter para string primeiro para garantir que todos os valores sejam processáveis
                    df[col] = df[col].fillna("NA").astype(str)
                    uniques = sorted(df[col].unique())
                    mapping = {v: i for i, v in enumerate(uniques)}
                    self.ordinal_maps[col] = mapping
                    df[col] = df[col].map(mapping)
                    cols_aplicadas.append(col)
                    print(f"   ✓ Aplicada codificação ORDINAL à coluna '{col}'")

                elif strategy == "ONE_HOT":
                    # Converter para string primeiro para garantir que todos os valores sejam processáveis
                    df[col] = df[col].fillna("NA").astype(str)
                    # Usar get_dummies para codificação one-hot
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                    # Concatenar e remover coluna original
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    cols_aplicadas.append(col)
                    print(f"   ✓ Aplicada codificação ONE_HOT à coluna '{col}' ({dummies.shape[1]} novas colunas)")

                elif strategy == "NENHUMA_ACAO_NECESSARIA":
                    print(f"   - Nenhuma ação necessária para '{col}' conforme especificado no plano")
                    continue

                else:
                    print(f"⚠️ Estratégia de transformação '{strategy}' para coluna '{col}' não implementada.")
                    # Tentar aplicar codificação padrão (ONE_HOT)
                    df[col] = df[col].fillna("NA").astype(str)
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    cols_aplicadas.append(col)
                    print(f"   ✓ Aplicada codificação padrão (ONE_HOT) à coluna '{col}'")
            
            except Exception as e:
                print(f"⚠️ Erro ao processar coluna '{col}': {str(e)}")
                # Remover a coluna problemática se não puder ser processada
                df.drop(columns=[col], inplace=True, errors='ignore')
                print(f"   ✗ Coluna '{col}' removida devido a erros no processamento")
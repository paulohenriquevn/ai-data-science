import pandas as pd

class Prioritizer:
    """
    Prioritizer class to prioritize columns based on skewness and outlier ratio.
    """ 
    def __init__(self, skew_weight: float = 0.6, outlier_weight: float = 0.4):
        self.skew_weight = skew_weight
        self.outlier_weight = outlier_weight

    def prioritize(self, dist_analysis: list, outlier_analysis: list) -> pd.DataFrame:
        df_dist = pd.DataFrame(dist_analysis)
        df_out = pd.DataFrame(outlier_analysis)

        df_dist = df_dist[['column', 'statistics']].copy()
        df_dist['skewness'] = df_dist['statistics'].apply(lambda x: abs(x.get('skewness', 0)))

        df_out = df_out[['column', 'statistics']].copy()
        df_out['outlier_ratio'] = df_out['statistics'].apply(lambda x: x.get('outlier_ratio', 0))

        df = pd.merge(df_dist[['column', 'skewness']], df_out[['column', 'outlier_ratio']], on='column', how='outer')
        df['priority_score'] = df['skewness'] * self.skew_weight + df['outlier_ratio'] * self.outlier_weight

        df = df.sort_values(by='priority_score', ascending=False).reset_index(drop=True)
        return df
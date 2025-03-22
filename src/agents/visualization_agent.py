# agents/visualization_agent.py
import matplotlib.pyplot as plt
import seaborn as sns
from .base_agent import EDAAgent
import pandas as pd

class VisualizationAgent(EDAAgent):
    def analyze(self, data: pd.DataFrame, context: Dict[str, Any] = None, insights: dict = None) -> dict:
        # Gerar gráficos baseados nos insights
        sns.pairplot(data.select_dtypes(include='number'))
        plt.savefig('pairplot.png')
        return {'visualizations': ['pairplot.png', 'correlation_heatmap.png']}
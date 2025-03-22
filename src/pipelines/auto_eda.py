# pipelines/auto_eda.py
class AutoEDA:
    def __init__(self):
        self.steps = [
            DataIngestor(),
            QualityDiagnosticAgent(),
            FeatureScorer(),
            StatisticalSignificanceAnalyzer(),
            VisualizationAgent(),
            # HypothesisAgent(),
            # ReportGenerator()
        ]
    
    def run_pipeline(self, file_path: str):
        data = None
        all_results = {}
        
        for step in self.steps:
            if isinstance(step, DataIngestor):
                data = step.analyze(file_path)
            else:
                result = step.analyze(data, all_results)
                all_results[step.__class__.__name__] = result
        
        return all_results
# agents/quality_diagnostic.py
from src.analyzers import (
    MissingValuesAnalyzer,
    OutlierAnalyzer,
    DuplicatesAnalyzer,
    DistributionAnalyzer
)

class QualityDiagnosticAgent(EDAAgent):
    def analyze(self, data: pd.DataFrame, context: Dict[str, Any] = None) -> dict:
        results = {}
        
        analyzers = [
            MissingValuesAnalyzer(),
            OutlierAnalyzer(),
            DuplicatesAnalyzer(),
            DistributionAnalyzer()
        ]
        
        for analyzer in analyzers:
            result = analyzer.analyze(data)
            results[analyzer.__class__.__name__] = result
        
        return results
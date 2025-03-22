# agents/base_agent.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class EDAAgent(ABC):
    def __init__(self, dependencies=None):
        self.dependencies = dependencies or []
        self.results = None

    @abstractmethod
    def analyze(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        return {}
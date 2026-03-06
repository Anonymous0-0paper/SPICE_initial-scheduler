from abc import ABC, abstractmethod
from typing import Dict, Any

from salsa.utils.typing import microservice_id


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, current_data_df: Dict[microservice_id, float], context: Dict[str, Any]):
        pass

    @abstractmethod
    def reset(self):
        pass
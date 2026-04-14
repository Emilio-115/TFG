from dataclasses import dataclass
import numpy as np

@dataclass
class Results:
    classif_report: str | dict
    conf_matrix: np.ndarray
    ap_0: float
    ap_1: float
    roc_auc_0: float
    roc_auc_1: float
    
    def __str__(self):
        return f"Report:\n{self.classif_report}\nMatrix:\n{self.conf_matrix}"
    def __repr__(self):
        return self.__str__()
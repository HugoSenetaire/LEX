from .classification_only import ordinaryPredictionTraining, trainingWithSelection
from .separate_loss import SEPARATE_LOSS
from .single_loss import SINGLE_LOSS
from .selection_only import selectionTraining

list_complete_trainer = {
    "ordinaryPredictionTraining": ordinaryPredictionTraining,
    "trainingWithSelection": trainingWithSelection,
    "SEPARATE_LOSS": SEPARATE_LOSS,
    "SINGLE_LOSS": SINGLE_LOSS,
    "selectionTraining" : selectionTraining,
}

def get_complete_trainer(module_name):
    if module_name in list_complete_trainer :
        return list_complete_trainer[module_name]
    else :
        raise ValueError("module_name not in list_complete_trainer")
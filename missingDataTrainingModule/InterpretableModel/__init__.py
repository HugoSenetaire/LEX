from .fixed_selection_training import PredictionCompleteModel, trueSelectionCompleteModel, EVAL_X
from .interpretation_training import COUPLED_SELECTION, DECOUPLED_SELECTION

list_selection_modules = {
    "PredictionCompleteModel": PredictionCompleteModel,
    "trueSelectionCompleteModel": trueSelectionCompleteModel,
    "EVAL_X": EVAL_X,
    "COUPLED_SELECTION": COUPLED_SELECTION,
    "DECOUPLED_SELECTION": DECOUPLED_SELECTION,
}

def get_interpretable_module(module_name):
    return list_selection_modules[module_name]
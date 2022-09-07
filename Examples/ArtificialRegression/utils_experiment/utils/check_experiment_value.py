import os

def check_experiment_value(path):
    path_prediction = os.path.join(path, "prediction_module_best.pt")
    path_selection = os.path.join(path, "selection_module_best.pt")
    if os.path.exists(path_prediction) or os.path.exists(path_selection):
        return True
    else :
        return False
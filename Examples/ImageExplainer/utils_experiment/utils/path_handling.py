
import os

def create_name(args, dataset_name, name_experiment, count):
    current_name_experiment = os.path.join(name_experiment, dataset_name)
    origin_path = args.args_output.path
    path_global = os.path.join(origin_path, current_name_experiment)
    aux_string = f"{args.args_trainer.complete_trainer}_{args.args_trainer.monte_carlo_gradient_estimator}_{args.args_selection.regularization}_{args.args_selection.loss_regularization}_{args.args_selection.batched}_{args.args_classification.imputation}"
    current_path = os.path.join(path_global, aux_string)
    current_path = os.path.join(current_path, f"{args.args_classification.classifier}_{args.args_selection.selector}_{args.args_selection.rate}_MEANIMPUTATION_{count}")
    args.args_output.path = current_path

    return args
import keras_tuner as kt
from typing import Callable
import os
import optuna

def get_keras_tuner(model_builder: Callable, fold_idx, prolapse_name, experiment):

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    tuner_dir = os.path.join(BASE_DIR, "results", experiment, prolapse_name, "keras_tuner_results")
    os.makedirs(tuner_dir, exist_ok=True)
    project_name = f"tuner_fold_{fold_idx}"

    return kt.BayesianOptimization(
        hypermodel=model_builder,
        objective=kt.Objective("val_pr_auc", direction="max"),
        max_trials=10,
        directory=tuner_dir,
        project_name=project_name,
        overwrite=False
    )

def get_optuna_study(fold_idx, prolapse_name, experiment,
                     direction="maximize", persist=True, reset=False):
    study_name = f"study_fold_{fold_idx}"

    if persist:
        BASE_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        study_dir = os.path.join(BASE_DIR, "results", experiment, prolapse_name, "optuna_results")
        os.makedirs(study_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"

        if reset:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f"Estudio '{study_name}' eliminado.")
            except KeyError:
                pass
    else:
        storage = None

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=persist and not reset,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    return study
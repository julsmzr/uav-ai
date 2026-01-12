import os
import csv
import shutil
import time

import numpy as np
import pandas as pd

from uav.experiments.data import TempTrainingContext, Modality

from ultralytics import YOLO # type: ignore


def current_milli_time() -> int:
    """Returns current time in milliseconds."""
    return round(time.time() * 1000)

# run 5x5 fold rskf for single seed
def run_repeated_k_fold(model_seed: int, splits: list, experiment_name: str, epochs: int, model_weight_path: str, run_dir: str):
    """Trains and evaluates YOLO model on all RSKF splits, yielding metrics for each fold."""
    for split in splits:
        fold = split['fold']
        print(f"Processing fold {fold}")
        
        def exp_name(template: str, fold_idx: str, ending: str) -> str:
            """Generates experiment name with fold index and train/val suffix."""
            return "%s-f_%s_%s" % (template, fold_idx, ending)
        
        expname_train = exp_name(experiment_name, fold, "train")
        expname_val = exp_name(experiment_name, fold, "val")

        tpath = os.path.join(run_dir, expname_train)
        vpath = os.path.join(run_dir, expname_val)

        if os.path.exists(os.path.join(tpath, "results.csv")) and os.path.exists(os.path.join(vpath, "predictions.json")):
            continue
        elif os.path.exists(tpath) or os.path.exists(vpath):
            if os.path.exists(tpath):
                try:
                    shutil.rmtree(tpath)
                except OSError as e:
                    print(f"Warning: Could not remove {tpath}: {e}")
            if os.path.exists(vpath):
                try:
                    shutil.rmtree(vpath)
                except OSError as e:
                    print(f"Warning: Could not remove {vpath}: {e}")
        
        with TempTrainingContext(split['filepaths'], Modality.VISIBLE, split['train_idx'], split['test_idx']) as temp_ctx:

            cfg = os.path.join(temp_ctx, "cfg.yaml")
            model = YOLO(model_weight_path)

            print("Running", experiment_name, "...")
            pre_train_timestamp = current_milli_time()
            model.train(
                data=cfg, 
                epochs=epochs,  
                imgsz=640,  
                device="mps",  
                seed=model_seed,
                name=expname_train,
                project=run_dir,
            )
            post_train_timestamp = current_milli_time()
            model.val(
                data=cfg,
                imgsz=640,
                device="mps",
                seed=model_seed,
                name=expname_val,
                project=run_dir,
                save_json=True
            )   
            post_eval_timestamp = current_milli_time()

            df = pd.read_csv(os.path.join(tpath, "results.csv"))
            last_row = df.iloc[-1]

            metrics = [
                'metrics/precision(B)',
                'metrics/recall(B)',
                'metrics/mAP50(B)',
                'metrics/mAP50-95(B)'
            ]
            
            metric_values = [f"{last_row[metric]:.6f}" for metric in metrics]
            training_time_ms = post_train_timestamp - pre_train_timestamp
            eval_time_ms = post_eval_timestamp - post_train_timestamp
            yield [experiment_name] + metric_values + [training_time_ms, eval_time_ms]

def append_results(results_csv_filepath: str, result_row: list) -> None:
    """Appends experiment result row to CSV file, creating header if file doesn't exist."""
    file_exists = os.path.exists(results_csv_filepath)

    with open(results_csv_filepath, "a", newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                "experiment_name",
                "precision",
                "recall",
                "mAP50",
                "mAP50-95",
                "training_time_ms",
                "eval_time_ms",
            ]
            writer.writerow(header)
            
        writer.writerow(result_row)

def run_experiments(experiment_rskf_file_npy: str, metrics_file_txt: str, model_seeds: list[int], epochs: int, model_weight_path: str, run_dir: str) -> None:
    """Runs experiments for all modalities and model seeds."""

    # len: n_repeats * n_splits of {fold idx, filepaths, train indices, test indices}
    splits = np.load(experiment_rskf_file_npy, allow_pickle=True)

    for modality in [
        Modality.VISIBLE,
        Modality.INFRARED,
        Modality.HYBRID
    ]:
        for model_seed in model_seeds:
            experiment_name = "%s-%s" % (modality.value, model_seed)
            try:
                for result_row in run_repeated_k_fold(model_seed, splits, experiment_name, epochs, model_weight_path, run_dir):
                    append_results(metrics_file_txt, result_row)
            except KeyboardInterrupt:
                print("Run cancelled via KeyboardInterrupt.")
            except Exception as e:
                print("unexpected exception!", e)

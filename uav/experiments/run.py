import numpy as np
import os
import shutil
import pandas as pd
import re

from ultralytics import YOLO # type: ignore

from uav.experiments.data import TempTrainingContext, Modality

# run 5x5 fold rskf for single seed
def run_repeated_k_fold(model_seed: int, n_splits, splits: list, experiment_name: str):

    for split in splits:

        fold = split['fold']
        print(f"Processing fold {fold}")
        
        def exp_name(template: str, fold_idx: str, ending: str) -> str:
            return "%s-f_%s_%s" % (template, fold_idx, ending)
        
        expname_train = exp_name(experiment_name, fold, "train")
        expname_val = exp_name(experiment_name, fold, "val")

        tpath = os.path.join("results", "exp_train_runs", expname_train)
        vpath = os.path.join("results", "exp_train_runs", expname_val)
        if os.path.exists(os.path.join(tpath, "results.csv")) and os.path.exists(os.path.join(vpath, "predictions.json")):
            print("skipping", expname_train)
            continue
        elif os.path.exists(vpath) or os.path.exists(vpath):
            shutil.rmtree(tpath)
            shutil.rmtree(vpath)
        
        with TempTrainingContext(split['filepaths'], Modality.VISIBLE, split['train_idx'], split['test_idx']) as temp_ctx:

            cfg = os.path.join(temp_ctx, "cfg.yaml")
            model = YOLO("models/weights/yolo12n.pt")

            print("Running", experiment_name, "...")
            model.train(
                data=cfg, 
                epochs=20,  
                imgsz=640,  
                device="mps",  
                seed=model_seed,
                name=expname_train,
                project="results/exp_train_runs",
            )

            model.val(
                data=cfg,
                imgsz=640,
                device="mps",
                seed=model_seed,
                name=expname_val,
                project="results/exp_train_runs",
                save_json=True
            )

            df = pd.read_csv(os.path.join(tpath, "results.csv"))
            last_row = df.iloc[-1]

            metrics = [
                'metrics/precision(B)',
                'metrics/recall(B)',
                'metrics/mAP50(B)',
                'metrics/mAP50-95(B)'
            ]
            
            metric_values = [f"{last_row[metric]:.6f}" for metric in metrics]
            yield "%s,%s\n" % (experiment_name, ",".join(metric_values))

def append_results(res_path_txt: str, data: str) -> None:
    with open(res_path_txt, "a") as f:
        f.write(data)

# for all 5 model seeds run experiments
def run_experiments(n_splits: int, model_seeds: list[int]):

    # len: n_repeats * n_splits of {fold idx, filepaths, train indices, test indices}
    splits = np.load('experiments/rskf_splits.npy', allow_pickle=True)

    for modality in [
        Modality.VISIBLE,
        Modality.INFRARED,
        Modality.HYBRID
    ]:
        for model_seed in model_seeds:
            experiment_name = "%s-%s" % (modality.value, model_seed)
            for run_results in run_repeated_k_fold(model_seed, n_splits, splits, experiment_name):
                append_results("results/metrics.txt", run_results)

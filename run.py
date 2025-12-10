from uav.experiments.run import run_experiments


SEEDS = [    
4140620135, # StratifiedShuffleSplit
1132905197, # RepeatedStratifiedKFold
3754523883, # |
3492508408, # |
3378715402, # Model Seeds
4213948446, # |
1521997286, # |
]

run_experiments(
    experiment_rskf_file_npy="experiments/rskf_splits.npy",
    metrics_file_txt="results/metrics.csv",
    model_seeds=SEEDS[2:7],
    epochs=20,
    model_weight_path="models/weights/yolo12n.pt",
    run_dir="results/exp_train_runs"
    )

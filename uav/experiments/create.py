import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold


def create_experiments(seeds: list[int], source_dir: str, target_filepath: str, fold_size: int, n_splits: int = 5, n_repeats: int = 2) -> None:
    """Create the 'experiments.yml' file to allow later execution of the experiments one by one."""

    assert len(seeds) == 2

    dataset_images_dir = os.path.join(source_dir, "images")
    dataset_labels_dir = os.path.join(source_dir, "labels")
    
    sequences = os.listdir(dataset_images_dir)
    X, y_stratify = [], []  # y_stratify for stratification labels
 
    for sequence_name in tqdm(sequences, desc="Processing Sequences"):
        sequence_images_dir = os.path.join(dataset_images_dir, sequence_name)
        sequence_labels_dir = os.path.join(dataset_labels_dir, sequence_name)

        image_files = sorted([f for f in os.listdir(sequence_images_dir) if f.endswith('.jpg') and '-vz-' in f])
        for img_file in image_files:
            img_path = os.path.join(sequence_images_dir, img_file)
            X.append(img_path)
            
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(sequence_labels_dir, label_file)
            
            # Create binary stratification labels: 1 for object, 0 for empty
            if os.path.getsize(label_path) == 0:
                y_stratify.append(0) 
            else:
                y_stratify.append(1)
              
    X = np.array(X)
    y_stratify = np.array(y_stratify)

    print(f"Original X shape: {X.shape}")
    print(f"Original stratification labels: {np.unique(y_stratify, return_counts=True)}")
        
    total_images = n_splits * fold_size

    if len(X) > total_images:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=total_images, random_state=seeds[0]) 
        
        for sample_idx, _ in sss.split(X, y_stratify):
            X = X[sample_idx]
            y_stratify = y_stratify[sample_idx]
    
    print(f"Sampled X shape: {X.shape}")
    print(f"Sampled stratification labels: {np.unique(y_stratify, return_counts=True)}")
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seeds[1])
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y_stratify)):
        folds.append({
            'fold': fold_idx,
            'filepaths': X,
            'train_idx': train_idx,
            'test_idx': test_idx,
        })
    
    
    np.save(target_filepath, folds)
    print("Saved RSKF config to %s." % target_filepath)

import tempfile
import os
from enum import Enum
import shutil
import yaml

class Modality(Enum):
    VISIBLE = "vz"
    INFRARED = "ir"
    HYBRID = "hy"

class TempTrainingContext:
    def __init__(self, filepaths: list[str], modality: Modality, train_idx: list[int], test_idx: list[int]):
        self.temp_dir = None
        self.filepaths = filepaths
        self.modality = modality
        self.train_idx = train_idx
        self.test_idx = test_idx

        if modality == Modality.VISIBLE:
            self.img_filepaths = self.filepaths
        elif modality == Modality.INFRARED:
            self.img_filepaths = [path.replace("vz", "ir") for path in self.filepaths]        
        elif modality == Modality.HYBRID:
            split_idx = int(len(filepaths)/2)
            self.img_filepaths = self.filepaths[:split_idx] + [path.replace("vz", "ir") for path in self.filepaths[split_idx:]]   

        self.label_filepaths = [path.replace("images", "labels").replace(".jpg", ".txt") for path in self.img_filepaths]

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()

        dataset_name = self.temp_dir.split('/')[-1]

        for split, idx in zip(["train", "val"],[self.train_idx, self.test_idx]):

            split_images = [self.img_filepaths[i] for i in range(len(self.img_filepaths)) if i in idx]
            split_labels = [self.label_filepaths[i] for i in range(len(self.label_filepaths)) if i in idx] 

            images_dir = os.path.join(self.temp_dir, split, "images")
            labels_dir = os.path.join(self.temp_dir, split, "labels")

            os.makedirs(images_dir)
            os.makedirs(labels_dir)

            for img, lbl in zip(split_images, split_labels):
                img_name = img.split('/')[-1]
                lbl_name = lbl.split('/')[-1]

                shutil.copy2(img, os.path.join(images_dir, img_name))
                shutil.copy2(lbl, os.path.join(labels_dir, lbl_name))

        data = {
            'path': self.temp_dir,
            'train': 'train',
            'val': 'val',
            'names': [{0: 'uav'}]
        }

        with open(os.path.join(self.temp_dir, 'cfg.yaml'), 'w') as outfile:
            yaml.dump(data, outfile)

        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

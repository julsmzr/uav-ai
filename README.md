# Evaluation of YOLOv12 for Multi-Modal Real-Time Visual UAV Detection and Tracking

This project was created during the Advanced Topics in Artificial Intelligence course at PWR under supervision of Maciej Huk

## Development

### Setup

```bash
git submodule update --init

python3 -m venv venv
source venv/bin/activate

python3 -m pip install -r requirements.txt
python3 -m pip install -e models/ultralytics/ultralytics
```

### Notebooks
To replicate experiments and results, use the following notebooks.

1. <b>notebooks/1_setup.ipynb </b> in order to set up thPe Anti-UAV300 dataset
2. <b>notebooks/2_experiments.ipynb</b> to replicate the ran experiments and 
3. <b>notebooks/3_evaluation.ipynb</b> to run the statistical analysis.

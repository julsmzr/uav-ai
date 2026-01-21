# Evaluation of YOLOv12 for Multi-Modal Real-Time Visual UAV Detection and Tracking

This project was created during the Advanced Topics in Artificial Intelligence course at PWR under supervision of Maciej Huk. The final project report can be found in [docs/project_report.pdf](docs/project_report.pdf).

## Research Summary

### Objective
Comparative analysis of YOLOv12-nano performance across **Visible (VZ)**, **Infrared (IR)**, and **Hybrid (HY)** modalities for real-time UAV detection on the Anti-UAV300 dataset.

### Methodology
- **Dataset**: Anti-UAV300 (318 sequences, 593,802 frames, ~96.6% annotated)
- **Model**: YOLOv12-nano pre-trained on COCO
- **Validation**: Repeated Stratified K-Fold (5 splits × 5 repeats = 25 folds)
- **Training**: 5 seeded weight initializations × 25 folds = **125 training runs per modality** (375 total)
- **Statistical Analysis**: Friedman test, Wilcoxon signed-rank test with Hommel correction, effect size (η²)
- **Multi-implementation verification**: SciPy, R, Pingouin, STAC, Statsmodels

### Key Results

| Metric | Visible (VZ) | Infrared (IR) | Hybrid (HY) |
|--------|--------------|---------------|-------------|
| Precision | **96.53 ± 0.57%** | **96.53 ± 0.54%** | 96.32 ± 0.63% |
| Recall | **94.01 ± 0.60%** | **93.94 ± 0.66%** | 93.69 ± 0.78% |
| mAP50 | **96.69 ± 0.39%** | **96.62 ± 0.59%** | 96.44 ± 0.58% |
| mAP50-95 | **60.20 ± 0.95%** | **60.15 ± 0.96%** | 59.60 ± 1.11% |

**Statistical Analysis:**
- **Modality Equivalence**: VZ and IR achieved statistically equivalent performance across all metrics (*p* > 0.05)
- **Significant Difference**: Only mAP50-95 showed overall modality effect (Friedman *p* = 0.013, η² = 0.173 - large effect)
- **Pairwise Comparisons**: No pairwise differences survived Hommel correction after multiple testing adjustment
- **Hybrid Degradation**: Hybrid modality showed subtle performance decrease, particularly in localization accuracy

### Conclusion
**Single-modality systems (either Visible or Infrared) are sufficient and preferable to naive multi-modal fusion.** Infrared performs equivalently to RGB, validating thermal imaging as a viable alternative under low-light conditions. The interleaved Hybrid approach offers no benefit and may degrade localization accuracy without dedicated fusion architectures.

## Development

### Setup

```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -r requirements.txt
python3 -m pip install -e models/ultralytics/ultralytics
```

### Notebooks
To replicate experiments and results, use the following notebooks:

1. **1_setup.ipynb** - Downloads and processes the Anti-UAV300 dataset
2. **2_experiments.ipynb** - Runs training experiments across all modalities
3. **3_evaluation.ipynb** - Performs statistical analysis and generates visualizations

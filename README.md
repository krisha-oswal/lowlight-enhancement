# Low-Light Image Enhancement for Object Detection: A Task-Driven Evaluation Framework

## Project Overview
This research framework systematically evaluates whether classical and learned low-light image enhancement methods improve downstream object detection performance on the ExDark dataset.

## Research Questions
1. Does enhancement consistently improve detection (mAP)?
2. Which methods work best under what conditions?
3. Does perceptual quality (PSNR/SSIM) correlate with task performance (mAP)?
4. Can we automatically select the best enhancement for a given image?

## Project Structure
```
lowlight_detection/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── data/
│   └── exdark_loader.py         # ExDark dataset loader & annotation parser
├── preprocessing/
│   ├── base_enhancer.py         # Abstract enhancer interface
│   ├── histogram_eq.py          # Histogram Equalization
│   ├── clahe.py                 # CLAHE
│   ├── gamma_correction.py      # Gamma Correction
│   ├── retinex.py               # Retinex (MSR + SSR)
│   ├── zero_dce.py              # Zero-DCE (deep learning)
│   └── enhancer_factory.py      # Factory pattern for enhancers
├── detection/
│   └── yolov8_detector.py       # YOLOv8 inference pipeline
├── evaluation/
│   ├── metrics.py               # mAP, Precision, Recall computation
│   └── image_characterizer.py   # Brightness, contrast, entropy, noise
├── selector/
│   └── enhancement_selector.py  # Oracle labeling + ML selector training
├── visualization/
│   └── plot_results.py          # All plots and figures
├── utils/
│   └── io_utils.py              # Caching, saving, loading results
├── run_experiment.py            # Main experiment orchestrator
├── train_selector.py            # Train the enhancement selector
├── analyze_results.py           # Generate all analysis and plots
└── requirements.txt
```

## Setup & Installation
```bash
pip install -r requirements.txt
```

## Running the Experiments

### Step 1: Run the full detection pipeline
```bash
python run_experiment.py --config configs/config.yaml
```

### Step 2: Train the enhancement selector
```bash
python train_selector.py --config configs/config.yaml
```

### Step 3: Generate analysis & plots
```bash
python analyze_results.py --config configs/config.yaml
```

## Dataset
Download ExDark from: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
Place in: `data/ExDark/`

Expected structure:
```
data/ExDark/
├── images/
│   ├── Bicycle/
│   ├── Boat/
│   └── ...
└── annotations/
    ├── Bicycle/
    └── ...
```

## Enhancement Methods
| Method | Type | Key Parameter |
|--------|------|---------------|
| Baseline | None | - |
| Histogram Equalization | Classical | - |
| CLAHE | Classical | clip_limit=2.0, tile=(8,8) |
| Gamma Correction | Classical | gamma=1.5 |
| Retinex (MSR) | Classical | sigma=[15,80,250] |
| Zero-DCE | Deep Learning | pretrained weights |

## Citation
If you use this framework, please cite:
```
[Your paper citation here]
```

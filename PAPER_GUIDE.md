# Research Paper Writing Guide
## "Does Low-Light Image Enhancement Actually Improve Object Detection?"
### IEEE Transactions Format — Detailed Outline

---

## 1. ABSTRACT (150–250 words)

Structure:
1. **Motivation** (1–2 sentences): Low-light image enhancement is typically evaluated using perceptual metrics (PSNR, SSIM), yet real-world deployment demands task performance. The disconnect between perceptual quality and detection accuracy remains understudied.
2. **Method** (2–3 sentences): We systematically evaluate five enhancement methods (HE, CLAHE, Gamma, Retinex-MSR, Zero-DCE) on the ExDark dataset using YOLOv8 as the detector. We compute mAP@0.5, precision, and recall per-method, per-class, and per image condition.
3. **Novel contribution** (1–2 sentences): We propose a lightweight Enhancement Selector that predicts the best preprocessing method from image features alone, without running detection.
4. **Key finding** (1–2 sentences): Enhancement does not uniformly improve detection; the benefit is highly condition-dependent. CLAHE and Gamma provide the most consistent gains; perceptual metrics poorly predict detection performance (r < 0.3).
5. **Conclusion** (1 sentence): Our selector reduces the need for exhaustive trial and approaches oracle performance.

---

## 2. INTRODUCTION

**Paragraph 1 — Hook**: Object detectors deployed in surveillance, autonomous driving, and robotics must operate in degraded illumination. Low-light conditions reduce feature contrast, increase noise, and shift pixel distributions away from training data.

**Paragraph 2 — Prior work gap**: Existing low-light enhancement literature (RetinexNet, Zero-DCE, LIME, MBLLEN) optimizes for PSNR/SSIM against reference images. But PSNR is a pixel-fidelity metric, not a task metric. A detector may perform worse on a perceptually better image if enhancement destroys structural gradients.

**Paragraph 3 — Problem statement**: We pose the question: *Does low-light enhancement actually help downstream object detection, and if so, which method works best for which image conditions?*

**Paragraph 4 — Contributions** (use bullet list in paper):
- Systematic task-driven evaluation on ExDark using YOLOv8 (6 methods × 3,000+ images × 12 classes)
- Correlation study: PSNR/SSIM vs mAP reveals perceptual quality ≠ detection quality
- Failure case taxonomy: conditions under which enhancement degrades detection
- Enhancement Selector: a lightweight ML model predicting best method from image features

**Paragraph 5 — Paper structure**: "The rest of this paper is organized as follows..."

---

## 3. RELATED WORK

### 3.1 Low-Light Image Enhancement
- **Classical methods**: HE (Pizer, 1987), CLAHE (Zuiderveld, 1994), gamma correction, Retinex (Land & McCann, 1971; Jobson et al., 1997)
- **Learning-based methods**: LLNet (Lore et al., 2017), RetinexNet (Wei et al., 2018), Zero-DCE (Guo et al., 2020), MBLLEN (Lv et al., 2018), SNR-Aware (Xu et al., 2022)
- **Evaluation metrics**: PSNR, SSIM, LPIPS, NIQE, BRISQUE

### 3.2 Object Detection in Low Light
- Two-stage: PANDA, NightOwls dataset evaluations
- One-stage: YOLO variants, DETR
- Domain adaptation for nighttime: Dark Face, ExDark benchmarks

### 3.3 Preprocessing for Detection
- Prior work typically treats enhancement as an isolated module
- Few studies close the loop between enhancement quality and detection mAP
- Cite: "Enhancement-Detection closed-loop evaluation" studies if available

### 3.4 Enhancement Selection
- Cite: NAS-based preprocessing selection, reinforcement learning for pipeline selection
- Our method: supervised oracle-based classifier (simpler, more interpretable)

---

## 4. DATASET & EXPERIMENTAL SETUP

### 4.1 ExDark Dataset
- 7,363 images, 12 object categories, 10 illumination conditions
- Conditions: indoor (lamp), outdoor (natural night, street light, etc.)
- Ground truth: bounding box annotations, per-image illumination metadata

### 4.2 Enhancement Methods
Present Table 1: Method | Type | Key Parameters | Complexity (FLOPs or ms/image)

### 4.3 Detection Model
- YOLOv8n pretrained on COCO, inference-only (no fine-tuning)
- Design justification: using pretrained COCO weights tests whether enhancement bridges the domain gap between dark ExDark and bright COCO training data
- Fixed: conf=0.25, IoU=0.45, input size 640

### 4.4 Evaluation Protocol
- Primary: mAP@0.5 (VOC protocol, area-under-PR-curve)
- Secondary: Precision, Recall (per-image and aggregate)
- Quality: PSNR, SSIM (enhanced vs. original)
- Image features: 22-dimensional characterization vector

---

## 5. METHODOLOGY

### 5.1 Enhancement Pipeline
Diagram (Fig. 1): Input image → Enhancement → YOLOv8 → Detections → mAP

### 5.2 Image Characterization
List 22 features with equations:
- Brightness: μ = (1/N) Σ I(x,y)
- Entropy: H = -Σ p_i log₂(p_i)
- Noise: σ²_L = Var(∇²I) / 36
- etc.

### 5.3 Perceptual Quality Metrics
- PSNR(O, E) = 20·log₁₀(255 / √MSE(O,E))
- SSIM equation

### 5.4 Enhancement Selector
**Fig 2**: System diagram: Image → Feature Extraction → Classifier → Best Method → Enhanced → Detector

Oracle labeling: L*(i) = argmax_m recall(i, m)

Training: RF, XGBoost, MLP with 5-fold CV. Feature matrix X ∈ ℝ^{N×22}, labels y ∈ {baseline, HE, CLAHE, Gamma, Retinex, Zero-DCE}^N

---

## 6. EXPERIMENTS & RESULTS

### 6.1 Overall Detection Performance (Table 2)
Columns: Method | mAP@0.5 | Precision | Recall | PSNR (vs original)
Best numbers in bold.

### 6.2 Per-Class Analysis (Table 3 + Fig. heatmap)
Key observations:
- Which classes benefit most from which method?
- Example: "People" benefits from CLAHE (local contrast in dark scenes)
- "Bottle" may be hurt by Retinex (color distortion of transparent objects)

### 6.3 Correlation Study (Fig. scatter plots)
Report: Pearson r between PSNR and mAP, SSIM and mAP, per-method and pooled.
Key claim: "PSNR explains less than X% of variance in detection recall (r² < 0.1)"

### 6.4 Failure Case Analysis (Fig. bar chart)
- HE fails most on: already-bright images (over-saturation)
- Retinex fails on: high-contrast scenes (halos around edges)
- Zero-DCE (random weights): may perform comparably on some conditions

### 6.5 Enhancement Selector (Table 4)
Columns: Classifier | CV Acc | Test Acc | mAP Gap vs Oracle

"Our best selector (Random Forest) achieves X% accuracy, reducing the performance gap with the oracle from Y% to Z%"

---

## 7. DISCUSSION

### 7.1 Does Enhancement Help?
- Main finding: YES, but conditionally. ~X% of images benefit; ~Y% are neutral; ~Z% are hurt.
- Enhancement helps most when: dark_pixel_ratio > 0.6, brightness_mean < 60
- Enhancement hurts most when: noise_variance > threshold (amplifies noise), image already has good contrast

### 7.2 Perceptual vs. Task Quality
- PSNR/SSIM are poor proxies for detection performance
- Zero-DCE achieves high PSNR changes but not always better mAP
- This challenges the assumption of using perceptual benchmarks for task-driven evaluation

### 7.3 Selector Analysis
- Most informative features: dark_pixel_ratio, entropy, brightness_mean
- Selector gap to oracle: interpret as "headroom for future improvement"
- Practical value: selector adds <1ms overhead, vs running all 6 methods

### 7.4 Limitations
- YOLOv8 pretrained on COCO (day images): fine-tuning on ExDark would change results
- Zero-DCE without pretrained weights is suboptimal
- Selector trained on ExDark; generalization to other datasets untested

---

## 8. CONCLUSION

Summary of contributions, findings, and practical implications. Future work:
- End-to-end trainable enhancement+detection
- Reinforcement learning for adaptive enhancement selection
- Extension to video (temporal consistency)

---

## FIGURES & TABLES CHECKLIST

| # | Type | Description |
|---|------|-------------|
| Fig. 1 | Pipeline diagram | Full system: image→enhance→detect→evaluate |
| Fig. 2 | Selector diagram | Feature extraction → classifier → prediction |
| Fig. 3 | Bar chart | mAP comparison across methods |
| Fig. 4 | Heatmap | Per-class AP × method |
| Fig. 5 | Scatter | PSNR vs Recall (correlation study) |
| Fig. 6 | Scatter | SSIM vs Recall |
| Fig. 7 | Bar chart | Failure case counts per method |
| Fig. 8 | Confusion matrix | Selector predictions |
| Fig. 9 | Bar chart | Feature importances |
| Fig. 10 | Bar chart | Oracle/Selector/Fixed/Baseline comparison |
| Table 1 | Table | Enhancement methods overview |
| Table 2 | Table | Overall detection metrics |
| Table 3 | Table | Per-class AP by method |
| Table 4 | Table | Selector results |

---

## IEEE FORMAT NOTES

- Use IEEEtran.cls document class
- Two-column format, 10pt font
- Figures: at least 300 DPI, saved as PDF or EPS
- Citations: numeric style [1], use IEEEbib.bst
- Math in dedicated equation environments
- Algorithm pseudocode in algorithm2e or algorithmicx

## KEY REFERENCES TO CITE

1. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," CVPR 2020
2. Jobson et al., "A multiscale retinex for bridging the gap between color images and the human observation of scenes," IEEE TIP, 1997
3. Zuiderveld, "Contrast limited adaptive histogram equalization," Graphics Gems IV, 1994
4. Redmon & Farhadi, "YOLOv3: An Incremental Improvement," arXiv 2018
5. Jocher et al., "Ultralytics YOLOv8," GitHub 2023
6. Loh & Chan, "Getting to Know Low-light Images with the Exclusively Dark Dataset," CVIU 2019
7. Wang et al., "Image quality assessment: From error visibility to structural similarity," IEEE TIP, 2004

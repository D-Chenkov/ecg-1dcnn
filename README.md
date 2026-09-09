# ecg-1dcnn - Project #3 (biosignals)

A 1D-ResNet that classifies ECG heartbeats into the 5 AAMI arrhythmia classes, built on the MIT-BIH heartbeat dataset. Portfolio Project #3 of my QA-Automation to ML-Engineer transition; connects a CNN background to a deployable biosignals model, reusing the MLOps stack from Project #1 (MLflow, FastAPI, Docker).

STATUS: trained + evaluated (macro-F1 0.914 on test); served via FastAPI, containerized.

## Data
"ECG Heartbeat Categorization Dataset" (Kachuee et al.) on Kaggle. Download `mitbih_train.csv` and `mitbih_test.csv` into `data/` (gitignored). Each row = 187 signal samples of one heartbeat + a class label (0=N normal, 1=S supraventricular, 2=V ventricular, 3=F fusion, 4=Q unknown). The set is heavily imbalanced (class N dominates, ~83%).

## Structure
```
ecg-1dcnn/
  data/                 the CSVs (gitignored)
  notebooks/            preprocessing + EDA
  src/
    dataset.py          ECGBeatDataset (loads the CSV)
    model.py            ECG1DCNN (1D-ResNet)
    train.py            training loop (seed, val split, weighting, early stop, AMP, scheduler) + MLflow
    evaluate.py         per-class precision/recall + confusion matrix
  app/main.py           FastAPI /predict + /health
  tests/test_app.py     pytest (hermetic: throwaway model)
  Dockerfile            non-root serving image
  requirements.txt
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the Kaggle CSVs into data/
python src/train.py --epochs 40 --batch_size 512 --weighting class_balanced
python src/evaluate.py                       # per-class report on test (run once)
```

## Serving
```bash
uvicorn app.main:app --port 8080             # needs ecg_1dcnn.pth (from train.py)
curl -X POST localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"signal": [ ... 187 floats ... ]}'
# {"class_id": 0, "class_code": "N", "class_name": "normal", "confidence": 0.99}

pytest -q                                    # API tests (no trained model needed)
docker build -t ecg-1dcnn:v1 . && docker run -p 8080:8080 ecg-1dcnn:v1
```

## Model card
- **Task:** 5-class ECG heartbeat / arrhythmia classification (AAMI: N, S, V, F, Q).
- **Dataset + preprocessing:** Kaggle MIT-BIH heartbeat set (Kachuee et al.); beats pre-segmented and resampled to 187 samples, values ~[0,1]. Minority-class augmentation at train time (amplitude scale + Gaussian noise + small time shift). Class-balanced loss weighting (Cui et al. effective number).
- **Architecture:** 1D-ResNet - conv stem (k7) + MaxPool, then 3 residual blocks (32 -> 64 -> 128, stride-2 downsampling) with skip connections, global average pool, linear head. Trained with Adam, mixed precision (AMP), ReduceLROnPlateau, and early stopping on validation macro-F1 (stopped at epoch 67).
- **Metrics (test set, 21,892 beats) - lead with macro-F1, not accuracy:**

  | Class | Precision | Recall | F1 | Support |
  |-------|-----------|--------|-----|---------|
  | N (normal) | 0.989 | 0.997 | 0.993 | 18,118 |
  | S (supraventricular) | 0.919 | 0.759 | 0.832 | 556 |
  | V (ventricular) | 0.971 | 0.957 | 0.964 | 1,448 |
  | F (fusion) | 0.840 | 0.747 | 0.791 | 162 |
  | Q (unknown) | 0.994 | 0.986 | 0.990 | 1,608 |
  | **macro avg** | **0.943** | **0.889** | **0.914** | |
  | accuracy | | | 0.986 | |

  Selection was done on a validation split; the test set was evaluated once. macro-F1 improved from 0.715 (shallow CNN, no augmentation) to 0.914 after adding residual depth + minority augmentation - loss weighting alone did not move it (verified in a controlled comparison).
- **Limitations (important):**
  - **Intra-patient leakage.** The Kaggle split is a beat-level random split, so heartbeats from the same patient can appear in both train and test. This inflates scores relative to a **patient-wise (inter-patient / AAMI) split**, where no patient is shared. Expect materially lower numbers under a patient-wise evaluation; the 0.986 accuracy should be read as "for this benchmark split," not as unseen-patient performance.
  - Single-lead, pre-segmented beats (no raw-signal detection/segmentation in the loop).
  - Rare classes (S, F) still have the lowest recall; not a clinical-grade model - a screening/portfolio demonstration.

## Roadmap
- [x] Repo scaffold
- [ ] Preprocessing notebook + EDA (class balance, example beats)  # outline in notebooks/
- [x] Train 1D-ResNet, log to MLflow (seed, val split, early stopping, AMP)
- [x] Per-class precision/recall + confusion matrix (macro-F1 0.914)
- [x] FastAPI `/predict` + `/health` + Dockerfile + tests
- [ ] Patient-wise (inter-patient) split to report honest generalization # Just made a report on that, but not actually fixing here...yet.
- [ ] (stretch) raw-signal WFDB segmentation; ensembling / focal loss

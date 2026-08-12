# ecg-1dcnn - Project #3 (biosignals)

A 1D-CNN that classifies ECG heartbeats into the 5 AAMI arrhythmia classes, built on the MIT-BIH heartbeat dataset. Portfolio Project #3 of my QA-Automation to ML-Engineer transition; connects a CNN background to a deployable biosignals model, reusing the MLOps stack from Project #1 (MLflow, FastAPI, Docker).

STATUS: Iterating model and training loop; will deploy to FastAPI and Docker soon.

## Data
"ECG Heartbeat Categorization Dataset" (Kachuee et al.) on Kaggle. Download `mitbih_train.csv` and `mitbih_test.csv` into `data/` (gitignored). Each row = 187 signal samples of one heartbeat + a class label (0=N normal, 1=S supraventricular, 2=V ventricular, 3=F fusion, 4=Q unknown). The set is heavily imbalanced (class N dominates).

## Structure
```
ecg-1dcnn/
  data/                 the CSVs (gitignored)
  notebooks/            preprocessing + EDA
  src/
    dataset.py          ECGBeatDataset (loads the CSV)
    model.py            ECG1DCNN
    train.py            training loop + MLflow logging
    evaluate.py         per-class precision/recall + confusion matrix
  app/main.py           FastAPI serving stub (Week 7)
  requirements.txt
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the Kaggle CSVs into data/
python src/train.py --epochs 10
python src/evaluate.py
```

## Model card
- Dataset + preprocessing:
- Architecture: 1D-CNN (Conv1d/BatchNorm/ReLU/MaxPool -> global avg pool -> Linear)
- Metrics (PER CLASS - imbalanced data, so accuracy alone is misleading):
- Limitations:

## Roadmap
- [x] Repo scaffold
- [ ] Preprocessing notebook + EDA (class balance, example beats)
- [ ] Train 1D-CNN, log to MLflow
- [ ] Per-class precision/recall + confusion matrix (macro-F1, class weights)
- [ ] (Week 7) FastAPI /predict + Dockerfile + finish model card
- [ ] (stretch) raw-signal WFDB segmentation; ResNet-style skip connections for a deeper net

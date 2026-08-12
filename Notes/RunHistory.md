# Run History

## Run 1: Baseline

### Model
- 1D-CNN
- Conv1d(1, 32, kernel_size=5, padding=2)
- BatchNorm1d(32), ReLU(), MaxPool1d(2)
- Conv1d(32, 64, kernel_size=5, padding=2)
- BatchNorm1d(64), ReLU(), MaxPool1d(2)
- Conv1d(64, 128, kernel_size=3, padding=1)
- BatchNorm1d(128), ReLU()
- AdaptiveAvgPool1d(1)
- Linear(128, 5)

### Confusion Matrix
```
[[12850   598  4585     6    79]
 [   99   346   109     1     1]
 [    6     0  1440     0     2]
 [    2     0   154     6     0]
 [    3     0    65     0  1540]]
```

### Metrics
```
                precision    recall  f1-score   support

    N (normal)      0.992     0.709     0.827     18118
S (supravent.)      0.367     0.622     0.461       556
  V (ventric.)      0.227     0.994     0.369      1448
    F (fusion)      0.462     0.037     0.069       162
   Q (unknown)      0.949     0.958     0.954      1608

      accuracy                          0.739     21892
     macro avg      0.599     0.664     0.536     21892
  weighted avg      0.918     0.739     0.791     21892
```

### Notes:

Clearly overcorrected for the imbalanced dataset by heavily penalizing the minority classes.
Going to try to soften the weighting from raw inverse-frequency to something reasonable like sqrt-inverse frequency or class balanced. Let's see what works.


## Run 2: 

Model:

`` Same as before


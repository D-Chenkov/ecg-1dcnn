# Run History

## Run 1: Baseline 'raw inverse-frequency' weighting

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


## Run 2: 'sqrt-inverse frequency' weighting

Model:

`` Same as before

### Confusion Matrix

```
[[14375   732    79     1  2931]
 [  111   384     6     0    55]
 [   59    11  1160     0   218]
 [   15     0    12    56    79]
 [    8     2     1     0  1597]]
```

### Metrics
```
                precision    recall  f1-score   support

    N (normal)      0.987     0.793     0.880     18118
S (supravent.)      0.340     0.691     0.456       556
  V (ventric.)      0.922     0.801     0.857      1448
    F (fusion)      0.982     0.346     0.511       162
   Q (unknown)      0.327     0.993     0.492      1608

      accuracy                          0.803     21892
     macro avg      0.712     0.725     0.639     21892
  weighted avg      0.897     0.803     0.836     21892
```


### Notes: 
Still overcorrected, marginally better than before. Going to try to use class balanced weighting instead of sqrt-inverse frequency.

We lost normal precision, but gained normal recall.
Fusion got way better precision and recall. 
Macro avg precision and f1 score improved, however, weighted avg precision and f1 score decreased.

## Run 3: 'class balanced' weighting

Model:

`` Same as before

### Confusion Matrix
```
[[16004  1557   270   262    25]
 [   95   448     6     7     0]
 [   36    22  1346    43     1]
 [   11     0     8   143     0]
 [   33    17    20     7  1531]]
```

### Metrics
```
                precision    recall  f1-score   support

    N (normal)      0.989     0.883     0.933     18118
S (supravent.)      0.219     0.806     0.345       556
  V (ventric.)      0.816     0.930     0.869      1448
    F (fusion)      0.310     0.883     0.458       162
   Q (unknown)      0.983     0.952     0.967      1608

      accuracy                          0.889     21892
     macro avg      0.663     0.891     0.715     21892
  weighted avg      0.953     0.889     0.913     21892
```

### Notes:

Class balanced weighting is working better than the other two.
Normal precision and recall are both very high.
Fusion precision and recall are very high.
Macro avg precision and f1 score are very high.
Weighted avg precision and f1 score are very high.

Going to try to use class balanced weighting for the final model.

## Next steps:

- Run more epochs with class balanced weighting.
- Try optimizers like Optuna, AdamW, etc.
- Try different models even deeper or wider or with more layers or different architecture.
- Just more experimentation in general.

## Run 4: 'class balanced' weighting with 100 epochs

Model:

`` Same as before

Specs:

-- GPU: RTX 4090
-- Batch size: 512
-- Learning rate: 0.001
-- Epochs: 100
-- Weight decay: 0.0001
-- Optimizer: AdamW
-- Scheduler: CosineAnnealingLR
-- Early stopping: 10 epochs without improvement


### Metrics
```
                precision    recall  f1-score   support

    N (normal)      0.989     0.997     0.993     18118
S (supravent.)      0.919     0.759     0.832       556
  V (ventric.)      0.971     0.957     0.964      1448
    F (fusion)      0.840     0.747     0.791       162
   Q (unknown)      0.994     0.986     0.990      1608

      accuracy                          0.986     21892
     macro avg      0.943     0.889     0.914     21892
  weighted avg      0.985     0.986     0.985     21892
```

### Confusion Matrix
```
confusion matrix (rows = true, cols = pred):
[[18062    29    18     4     5]
 [  128   422     5     0     1]
 [   34     7  1386    18     3]
 [   22     1    18   121     0]
 [   21     0     1     1  1585]]
 ```

 ### Notes

 Great improvements, we got the weighted average we needed. These numbers are fantastic! 
 early stop @ epoch 67 (no val gain in 7)

# TODO List

- [x] 重命名损失函数
- [x] test模块，重新写一个，包括log、结果保存、argparser
- [x] train模块，重新写一个，包括log、结果保存、argparser
- [x] 重构模型的输出，返回`dict`各自段重命名



```
[INFO] [139871478830912] 2025-02-16 19:54-train.py(line: 103) : Epoch[1/300] Iter[20/125] - Loss: 1.3884  Overlap Loss: 1.3710  Nonoverlap Loss: 0.0174  LR: 0.00010000  
[INFO] [139871478830912] 2025-02-16 19:55-train.py(line: 103) : Epoch[1/300] Iter[40/125] - Loss: 0.9565  Overlap Loss: 0.9560  Nonoverlap Loss: 0.0005  LR: 0.00010000  
[INFO] [139871478830912] 2025-02-16 19:55-train.py(line: 103) : Epoch[1/300] Iter[60/125] - Loss: 0.9121  Overlap Loss: 0.9121  Nonoverlap Loss: 0.0000  LR: 0.00010000  
[INFO] [139871478830912] 2025-02-16 19:56-train.py(line: 103) : Epoch[1/300] Iter[80/125] - Loss: 0.8344  Overlap Loss: 0.8344  Nonoverlap Loss: 0.0000  LR: 0.00010000  
[INFO] [139871478830912] 2025-02-16 19:56-train.py(line: 103) : Epoch[1/300] Iter[100/125] - Loss: 0.7992  Overlap Loss: 0.7992  Nonoverlap Loss: 0.0000  LR: 0.00010000  
[INFO] [139871478830912] 2025-02-16 19:57-train.py(line: 103) : Epoch[1/300] Iter[120/125] - Loss: 0.8344  Overlap Loss: 0.8344  Nonoverlap Loss: 0.0000  LR: 0.00010000  
[INFO] [139871478830912] 2025-02-16 19:57-train.py(line: 103) : Epoch[2/300] Iter[15/125] - Loss: 0.6342  Overlap Loss: 0.6342  Nonoverlap Loss: 0.0000  LR: 0.00009700  
[INFO] [139871478830912] 2025-02-16 19:58-train.py(line: 103) : Epoch[2/300] Iter[35/125] - Loss: 0.7374  Overlap Loss: 0.7374  Nonoverlap Loss: 0.0000  LR: 0.00009700  
[INFO] [139871478830912] 2025-02-16 19:58-train.py(line: 103) : Epoch[2/300] Iter[55/125] - Loss: 0.7643  Overlap Loss: 0.7643  Nonoverlap Loss: 0.0000  LR: 0.00009700  
[INFO] [139871478830912] 2025-02-16 19:59-train.py(line: 103) : Epoch[2/300] Iter[75/125] - Loss: 0.7631  Overlap Loss: 0.7631  Nonoverlap Loss: 0.0000  LR: 0.00009700  
[INFO] [139871478830912] 2025-02-16 19:59-train.py(line: 103) : Epoch[2/300] Iter[95/125] - Loss: 0.7844  Overlap Loss: 0.7844  Nonoverlap Loss: 0.0000  LR: 0.00009700  
[INFO] [139871478830912] 2025-02-16 19:59-train.py(line: 103) : Epoch[2/300] Iter[115/125] - Loss: 0.7525  Overlap Loss: 0.7524  Nonoverlap Loss: 0.0000  LR: 0.00009700  
[INFO] [139871478830912] 2025-02-16 20:00-train.py(line: 103) : Epoch[3/300] Iter[10/125] - Loss: 0.3806  Overlap Loss: 0.3806  Nonoverlap Loss: 0.0000  LR: 0.00009409  
[INFO] [139871478830912] 2025-02-16 20:00-train.py(line: 103) : Epoch[3/300] Iter[30/125] - Loss: 0.7681  Overlap Loss: 0.7563  Nonoverlap Loss: 0.0118  LR: 0.00009409  
[INFO] [139871478830912] 2025-02-16 20:01-train.py(line: 103) : Epoch[3/300] Iter[50/125] - Loss: 0.7217  Overlap Loss: 0.7217  Nonoverlap Loss: 0.0000  LR: 0.00009409  
[INFO] [139871478830912] 2025-02-16 20:01-train.py(line: 103) : Epoch[3/300] Iter[70/125] - Loss: 0.6828  Overlap Loss: 0.6827  Nonoverlap Loss: 0.0000  LR: 0.00009409  
[INFO] [139871478830912] 2025-02-16 20:02-train.py(line: 103) : Epoch[3/300] Iter[90/125] - Loss: 0.8041  Overlap Loss: 0.7583  Nonoverlap Loss: 0.0458  LR: 0.00009409  
[INFO] [139871478830912] 2025-02-16 20:02-train.py(line: 103) : Epoch[3/300] Iter[110/125] - Loss: 0.7111  Overlap Loss: 0.7111  Nonoverlap Loss: 0.0000  LR: 0.00009409  
[INFO] [139871478830912] 2025-02-16 20:03-train.py(line: 103) : Epoch[4/300] Iter[5/125] - Loss: 0.2072  Overlap Loss: 0.2072  Nonoverlap Loss: 0.0000  LR: 0.00009127  
[INFO] [139871478830912] 2025-02-16 20:03-train.py(line: 103) : Epoch[4/300] Iter[25/125] - Loss: 0.7194  Overlap Loss: 0.7193  Nonoverlap Loss: 0.0001  LR: 0.00009127  
[INFO] [139871478830912] 2025-02-16 20:03-train.py(line: 103) : Epoch[4/300] Iter[45/125] - Loss: 0.6660  Overlap Loss: 0.6658  Nonoverlap Loss: 0.0002  LR: 0.00009127  
[INFO] [139871478830912] 2025-02-16 20:04-train.py(line: 103) : Epoch[4/300] Iter[65/125] - Loss: 0.7277  Overlap Loss: 0.7277  Nonoverlap Loss: 0.0000  LR: 0.00009127  
[INFO] [139871478830912] 2025-02-16 20:04-train.py(line: 103) : Epoch[4/300] Iter[85/125] - Loss: 0.6508  Overlap Loss: 0.6508  Nonoverlap Loss: 0.0000  LR: 0.00009127  
[INFO] [139871478830912] 2025-02-16 20:04-train.py(line: 103) : Epoch[4/300] Iter[105/125] - Loss: 0.6665  Overlap Loss: 0.6665  Nonoverlap Loss: 0.0000  LR: 0.00009127
```


# Graphormer-Spatial  — HAR about NTU-RGB-D-120
A pure Graphormer implementation for Human Action Recognition (HAR) from 3-D skeletons.Trains with PyTorch ≥ 2.1 and runs on CPU, GPU.


---

## Advantages
  
-  Token Dropout and progressive unfreezing exposed through the CLI.
  
-  Resume from any checkpoint with --resume best_graphormer.pth.
  
-  Live LR hot-reload via hyper_lr.txt — no need to restart training.
  
-  Automatic confusion matrix and loss curves saved under ./metrics/.
  
- 
-  Multi-GPU support (DataParallel) and AMP mixed-precision ready.

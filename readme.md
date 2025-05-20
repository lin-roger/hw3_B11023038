# 機器學習-作業三

## 方法outline

### 預處理

1. 剪裁到only隆凸與氣切管
2. data augmentation:
    * horizontal flipping(p=0.5)
    * random radiation transformation(p=1.0)
    * Contrast-Limited Adaptive Histogram Equalization(p=0.5)

### model 

* 架構: U-Net, U-Net++, U2-Net
* $\mathcal{L}$: Focal Loss

### framework

pytorch
segmentation-models-pytorch
# UNet

[UNet](https://arxiv.org/abs/1505.04597) implementation for Semantic Segmentation using PyTorch

#### Train
* Run `python main.py` for training

#### Dataset structure (similar to CamVid dataset)
    ├── Dataset folder 
        ├── train
            ├── 1111.png
            ├── 2222.png
        ├── train_labels
            ├── 1111_L.png
            ├── 2222_L.png
        ├── class_dict.csv
 
#### Note 
* default feature extractor is [EfficientNetV2-S](https://arxiv.org/pdf/2104.00298.pdf)
* changing configuration of training, change parameters in `utils/config.py`
* default loss function is `weighted cross entropy`

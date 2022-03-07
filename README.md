# DeepSAR: Vessel Detection in SAR Imagery with Noisy Labels
This repository is the official implementation of [DeepSAR: Vessel Detection in SAR Imagery with Noisy Labels](https://github.com/manupillai308/DeepSAR). 

## Abstract
> Unlike traditional optoelectronic satellite imaging, Synthetic Aperture Radar (SAR) allows remote sensing applications to operate under all weather conditions. This makes it uniquely valuable for detecting ships/vessels involved in illegal, unreported, and unregulated (IUU) fishing. While recent work has shown significant improvement in this domain, detecting small objects using noisy point annotations remains an unexplored area. In order to meet the unique challenges of this problem, we propose a progressive training methodology that utilizes two different spatial sampling strategies. Firstly, we use stochastic sampling of background points to reduce the impact of class imbalance and missing labels, and secondly, during the refinement stage, we use hard negative sampling to improve the model. Experimental results on the challenging xView3 dataset show that our method outperforms conventional small object localization methods in a large, noisy dataset of SAR images.

![system_arch](/figures/system_arch.png)


## Requirements

To install requirements:

```shell
conda env create -f environment.yml
```

### Dataset
Download the [xView3](https://iuu.xview.us/) dataset and place it inside a directory with the following structure:

    .
    ├── train            
    │   ├── 00a035722196ee86t
    │   │   ├── bathymetry.tif
    │   │   ├── owiMask.tif
    │   │   ├── owiWindDirection.tif
    │   │   ├── owiWindQuality.tif
    │   │   ├── owiWindSpeed.tif
    │   │   ├── VH_dB.tif
    │   │   ├── VV_dB.tif
    │   ├── 00a035722196ee86t         
    │   └── ....
    ├── valid                    
    │   ├── 7b7e837a7ac5a880v
    │   └── ....
    ├── train.csv  
    └── validation.csv


## Training

To train the model in the paper with Progressive Training Strategy, run this command:

```shell
# stochastic sampling
python train-SS.py --result-dir "./result" --data-path "./dataset/train" --label-path "./dataset/train.csv"
```
```shell
# hard negative sampling
python train-HNS.py --result-dir "./result" --data-path "./dataset/train" --label-path "./dataset/train.csv" --finetune 0 --rpn-path "RPN_trained_model_3_epochs.pth"
```
```shell
# refinement
python train-HNS.py --result-dir "./result" --data-path "./dataset/train" --label-path "./dataset/train.csv" --finetune 1 --dn-path "DN1_trained_model_3_epochs.pth"
```

## Evaluation

To evaluate the model on xView3, run:

```shell
python evaluate.py --result-dir "./result" --data-path "./dataset/valid" --model-name "DN2_trained_model_3_epochs.pth"
```

## Results

All the evaluations are performed according to the metrics provided by the [xView3](https://iuu.xview.us/) Challenge. We test both small object localization models and generic object localization models.

### Quantitative

| Method        | Detection F1  | Close-to-Shore F1 | Vessel Classification F1 | Fishing Classification F1 |
| ------------------ |---------------- | -------------- | -------------- | -------------- |
| DeepLabv3   |       0.0068       |     0.0047        |   0.7177         |     0.1212       |
| DeepLabv3+   |      0.0102        |     0.0042        |   0.7709         |    0.1465        |
| DenseASPP  |      0.1893        |      0.0551       |      0.7138      |      0.2259      |
| PSPNet   |      0.4169        |     0.0741        |    0.7631        |      0.3838      |
| FarSeg   |      0.5103        |     0.0534        |     0.7796       |      0.5622      |
| FactSeg   |     0.5634         |    0.0677         |    0.8136        |     0.6586       |
| **DeepSAR (Ours)**   |    **0.6207**         |     **0.1081**        |    **0.8669**        |    **0.7701**        |


### Qualitative


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

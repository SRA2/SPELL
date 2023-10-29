:exclamation:**Please consider using the most recent version of our graph learning framework: [GraVi-T](https://github.com/IntelLabs/GraVi-T)**

# SPELL
Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection (ECCV 2022)\
[**paper**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950367.pdf) | [**poster**](https://drive.google.com/file/d/1q4ds3p1X7mfdpvROMYrBChrt2Zr55sfx/view?usp=sharing) | [**presentation**](https://youtu.be/wqb3crJ47KM)

## Overview
SPELL is a novel spatial-temporal graph learning framework for active speaker detection (ASD). It can model a minute-long temporal context without relying on computationally expensive networks. Through extensive experiments on the AVA-ActiveSpeaker dataset, we demonstrate that learning graph-based representations significantly improves the detection performance thanks to its explicit spatial and temporal structure. Specifically, SPELL outperforms all previous state-of-the-art approaches while requiring significantly lower memory and computation resources.

## Ego4D Challenges
SPELL and its improved version ([STHG](https://arxiv.org/abs/2306.10608)) achieved 1st place in the Ego4D Challenges [@ECCV22](https://ego4d-data.org/workshops/eccv22/) and [@CVPR23](https://ego4d-data.org/workshops/cvpr23/), respectively. We summarize ASD performance comparisons on the validation set of the Ego4D dataset:
|  ASD Model  |  ASD mAP(%)&#8593;  |  ASD mAP@0.5(%)&#8593;  |
|:------------|:-------------------:|:-----------------------:|
|  RegionCls  |   -                 |  24.6                   |
|  TalkNet    |   -                 |  50.6                   |
|  SPELL      |   71.3              |  60.7                   |
|  [STHG](https://arxiv.org/abs/2306.10608)  |   **75.7**   |  **63.7**  |

:bulb:In this table, We report two metrics to evaluate ASD performance: mAP quantifies the ASD results by assuming that the face bound-box detections are the ground truth (i.e. assuming the perfect face detector), whereas mAP@0.5 quantifies the ASD results on the detected face bounding boxes (i.e. a face detection is considered positive only if the IoU between a detected face bounding box and the ground-truth exceeds 0.5). For more information, please refer to our technical reports for the challenge.

:bulb:We computed mAP@0.5 by using [Ego4D's official evaluation tool](https://github.com/EGO4D/audio-visual/tree/main/active-speaker-detection/active_speaker/active_speaker_evaluation)

## ActivityNet 2022
SPELL achieved 2nd place in the [AVA-ActiveSpeaker Challenge](https://research.google.com/ava/challenge.html) at ActivityNet 2022. For the challenge, we used a visual input spanning a longer period of time (23 consecutive face-crops instead of 11). We also found that using a larger `channel1` can further boost the performance.\
[**tech report**](https://static.googleusercontent.com/media/research.google.com/en//ava/2022/S2_SPELL_ActivityNet_Challenge_2022.pdf) | [**presentation**](https://youtu.be/WCOOxsY0z34)

## Dependency
We used python=3.6, pytorch=1.9.1, and torch-geometric=2.0.3 in our experiments.

## Code Usage
1) Download the audio-visual features and the annotation csv files from [Google Drive](https://drive.google.com/drive/folders/1_vr3Wxf6yZRA3IjWgelnf0TQqzKzDNeu?usp=sharing). The directories should look like as follows:
```
|-- features
    |-- resnet18-tsm-aug
        |-- train_forward
        |-- val_forward
    |-- resnet50-tsm-aug
        |-- train_forward
        |-- val_forward
|-- csv_files
    |-- ava_activespeaker_train.csv
    |-- ava_activespeaker_val.csv
```

2) Run `generate_graph.py` to create the spatial-temporal graphs from the features:
```
python generate_graph.py --feature resnet18-tsm-aug
```
Although this script takes some time to finish in its current form, it can be modified to run in parallel and create the graphs for multiple videos at once. For example, you can change the `files` variable in line 81 of `data_loader.py`.

3) Use `train_val.py` to train and evaluate the model:
```
python train_val.py --feature resnet18-tsm-aug
```
You can change the `--feature` argument to `resnet50-tsm-aug` for SPELL with ResNet-50-TSM.

## Note
- We used the official code of [Active Speakers in Context (ASC)](https://github.com/fuankarion/active-speakers-context) to extract the audio-visual features (Stage-1). Specifically, we used `STE_train.py` and `STE_forward.py` of the ASC repository to train our two-stream ResNet-TSM encoders and extract the audio-visual features. We did not use any other components such as the postprocessing module or the context refinement modules. Please refer to `models_stage1_tsm.py` and the checkpoints from this [link](https://drive.google.com/drive/folders/1-EiPau0uzRA7pesuD5D-f6LZD6mxmYhz?usp=sharing) to see how we implanted the TSM into the two-stream ResNets.

## Citation
ECCV 2022 paper:
```bibtex
@inproceedings{min2022learning,
  title={Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  booktitle={European Conference on Computer Vision},
  pages={371--387},
  year={2022},
  organization={Springer}
}
```

Technical report for AVA-ActiveSpeaker challenge 2022:
```bibtex
@article{minintel,
  title={Intel Labs at ActivityNet Challenge 2022: SPELL for Long-Term Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  journal={The ActivityNet Large-Scale Activity Recognition Challenge},
  year={2022},
  note={\url{https://research.google.com/ava/2022/S2_SPELL_ActivityNet_Challenge_2022.pdf}}
}
```

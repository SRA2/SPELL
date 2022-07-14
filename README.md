# SPELL
Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection (ECCV 2022)\
**paper** | **poster** | **presentation** (will be updated soon)

## ActivityNet 2022 (AVA-ActiveSpeaker Challenge)
SPELL achieved 2nd place in the [AVA-ActiveSpeaker Challenge](https://research.google.com/ava/challenge.html) at ActivityNet 2022. For the challenge, we used a visual input spanning a longer period of time (23 consecutive face-crops instead of 11). We also found that using a larger `channel1` can further boost the performance.\
[**tech report**](https://static.googleusercontent.com/media/research.google.com/en//ava/2022/S2_SPELL_ActivityNet_Challenge_2022.pdf) | [**presentation**](https://youtu.be/WCOOxsY0z34)

## Dependency
We used python=3.6, pytorch=1.9.1, and torch-geometric=2.0.3 in our experiments.

## Code Usage
1) Download the audio-visual features and the annotation csv files from [Google Drive](https://drive.google.com/drive/folders/1fYALbElvIKjqeS8uGTHSeqtOhA6FXuRi?usp=sharing). The directories should look like as follows:
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

3) Use `train_val.py` to train and evaluate the model:
```
python train_val.py --feature resnet18-tsm-aug
```
You can change the `--feature` argument to `resnet50-tsm-aug` for SPELL with ResNet-50-TSM.

## Citation
ECCV 2022 paper:
```
will be updated soon
```

Technical report for AVA-ActiveSpeaker challenge 2022:
```bibtex
@article{minintel,
  title={Intel Labs at ActivityNet Challenge 2022: SPELL for Long-Term Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  year={2022}
}
```

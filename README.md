# Multimodal VB Fracture Detector: A framework for clinical multimodal predicitve models 

The multimodal VB fracture detector leverages different modalities, including fracture events from clinical notes extracted with [BERT-EE](https://github.com/wilsonlau-uw/BERT-EE), vertebral body (VB) classification results on radiographs from an imaging analysis pipeline, and patient demographics data to predict the presence of VB fractures at the patient-level. Specifically the task is a binary classification of fracture vs. no fracture. 

The model architectures in this repository are mainly a joint fusion-like and late fusion-like approach, as it leverages outputs from other models plus patient demographics data. In the joint fusion-like approach, each of the modalities described above were passed through its own respective neural network, and the output features were then concatenated prior to final classification. Multiple approaches were performed, such as backpropagating independent losses for each modality or applying attention mechanisms to update the weighs from each modality. Each of these model architectures can be found in `multimodal/joint_fusion.py`. In the late fusion-like approach, the outputs and patient demographics data were concatenated prior to passing to a neural network. Each of these model architectures can be found in `multimodal/late_fusion.py`.

![Joint Fusion CNN Losses](https://github.com/brian-cy-chang/Multimodal_VB-Fracture-Detector/blob/main/assets/jointfusion_cnn_losses.jpg?raw=true)

## Installation
The models require installing a local environment with `python>=3.9` and `torch>=2.2.2` and `torchvision>=0.17.2` using [Poetry](https://python-poetry.org/). Please follow the instructions below to set up the environment. 

```bash
git clone https://github.com/brian-cy-chang/Multimodal_VB-Fracture-Detector.git

cd Multimodal_VB-Fracture-Detector & poetry install
```
Once the environment is set up, activate the environment.
```bash
poetry shell
```

Note:
1. Please refer to the official [Poetry documentation](https://python-poetry.org/docs/) for installation.
2. Installing PyTorch requires the proper NVIDIA libraries and drivers. Please install the correct [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.

## Getting Started
Two modes are available: `train` or `predict`. Please refer to the configuration file, `config.ini`, for each parameter setting. Command line parameters will override those set in the configuration file.

Model training can be done *de novo* or from a model checkpoint specified in the configuration file.

```Python
python main.py \
    [--parameter] # to override config.ini parameters
```

## Acknowledgments
This framework leverages [BERT-EE](https://github.com/wilsonlau-uw/BERT-EE) to produce the fracture events from clinical notes.

@article{Wilson-Lau-2021-event-extraction,
    title = "Event-based clinical findings extraction from radiology reports with pre-trained language model",
    author = "Wilson Lau, Kevin Lybarger, Martin L. Gunn, Meliha Yetisgen",    
    url = "https://link.springer.com/article/10.1007/s10278-022-00717-5"
    }




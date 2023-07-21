# Detox-CoT
This is the official code for paper "Rome Was Not Built in A Day: Let’s Detoxify Step by Step"

# Overview
<p align="center"><img src="./assets/detox_chain.png" alt="Logo"></p>

# Quick Start

### Requirements
```
conda env create -f environment.yaml
```

### Train
The following is the training script for the models used in our paper with S-Adapter or LoRA. Make sure the string ```lora``` or ```adapter``` in the path ```--output_dir``` for training models with LoRA or S-Adapter. We conduct the experiment with 8 NVIDIA-A100(40GB).
* LlaMA & Alpaca
```
cd llama
sh ./train.sh
```

* FLAN-T5 XL
```
cd flan_t5_xl
sh ./train.sh
```

* GPT2 XL
```
cd gpt_xl
sh ./train.sh
```

### Inference
```
cd inference
sh ./inference.sh
```

### Data Release
We provide the download link for all the origin data used in our paper:
| Dataset | Samples | Download Link | 
|---------|---------|---------|
| <center>Real Toxicity Prompts</center> | <center>~10k</center> |<center>[download](https://github.com/allenai/real-toxicity-prompts)</center>|
| <center>Jigsaw Toxic Comment Classification Challenge</center> | <center>~16k(Train)</center> |<center>[download](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)</center>|

Our chain-of-thought datasets for all models are coming soon.



### Model Release
Coming Soon

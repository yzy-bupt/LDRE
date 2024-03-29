# LDRE

### LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval

This is the **anonymous  repository** of the paper "***LDRE**: **L**LM-based **D**ivergent **R**easoning and **E**nsemble for Zero-Shot Composed Image Retrieval*".

## Overview

![framework](https://github.com/yzy-bupt/LDRE/blob/main/images/framework.png)

### Abstract

Zero-Shot Composed Image Retrieval (ZS-CIR) has garnered increasing interest in recent years, which aims to retrieve a target image based on a query composed of a reference image and a modification text without training samples. Specifically, the modification text describes the distinction between the two images. To conduct ZS-CIR, the prevailing methods employ pre-trained image-to-text models to transform the query image and text into a single text, which is then projected into the common feature space by CLIP to retrieve the target image. However, these methods neglect that ZS-CIR is a typical *fuzzy retrieval* task, where the semantics of the target image are not strictly defined by the query image and text. To overcome this limitation, this paper proposes a training-free LLM-based Divergent Reasoning and Ensemble (LDRE) method for ZS-CIR to capture diverse possible semantics of the composed result. Firstly, we employ a pre-trained captioning model to generate dense captions for the reference image, focusing on different semantic perspectives of the reference image. Then, we prompt Large Language Models (LLMs) to conduct divergent compositional reasoning based on the dense captions and modification text, deriving divergent edited captions that cover the possible semantics of the composed target. Finally, we design a divergent caption ensemble to obtain the ensemble caption feature weighted by semantic relevance scores, which is subsequently utilized to retrieve the target image in the CLIP feature space. Extensive experiments on three public datasets demonstrate that our proposed LDRE achieves the new state-of-the-art performance. The code is publicly available at https://anonymous.4open.science/r/LDRE-66A2.

## Getting Started

We recommend using the [**Anaconda**](https://www.anaconda.com/) package manager to avoid dependency/reproducibility problems.
For Linux systems, you can find a conda installation guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Installation

1. Clone the repository, click  `Download file`

2. Install Python dependencies

```sh
conda create -n LDRE -y python=3.8.18
conda activate LDRE
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
pip install transformers==4.26.1 tqdm==4.66.1 openai==0.28 salesforce-lavis==1.0.2 open_clip_torch==2.24.0
pip install git+https://github.com/openai/CLIP.git
```

### Data Preparation

#### CIRCO

Download the CIRCO dataset following the instructions in the [**official repository**](https://github.com/miccunifi/CIRCO).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRCO
│   ├── annotations
|   |   ├── [val | test].json

│   ├── COCO2017_unlabeled
|   |   ├── annotations
|   |   |   ├──  image_info_unlabeled2017.json
|   |   ├── unlabeled2017
|   |   |   ├── [000000243611.jpg | 000000535009.jpg | ...]
```

#### CIRR

Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

#### FashionIQ

Download the FashionIQ dataset following the instructions in the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── images
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]
```

## LDRE

This section provides instructions for reproducing the results of the LDRE method.

### 1. Dense Caption Generation

In order to ensure comprehensive coverage of possible semantics of the composed results for fuzzy retrieval, we propose a dense caption generator to generate dense  captions. To obtain dense captions and avoid repetition, we employ nucleus sampling during the caption generation process to enhance the diversity of the generated captions.

Run the following command to generate dense  captions:

```sh
python src/dense_caption_generator.py
```

### 2. Multi-Prompt Editing Reasoning

We harness the reasoning capabilities of existing LLMs. Instead of merging the reference image caption and modification text by a fixed template, our objective is to derive cohesive, unified, and divergent edited captions by LLMs.

Run the following command to reason and edit for combination:

```sh
python src/reasoning_and_editing.py
```

### 3. Divergent Caption Ensemble

To effectively integrate complementary information in divergent edited captions and filter out noise, we design a semantic relevance scorer to measure the relevance scores of edited captions and use them as weights to combine the final ensemble caption feature. To generate the predictions file to be uploaded on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) or on the [CIRCO Evaluation Server](https://circo.micc.unifi.it/) run the following command:

```sh
python src/divergent_caption_ensemble.py
```

The predictions file will be saved in the `data/test_submissions/{dataset}/` folder.

We have provided the experimental results of our LDRE for your evaluation on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) or on the [CIRCO Evaluation Server](https://circo.micc.unifi.it/), in the `data/test_submissions/{dataset}/` folder, which have achieved state-of-the-art (SOTA) results as shown in the paper.

### 4. Qualitative Results

Qualitative results on the test set of CIRCO:

![framework](https://github.com/yzy-bupt/LDRE/blob/main/images/examples.png)
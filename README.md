# Behind the Magic, MERLIM: Multi-modal Evaluation Benchmark for Large Image-Language Models

Large Vision and Language Models have enabled significant advances in fully supervised and zero-shot visual tasks. These large architectures serve as the baseline to what is currently known as Instruction Tuning Large Vision and Language models (IT-LVLMs). IT-LVLMs are general-purpose multi-modal assistants whose responses are modulated by natural language instructions and visual data. Despite this versatility, IT-LVLM effectiveness in fundamental computer vision problems remains unclear, primarily due to the absence of a standardized evaluation benchmark. This paper introduces a Multi-modal Evaluation Benchmark named MERLIM, a scalable test-bed to assess the capabilities of IT-LVLMs on fundamental computer vision tasks. MERLIM contains over 300K image-question pairs and has a strong focus on detecting cross-modal “hallucination” events in IT-LVLMs. Our results bring important insights on the performance of state-of-the-art IT-LVMLs including limitations at identifying fine-grained visual concepts, object hallucinations across tasks, and biases towards the language query. Our findings also suggest that these models have weak visual grounding, but manage to make adequate guesses from global visual patterns or language biases contained in the LLM component.

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain a third-party dataset (MS-COCO), models, and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.

## Contents
- [Download Dataset](#download-dataset)
- [Install](#install)
- [Evaluation](#evaluation)

## Download Dataset

We provide the edited version of the images of MS-COCO Validation, which we computed to evaluate the hidden hallucinations of the models. We encourage you to download the corresponding original images using the official MS-COCO library. We provide a script to facilitate this process. 

1. Download the edited images using the following link.
From the command line (Linux)
```bash
wget https://p-lux3.pcloud.com/cBZ0MHubAZgVjTAR7ZZZ70Yc7kZ2ZZ89JZkZyxE68QZd4ZIpZd8ZdHZEzZszZgLZDQZq8ZGLZQ4ZyHZzQZlpZLSlu0ZtXYmuw9KBSFicN5G95DEFSky8R1V/coco_aug.zip
```
Also, It can be downloaded manually from [this website](https://u.pcloud.link/publink/show?code=kZLSlu0ZAvE3vnlBm5LExM9TCG4AlLttiNHy)

2. Unzip the file to extract the images.
```bash
unzip coco_aug.zip
```
3. Download the validation instance file of COCO (instances_val2017.json) from the official website.
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
4. Install the official MS-COCO library.
```bash
pip install pycocotools
```
5. Run the code to download the original validation image.
```bash
python download_data.py --img_dst_dir 'DESTINATION PATH FOR THE IMAGES' --annFile 'PATH TO THE FILE (instances_val2017.json)'
```

# Behind the Magic, MERLIM: Multi-modal Evaluation Benchmark for Large Image-Language Models

Large Vision and Language Models have enabled significant advances in fully supervised and zero-shot visual tasks. These large architectures serve as the baseline to what is currently known as Instruction Tuning Large Vision and Language models (IT-LVLMs). IT-LVLMs are general-purpose multi-modal assistants whose responses are modulated by natural language instructions and visual data. Despite this versatility, IT-LVLM effectiveness in fundamental computer vision problems remains unclear, primarily due to the absence of a standardized evaluation benchmark. This paper introduces a Multi-modal Evaluation Benchmark named MERLIM, a scalable test-bed to assess the capabilities of IT-LVLMs on fundamental computer vision tasks. MERLIM contains over 300K image-question pairs and has a strong focus on detecting cross-modal “hallucination” events in IT-LVLMs. Our results bring important insights on the performance of state-of-the-art IT-LVMLs including limitations at identifying fine-grained visual concepts, object hallucinations across tasks, and biases towards the language query. Our findings also suggest that these models have weak visual grounding, but manage to make adequate guesses from global visual patterns or language biases contained in the LLM component.

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain a third-party dataset (COCO), models, and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.

## Contents
- [Download Dataset](#download-dataset)
- [Install](#install)
- [Evaluation](#evaluation)

## Download Dataset

We provide the edited version of the images of COCO Validation, which we computed to evaluate the hidden hallucinations of the models. We encourage you to download the original images using the COCO official library. We provide a script to facilitate this process. 
```bash
wget https://u.pcloud.link/publink/show?code=kZLSlu0ZAvE3vnlBm5LExM9TCG4AlLttiNHy
```

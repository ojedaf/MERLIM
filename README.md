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
cd data & wget https://p-lux3.pcloud.com/cBZ0MHubAZgVjTAR7ZZZ70Yc7kZ2ZZ89JZkZyxE68QZd4ZIpZd8ZdHZEzZszZgLZDQZq8ZGLZQ4ZyHZzQZlpZLSlu0ZtXYmuw9KBSFicN5G95DEFSky8R1V/coco_aug.zip
```
Also, It can be downloaded manually from [this website](https://u.pcloud.link/publink/show?code=kZLSlu0ZAvE3vnlBm5LExM9TCG4AlLttiNHy)

2. Unzip the file to extract the images.
```bash
unzip coco_aug.zip
```
3. Download the validation instance file of COCO (instances_val2017.json) from the official website.
```bash
cd data & wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
```bash
unzip annotations_trainval2017.zip
```
4. Create conda env from the requirements file.
```bash
conda create --name MERLIM_ENV --file requirements.txt
```
```bash
conda activate MERLIM_ENV
```
5. Run the code to download the original validation image in the same folder as the edited images.
```bash
python download_data.py --img_dst_dir 'DESTINATION PATH FOR THE IMAGES' --annFile 'PATH TO THE FILE (instances_val2017.json)'
```

## Install

1. Install the method to be evaluated. Begin by installing the method you intend to evaluate. Ensure that all necessary dependencies are installed according to the official implementation guidelines. Below, you will find links to the official repositories for several methods.

2. Develop the Evaluation Class. Extend the provided base method (./eval_methods/base_method.py) to create an evaluation class specific to your method. For guidance, refer to the implementations in BLIP3.py and BLIP2.py, which were used to evaluate XGen-MM (Phi-3 Mini), and BLIP2 and InstructBLIP, respectively.

## Evaluation

1. Defining the Evaluation Data.
    - **in_data_v3:** This dataset is used to evaluate Object Counting and Object Recognition tasks.
    - **rel_task_set_eval_curated_v3:** This dataset is designed to evaluate the Inter-object Relationship Understanding task within a curated set of relationships.
    - **rel_task_set_eval_random_v3:** This dataset is intended to evaluate the Inter-object Relationship Understanding task within a randomly selected set of relationships.

2. Run the Evaluation Task. 
```bash
python run_task.py --name_class 'NAME_OF_THE_CORRESPONDING_CLASS/METHOD_TO_EVAL_IN_EVAL_METHODS_FOLDER' --name_model 'NAME_OF_THE_MODEL' --model_type 'TYPE_OF_LLM_MODEL' --name_data 'NAME_OF_THE_EVALUATION_DATA_ACCORDING_TO_THE_TASK' --main_img_dir 'IMAGE_FOLDER' --main_data_dir 'FOLDER_THAT_CONTAINS_THE_EVAL_DATA' --type_task 'TASK_TO_EVAL' --exp_name 'NAME_OF_THE_EXP' --num_question 'MUST_BE_AN_INT_VALUE_FROM_0_TO_4_TO_SEL_THE_QUESTION_FOR_THE_OBJECT_RECOGNITION_TASK' --num_steps2save 'INT_VALUE_THAT_MEANS_THE_FREQUENCY_TO_SAVE' --cfg-path 'CONF_FILE_PATH_FOR_MINIGPT4' --model_path 'MODEL_PATH_FOR_LAVA/OTHERS'
```
NOTE: 
- For **InstructBLIP/BLIP2**, ensure you use the **name_model and model_type** as specified in the **official repository**.
- If type_task is **'classification'**, set **name_data** to **'in_data_v3.pkl'**.
- If type_task is **'count'**, set name_data to **'in_data_v3.pkl'**.
- If type_task is **'reasoning'**, set name_data to **'rel_task_set_eval_curated_v3.pkl'** for **evaluating curated relationships**, or **'rel_task_set_eval_random_v3.pkl'** for **evaluating random relationships**.

3. Get Metrics.


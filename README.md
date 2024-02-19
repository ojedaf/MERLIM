# Behind the Magic, MERLIM: Multi-modal Evaluation Benchmark for Large Image-Language Models

```diff
- The Code will be available SOON.
```
[[Paper]](https://arxiv.org/abs/2312.02219)


Large Vision and Language Models have enabled significant advances in fully supervised and zero-shot vision tasks. These large pre-trained architectures serve as the baseline to what is currently known as Instruction Tuning Large Vision and Language models (IT-LVLMs). IT-LVLMs are general-purpose multi-modal assistants whose responses are modulated by natural language instructions and arbitrary visual data. Despite this versatility, IT-LVLM effectiveness in fundamental computer vision problems remains unclear, primarily due to the absence of a standardized evaluation benchmark. This paper introduces a Multi-modal Evaluation Benchmark named MERLIM, a scalable test-bed to assess the performance of IT-LVLMs on fundamental computer vision tasks. MERLIM contains over 300K image-question pairs and has a strong focus on detecting cross-modal “hallucination” events in IT-LVLMs. Our results show that state-of-the-art IT-LVMLs are still limited at identifying fine-grained visual concepts, object hallucinations are common across tasks, and their results are strongly biased by small variations in the input query, even if the queries have the very same semantics. Our findings also suggest that these models lack direct visual groundings, but can still make adequate guesses from global visual patterns or textual biases contained in the LLM component.



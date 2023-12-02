# Behind the Magic, MERLIM: Multi-modal Evaluation Benchmark for Large Image-Language Models

```diff
- The Code will be available SOON.
```
[[Paper]]()

Large Vision and Language Models have enabled significant advances in fully supervised and zero-shot vision tasks. These large pre-trained architectures serve as the baseline to what is currently known as Instruction Tuning Large Vision and Language models (IT-LVLMs), which allow the direct querying of visual data with natural language, generating the corresponding output in text format. IT-LVLMs are general-purpose multi-modal assistants whose responses are modulated by natural language instructions and arbitrary visual data. Despite this versatility, IT-LVLM effectiveness in fundamental computer vision problems remains unclear, primarily due to the absence of a standardized evaluation benchmark to quantify the IT-LVMLs zero-shot performance in machine vision tasks. This paper introduces a Multi-modal Evaluation Benchmark named MERLIM, a scalable test-bed to assess the performance of IT-LVLMs on fundamental computer vision tasks. MERLIM contains over 279K image-question pairs and allows to translate the open-set language predictions into the closed-set ground-truth of computer vision tasks with a strong focus on detecting cross-modal “hallucination” events in IT-LVLMs, where the language output refers to visual concepts that lack any effective grounding in the image. Our results show that state-of-the-art IT-LVMLs are still limited at identifying fine-grain visual concepts, object hallucinations are common, and their results are strongly biased by small variations in the input query, even if the semantics remain stable. Furthermore, our findings suggest that these models have weak visual groundings but don't suffer a significant performance penalty as they can make adequate guesses by global visual patterns or textual biases contained in the LLM component.



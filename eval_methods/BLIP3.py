from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch
from .base_method import BaseEvalMethod


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids      
    
class BLIP3(BaseEvalMethod):
    def __init__(self, args, *argv):
        super(BLIP3, self).__init__()
        model_path = args.model_path

        self.model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, legacy=False)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        print('BLIP3 Ready')

    def vis_processors(self, raw_image):
        tensor_img = self.image_processor([raw_image], return_tensors="pt", image_aspect_ratio='anyres')
        tensor_img['pixel_values'] = tensor_img['pixel_values'][0]
        return tensor_img

    # define the prompt template
    def apply_prompt_template(self, prompt):
        s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
            )
        return s
    
    def generate(self, inputs, queries, *argv):
        raw_image_sizes = argv[0].get('raw_image_size') #should be a list
        raw_image_sizes = [raw_image_size.cpu().item() for raw_image_size in raw_image_sizes]
        raw_image_sizes = [tuple(raw_image_sizes)]
        if type(queries) == list:
            prompts = [self.apply_prompt_template(query) for query in queries]
        else:
            prompts = [self.apply_prompt_template(queries)]
        language_inputs = self.tokenizer(prompts, return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        generated_text = self.model.generate(**inputs, image_size=raw_image_sizes,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,
                                        stopping_criteria = [EosListStoppingCriteria()],
                                        )
        prediction = self.tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        return [prediction]


    

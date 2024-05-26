from lavis.models import load_model_and_preprocess
from .base_method import BaseEvalMethod

class BLIP2(BaseEvalMethod):
    def __init__(self, args, *argv):
        super(BLIP2, self).__init__()

        name_model = args.name_model
        model_type = args.model_type
        device = argv[0].get('device')

        self.model, self.image_processor, _ = load_model_and_preprocess(name=name_model, model_type=model_type, is_eval=True, device=device)
        self.type_task=args.type_task
        print('InstructBLIP/BLIP2 Ready')

    def vis_processors(raw_image):
        return self.image_processor["eval"](raw_image)
    
    def generate(self, inputs, queries, *argv):
        if self.name_model == 'blip2_t5' or self.name_model == 'blip2_opt':
            blip_2_prompt_temp = 'Question: {} Answer:'
            queries = [blip_2_prompt_temp.format(query.lower()) for query in queries]  
        prediction = self.model.generate({"image": inputs, "prompt": queries})
        return prediction

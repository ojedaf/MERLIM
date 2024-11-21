import argparse
import pickle
from utils.dataset_task import CocoDataset
from torch.utils.data import DataLoader
from eval_methods.init_method import create_method 
from accelerate import Accelerator 

from textblob import TextBlob
import os

def parser_fn(in_pred_ans, in_gt_ans, orig_pred_ans, orig_gt_ans, categories, questions, id_imgs, id_orig_imgs):
    new_in_pred_ans, new_in_gt_ans, new_orig_pred_ans, new_orig_gt_ans, new_categories, new_questions, new_id_imgs, new_id_orig_imgs = [], [], [], [], [], [], [], []
    sel_idx = []
    for i, in_pred in enumerate(in_pred_ans):
        try:
            in_pred = int(in_pred)
            in_gt = int(in_gt_ans[i])
            orig_pred = int(orig_pred_ans[i])
            orig_gt = int(orig_gt_ans[i])
            new_in_pred_ans.append(in_pred)
            new_in_gt_ans.append(in_gt)
            new_orig_pred_ans.append(orig_pred)
            new_orig_gt_ans.append(orig_gt)
            new_categories.append(categories[i])
            new_questions.append(questions[i])
            new_id_imgs.append(id_imgs[i])
            new_id_orig_imgs.append(id_orig_imgs[i])
            sel_idx.append(i)
        except ValueError as e:
            pass
    return new_in_pred_ans, new_in_gt_ans, new_orig_pred_ans, new_orig_gt_ans, new_categories, new_questions, new_id_imgs, new_id_orig_imgs, sel_idx

def eval_counting_fn(model, batch, accelerator, num_processes):
    in_image, in_gt_ans, img_name, orig_image, orig_gt_ans, orig_img_name, question, cat_labels, in_image_size, orig_image_size = batch
    questions = list(question)
    
    if num_processes > 1:
        in_kwargs = {'raw_image_size': in_image_size, 'img_names': img_name}
        in_pred_ans = model.module.generate(in_image, questions, in_kwargs)
        orig_kwargs = {'raw_image_size': orig_image_size, 'img_names': orig_img_name}
        orig_pred_ans = model.module.generate(orig_image, questions, orig_kwargs)
    else:
        in_kwargs = {'raw_image_size': in_image_size, 'img_names': img_name}
        in_pred_ans = model.generate(in_image, questions, in_kwargs)
        orig_kwargs = {'raw_image_size': orig_image_size, 'img_names': orig_img_name}
        orig_pred_ans = model.generate(orig_image, questions, orig_kwargs)

    # Filt the int answer
    in_pred_ans, in_gt_ans, orig_pred_ans, orig_gt_ans, cat_labels, questions, img_name, orig_img_name, sel_idx = parser_fn(in_pred_ans, in_gt_ans,
                                                                                                                            orig_pred_ans, 
                                                                                                                            orig_gt_ans,
                                                                                                                            cat_labels, questions, 
                                                                                                                            img_name, orig_img_name)
    in_rev_questions, in_rev_pred_ans = [], []
    orig_rev_questions, orig_rev_pred_ans = [], []
    if sel_idx:
        for i, pred in enumerate(in_pred_ans):
            cat_label = cat_labels[i]
            orig_pred = orig_pred_ans[i]
            pl_label = TextBlob(cat_label)
            pl_label = pl_label.words.pluralize()[0]
            in_rev_que = 'Is there 1 {}?'.format(cat_label) if pred == 1 else 'Are there {} {}?'.format(pred, pl_label)
            orig_rev_que = 'Is there 1 {}?'.format(cat_label) if orig_pred == 1 else 'Are there {} {}?'.format(orig_pred, pl_label)
            in_rev_questions.append(in_rev_que)
            orig_rev_questions.append(orig_rev_que)

        if num_processes > 1:
            in_kwargs = {'raw_image_size': in_image_size[sel_idx], 'img_names': img_name}
            in_rev_pred_ans = model.module.generate(in_image[sel_idx], in_rev_questions, in_kwargs)
            orig_kwargs = {'raw_image_size': orig_image_size[sel_idx], 'img_names': orig_img_name}
            orig_rev_pred_ans = model.module.generate(orig_image[sel_idx], orig_rev_questions, orig_kwargs)
        else:
            in_kwargs = {'raw_image_size': in_image_size[sel_idx], 'img_names': img_name}
            in_rev_pred_ans = model.generate(in_image[sel_idx], in_rev_questions, in_kwargs)
            orig_kwargs = {'raw_image_size': orig_image_size[sel_idx], 'img_names': orig_img_name}
            orig_rev_pred_ans = model.generate(orig_image[sel_idx], orig_rev_questions, orig_kwargs)

    gathered_in_gt_ans = accelerator.gather_for_metrics(in_gt_ans)
    gathered_orig_gt_ans = accelerator.gather_for_metrics(orig_gt_ans)
    gathered_in_pred_ans = accelerator.gather_for_metrics(in_pred_ans)
    gathered_in_rev_pred_ans = accelerator.gather_for_metrics(in_rev_pred_ans)
    gathered_orig_pred_ans = accelerator.gather_for_metrics(orig_pred_ans)
    gathered_orig_rev_pred_ans = accelerator.gather_for_metrics(orig_rev_pred_ans)
    gathered_questions = accelerator.gather_for_metrics(questions)
    gathered_in_rev_questions = accelerator.gather_for_metrics(in_rev_questions)
    gathered_orig_rev_questions = accelerator.gather_for_metrics(orig_rev_questions)
    gathered_img_name = accelerator.gather_for_metrics(img_name)
    gathered_orig_img_name = accelerator.gather_for_metrics(orig_img_name)
    gathered_cat_labels = accelerator.gather_for_metrics(cat_labels)

    gathered_results = (gathered_in_gt_ans, gathered_orig_gt_ans, gathered_in_pred_ans, gathered_orig_pred_ans,
                        gathered_in_rev_pred_ans, gathered_orig_rev_pred_ans, gathered_img_name, gathered_questions, 
                        gathered_in_rev_questions, gathered_orig_rev_questions, gathered_orig_img_name, gathered_cat_labels)
    
    return gathered_results
    
def eval_reasoning_fn(model, batch, accelerator, num_processes):
    in_image, in_gt_ans, img_name, orig_image, orig_gt_ans, orig_img_name, question, neg_question, in_neg_gt_ans, orig_neg_gt_ans, in_image_size, orig_image_size = batch
    questions = list(question)
    neg_questions = list(neg_question)

    if num_processes > 1:
        in_kwargs = {'raw_image_size': in_image_size, 'img_names': img_name}
        in_pred_ans = model.module.generate(in_image, questions, in_kwargs)
        in_neg_pred_ans = model.module.generate(in_image, neg_questions, in_kwargs)
        orig_kwargs = {'raw_image_size': orig_image_size, 'img_names': orig_img_name}
        orig_pred_ans = model.module.generate(orig_image, questions, orig_kwargs)
        orig_neg_pred_ans = model.module.generate(orig_image, neg_questions, orig_kwargs)
    else:
        in_kwargs = {'raw_image_size': in_image_size, 'img_names': img_name}
        in_pred_ans = model.generate(in_image, questions, in_kwargs)
        in_neg_pred_ans = model.generate(in_image, neg_questions, in_kwargs)
        orig_kwargs = {'raw_image_size': orig_image_size, 'img_names': orig_img_name}
        orig_pred_ans = model.generate(orig_image, questions, orig_kwargs)
        orig_neg_pred_ans = model.generate(orig_image, neg_questions, orig_kwargs)
    
    gathered_in_gt_ans = accelerator.gather_for_metrics(in_gt_ans)
    gathered_in_neg_gt_ans = accelerator.gather_for_metrics(in_neg_gt_ans)
    gathered_orig_gt_ans = accelerator.gather_for_metrics(orig_gt_ans)
    gathered_orig_neg_gt_ans = accelerator.gather_for_metrics(orig_neg_gt_ans)
    gathered_in_pred_ans = accelerator.gather_for_metrics(in_pred_ans)
    gathered_in_neg_pred_ans = accelerator.gather_for_metrics(in_neg_pred_ans)
    gathered_orig_pred_ans = accelerator.gather_for_metrics(orig_pred_ans)
    gathered_orig_neg_pred_ans = accelerator.gather_for_metrics(orig_neg_pred_ans)
    gathered_questions = accelerator.gather_for_metrics(questions)
    gathered_neg_questions = accelerator.gather_for_metrics(neg_questions)
    gathered_img_name = accelerator.gather_for_metrics(img_name)
    gathered_orig_img_name = accelerator.gather_for_metrics(orig_img_name)

    gathered_results = (gathered_in_gt_ans, gathered_in_neg_gt_ans, gathered_orig_gt_ans, gathered_orig_neg_gt_ans,
                        gathered_in_pred_ans, gathered_in_neg_pred_ans, gathered_orig_pred_ans, gathered_orig_neg_pred_ans, 
                        gathered_questions, gathered_neg_questions, gathered_img_name, gathered_orig_img_name)
    
    return gathered_results
    
def eval_classification_fn(model, batch, accelerator, num_processes):
    in_image, in_gt_ans, img_name, orig_image, orig_gt_ans, orig_img_name, question, in_image_size, orig_image_size = batch  

    questions = list(question)
    if num_processes > 1:
        in_kwargs = {'raw_image_size': in_image_size, 'img_names': img_name}
        in_pred_ans = model.module.generate(in_image, questions, in_kwargs)
        orig_kwargs = {'raw_image_size': orig_image_size, 'img_names': orig_img_name}
        orig_pred_ans = model.module.generate(orig_image, questions, orig_kwargs)
    else:
        in_kwargs = {'raw_image_size': in_image_size, 'img_names': img_name}
        in_pred_ans = model.generate(in_image, questions, in_kwargs)
        orig_kwargs = {'raw_image_size': orig_image_size, 'img_names': orig_img_name}
        orig_pred_ans = model.generate(orig_image, questions, orig_kwargs)

    in_gt_ans = [elem.split(',') for elem in in_gt_ans]
    orig_gt_ans = [elem.split(',') for elem in orig_gt_ans]
    gathered_in_gt_ans = accelerator.gather_for_metrics(in_gt_ans)
    gathered_orig_gt_ans = accelerator.gather_for_metrics(orig_gt_ans)
    gathered_in_pred_ans = accelerator.gather_for_metrics(in_pred_ans)
    gathered_orig_pred_ans = accelerator.gather_for_metrics(orig_pred_ans)
    gathered_img_name = accelerator.gather_for_metrics(img_name)
    gathered_questions = accelerator.gather_for_metrics(questions)
    gathered_orig_img_name = accelerator.gather_for_metrics(orig_img_name)
    gathered_results = (gathered_in_gt_ans, gathered_orig_gt_ans, gathered_in_pred_ans, gathered_orig_pred_ans, gathered_img_name, 
                        gathered_questions, gathered_orig_img_name)
    return gathered_results

def test_function(args):
    name_model=args.name_model
    accelerator=Accelerator()
    num_processes = accelerator.num_processes
    print('num_processes: ',num_processes)
    device = accelerator.device
    main_data_dir=args.main_data_dir
    name_data=args.name_data
    main_img_dir=args.main_img_dir
    model_type=args.model_type
    batch_size=int(args.batch_size)
    type_task=args.type_task
    exp_name=args.exp_name


    model = create_method(args, {'device':device})
    
    path_data=os.path.join(main_data_dir, name_data)
        
    if type_task == 'count':
        question = 'How many {} are there? Just answer the number.'
        #question = 'How many {} are there? Only answer the number, no words and provide the shortest answer possible.' #FOR BLIP2_T5
        dataset = CocoDataset(path_data, type_task, question, main_img_dir=main_img_dir, transform=model.vis_processors, name_model=name_model)
        forward_pass_fn = eval_counting_fn
        name_file = 'count_{}_{}_{}.pkl'.format(exp_name, name_model, model_type)
        print('question: ',question)
    elif type_task == 'reasoning':
        dataset = CocoDataset(path_data, type_task, None, main_img_dir=main_img_dir, transform=model.vis_processors, name_model=name_model)
        forward_pass_fn = eval_reasoning_fn
        name_file = 'reasoning_{}_{}_{}.pkl'.format(exp_name, name_model, model_type)
    else:
        num_question=int(args.num_question)
        question_0 = 'List the objects that appear in the image.'
        question_1 = 'Enumerate the items visible in the picture.'
        question_2 = 'Itemize the things seen in the image.'
        question_3 = 'Detail the items shown in the picture.'
        question_4 = 'Identify the objects within the image.'
        questions = [question_0, question_1, question_2, question_3, question_4]
        question = questions[num_question]
        print('question: ',question)
        dataset = CocoDataset(path_data, type_task, question, main_img_dir=main_img_dir, transform=model.vis_processors)
        forward_pass_fn = eval_classification_fn
        name_file = 'cls_{}_{}_{}_{}.pkl'.format(exp_name, num_question, name_model, model_type)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if model != None:
        model, test_dataloader = accelerator.prepare(model, test_dataloader) 
    results = []
    for idxb, batch in enumerate(test_dataloader):
        print(f'batch: {idxb}/{len(test_dataloader)}')
        result = forward_pass_fn(model, batch, accelerator, num_processes)
        results.append(result)

        if (model == None or accelerator.is_main_process) and (idxb + 1) % args.num_steps2save == 0:
            test_pred_pkl = os.path.join(main_data_dir, name_file)
            print('Path to save: ',test_pred_pkl)
            with open(test_pred_pkl, 'wb') as handle:
                pickle.dump(results, handle)
    
    if model == None or accelerator.is_main_process:
        test_pred_pkl = os.path.join(main_data_dir, name_file)
        print('Path to save: ',test_pred_pkl)
        with open(test_pred_pkl, 'wb') as handle:
            pickle.dump(results, handle)

    create_method

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--name_class", default="BLIP2", help="Name of the file and the class of model to eval. It must be in the eval_methods folder. The name of File and Class must be the same", type=str)
    parser.add_argument("--name_model", default="blip2_vicuna_instruct", type=str)
    parser.add_argument("--model_type", default='vicuna7b', type=str)
    parser.add_argument("--name_data", default='in_data_v3.pkl', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--main_img_dir", default='./data/coco_aug/', type=str)
    parser.add_argument("--main_data_dir", default='./data/', type=str)
    parser.add_argument("--type_task", default='classification', type=str)
    parser.add_argument("--exp_name", default='test_pred', type=str)
    parser.add_argument("--num_question", default=0, type=int)
    parser.add_argument("--num_steps2save", default=20, type=int)
    parser.add_argument("--cfg-path", help="path to configuration file of MiniGPT.", default='./MiniGPT-4/eval_configs/minigpt4_eval.yaml', type=str)
    parser.add_argument("--model_path", default='./llava/llava-v1.5-7b', type=str)
    args=parser.parse_args()

    print('########## Start {} Task Eval ###########'.format(args.type_task.upper()))
    print('model_type: ',args.model_type)
    print('batch_size: ',args.batch_size)
    print('type_task: ',args.type_task)
    print('name_data: ',args.name_data)
    
    test_function(args)
     
    print('########## Complete Eval ###########')
    
    

if __name__ == "__main__":
    main()
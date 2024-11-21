import os
import pickle
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset
from textblob import TextBlob

class CocoDataset(Dataset):
    def __init__(self, path_data, type_task, question, main_img_dir='./data/coco_aug/', input_dim=512, transform=None, 
                 path_coco_data='./data/annotations/', dataType='val2017'):

        self.input_dim = input_dim
        self.type_task = type_task
        self.question = question
        self.main_img_dir = main_img_dir
        with open(path_data, 'rb') as f:
            data = pickle.load(f)
        self.list_data = []
        for _, in_imgs in data.items():
            self.list_data.extend(in_imgs)
        self.transform = transform
        annFile=os.path.join(path_coco_data, 'instances_{}.json'.format(dataType))
        coco=COCO(annFile)
        imgIds=sorted(coco.getImgIds())
        imgs = coco.loadImgs(imgIds)
        cats=coco.loadCats(coco.getCatIds())
        dict_cats = {cat['id']: cat for cat in cats}
        self.dict_cls = {}
        for img in imgs:
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)
            list_cls = []
            self.dict_cls[img['id']] = {}
            for ann in anns:
                cat_id = ann['category_id']
                name = dict_cats[cat_id]['name']
                if not cat_id in self.dict_cls[img['id']]:
                    self.dict_cls[img['id']][cat_id] = {'name': name, 'num': 1}
                else:
                    self.dict_cls[img['id']][cat_id]['num']+=1

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        elem = self.list_data[idx]
        img_name = elem['in_img']
        orig_img_name = elem['orig_img']
        img_path = os.path.join(self.main_img_dir, img_name)
        orig_img_path = os.path.join(self.main_img_dir, orig_img_name)
        if self.transform == None:
            in_image, in_image_size = '', ''
            orig_image, orig_image_size = '', ''
        else:
            raw_image = Image.open(img_path).convert("RGB")
            orig_raw_image = Image.open(orig_img_path).convert("RGB")
            in_image = self.transform(raw_image)
            orig_image = self.transform(orig_raw_image)
            in_image_size = raw_image.size
            orig_image_size = orig_raw_image.size

        if self.type_task == 'classification':
            org_cls = self.dict_cls[int(orig_img_name.split('.')[0])]
            orig_gt_ans = [elem['name'] for elem in list(org_cls.values())]
            orig_gt_ans = ",".join(orig_gt_ans)
            cat_label = elem['removed_obj']['cat_label']
            num_rev_obj = elem['removed_obj']['num_rem_obj']
            in_gt_ans = []
            for elem_cls in list(org_cls.values()):
                if cat_label != elem_cls['name'] or (cat_label == elem_cls['name'] and num_rev_obj>0):
                    in_gt_ans.append(elem_cls['name'])
            in_gt_ans = ",".join(in_gt_ans)
            question = self.question
            return in_image, in_gt_ans, img_name, orig_image, orig_gt_ans, orig_img_name, question, in_image_size, orig_image_size   
        elif self.type_task == 'reasoning':
            question = elem['question']
            question_neg = elem['question_neg']
            orig_gt_ans = elem['orig answer']
            orig_neg_gt_ans = elem['orig neg answer']
            in_gt_ans = elem['inp answer']
            in_neg_gt_ans = elem['inp neg answer']
            return in_image, in_gt_ans, img_name, orig_image, orig_gt_ans, orig_img_name, question, question_neg, in_neg_gt_ans, orig_neg_gt_ans, in_image_size, orig_image_size
        else:
            cat_label = elem['removed_obj']['cat_label']
            pl_label = TextBlob(cat_label)
            pl_label = pl_label.words.pluralize()[0]
            question_temp = self.question
            question = question_temp.format(pl_label)
            in_gt_ans = int(elem['removed_obj']['num_rem_obj'])
            orig_gt_ans = in_gt_ans + 1  # Because we just deleted one object.
            return in_image, in_gt_ans, img_name, orig_image, orig_gt_ans, orig_img_name, question, cat_label, in_image_size, orig_image_size
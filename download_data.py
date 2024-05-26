import argparse
import os
from pycocotools.coco import COCO
import skimage.io as io
from skimage.io import imsave
from multiprocessing import Pool

def download(img):
    if not os.path.exists(img['dst_path']):
        I = io.imread(img['coco_url'])
        imsave(img['dst_path'], I)

def get_coco_images(args):
    list_imgs = []
    coco=COCO(args.annFile)
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    for img in images:
        img['dst_path'] = os.path.join(args.img_dst_dir, str(img['id'])+'.jpg')
        list_imgs.append(img)
    return list_imgs

def run_multiprocess(operation, input, pool):
    pool.map(operation, input)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--num_process", default=10, type=int)
    parser.add_argument("--img_dst_dir", default='./data/coco_aug', type=str)
    parser.add_argument("--annFile", default='./data/annotations/instances_val2017.json', type=str)

    args=parser.parse_args()
    
    print('########## Downloading data ###########')
    if not os.path.isdir(args.img_dst_dir):
        os.mkdir(args.img_dst_dir) 
    list_imgs = get_coco_images(args)
    processes_pool = Pool(args.num_process)
    run_multiprocess(download, list_imgs, processes_pool) 
    print('########## Complete ###########')
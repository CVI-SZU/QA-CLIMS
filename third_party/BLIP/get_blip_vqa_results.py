import json
from threading import Thread
from models.blip_vqa import blip_vqa
from PIL import Image
import torch
import os.path as osp
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from tqdm import tqdm
import pickle
from nltk.stem import WordNetLemmatizer
from pycocotools.coco import COCO
from argparse import ArgumentParser

QUESTIONS = {
    'fg': [
        'What kind of {} is in the photo?',
        'What type of {} is in the photo?',
        'What is this {} also called?',
        'What is this {} usually called?',
        'What is another word for this {}?',
        'What is another name for this {}?',
    ],
    'bg': [
        'what is above the {}?',
        'what is under the {}?',
        'what is behind the {}?',
        'what is around the {}?',
        'what is next to the {}?',
        'what is the left side of {}?',
        'what is the right side of {}?',
        'what scene is the {} in?',
        'what environment is the {} in?',
        'what place is the {} in?',
    ],
}

voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
              'dog',
              'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

coco_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco_category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                     "13": 12,
                     "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22,
                     "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32,
                     "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42,
                     "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52,
                     "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62,
                     "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72,
                     "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}


def load_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def load_model(model_path, image_size, device):
    model = blip_vqa(pretrained=model_path, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    return model


def vqa(model, images, questions):
    with torch.no_grad():
        answers = model(images, questions, train=False, inference='generate')
    return answers


def get_voc_image_names(voc_cls_label_path) -> (dict, list):
    print('calcing clims npy img names...')
    clims_cls_label: dict = np.load(voc_cls_label_path, allow_pickle=True).item()
    results = {}
    img_names = []
    for img_name_int in tqdm(clims_cls_label.keys()):
        img_name_str = str(img_name_int)
        img_name = f'{img_name_str[:4]}_{img_name_str[4:]}'
        clims_labels = np.nonzero(clims_cls_label[img_name_int])[0]
        for clims_label in clims_labels:
            clims_label = voc_labels[clims_label]
            if not results.__contains__(clims_label):
                results[clims_label] = []
            results[clims_label].append(img_name)
            img_names.append((clims_label, img_name))
    print(f'process voc img names done, total {len(img_names)}')
    return results, img_names


def get_coco_image_names(coco):
    img_ids = coco.getImgIds()
    results = {}
    img_names = []
    for img_id in img_ids:
        img_name = coco.loadImgs(img_id)[0]['file_name'].split('.')[0]
        annids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)
        cat_ids = [ann['category_id'] for ann in anns]
        cat_ids = list(set(cat_ids))
        cat_ids = [coco_category_map[str(cat_id)] - 1 for cat_id in cat_ids]
        for cat_id in cat_ids:
            cat_label = coco_labels[cat_id]
            if not results.__contains__(cat_label):
                results[cat_label] = []
            results[cat_label].append(img_name)
            img_names.append((cat_label, img_name))
    print(f'process coco img names done, total {len(img_names)} image-label pairs')
    return results, img_names


def vqa_all_img_bg_main(
        data='voc',
        # questions
        questions=None,
        # voc
        voc_cls_labels_npy_path='/PATH/TO/QA-CLIMS/voc12/cls_labels.npy',
        voc_img_root_path='/PATH/TO/VOC2012/JPEGImages',
        # coco
        coco_annotation_path='/PATH/TO/COCO/annotations/instances_train2014.json',
        coco_img_root_path='/PATH/TO/COCO/train2014',
        # blip model
        blip_model_path='./checkpoints/model_base_vqa_capfilt_large.pth',
        # result
        target_result_path='/PATH/TO/QA-CLIMS/vqa/voc_vqa_bg_blip.npy',
        thread_num=1,
        gpu_num=1,
        image_size=480,
):
    assert questions is not None
    print('questions:', questions)

    if data == 'voc':
        assert os.path.exists(voc_cls_labels_npy_path)
        assert os.path.exists(voc_img_root_path)
        voc_img_name_list, img_names = get_voc_image_names(voc_cls_labels_npy_path)
        img_root = voc_img_root_path
    elif data == 'coco':
        coco = COCO(coco_annotation_path)
        coco_img_name_list, img_names = get_coco_image_names(coco)
        img_root = coco_img_root_path
    else:
        raise Exception('data should be voc or coco')

    img_names_split = np.array_split(img_names, thread_num)
    img_names_split = [list(x) for x in img_names_split]

    WordNetLemmatizer().lemmatize('start')

    print('multiprocess vqa...')
    progress_bar = tqdm(total=len(img_names))
    img_results = {}

    def _work(process_id, _img_names, _progress_bar):
        _lemmatizer = WordNetLemmatizer()
        device = torch.device(f'cuda:{process_id % gpu_num}')
        print(f'[{process_id}] Loading BLIP model on {device}...')
        _model = load_model(blip_model_path, image_size, device)

        _img_names = _img_names[process_id]
        print(f'[{process_id}] {len(_img_names)} images')

        for _label, _img_name in _img_names:
            image = load_image(osp.join(img_root, _img_name + '.jpg'), image_size, device)

            _answers = []
            for q in questions:
                a = vqa(_model, image, q.format(_label))
                _answers.append(a[0])
            _lemmatized_answers = [_lemmatizer.lemmatize(a) for a in _answers]

            if not img_results.__contains__(str(_img_name)):
                img_results[str(_img_name)] = []
            img_results[str(_img_name)].append(dict(label=_label,
                                                    vqa=list(zip(questions, _answers)),
                                                    answers=_lemmatized_answers))
            progress_bar.update(1)

    thread_list = []
    for i in range(thread_num):
        thread_list.append(Thread(target=_work, args=(i, img_names_split, progress_bar)))

    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()

    print('saving results...')
    target_result = {}
    # target format:
    # {
    #     'img_name': { 'label': {vqa, answers}, 'label': {vqa, answers}, ... },
    #     'img_name': { 'label': {vqa, answers}, 'label': {vqa, answers}, ... },
    #     ...
    # }
    for img_name, results in tqdm(img_results.items()):
        target_result[img_name] = {}
        for result in results:
            label = result['label']
            target_result[img_name][label] = dict(vqa=result['vqa'], answers=result['answers'])

    np.save(target_result_path, target_result)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['voc'])
    parser.add_argument('question_type', type=str, choices=['fg', 'bg'])
    parser.add_argument('--voc_cls_labels_npy_path', type=str, default='/PATH/TO/QA-CLIMS/voc12/cls_labels.npy')
    parser.add_argument('--voc_img_root_path', type=str, default='/PATH/TO/VOC2012/JPEGImages')
    parser.add_argument('--coco_annotation_path', type=str, default='/PATH/TO/COCO/annotations/instances_train2014.json')
    parser.add_argument('--coco_img_root_path', type=str, default='/PATH/TO/COCO/train2014')
    parser.add_argument('--blip_model_path', type=str, default='./checkpoints/model_base_vqa_capfilt_large.pth')
    parser.add_argument('--target_result_path', type=str, default='/PATH/TO/QA-CLIMS/vqa/voc_vqa_bg_blip.npy')
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vqa_all_img_bg_main(
        data=args.dataset,
        questions=QUESTIONS[args.question_type],
        voc_cls_labels_npy_path=args.voc_cls_labels_npy_path,
        voc_img_root_path=args.voc_img_root_path,
        coco_annotation_path=args.coco_annotation_path,
        coco_img_root_path=args.coco_img_root_path,
        blip_model_path=args.blip_model_path,
        target_result_path=args.target_result_path,
        thread_num=args.thread_num,
        gpu_num=args.gpu_num,
    )

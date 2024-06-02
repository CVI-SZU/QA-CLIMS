
import numpy as np
import os.path as osp
import mscoco.dataloader
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion
from misc.cat_names import category_dict
from tqdm import tqdm


def print_iou(iou, dname='coco'):
    iou_dict = {}
    for i in range(len(iou) - 1):
        iou_dict[category_dict[dname][i]] = iou[i + 1]
    print(iou_dict)

    return iou_dict

def run(args):
    dataset = mscoco.dataloader.COCOSegmentationDataset(image_dir=osp.join(args.mscoco_root, 'train2014/'),
                                                        anno_path=osp.join(args.mscoco_root,
                                                                           'annotations/instances_train2014.json'),
                                                        masks_path=osp.join(args.mscoco_root, 'mask/train2014'),
                                                        crop_size=512)
    eval_9999 = args.eval_9999
    if eval_9999:
        print('eval 10000 images!')
    preds = []
    labels = []
    n_images = 0
    num = len(dataset) if not eval_9999 else 9999
    pbar = tqdm(total=num, desc='evaluating', ascii=True)
    for i, pack in enumerate(dataset):
        # if i % 5000 == 0:
        #     print(i, '/', num)
        if i == 9999 and eval_9999:
            break
        filename = pack['name'].split('.')[0]

        n_images += 1
        cam_dict = np.load(osp.join(args.cam_out_dir, filename + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels].astype(np.uint8)
        preds.append(cls_labels.copy())

        label = dataset.get_label_by_name(filename)
        labels.append(label)
        pbar.update(1)

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    # print(iou)
    iou_dict = print_iou(iou, 'coco')

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    print('among_predfg_bg', float((resj[1:].sum() - confusion[1:, 1:].sum()) / (resj[1:].sum())))

    if args.eval_cam_best:
        threshs = np.arange(10, 60, 2)
        _thresh_max_iou = 0
        _thresh_max_iou_thres = 0
        _max_miou_dict = {}
        for thres in tqdm(threshs, desc='find best thresh', ascii=True):
            _preds = []
            for i, pack in enumerate(dataset):
                if i == 9999 and eval_9999:
                    break
                _filename = pack['name'].split('.')[0]
                cam_dict = np.load(osp.join(args.cam_out_dir, _filename + '.npy'), allow_pickle=True).item()
                _cams = np.pad(cam_dict['high_res'], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres / 100)
                _keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                _cls_labels = np.argmax(_cams, axis=0)
                _cls_labels = _keys[_cls_labels]
                _preds.append(_cls_labels.copy())

            _confusion = calc_semantic_segmentation_confusion(_preds, labels)
            _gtj = _confusion.sum(axis=1)
            _resj = _confusion.sum(axis=0)
            _gtjresj = np.diag(_confusion)
            _denominator = _gtj + _resj - _gtjresj
            _iou = _gtjresj / _denominator
            print(f'thres: {thres / 100}, miou: {np.nanmean(_iou)}')
            if _thresh_max_iou < np.nanmean(_iou):
                _thresh_max_iou = np.nanmean(_iou)
                _thresh_max_iou_thres = thres
                _max_miou_dict = _iou
            else:
                break
        print("Best miou:", _thresh_max_iou, "thres:", _thresh_max_iou_thres / 100)
        _max_miou_dict = print_iou(_max_miou_dict, 'coco')


    hyper = [float(h) for h in args.hyper.split(',')]
    assert len(hyper) == 3

    name = f'{hyper[0]}_{hyper[1]}_{hyper[2]}_ncet{args.nce_t}_ep({args.clims_num_epoches})_lr({args.clims_learning_rate})'
    with open(args.work_space + '/eval_result.txt', 'a') as file:
        file.write(name + f' {args.clip} th: {args.cam_eval_thres}, mIoU: {np.nanmean(iou)} {iou_dict} \n')
        if args.eval_cam_best:
            file.write(f"\t└ Best miou: {_thresh_max_iou} thres: {_thresh_max_iou_thres / 100} \n")
            file.write(f"\t└-- {_max_miou_dict} \n")

    return np.nanmean(iou)

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp

import voc12.dataloader
from misc import torchutils, imutils
import cv2
cudnn.enabled = True
from tqdm import tqdm

from step.make_clims import make_cam, colormap, denormalize, get_numpy_from_tensor

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
vis_cam = True
vis_num = 10000
def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    if process_id == 0:
        pbar = tqdm(total=len(data_loader), desc=f"Process {process_id}", dynamic_ncols=False, ascii=True)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            
            np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if vis_cam and iter < vis_num:

                cam = torch.sum(highres_cam, dim=0)
                cam = cam.unsqueeze(0).unsqueeze(0)

                cam = make_cam(cam).squeeze()
                cam = get_numpy_from_tensor(cam)

                image = np.array(pack['img'][0])[0]
                image = image[0]
                image = denormalize(image, imagenet_mean, imagenet_std)
                h, w, c = image.shape

                cam = (cam * 255).astype(np.uint8)
                cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                cam = colormap(cam)

                image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
                cv2.imwrite(f'vis/{args.work_space}/{img_name}.png', image.astype(np.uint8))

            # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
            #     print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')
            if process_id == 0:
                pbar.update(1)


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    if not os.path.exists(f'vis/{args.work_space}'):
        os.makedirs(f'vis/{args.work_space}')

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
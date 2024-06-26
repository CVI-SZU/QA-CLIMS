import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import os.path as osp
import numpy as np
import importlib
import os
import imageio
from tqdm import tqdm

import mscoco.dataloader
from misc import torchutils, indexing
from PIL import Image

cudnn.enabled = True


def _work(process_id, gpu_ids, model, dataset, args):
    gpu_id = gpu_ids[process_id]

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // len(gpu_ids), pin_memory=False)

    if process_id == 0:
        pbar = tqdm(total=len(data_loader), desc=f"Process {process_id}", dynamic_ncols=False, ascii=True)

    with torch.no_grad(), cuda.device(gpu_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0].split('.')[0]
            # if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.png')):
            #     continue
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            # cams = np.power(cam_dict['cam'], 1.5) # AdvCAM
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            if keys.shape[0] == 1:

                conf = np.zeros_like(pack['img'][0])[0, 0]
                imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), conf.astype(np.uint8))
                continue

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')
            if process_id == 0:
                pbar.update(1)


def run(args):
    n_process_per_gpu = 1
    n_gpus = torch.cuda.device_count()
    n_process_total = n_gpus * n_process_per_gpu

    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)

    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = mscoco.dataloader.COCOClassificationDatasetMSF(
        image_dir=osp.join(args.mscoco_root, 'train2014/'),
        anno_path=osp.join(args.mscoco_root, 'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy',
        scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_process_total)

    gpu_id_per_process = torch.arange(n_gpus).repeat(n_process_per_gpu).tolist()

    print("n_gpus: %d, n_process_per_gpu: %d, n_process_total: %d" % (n_gpus, n_process_per_gpu, n_process_total))


    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_process_total,
                          args=(gpu_id_per_process, model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()

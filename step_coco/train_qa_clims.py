
import torch
from torch import distributed
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os.path as osp

import importlib
from misc.cat_names import category_dict
from clip_loss import InfoNCELossFG, InfoNCELossBG
import mscoco.dataloader
from misc import pyutils, torchutils, clip_adapter

from net.vqa_bg import BG_VQA
from net.vqa_fg import FG_VQA
from visual_questions import QUESTIONS

from net.mask_adapter import MaskAdapter_DynamicThreshold


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


# transform multi-hot label to class index label
def preprocess(labels):
    new_labels = []
    for n in range(labels.size(0)):
        for idx in range(0, labels.size(1)):
            temp = torch.zeros(1, labels.size(1)).long()
            if labels[n, idx] == 1:
                temp[0, idx] = 1
            new_labels.append(temp)
    return torch.cat(new_labels, dim=0).cuda()


def run(args):
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=80)

    # ================ Dataloader =================
    # initialize backbone network with baseline CAM
    if (not args.use_distributed_train) or \
            (args.use_distributed_train and args.local_rank == 0):
        model.load_state_dict(torch.load(
            'cam-baseline-coco14/res50_cam.pth'), strict=True)

    train_dataset = mscoco.dataloader.COCOClassificationDataset(
        image_dir=osp.join(args.mscoco_root, 'train2014/'),
        anno_path=osp.join(args.mscoco_root, 'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy',
        resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random"
    )

    if args.use_distributed_train:
        print('Using GPU num:', torch.cuda.device_count())
        distributed.init_process_group(backend="nccl")
        print('Torch Distributed world_size', torch.distributed.get_world_size())
        torch.cuda.set_device(args.local_rank)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                       shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                       sampler=train_sampler)
        max_step = (len(train_dataset) // args.cam_batch_size // torch.cuda.device_count()) * args.clims_num_epoches
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches

    # ================ CLIP =================
    device = f"cuda:{args.local_rank if args.local_rank != -1 else 0}" if torch.cuda.is_available() else "cpu"
    clip_model, clip_input_size = clip_adapter.load_clip_model(args.clip,
                                                               device=device,
                                                               mask_adapted=args.use_mask_clip)

    # ================ BG VQA answer feats =================
    # Note: run `tools/gen_text_feats_cache.py` first to generate cache file
    bg_vqa_tool = BG_VQA(
        args.vqa_bg_file,
        QUESTIONS['bg'],
        clip_model,
        args.clip,
        device=device,
        cache_path=args.vqa_bg_cache_file,
    )
    bg_vqa_tool.load_cache()

    # ================ FG VQA answer feats =================
    # Note: run `tools/gen_text_feats_cache.py` first to generate cache file
    fg_vqa_module = FG_VQA(
        args.vqa_fg_file,
        QUESTIONS['fg'],
        category_dict['coco'],
        clip_model,
        args.clip,
        cache_path=args.vqa_fg_cache_file,
    )

    # ================ Foreground Adaptive Thresholding =================
    mask_adapter = MaskAdapter_DynamicThreshold(
        alpha=args.mask_adapter_alpha,
    )

    # ================ Model =================
    if args.use_distributed_train:
        model = model.cuda(args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank, )
        param_groups = model.module.trainable_parameters()
    else:
        model = torch.nn.DataParallel(model).cuda()
        param_groups = model.module.trainable_parameters()

    # ================ Optimizer =================
    params = [
        {
            'params': param_groups[0],
            'lr': args.clims_learning_rate,
            'weight_decay': args.cam_weight_decay
        },
        {
            'params': param_groups[1],
            'lr': 10 * args.clims_learning_rate,
            'weight_decay': args.cam_weight_decay
        },
    ]
    optimizer = torchutils.PolyOptimizer(params,
                                         lr=args.clims_learning_rate,
                                         weight_decay=args.cam_weight_decay,
                                         max_step=max_step)
    print(
        f'init optimizer with lr={args.clims_learning_rate}, weight_decay={args.cam_weight_decay}, max_step={max_step}')
    model.train()

    # ================ Loss =================
    NCELoss = InfoNCELossFG(temperature=args.nce_t)
    NCELoss_BB = InfoNCELossBG(temperature=args.nce_t)

    # ================ Misc =================
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()
    hyper = [float(h) for h in args.hyper.split(',')]
    assert len(hyper) == 3
    print('Hyper:', hyper)

    # ================ Training =================
    # DDP wait for all processes to be ready
    if args.use_distributed_train:
        torch.distributed.barrier()
    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))
        if args.use_distributed_train:
            train_data_loader.sampler.set_epoch(ep)

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)

            fg_label = preprocess(label.cpu())

            x = model(img)
            N, _, _, _ = x.size()
            optimizer.zero_grad()

    # ================ CAM & Mask prepare =================
            # foreground indices
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()

            if len(fg_indices.shape) == 0:
                fg_indices = fg_indices.unsqueeze(0)

            def train_step():
                cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True) \
                    .reshape(N * 80, 1, clip_input_size, clip_input_size)
                cam_224_mask = mask_adapter(cam_224)  # [bs * 80, 1, 224, 224]

                img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

                fg_224_eval = []
                bg_224_eval = []
                fg_mask = []
                bg_mask = []
                temp_idx = torch.nonzero(label == 1, as_tuple=False)
                for j in range(temp_idx.shape[0]):
                    fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])
                    bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])
                    fg_mask.append(cam_224_mask[fg_indices[j]])
                    bg_mask.append(1 - cam_224_mask[fg_indices[j]])

                fg_224_eval = torch.stack(fg_224_eval, dim=0)
                bg_224_eval = torch.stack(bg_224_eval, dim=0)
                fg_mask = torch.stack(fg_mask, dim=0)
                bg_mask = torch.stack(bg_mask, dim=0)

        # ================ VQA answers prepare =================
                img_names = pack['name']
                fg_label_idxes = [np.nonzero(_.cpu().numpy())[0][0] for _ in fg_label[fg_indices]]
                fg_labels = [category_dict['coco'][_] for _ in fg_label_idxes]
                fg_image_name_idxes = fg_indices.cpu().numpy() // len(category_dict['coco'])
                fg_image_names = [img_names[_] for _ in fg_image_name_idxes]

        # ================ FG & BG masked image feature prepare =================
                if args.use_mask_clip:
                    fg_img_features = clip_model.encode_image(fg_224_eval, m=fg_mask)
                    bg_img_features = clip_model.encode_image(bg_224_eval, m=bg_mask)
                else:
                    fg_img_features = clip_model.encode_image(fg_224_eval * fg_mask)
                    bg_img_features = clip_model.encode_image(bg_224_eval * bg_mask)

        # ================ Loss Calculate =================
                L_FRC = torch.tensor(.0, requires_grad=True, device=cam_224.device)
                L_BRC = torch.tensor(.0, requires_grad=True, device=cam_224.device)
                for _i, (_lb, _im) in enumerate(zip(fg_labels, fg_image_names)):
                    fg_vqa_feat = fg_vqa_module(_lb, _im)
                    bg_vqa_feat = bg_vqa_tool.feats(_lb, _im)
                    if bg_vqa_feat is None:
                        continue
                    L_FRC = L_FRC + NCELoss(fg_img_features[_i], fg_vqa_feat, bg_vqa_feat)
                    L_BRC = L_BRC + NCELoss_BB(bg_img_features[_i], fg_vqa_feat, bg_vqa_feat)

                L_REG = torch.mean(x)
                return L_FRC, L_BRC, L_REG

            if fg_indices.shape[0] == 0:
                L_FRC, L_BRC, L_REG = (
                    torch.tensor(.0, requires_grad=True, device=img.device),
                    torch.tensor(.0, requires_grad=True, device=img.device),
                    torch.tensor(.0, requires_grad=True, device=img.device))
            else:
                L_FRC, L_BRC, L_REG = train_step()

            loss = hyper[0] * L_FRC + hyper[1] * L_BRC + hyper[2] * L_REG

            loss.backward()
            optimizer.step()

            if args.use_distributed_train:
                loss = reduce_mean(loss, distributed.get_world_size())
                if args.local_rank != 0:
                    continue

            avg_meter.add({'loss': loss.item(),
                           'L_FRC': L_FRC.item(),
                           'L_BRC': L_BRC.item(),
                           'L_REG': L_REG.item()})

            if (optimizer.global_step - 1) % 200 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'L_FRC:%.4f' % (avg_meter.pop('L_FRC')),
                      'L_BRC:%.4f' % (avg_meter.pop('L_BRC')),
                      'L_REG:%.4f' % (avg_meter.pop('L_REG')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        timer.reset_stage()

    # ================ Save Model checkpoint =================
    if args.use_distributed_train:
        distributed.barrier()
        if args.local_rank == 0:
            torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
        torch.cuda.empty_cache()
        distributed.destroy_process_group()
    else:
        torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
        torch.cuda.empty_cache()

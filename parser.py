import argparse
import os.path as osp
import os
from misc import pyutils
import random
import torch
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def get_parser():
    parser = argparse.ArgumentParser()

    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--seed", default=-1, type=int, help="Set -1 to use random seed.")

    # Dataset
    parser.add_argument("--voc12_root", default='/path/to/VOC2012', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    parser.add_argument("--num_cores_eval", default=8, type=int)

    # QA-CLIMS
    parser.add_argument("--clims_network", default="net.resnet50_clims", type=str)
    parser.add_argument("--clims_num_epoches", default=15, type=int)
    parser.add_argument("--clims_learning_rate", default=0.00035, type=float)
    parser.add_argument('--hyper', default='10,8,0.2', type=str)
    parser.add_argument('--clip', default='ViT-L/14', type=str)

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.3, type=float)
    parser.add_argument("--conf_bg_thres", default=0.1, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.2)

    # Output Path
    parser.add_argument("--work_space", default="experiments/test", type=str)  # set your path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
    parser.add_argument("--clims_weights_name", default="res50_qa_clims", type=str)
    parser.add_argument("--irn_weights_name", default="res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="sem_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--train_qa_clims_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_clims_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=False)
    parser.add_argument("--train_irn_pass", type=str2bool, default=False)
    parser.add_argument("--make_sem_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_sem_seg_pass", type=str2bool, default=False)

    # DDP
    parser.add_argument("--use_distributed_train",
                        type=str2bool, default=False)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='DONT CHANGE! for distributed train')

    # NCELoss
    parser.add_argument("--nce_t", type=float, default=0.7)

    # VQA
    parser.add_argument("--vqa_fg_file", type=str, default='vqa/voc_vqa_fg_blip.npy')
    parser.add_argument("--vqa_bg_file", type=str, default='vqa/vqa/voc_vqa_bg_blip.npy')
    parser.add_argument("--vqa_fg_cache_file", type=str, default='vqa/voc_vqa_fg_blip_ViT-L-14_cache.npy')
    parser.add_argument("--vqa_bg_cache_file", type=str, default='vqa/voc_vqa_bg_blip_ViT-L-14_cache.npy')

    # mask-adapted CLIP
    parser.add_argument("--use_mask_clip", type=str2bool, default=True)

    return parser


def parse_args(parser):
    args = parser.parse_args()

    args.log_name = osp.join(args.work_space, args.log_name)
    args.cam_weights_name = osp.join(args.work_space, args.cam_weights_name)
    args.irn_weights_name = osp.join(args.work_space, args.irn_weights_name)
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space, args.ir_label_out_dir)
    args.sem_seg_out_dir = osp.join(args.work_space, args.sem_seg_out_dir)
    args.clims_weights_name = osp.join(args.work_space, args.clims_weights_name)

    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if hasattr(args, 'mscoco_root'):
        assert os.path.exists(args.mscoco_root), "MSCOCO root not found"
    elif hasattr(args, 'voc12_root'):
        assert os.path.exists(args.voc12_root), "VOC12 root not found"

    if args.seed != -1:
        seed_torch(args.seed)
    print(f'[pytorch seed: {torch.initial_seed()}]')

    return args

from misc import pyutils
from parser import get_parser, parse_args, str2bool


if __name__ == '__main__':

    parser = get_parser()

    parser.set_defaults(voc12_root='/path/to/VOC2012/')
    parser.set_defaults(clims_learning_rate=0.00035)
    parser.set_defaults(hyper='10,8,0.2')
    parser.set_defaults(clip='ViT-B/32')
    parser.set_defaults(nce_t=0.7)
    parser.set_defaults(vqa_fg_file='vqa/voc_vqa_fg_blip.npy')
    parser.set_defaults(vqa_bg_file='vqa/voc_vqa_bg_blip.npy')
    parser.set_defaults(vqa_fg_cache_file='vqa/voc_vqa_fg_ViT-L-14_cache.npy')
    parser.set_defaults(vqa_bg_cache_file='vqa/voc_vqa_bg_ViT-L-14_cache.npy')

    parser.add_argument("--eval_cam_best", type=str2bool, default=False)
    parser.add_argument("--mask_adapter_alpha", type=float, default=0.1)

    args = parse_args(parser)

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('\n>>step.train_cam:')
        step.train_cam.run(args)

    if args.train_qa_clims_pass is True:
        import step.train_qa_clims

        timer = pyutils.Timer('\n>>step.train_qa_clims:')
        step.train_qa_clims.run(args)

    if args.use_distributed_train is True:
        if args.local_rank != 0:
            print(">>local_rank: %d, exit." % args.local_rank)
            exit(0)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('\n>>step.make_cam:')
        step.make_cam.run(args)
    
    if args.make_clims_pass is True:
        import step.make_clims

        timer = pyutils.Timer('\n>>step.make_clims:')
        step.make_clims.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('\n>>step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('\n>>step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('\n>>step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('\n>>step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('\n>>step.eval_sem_seg:')
        step.eval_sem_seg.run(args)


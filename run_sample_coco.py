from misc import pyutils
from parser import get_parser, parse_args, str2bool


if __name__ == '__main__':

    parser = get_parser()

    parser.add_argument("--mscoco_root", type=str, default='/path/to/COCO2014/')
    parser.set_defaults(clims_learning_rate=0.00035)
    parser.set_defaults(hyper='30,24,0.2')
    parser.set_defaults(clip='ViT-B/32')
    parser.set_defaults(nce_t=0.7)
    parser.set_defaults(vqa_fg_file='vqa/coco_vqa_fg_blip.npy')
    parser.set_defaults(vqa_bg_file='vqa/coco_vqa_bg_blip.npy')
    parser.set_defaults(vqa_fg_cache_file='vqa/coco_vqa_fg_ViT-L-14_cache.npy')
    parser.set_defaults(vqa_bg_cache_file='vqa/coco_vqa_bg_ViT-L-14_cache.npy')

    parser.add_argument("--eval_cam_best", type=str2bool, default=False)
    parser.add_argument("--mask_adapter_alpha", type=float, default=0.1)

    args = parse_args(parser)

    # if args.train_cam_pass is True:
    #     import step_coco.train_cam
    #
    #     timer = pyutils.Timer('\n>>step_coco.train_cam:')
    #     step_coco.train_cam.run(args)

    if args.train_qa_clims_pass is True:
        import step_coco.train_qa_clims

        timer = pyutils.Timer('\n>>step_coco.train_qa_clims:')
        step_coco.train_qa_clims.run(args)

    if args.use_distributed_train is True:
        if args.local_rank != 0:
            print(">>local_rank: %d, exit." % args.local_rank)
            exit(0)

    # if args.make_cam_pass is True:
    #     import step_coco.make_cam
    #
    #     timer = pyutils.Timer('\n>>step_coco.make_cam:')
    #     step_coco.make_cam.run(args)
    
    if args.make_clims_pass is True:
        import step_coco.make_clims

        timer = pyutils.Timer('\n>>step_coco.make_clims:')
        step_coco.make_clims.run(args)

    if args.eval_cam_pass is True:
        import step_coco.eval_cam

        timer = pyutils.Timer('\n>>step_coco.eval_cam:')
        step_coco.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step_coco.cam_to_ir_label

        timer = pyutils.Timer('\n>>step_coco.cam_to_ir_label:')
        step_coco.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step_coco.train_irn

        timer = pyutils.Timer('\n>>step_coco.train_irn:')
        step_coco.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step_coco.make_sem_seg_labels
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('\n>>step_coco.make_sem_seg_labels:')
        step_coco.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step_coco.eval_sem_seg

        timer = pyutils.Timer('\n>>step_coco.eval_sem_seg:')
        step_coco.eval_sem_seg.run(args)


from net.vqa_bg import BG_VQA
import clip
from net.vqa_fg import FG_VQA

from visual_questions import QUESTIONS
from misc.cat_names import category_dict

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['voc', 'coco'])
    parser.add_argument('--vqa_fg_file', type=str,)
    parser.add_argument('--vqa_bg_file', type=str,)
    parser.add_argument('--vqa_fg_cache_file', type=str,)
    parser.add_argument('--vqa_bg_cache_file', type=str,)
    # parser.add_argument('--prompt', type=str, default='a photo of {}.')
    parser.add_argument('--clip', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-L/14'])
    return parser.parse_args()


def gen_bg_text_feats_cache(device, clip_model, clip_name, vqa_file_path, cache_file_path):

    bg_vqa_tool = BG_VQA(
        vqa_file_path,
        QUESTIONS['bg'],
        clip_model,
        clip_name,
        device=device,
        cache_path=cache_file_path,
    )

    bg_vqa_tool.gen_cache()


def gen_fg_text_feats_cache(device, clip_model, clip_name, vqa_file_path, cache_file_path, dataset_name):
    fg_vqa_module = FG_VQA(
        vqa_file_path,
        QUESTIONS['fg'],
        category_dict[dataset_name],
        clip_model,
        clip_name,
        modify_cache=True,
        cache_path=cache_file_path,
    )


if __name__ == '__main__':
    args = parse_args()

    device = 'cuda:0'
    clip_name = args.clip
    clip_model, preprocess = clip.load(clip_name, device='cpu')
    clip_model.to(device)
    clip_model.eval()

    if args.vqa_bg_file is not None:
        print("\n\t\t=== Generating background text features cache ===\n")
        gen_bg_text_feats_cache(
            device=device,
            clip_model=clip_model,
            clip_name=clip_name,
            vqa_file_path=args.vqa_bg_file,
            cache_file_path=args.vqa_bg_cache_file,
        )

    if args.vqa_fg_file is not None:
        print("\n\t\t=== Generating foreground text features cache ===\n")
        gen_fg_text_feats_cache(
            device=device,
            clip_model=clip_model,
            clip_name=clip_name,
            vqa_file_path=args.vqa_fg_file,
            cache_file_path=args.vqa_fg_cache_file,
            dataset_name=args.dataset,
        )

import numpy as np
from net.vqa_bg import gen_clip_features
import clip
from visual_questions import TEXT_PROMPT

fg_path = 'vqa/voc_vqa_fg_ViT-L-14_cache.npy'
new_fg_path = 'vqa/voc_vqa_fg_blip_ViT-L-14_cache.npy'

fg = np.load(fg_path, allow_pickle=True).item()
new_fg = np.load(new_fg_path, allow_pickle=True).item()

# device = 'cuda:0'
# clip_name = 'ViT-L/14'
# clip_model, preprocess = clip.load(clip_name, device='cpu')
# clip_model.to(device)
# clip_model.eval()

img_names = list(fg.keys())
for img_name in img_names:
    labels = list(fg[img_name].keys())
    new_labels = list(new_fg[img_name].keys())
    if labels != new_labels:
        print(img_name, labels, new_labels)
        break
    for label in labels:
        answers = list(fg[img_name][label].keys())
        new_answers = list(new_fg[img_name][label].keys())
        if answers != new_answers:
            print(img_name, label, answers, new_answers)
            break
        for answer in answers:
            if fg[img_name][label][answer].shape != new_fg[img_name][label][answer].shape:
                print(img_name, label, answer, fg[img_name][label][answer].shape, new_fg[img_name][label][answer].shape)
                break
            # gen_feat = gen_clip_features([answer], TEXT_PROMPT, clip_model)
            print(img_name, label, answer, fg[img_name][label][answer].shape, new_fg[img_name][label][answer].shape)
            # 两个tensor的值应该相等
            # assert np.all(fg[img_name][label][answer] == new_fg[img_name][label][answer])
            # print(gen_feat)
            print(fg[img_name][label][answer])
            print(new_fg[img_name][label][answer])
            exit(-1)

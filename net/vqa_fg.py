import json
import os.path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from net.vqa_bg import gen_clip_features
from visual_questions import TEXT_PROMPT

supply_cat = {
    'dining table': ['table', 'dining table'],
    'tv monitor': ['monitor', 'tv monitor', ],
    'potted plant': ['plant', 'potted plant', ],
}


def fg_qa_prompt(questions: list, answers: list, label: str):
    exp_label = supply_cat.get(label, [label])
    new_answers = []
    for q, a in zip(questions, answers):
        if not any([e in a for e in exp_label]):
            a = f'{a} {label}'
        # if label not in a:
        #     a = f'{a} {label}'
        new_answers.append(a)
    return new_answers


class FG_VQA(nn.Module):
    def __init__(self, vqa_file_path: str, questions: list, label_names: list,
                 clip_model, clip_name: str,
                 prompt: str = TEXT_PROMPT,
                 modify_cache: bool = False,
                 label_weight_init: float = 0.5,
                 dtype: torch.dtype = torch.float32,
                 cache_path: str = None
                 ):
        super().__init__()
        self.questions = questions
        self.label_names = label_names
        self.question_num = len(self.questions)
        self.clip_name = clip_name.replace('/', '-')
        self.dtype = dtype
        self.cache_path = cache_path
        print('===')
        print(f">> FG VQA INIT:")
        print(f">> questions: {self.questions}")
        print(f">> label_names: {self.label_names}")
        print(f">> question_num: {self.question_num}")
        print(f">> clip_name: {self.clip_name}")
        print(f">> dtype: {self.dtype}")
        print(f">> cache_path: {self.cache_path}")

        assert os.path.exists(vqa_file_path)
        qa_file = np.load(vqa_file_path, allow_pickle=True).item()

        if modify_cache:
            # generate fg vqa cache, called by tools/gen_text_feats_cache.py
            assert self.cache_path is not None
            cache = self._preproccess_qa(qa_file, prompt, clip_model, cache=None)
            np.save(self.cache_path, cache)
            print(f"save fg cache to {self.cache_path}")
            return

        assert os.path.exists(cache_path)
        print(f"load fg cache from {self.cache_path}")
        cache = np.load(self.cache_path, allow_pickle=True).item()
        cache = self._preproccess_qa(qa_file, prompt, clip_model, cache=cache)

        self._preprocess_label(self.label_names, prompt, clip_model)

        self.weight_num = len(self.label_names)
        qa_feat_init = 1 / self.question_num

        # label_feat_weights: [self.weight_num, 1], init to label_weight_init
        assert 0 <= label_weight_init <= 1
        self.label_feat_weights = nn.Parameter(torch.tensor(
            [[label_weight_init]],
            device='cuda', dtype=self.dtype
        ))
        self.label_feat_weights.requires_grad = False

        # qa_feat_weights: [self.weight_num, question_num], init to 1/question_num
        self.qa_feat_weights = nn.Parameter(torch.tensor(
            [[qa_feat_init for _ in range(self.question_num)]],
            device='cuda', dtype=self.dtype
        ))
        self.qa_feat_weights.requires_grad = False

        print(f"label_feat_weights: {self.label_feat_weights.shape} all init to {label_weight_init} ")
        print(f"qa_feat_weights: {self.qa_feat_weights.shape} all init to {qa_feat_init} ")

    def _preproccess_qa(self, qa_file: dict, prompt: str, clip_model, cache: dict = None):
        file_questions = [q for (q, a) in list(list(qa_file.values())[0].values())[0]['vqa']]
        q_ids = [file_questions.index(q) for q in self.questions]

        self.vqas = {}
        cache = cache if cache is not None else {}
        for img_name in tqdm(qa_file.keys(), desc='processing fg vqa', dynamic_ncols=False, ascii=True):
            self.vqas[img_name] = {}
            if img_name not in cache:
                cache[img_name] = {}
            for label in qa_file[img_name]:
                if label not in cache[img_name]:
                    cache[img_name][label] = {}
                answers = qa_file[img_name][label]['answers']
                origin_answers = [answers[_i] for _i in q_ids]

                answers = fg_qa_prompt(self.questions, origin_answers, label)

                answers_feat = []
                for a_i, ans in enumerate(answers):
                    if cache.get(img_name, {}).get(label, {}).get(ans, None) is not None:
                        ans_feat = cache[img_name][label][ans]
                    else:
                        ans_feat = gen_clip_features([ans], prompt, clip_model)
                        cache[img_name][label][ans] = ans_feat
                    answers_feat.append(ans_feat)

                answers_feat = torch.cat(answers_feat, dim=0)
                # answers_feat = gen_clip_features(answers, prompt, clip_model, use_mean=False)
                self.vqas[img_name][label] = answers_feat

        return cache

    def _preprocess_label(self, label_names: list, prompt: str, clip_model):
        self.labels = {}
        for label in label_names:
            label_feat = gen_clip_features([label], prompt, clip_model)
            self.labels[label] = label_feat

    def forward(self, label: str, img_name: str):

        label_feat = self.labels[label].to(self.dtype).cuda()
        qa_feat = self.vqas[img_name][label].to(self.dtype).cuda()

        label_w = self.label_feat_weights.t()
        qa_w = self.qa_feat_weights

        feat = label_w * label_feat + (1.0 - label_w) * torch.sum(qa_w @ qa_feat, dim=0)

        return feat

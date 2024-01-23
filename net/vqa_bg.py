import numpy as np
import torch
from tqdm import tqdm
import clip
import os

from visual_questions import TEXT_PROMPT


def gen_clip_features(texts: list, prompt: str, clip_model):
    with torch.no_grad():
        prompt_texts = [prompt.format(t) for t in texts]
        feats = None
        if len(prompt_texts) != 0:
            text_feature = clip_model.encode_text(clip.tokenize(prompt_texts).cuda())  # [L, C]
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            feats = text_feature.cpu()
        return feats


class BG_VQA:
    def __init__(self,
                 vqa_file_path: str,
                 questions: list,
                 clip_model, clip_name: str,
                 prompt: str = TEXT_PROMPT,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cuda',
                 cache_path: str = None
                 ):
        self.vqa_file_path = vqa_file_path
        self.questions = questions
        self.clip_name = clip_name.replace('/', '-')
        self.dtype = dtype
        self.device = device
        self.clip_model = clip_model
        self.prompt = prompt
        self.cache_file_path = cache_path
        self._update_profile()

        self.vqa_feats = None

        print('===')
        print(f">> BG VQA INIT:")
        print(f">> questions: {self.questions}")
        print(f">> clip_name: {self.clip_name}")
        print(f">> dtype: {self.dtype}")
        print(f">> cache_path: {self.cache_file_path}")

    def _update_profile(self):
        prop_list = ['clip_name', 'dtype', 'device', 'prompt',
                     'vqa_file_path', 'cache_file_path', 'questions']
        self.profile = {}
        for prop in prop_list:
            self.profile[prop] = getattr(self, prop)

    def _save_cache(self, cache_path):
        total_cache = dict(profile=self.profile, feats=self.vqa_feats)
        np.save(cache_path, total_cache)

    def _load_cache(self, cache_path):
        total_cache = np.load(cache_path, allow_pickle=True).item()
        self.profile = total_cache['profile']
        self.vqa_feats = total_cache['feats']
        self.profile['device'] = self.device
        self.profile['dtype'] = self.dtype

    def gen_cache(self):
        # generate bg vqa cache, called by tools/gen_text_feats_cache.py
        assert os.path.exists(self.vqa_file_path)
        assert self.cache_file_path is not None
        assert isinstance(self.prompt, str)
        assert self.clip_name is not None
        assert self.clip_model is not None

        vqas: dict = np.load(self.vqa_file_path, allow_pickle=True).item()

        self.vqa_feats = self._process_vqa(vqas, self.questions)

        self._update_profile()
        self._save_cache(self.cache_file_path)
        print(f"save bg cache to {self.cache_file_path}")

    def load_cache(self):
        assert os.path.exists(self.cache_file_path)
        print(f"load bg cache from {self.cache_file_path}")
        self._load_cache(self.cache_file_path)

    def _process_vqa(self,
                     vqas: dict,
                     questions: list,
                     ):
        vqa_file_questions = list(list(vqas.values())[0].values())[0]['vqa']
        vqa_file_questions = [q[0] for q in vqa_file_questions]
        q_ids = [vqa_file_questions.index(q) for q in questions]

        result = {}
        for img_name in tqdm(vqas.keys(), desc='processing bg vqa', ascii=True):
            result[img_name] = {}
            for label in vqas[img_name].keys():
                answers = vqas[img_name][label]['answers']
                answers = [answers[_i] for _i in range(len(answers)) if _i in q_ids]
                answers = list(set(answers))
                if label in answers:
                    answers.remove(label)

                result[img_name][label] = {
                    'answers': answers,
                    'answer_feats': gen_clip_features(answers, self.prompt, self.clip_model),
                }

        return result

    def feats(self, label_name: str, image_name: str):
        feat = self.vqa_feats[image_name][label_name]['answer_feats']
        if feat is None:
            print(f"BG no feat for {label_name} in {image_name}")
            return None
        if feat.dtype != self.dtype:
            feat = feat.to(self.dtype)

        return feat.to(self.device)

import torch.nn as nn


class FAKE_CLIP_ADAPTER(nn.Module):
    def __init__(self, mask_clip_model):
        super(FAKE_CLIP_ADAPTER, self).__init__()
        self.clip_model = mask_clip_model


class FAKE_OV_SEG(nn.Module):
    def __init__(self, mask_clip_model):
        super(FAKE_OV_SEG, self).__init__()
        self.clip_adapter = FAKE_CLIP_ADAPTER(mask_clip_model)


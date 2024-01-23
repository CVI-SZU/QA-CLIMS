import torch
from ovseg_utils import FAKE_OV_SEG
import third_party.CLIP.clip as mask_clip
import clip

OV_SEG_CKP_PATH = 'third_party/CLIP/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth'


def load_clip_model(clip_name: str,
                    device: str,
                    mask_adapted: bool = False,
                    dtype: torch.dtype = torch.float32
                    ):
    if mask_adapted:
        assert clip_name == 'ViT-L/14'  # Only ViT-L/14 is supported
        state_dict = torch.load(OV_SEG_CKP_PATH, map_location='cpu')['model']
        if dtype != torch.float32:
            assert device != 'cpu'
            assert dtype == torch.float16
            clip_model, preprocess = mask_clip.load(clip_name, device=device, mask_prompt_depth=3)
            for k, v in state_dict.items():
                state_dict[k] = v.half()
        else:
            clip_model, preprocess = mask_clip.load(clip_name, device='cpu', mask_prompt_depth=3)
        tmp_ov_seg_model = FAKE_OV_SEG(clip_model)
        tmp_ov_seg_model.load_state_dict(state_dict, strict=False)
        clip_model = tmp_ov_seg_model.clip_adapter.clip_model
        print(f'Loaded MASK Adapted CLIP model from {OV_SEG_CKP_PATH} at {device} with dtype {clip_model.dtype}')
    else:
        clip_model, preprocess = clip.load(clip_name, device='cpu')
    # clip_model.eval()
    clip_model.to(device)

    for p in clip_model.parameters():
        p.requires_grad = False

    clip_input_size = 288 if clip_name == 'RN50x4' else 224

    return clip_model, clip_input_size


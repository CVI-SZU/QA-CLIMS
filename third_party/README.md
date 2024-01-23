# Thirdy Party Resources

As described in our [paper](), we use the BLIP model to generate the VQA results of foreground and background questions.
And we use the mask-adapted CLIP model as the image encoder and the text encoder.


## CLIP
> The `CLIP` package is copied from [ov-seg](https://github.com/facebookresearch/ov-seg/tree/main/third_party/CLIP).

This package is no need to install. But **downloading the pre-trained model is necessary.**

### Download the pre-trained model

Please download the finetuned OV-SEG model with CLIP-ViT-L/14, 
which can be found at [their README](https://github.com/facebookresearch/ov-seg/blob/main/GETTING_STARTED.md#try-demo) 
or [here](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing).

```
ovseg_swinbase_vitL14_ft_mpt.pth
```

We recommend to put it at `CLIP/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth`.

---

## BLIP
> The `BLIP` package is copied from [BLIP](https://github.com/salesforce/BLIP/).
> 
> **Install BLIP only if you want to generate your own VQA results.**

### 1. Install BLIP
**Note:** Since the recommended environment of BLIP has conflict with our QA-CLIMS, **please follow [BLIP/README.md](BLIP/README.md) to install BLIP in a new environment**.

### 2. Download the pre-trained model

Please download the finetuned **BLIP w/ ViT-B and CapFilt-L** checkpoint for **VQA**, 
which can be found at [BLIP/README.md #Finetuned checkpoints](BLIP/README.md#Finetuned-checkpoints)
or [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth).

```
model_base_vqa_capfilt_large.pth
```

We recommend to put it at `BLIP/checkpoints/model_base_vqa_capfilt_large.pth`.


### 3. Generate VQA results

```shell
cd BLIP
# generate VOC2012 VQA results of foreground questions
python get_blip_vqa_results.py voc bg \
    --voc_cls_labels_npy_path <path/to/QA-CLIMS/voc12/voc_cls_labels.npy> \
    --voc_img_root_path <path/to/VOC2012/JPEGImages> \
    --blip_model_path checkpoints/model_base_vqa_capfilt_large.pth \
    --target_result_path <path/to/QA-CLIMS/vqa/voc_vqa_bg_blip.npy>
```


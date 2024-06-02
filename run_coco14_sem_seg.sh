
COCO14_ROOT=/path/to/COCO2014
WORK_SPACE=work_space/QA-CLIMS_coco14

VQA_FG_FILE=vqa/coco_vqa_fg_blip.npy
VQA_BG_FILE=vqa/coco_vqa_bg_blip.npy
VQA_FG_CACHE=vqa/coco_vqa_fg_blip_ViT-L-14_cache.npy
VQA_BG_CACHE=vqa/coco_vqa_bg_blip_ViT-L-14_cache.npy

BATCH_SIZE=4
LR=0.00001
NCE_T=0.7
HYPER=30,24,0.2
MASK_ADAPTER_ALPHA=0.1


##### train IRNet and generate pseudo ground truth
CUDA_VISIBLE_DEVICES=0,1 python run_sample_coco.py \
--work_space $WORK_SPACE \
--mscoco_root $COCO14_ROOT \
--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
--vqa_bg_file $VQA_BG_FILE --vqa_bg_cache_file $VQA_BG_CACHE \
--vqa_bg_cache_file $VQA_BG_CACHE \
--use_mask_clip True --clip ViT-L/14 \
--cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True



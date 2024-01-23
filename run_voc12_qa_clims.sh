
VOC12_ROOT=/path/to/VOC2012
WORK_SPACE=work_space/QA-CLIMS_voc12

VQA_FG_FILE=vqa/voc_vqa_fg_blip.npy
VQA_BG_FILE=vqa/voc_vqa_bg_blip.npy
VQA_FG_CACHE=vqa/voc_vqa_fg_blip_ViT-L-14_cache.npy
VQA_BG_CACHE=vqa/voc_vqa_bg_blip_ViT-L-14_cache.npy

BATCH_SIZE=4
LR=0.00035
NCE_T=0.7
HYPER=10,8,0.2
MASK_ADAPTER_ALPHA=0.1


##### train QA-CLIMS to generate initial CAMs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 \
run_sample.py \
--use_distributed_train True --work_space $WORK_SPACE \
--voc12_root $VOC12_ROOT \
--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
--vqa_bg_file $VQA_BG_FILE --vqa_bg_cache_file $VQA_BG_CACHE \
--use_mask_clip True --clip ViT-L/14 \
--train_qa_clims_pass True

CUDA_VISIBLE_DEVICES=0,1 python \
run_sample.py \
--work_space $WORK_SPACE \
--voc12_root $VOC12_ROOT \
--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
--vqa_bg_file $VQA_BG_FILE --vqa_bg_cache_file $VQA_BG_CACHE \
--vqa_bg_cache_file $VQA_BG_CACHE \
--use_mask_clip True --clip ViT-L/14 \
--make_clims_pass True --eval_cam_pass True





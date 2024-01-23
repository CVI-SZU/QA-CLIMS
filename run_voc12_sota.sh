
VOC12_ROOT=/data/xjheng/data/VOC2012
WORK_SPACE=work_space/QA-CLIMS_voc12_sota
VQA_FG_FILE=vqa/blip_vqa_fg[voc_cls_labels_img]_full+2+3+4+5_.npy
VQA_BG_FILE=vqa/blip_vqa_bg[voc_cls_labels_img]_full_.npy
VQA_FG_CACHE=vqa/voc_vqa_fg_ViT-L-14_cache.npy
VQA_BG_CACHE=vqa/voc_vqa_bg_ViT-L-14_cache.npy

BATCH_SIZE=4
LR=0.00035
NCE_T=0.7
HYPER=10,8,0.2
MASK_ADAPTER_ALPHA=0.1

##### prepare vqa cache
#PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python tools/gen_text_feats_cache.py voc \
#--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
#--vqa_bg_file $VQA_BG_FILE --vqa_bg_cache_file $VQA_BG_CACHE \
#--clip ViT-L/14


##### train QA-CLIMS to generate initial CAMs
#CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 \
#run_sample.py \
#--use_distributed_train True --work_space $WORK_SPACE \
#--voc12_root $VOC12_ROOT \
#--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
#--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
#--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
#--vqa_bg_cache_file $VQA_BG_CACHE \
#--use_mask_clip True --clip ViT-L/14 \
#--train_qa_clims_pass True

#CUDA_VISIBLE_DEVICES=6,7 python \
#run_sample.py \
#--work_space $WORK_SPACE \
#--voc12_root $VOC12_ROOT \
#--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
#--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
#--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
#--vqa_bg_cache_file $VQA_BG_CACHE \
#--use_mask_clip True --clip ViT-L/14 \
#--make_clims_pass True --eval_cam_pass True


##### train IRNet and generate pseudo ground truth
#CUDA_VISIBLE_DEVICES=6,7 python run_sample.py \
#--work_space $WORK_SPACE \
#--voc12_root $VOC12_ROOT \
#--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
#--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
#--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
#--vqa_bg_cache_file $VQA_BG_CACHE \
#--use_mask_clip True --clip ViT-L/14 \
#--cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True


CUDA_VISIBLE_DEVICES=6,7 python run_sample.py \
--work_space $WORK_SPACE \
--voc12_root $VOC12_ROOT \
--cam_batch_size $BATCH_SIZE --clims_learning_rate $LR --nce_t $NCE_T --hyper $HYPER \
--mask_adapter_alpha $MASK_ADAPTER_ALPHA \
--vqa_fg_file $VQA_FG_FILE --vqa_fg_cache_file $VQA_FG_CACHE \
--vqa_bg_cache_file $VQA_BG_CACHE \
--use_mask_clip True --clip ViT-L/14 \
--make_sem_seg_pass True --eval_sem_seg_pass True




import numpy as np

from visual_questions import QUESTIONS

fg_file = 'vqa/blip_vqa_fg[voc_cls_labels_img]_full+2+3+4+5_.npy'
new_fg_file = 'vqa/voc_vqa_fg_blip.npy'
bg_file = 'vqa/blip_vqa_bg[voc_cls_labels_img]_full_.npy'
new_bg_file = 'vqa/voc_vqa_bg_blip.npy'


def get_new_vqa(orig_vqa, questions):
    new_vqa = {}
    ori_q_len_list = []
    ori_ans_len_list = []
    new_q_len_list = []
    new_ans_len_list = []
    for img_id, img_results in orig_vqa.items():
        new_vqa[img_id] = {}
        for label, label_results in img_results.items():
            new_vqa[img_id][label] = {'vqa': [], 'answers': []}
            assert len(label_results['vqa']) == len(label_results['answers'])
            ori_q_len = len(label_results['vqa'])
            ori_ans_len = len(label_results['answers'])
            ori_q_len_list.append(ori_q_len)
            ori_ans_len_list.append(ori_ans_len)
            for vqa_item, ans in zip(label_results['vqa'], label_results['answers']):
                q, a = vqa_item
                if q in questions:
                    new_vqa[img_id][label]['vqa'].append(vqa_item)
                    new_vqa[img_id][label]['answers'].append(ans)
            new_q_len = len(new_vqa[img_id][label]['vqa'])
            new_ans_len = len(new_vqa[img_id][label]['answers'])
            new_q_len_list.append(new_q_len)
            new_ans_len_list.append(new_ans_len)
    # 每一个new_q_len应该相等，每一个new_ans_len应该相等
    assert len(set(new_q_len_list)) == 1
    assert len(set(new_ans_len_list)) == 1
    # new_q_len与ori_q_len 对应位置的差值应该相等
    assert len(set([new_q_len - ori_q_len for new_q_len, ori_q_len in zip(new_q_len_list, ori_q_len_list)])) == 1
    # new_ans_len与ori_ans_len 对应位置的差值应该相等
    assert len(set([new_ans_len - ori_ans_len for new_ans_len, ori_ans_len in zip(new_ans_len_list, ori_ans_len_list)])) == 1
    print('ori_q_len_list:', set(ori_q_len_list))
    print('ori_ans_len_list:', set(ori_ans_len_list))
    print('new_q_len_list:', set(new_q_len_list))
    print('new_ans_len_list:', set(new_ans_len_list))

    return new_vqa


fg = np.load(fg_file, allow_pickle=True).item()
# {'2008_000032': {'train': {'vqa': [(q, a), ...], 'answers': [a, ...]}, ...}, ...}
new_fg = get_new_vqa(fg, QUESTIONS['fg'])
np.save(new_fg_file, new_fg)

bg = np.load(bg_file, allow_pickle=True).item()
new_bg = get_new_vqa(bg, QUESTIONS['bg'])
np.save(new_bg_file, new_bg)

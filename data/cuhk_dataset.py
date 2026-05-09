import os
import json
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _safe_open_rgb(path: str, fallback_size=(224, 224)) -> Image.Image:
    """Open an image safely. If missing/corrupt, return a black image."""
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return Image.new('RGB', fallback_size, (0, 0, 0))


def split_dataset_missing(train_data, mode):
    ratio_config = {
        'easy': (0.5, 0.25, 0.25),
        'medium': (0.3, 0.35, 0.35),
        'hard': (0.1, 0.45, 0.45)
    }
    full_ratio, text_ratio, image_ratio = ratio_config[mode]

    person_dict = defaultdict(list)
    for ann in train_data:
        person_dict[ann['id']].append(ann)

    person_ids = list(person_dict.keys())
    np.random.seed(42)
    np.random.shuffle(person_ids)

    total = len(person_ids)
    split1 = int(total * full_ratio)
    split2 = split1 + int(total * text_ratio)

    full_group = set(person_ids[:split1])
    text_missing_group = set(person_ids[split1:split2])
    image_missing_group = set(person_ids[split2:])
    both_missing_group = set(person_ids[split2:])

    processed = []
    for ann in train_data:
        new_ann = ann.copy()
        pid = new_ann['id']
        if pid in full_group:
            pass
        elif pid in text_missing_group:
            # keep GT caption for private missing-modality recovery supervision
            new_ann['gt_captions'] = new_ann.get('captions', '')
            new_ann['captions'] = " "
            new_ann['processed_tokens'] = [" "]
        elif pid in image_missing_group:
            # keep GT image path for private missing-modality recovery supervision
            new_ann['gt_file_path'] = new_ann.get('file_path', '')
            new_ann['file_path'] = 'BLANK_IMG.jpg'
        # if pid in text_missing_group:
        #     new_ann['captions'] = " "
        #     new_ann['processed_tokens'] = [" "]
        #     # new_ann['file_path'] = 'BLANK_IMG.jpg'
        # elif pid in image_missing_group:
        #     new_ann['file_path'] = 'BLANK_IMG.jpg'
        processed.append(new_ann)
    #processed = [ann for ann in train_data if ann['id'] in full_group]

    return processed


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def split_CUHK_PEDE():
    root_dir = '/data0/chenqiang/reid_study/CADA/datasets/CUHK-PEDES'
    raw_dir = 'reid_raw.json'
    all_dir = 'caption_all.json'

    with open(os.path.join(root_dir,raw_dir),'r') as f:
        cap_list = json.load(f)

    train_list = []
    val_list = []
    test_list = []
    #counting = {i:[0,0] for i in range(1, 13003 + 1)}

    for info in cap_list:
        if info['split'] == 'train':
            info1 = info.copy()
            info2 = info.copy()
            info1['captions'] = info['captions'][0]
            info2['captions'] = info['captions'][1]
            train_list.append(info1)
            train_list.append(info2)
        elif info['split'] == 'test':
            test_list.append(info)
        else:
            val_list.append(info)

        #counting[info['id']][0] += 1
        #counting[info['id']][1] += int(len(info['captions']))


    """stat_img = {i:0 for i in range(30)}
    stat_cap = {i:0 for i in range(60)}
    for i in range(1, 13003 + 1):

        stat_img[counting[i][0]] += 1
        stat_cap[counting[i][1]] += 1

    sum_img = 0
    sum_cap = 0
    for i in range(30):
        sum_img += stat_img[i] * i
    for i in range(60):
        sum_cap += stat_cap[i] * i

    print(sum_img)
    print(sum_cap)"""


    # save_json(train_list, os.path.join(root_dir, "train_1.json"))
    return train_list, val_list, test_list


def split_ICFG_PEDE():
    root_dir = 'datasets/ICFG-PEDES'
    raw_dir = 'split.json'

    with open(os.path.join(root_dir, raw_dir), 'r') as f:
        cap_list = json.load(f)

    train_list = cap_list['train'].copy()
    val_list = cap_list['test'].copy()
    test_list = cap_list['test'].copy()

    return train_list, val_list, test_list

class mix_pede_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='', return_masks: bool = False, return_gt: bool = False):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        cuhk_path = os.path.join('datasets', 'CUHK-PEDES', 'imgs')
        icfg_path = os.path.join('datasets', 'ICFG-PEDES')
        train_list, _, _ = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.return_masks = return_masks
        self.return_gt = return_gt
        # Optional masks/GT are only used by private training code.
        if _env_flag('MGCN_RETURN_MISSING_GT'):
            self.return_masks = True
            self.return_gt = True

        self.img_ids = {}
        n = 0
        for i,ann in enumerate(self.annotation):
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            self.annotation[i]['file_path'] = os.path.join(cuhk_path,self.annotation[i]['file_path'])

        train_list, _, _ = split_ICFG_PEDE()
        for i, ann in enumerate(train_list):
            train_list[i]['id'] = train_list[i]['id'] + 20000
            img_id = train_list[i]['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            train_list[i]['file_path'] = os.path.join(icfg_path, train_list[i]['file_path'])
            train_list[i]['captions'] = ann['captions'][0]
        self.annotation = self.annotation + train_list


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = _safe_open_rgb(ann['file_path'])
        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt + pre_caption(captions, self.max_words)

        if not self.return_masks and not self.return_gt:
            return image, captions, self.img_ids[ann['id']]

        # mix dataset typically has no synthetic missing by default
        mask_img = 1
        mask_txt = 1
        gt_image = None
        gt_caption = None
        if self.return_gt:
            gt_caption = captions
            gt_image = image

        return image, captions, self.img_ids[ann['id']], mask_img, mask_txt, gt_image, gt_caption



class cuhk_pede_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt='', return_masks: bool = False, return_gt: bool = False, missing_mode: str = 'easy'):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        train_list, _, _ = split_CUHK_PEDE()

        train_list = split_dataset_missing(train_list, missing_mode)

        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.return_masks = return_masks
        self.return_gt = return_gt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # -------- observed (possibly missing) inputs --------
        image_path = self.image_root
        image = _safe_open_rgb(os.path.join(image_path, ann['file_path']))
        image = self.transform(image)

        captions_raw = ann.get('captions', '')
        captions = self.prompt + pre_caption(captions_raw, self.max_words)

        if not self.return_masks and not self.return_gt:
            return image, captions, self.img_ids[ann['id']]

        # -------- explicit missing masks --------
        is_img_missing = os.path.basename(ann.get('file_path', '')) == 'BLANK_IMG.jpg'
        # treat empty / whitespace-only caption as missing
        is_txt_missing = (str(captions_raw).strip() == '') or (str(captions_raw).strip() == ' ')
        mask_img = 0 if is_img_missing else 1
        mask_txt = 0 if is_txt_missing else 1

        # -------- optional GT (only meaningful when synthetic missing is used) --------
        gt_image = None
        gt_caption = None
        if self.return_gt:
            # If we injected missing, try to recover GT from cached fields; otherwise fall back to current input.
            if is_img_missing and ann.get('gt_file_path'):
                gt_image = _safe_open_rgb(os.path.join(image_path, ann['gt_file_path']))
                gt_image = self.transform(gt_image)
            else:
                gt_image = image

            if is_txt_missing and ann.get('gt_captions') is not None:
                gt_caption = self.prompt + pre_caption(str(ann.get('gt_captions', '')), self.max_words)
            else:
                gt_caption = captions

        return image, captions, self.img_ids[ann['id']], mask_img, mask_txt, gt_image, gt_caption


class cuhk_pede_caption_eval(Dataset):
    def __init__(self, transform, image_root, split):
        _, val_list, test_list = split_CUHK_PEDE()

        assert split in ['val','test']

        if split == 'val':
            self.annotation = val_list
        elif split == 'test':
            self.annotation = test_list
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root,ann['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['id']

        return image, img_id


class cuhk_pede_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, split, max_words=72):
        train_list, val_list, test_list = split_CUHK_PEDE()
        assert split in ['val', 'test']

        if split == 'val':
            self.annotation = val_list
        elif split == 'test':
            self.annotation = test_list
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.txt2pid  = []
        self.img2pid  = []

        person = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['file_path'])

            caps = ann['captions']

            self.text.append(pre_caption(caps[0],max_words))
            self.text.append(pre_caption(caps[1],max_words))
            pid = ann['id']
            self.img2pid.append(pid)
            self.txt2pid.append(pid)
            self.txt2pid.append(pid)
            if pid not in person.keys():
                person[pid] = {'image':[img_id],'text':[txt_id,txt_id+1]}
            else:
                person[pid]['image'].append(img_id)
                person[pid]['text'].append(txt_id)
                person[pid]['text'].append(txt_id+1)
            txt_id = txt_id + 2

        for pid in person.keys():
            for img_id in person[pid]['image']:
                self.img2txt[img_id] = person[pid]['text']
                assert self.img2pid[img_id] == pid
            for txt_id in person[pid]['text']:
                self.txt2img[txt_id] = person[pid]['image']
                assert self.txt2pid[txt_id] == pid

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


# eval for train_data
class cuhk_pede_trainset_eval(Dataset):
    def __init__(self, transform, image_root, max_words=50):
        train_list, val_list, test_list = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.txt2pid  = {}
        self.img2pid  = {}

        person = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['file_path'])

            self.text.append(pre_caption(ann['captions'],max_words))
            pid = ann['id']
            self.img2pid[img_id] = pid
            self.txt2pid[txt_id] = pid
            if pid not in person.keys():
                person[pid] = {'image':[img_id],'text':[txt_id]}
            else:
                person[pid]['image'].append(img_id)
                person[pid]['text'].append(txt_id)
            txt_id = txt_id + 1

        for pid in person.keys():
            for img_id in person[pid]['image']:
                self.img2txt[img_id] = person[pid]['text']
            for txt_id in person[pid]['text']:
                self.txt2img[txt_id] = person[pid]['image']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


if __name__ == '__main__':
    pass

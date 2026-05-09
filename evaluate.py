import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml

from data import create_dataset, create_loader
from models.cada import build_model
import utils


@torch.no_grad()
def evaluation(model, data_loader, device, config, itm=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    text_bs = int(config.get("text_batch_size", 256))
    text_ids = []
    text_embeds = []
    text_atts = []

    for i in range(0, len(texts), text_bs):
        text = texts[i: min(len(texts), i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=73,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            mode="text",
        )
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for image, _ in data_loader:
        image = image.to(device, non_blocking=True)
        image_feat = model.visual_encoder(image)
        image_embed = F.normalize(model.vision_proj(image_feat[:, 0, :]), dim=-1)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    sims_matrix = text_embeds @ image_embeds.t()

    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 500, header)):
        if itm:
            topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
            encoder_output = image_feats[topk_idx.cpu()].to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder(
                text_ids[i].repeat(config["k_test"], 1),
                attention_mask=text_atts[i].repeat(config["k_test"], 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score

    score_matrix_t2i = score_matrix_t2i + sims_matrix
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Evaluation time {total_time_str}")
    return score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_t2i, txt2img, img2txt, img2pid, txt2pid):
    img2pid = np.asarray(img2pid)
    txt2pid = np.asarray(txt2pid)
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in txt2img[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    indices = np.argsort(-scores_t2i, axis=1)
    pred_labels = img2pid[indices]
    matches = np.equal(txt2pid.reshape(-1, 1), pred_labels)

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = np.stack(tmp_cmc, 1) * matches
    ap = tmp_cmc.sum(1) / num_rel
    m_ap = ap.mean() * 100

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    ir_mean = (ir1 + ir5 + ir10) / 3

    return {
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "mAP": m_ap,
        "r_mean": ir_mean + m_ap,
    }


def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config["dataset"], config)
    num_classes = len(train_dataset.img_ids)
    print(f"loaded training identities for model head: {num_classes}")

    _, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        [None, None, None],
        batch_size=[config["batch_size_test"]] * 3,
        num_workers=[args.num_workers] * 3,
        is_trains=[False, False, False],
        collate_fns=[None, None, None],
    )

    print("Creating model")
    model = build_model(
        pretrained=config["pretrained"],
        mode="eval",
        num_classes=num_classes,
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
    )
    model = model.to(device)

    split_loader = test_loader if args.split == "test" else val_loader
    score_t2i = evaluation(model, split_loader, device, config, itm=not args.global_only)
    result = itm_eval(
        score_t2i,
        split_loader.dataset.txt2img,
        split_loader.dataset.img2txt,
        split_loader.dataset.img2pid,
        split_loader.dataset.txt2pid,
    )
    print(result)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.split}_evaluate.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MGCN checkpoints.")
    parser.add_argument("--config", default="./configs/retrieval_rstp.yaml")
    parser.add_argument("--output_dir", default="output/MGCN")
    parser.add_argument("--checkpoint", default="./output/rstp_checkpoint_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size_test", default=0, type=int)
    parser.add_argument("--k_test", default=32, type=int)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--global_only", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.Loader)
    if args.batch_size_test > 0:
        config["batch_size_test"] = args.batch_size_test
    config["pretrained"] = args.checkpoint
    config["k_test"] = args.k_test

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w", encoding="utf-8"))
    main(args, config)

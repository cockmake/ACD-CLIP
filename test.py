import argparse
import logging
import os
from glob import glob

import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DOMAINS, get_text_and_image_dataset
from utils import (
    get_multiple_adapted_text_embedding, metrics_eval_gpu,
)
from model.adapter import (
    ACDCLIP
)
from model.clip import create_model


def get_predictions(
        model: ACDCLIP,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device,
        dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].to(device).to(torch.int32)
        label = input_data["label"].to(device).to(torch.int32)
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # get text
        epoch_text_features = class_text_embeddings.unsqueeze(dim=1)  # [n_groups, 1, 768, 2]
        # forward image
        seg_tokens, det_tokens = model(image)  # [bs, patch_size, 768] * n_groups, [bs, 768] * n_groups
        seg_features = torch.stack(seg_tokens, dim=0)  # [n_groups, bs, patch_num, 768]
        det_features = torch.stack(det_tokens, dim=0)  # [n_groups, bs, 768]
        B = seg_features.shape[1]
        epoch_text_features = epoch_text_features.repeat(1, B, 1, 1)  # [n_groups, bs, 768, 2]
        cls_preds = [
            torch.matmul(
                det_features[i].unsqueeze(dim=1),  # [bs, 1, 768]
                epoch_text_features[i],  # [bs, 768, 2]
            ).squeeze(1) for i in range(det_features.shape[0])
        ]  # [bs, 2] * n_groups
        cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)  # [bs, 2]
        pred = F.softmax(cls_preds, dim=1)[:, 1]
        preds_image.append(pred)
        # [bs, img_size, img_size]
        seg_pred = model.vision_text_fusion_gate_seg(seg_features, epoch_text_features, test_mode=True,
                                                     domain=DOMAINS[dataset])
        preds.append(seg_pred)
    masks = torch.concatenate(masks, dim=0)  # [bs, 1, 518, 518]
    labels = torch.concatenate(labels, dim=0)  # [bs]
    preds = torch.concatenate(preds, dim=0)  # [bs, 518, 518]
    preds_image = torch.concatenate(preds_image, dim=0)  # [bs]
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Testing")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    # testing
    parser.add_argument("--n_groups", type=int, default=4, help="number of groups for adapter")

    parser.add_argument("--lora_rank", type=int, default=16, help="rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")

    parser.add_argument("--conv_lora_rank", type=int, default=8, help="rank for LoRA adapters")
    parser.add_argument("--conv_lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")
    parser.add_argument("--conv_kernel_size_list", type=int, nargs="+", default=[3, 5],
                        help="kernel size for convolutional LoRA adapters")

    parser.add_argument("--dataset", type=str, default="MPDD")
    parser.add_argument("--batch_size", type=int, default=84)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="ckpt/issue")

    args = parser.parse_args()
    # ========================================================
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("args: %s", vars(args))
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_device}" if use_cuda else "cpu")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=args.n_groups,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        conv_lora_rank=args.conv_lora_rank,
        conv_lora_alpha=args.conv_lora_alpha,
        conv_kernel_size_list=args.conv_kernel_size_list,
    ).to(device)
    model.eval()
    ckp_files = glob(args.save_path + "/adapter_*.pth")
    assert len(ckp_files) > 0, "adapter checkpoint not found"
    for file in ckp_files:
        checkpoint = torch.load(file)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")
        image_datasets = get_text_and_image_dataset(
            args.dataset,
            args.img_size,
            "test"
        )
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        with torch.no_grad():
            text_embeddings = get_multiple_adapted_text_embedding(model, args.dataset, device)

        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False
            )
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    dataset=args.dataset,
                )
            class_result_dict = metrics_eval_gpu(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)
        mean_vals = df[df.columns[1:]].mean()
        df.loc[len(df), df.columns[1:]] = mean_vals
        df.loc[len(df) - 1, "class name"] = "Average"
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()

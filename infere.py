import os
import argparse
import torch
from torch.utils.data import DataLoader

from main import get_args_parser
from data.dataset import build_dataset_single, build_dataset_multimodal_single
from model.flair_distill_flair import FLAIRMultiLayer

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize


def find_checkpoint(args):

    if getattr(args, "ckpt_path", None):
        if os.path.isfile(args.ckpt_path):
            return args.ckpt_path
        else:
            raise FileNotFoundError(f" ckpt_path not found: {args.ckpt_path}")

    candidates = [
        "checkpoint_best.pth",
        "checkpoint_last.pth",
        "checkpoint.pth",
    ]
    for name in candidates:
        path = os.path.join(args.output_dir, name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f" {args.output_dir} not found or does not contain checkpoint_best.pth / checkpoint_last.pth / checkpoint.pth. "
    )


def load_model_state(model, ckpt_path, device):
    print("\n================ Inference =================")
    print(f"{ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

    clean_state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print("-> load_state_dict complete:")
    print(f"   missing keys   : {len(missing)}")
    if len(missing) < 20:
        print(f"   {missing}")
    print(f"   unexpected keys: {len(unexpected)}")
    if len(unexpected) < 20:
        print(f"   {unexpected}")
    print("===========================================\n")


def build_dataloader(args, split, modality, pin_memory=True):

    if split == "test":
        dataset = build_dataset_single("test", args=args, mod=modality)
    elif split == "dev":
        dataset = build_dataset_single("dev", args=args, mod=modality)
    else:
        raise ValueError(f"not supported split: {split}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader


def compute_all_metrics(y_true, y_pred, y_prob, num_classes):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    assert y_prob.shape[1] == num_classes
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    precisions = []
    recalls = []
    specificities = []
    f1_pr_list = []
    f1_ss_list = []

    total = cm.sum()

    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = total - TP - FN - FP

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # sensitivity
        spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # specificity

        f1_pr = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_ss = 2 * rec * spe / (rec + spe) if (rec + spe) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spe)
        f1_pr_list.append(f1_pr)
        f1_ss_list.append(f1_ss)

        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "specificity": spe,
            "f1_pr": f1_pr,
            "f1_ss": f1_ss,
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_specificity = float(np.mean(specificities))
    macro_f1_pr = float(np.mean(f1_pr_list))
    macro_f1_ss = float(np.mean(f1_ss_list))

    accuracy = float((y_true == y_pred).mean())
    kappa = float(cohen_kappa_score(y_true, y_pred))

    y_true_onehot = label_binarize(y_true, classes=np.arange(num_classes))
    ap_per_class = average_precision_score(
        y_true_onehot, y_prob, average=None
    )
    mAP = float(np.nanmean(ap_per_class))

    metrics = {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_f1_pr": macro_f1_pr,
        "macro_f1_ss": macro_f1_ss,
        "ap_per_class": ap_per_class,
        "mAP": mAP,
        "accuracy": accuracy,
        "kappa": kappa,
    }
    return metrics


def main():

    base_parser = get_args_parser()

    parser = argparse.ArgumentParser(
        "Inference for MultiEYE FLAIRMultiLayer",
        parents=[base_parser],
        add_help=True,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=r"Your Checkpoint Path",
        help="checkpoint path for inference (overrides output_dir if specified)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="which split to perform inference on",
    )

    args = parser.parse_args()
    device = torch.device(args.device, int(args.device_id))
    concept_feat_path = os.path.join(args.concept_path, "concepts_raw.npy")
    args.classnames = [
        "NOR",
        "AMD",
        "CSC",
        "DR",
        "GLC",
        "MEM",
        "MYO",
        "RVO",
        "WAMD",
    ]

    model = FLAIRMultiLayer(args, device, concept_feat_path)
    model = model.to(device)

    ckpt_path = find_checkpoint(args)
    load_model_state(model, ckpt_path, device)

    data_loader = build_dataloader(args, split=args.split, modality=args.modality)

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    metrics = compute_all_metrics(
        y_true, y_pred, y_prob, num_classes=args.n_classes
    )

    print(f"====== Inference results on split = {args.split} ======")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Kappa        : {metrics['kappa']:.4f}")
    print(f"Macro P      : {metrics['macro_precision']:.4f}")
    print(f"Macro R      : {metrics['macro_recall']:.4f}")
    print(f"Macro Spec   : {metrics['macro_specificity']:.4f}")
    print(f"Macro P-R F1 : {metrics['macro_f1_pr']:.4f}")
    print(f"Macro S-S F1 : {metrics['macro_f1_ss']:.4f}")
    print(f"mAP          : {metrics['mAP']:.4f}")
    print("------------- Per-class -------------")
    for c in range(args.n_classes):
        m = metrics["per_class"][c]
        print(
            f"Class {c}: "
            f"P={m['precision']:.4f}, "
            f"R={m['recall']:.4f}, "
            f"Spec={m['specificity']:.4f}, "
            f"F1_PR={m['f1_pr']:.4f}, "
            f"F1_SS={m['f1_ss']:.4f}"
        )
    print("======================================")


if __name__ == "__main__":
    main()
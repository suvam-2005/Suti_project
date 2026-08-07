"""
train.py — Disaster Detection classifier (transfer learning, ResNet18)

Classifies an image into one of the disaster classes found as subfolders
under data/train and data/val. Works with any number of classes — add or
remove folders and it adapts automatically (no code changes needed).

Expected folder layout (rename/add subfolders as needed):

    data/
      train/
        flood/       *.jpg
        wildfire/    *.jpg
        cyclone/     *.jpg
        earthquake/  *.jpg
      val/
        flood/       *.jpg
        wildfire/    *.jpg
        cyclone/     *.jpg
        earthquake/  *.jpg

Usage:
    python train.py --data_dir data --epochs 15 --batch_size 32 --out_dir artifacts

Outputs (in --out_dir):
    best_model.pth         -> state_dict, for further training / our own backend
    model_scripted.pt      -> TorchScript, portable, loads with NO source code needed
    model.onnx              -> ONNX, runs anywhere (Python, JS/onnxruntime-web, mobile, C++)
    labels.json              -> class index -> class name mapping
    training_log.json         -> per-epoch metrics
"""

import argparse
import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os


def get_dataloaders(data_dir: str, batch_size: int, img_size: int = 224, num_workers: int = 0):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_datasets = {
        "train": datasets.ImageFolder(Path(data_dir) / "train", train_tf),
        "val": datasets.ImageFolder(Path(data_dir) / "val", val_tf),
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            # Use the caller-provided num_workers (safe default chosen in main()).
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ["train", "val"]
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    class_names = image_datasets["train"].classes  # alphabetical, e.g. ['cyclone','earthquake','flood','wildfire']

    return dataloaders, dataset_sizes, class_names


def build_model(num_classes: int, freeze_backbone: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze last residual block for fine-tuning — big accuracy boost over
        # a fully frozen backbone, still fast since most of the net is frozen
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def train_model(model, dataloaders, dataset_sizes, device, epochs, lr, out_dir: Path):
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log = []
    patience, patience_counter = 5, 0  # early stopping

    for epoch in range(epochs):
        t0 = time.time()
        epoch_stats = {"epoch": epoch + 1}

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss, running_corrects = 0.0, 0
            # iterate with index so we can print lightweight batch progress
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase], start=1):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # show a small heartbeat so the user knows progress is being made
                if batch_idx % 10 == 0 or batch_idx == len(dataloaders[phase]):
                    print(f"{phase}: batch {batch_idx}/{len(dataloaders[phase])}  loss={loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            epoch_stats[f"{phase}_loss"] = epoch_loss
            epoch_stats[f"{phase}_acc"] = epoch_acc

            if phase == "val":
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    torch.save(best_model_wts, out_dir / "best_model.pth")
                else:
                    patience_counter += 1

        epoch_stats["time_sec"] = round(time.time() - t0, 1)
        log.append(epoch_stats)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={epoch_stats['train_loss']:.4f} train_acc={epoch_stats['train_acc']:.4f} | "
            f"val_loss={epoch_stats['val_loss']:.4f} val_acc={epoch_stats['val_acc']:.4f} | "
            f"best_val_acc={best_acc:.4f}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs).")
            break

    model.load_state_dict(best_model_wts)
    return model, best_acc, log


def export_model(model, class_names, out_dir: Path, img_size: int = 224):
    model.eval()
    device = next(model.parameters()).device

    # 1. TorchScript — portable, runs with `torch.jit.load`, no model class needed
    example = torch.rand(1, 3, img_size, img_size).to(device)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(out_dir / "model_scripted.pt"))

    # 2. ONNX — framework-agnostic, runs anywhere (onnxruntime, browser via onnxruntime-web, mobile, C++)
    try:
        torch.onnx.export(
            model,
            example,
            str(out_dir / "model.onnx"),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
        )
    except ModuleNotFoundError as e:
        print(
            "ONNX export skipped: missing dependency. To enable ONNX export install: `pip install onnx onnxscript`"
        )
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # 3. Labels so any consumer knows how to map output indices to class names
    with open(out_dir / "labels.json", "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

    print(f"Exported: model_scripted.pt, model.onnx, labels.json -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train the Disaster Detection classifier.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of DataLoader worker processes (default: 0 on Windows, small positive on POSIX)")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # choose a safe default for num_workers (Windows: 0 to avoid spawn overhead)
    if args.num_workers is None:
        default_workers = 0 if os.name == "nt" else min(4, (os.cpu_count() or 1))
        num_workers = default_workers
    else:
        num_workers = args.num_workers

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size, num_workers=num_workers
    )
    print(f"Disaster classes detected: {class_names} | train={dataset_sizes['train']} val={dataset_sizes['val']}")

    model = build_model(num_classes=len(class_names), freeze_backbone=args.freeze_backbone).to(device)

    model, best_acc, log = train_model(model, dataloaders, dataset_sizes, device, args.epochs, args.lr, out_dir)
    print(f"Best validation accuracy: {best_acc:.4f}")

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    export_model(model, class_names, out_dir, args.img_size)


if __name__ == "__main__":
    main()

"""
PySpark augmentation + PyTorch training for LocalGlobalCrossAttentionNet.

Expected dataset:
data/
  Train/{Blotch_Apple,Normal_Apple,Rot_Apple,Scab_Apple}/...
  Val/{Blotch_Apple,Normal_Apple,Rot_Apple,Scab_Apple}/...
  Test/{Blotch_Apple,Normal_Apple,Rot_Apple,Scab_Apple}/...

Install:
    pip install pyspark torch torchvision timm pillow

Notes:
- Only Train is augmented by default. Val/Test remain unchanged for fair evaluation.
- Spark distributes image augmentation across CPU workers.
- PyTorch trains the neural network on CPU/GPU on the driver process.
"""
from __future__ import annotations

import time

import findspark
findspark.init()

import os
import random
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from global_local_cross_attention import build_model


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("../data_5k/data_5k").resolve()
AUG_ROOT = Path("data_augmented").resolve()

CLASS_NAMES = [
    "Blotch_Apple",
    "Normal_Apple",
    "Rot_Apple",
    "Scab_Apple",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
AUG_PER_IMAGE = 100       # number of new augmented copies per original Train image
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0         # on Windows, 0 is the safest default
SEED = 42


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def uri_to_local_path(uri: str) -> str:
    """Convert Spark file:/ URI to an OS path. Plain paths pass through."""
    if uri.startswith("file:"):
        parsed = urlparse(uri)
        path = unquote(parsed.path)
        # file:/C:/... -> C:/... on Windows
        if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path
    return uri


def is_image(path: str) -> bool:
    """Return True if PIL can open and verify the file as an image, regardless of extension."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 1) PySpark: discover the dataset without collecting image bytes
# -----------------------------------------------------------------------------
# PIL validates the file contents, so image discovery does not depend on
# .jpg/.png/etc. filename extensions.
is_image_udf = F.udf(lambda uri: is_image(uri_to_local_path(uri)), BooleanType())


def image_manifest_df(spark: SparkSession, split_root: Path):
    """
    Return ALL valid images found recursively inside the four class folders.

    Folder matching is case-insensitive, so names such as Blotch_apple and
    Normal_apple are accepted and normalized to the canonical class names.
    No image-extension filter is used.
    """
    df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(str(split_root))
        .select("path")
    )

    # (?i) makes class-folder matching case-insensitive.
    # Spark paths normally use '/', but this also accepts Windows-style '\\'.
    class_regex = (
        r"(?i)(?:/|\\)"
        r"(Blotch_Apple|Normal_Apple|Rot_Apple|Scab_Apple)"
        r"(?:/|\\)"
    )

    df = (
        df.withColumn(
            "folder_name",
            F.regexp_extract(F.col("path"), class_regex, 1),
        )
        .filter(F.col("folder_name") != "")
    )

    folder_lower = F.lower(F.col("folder_name"))
    df = df.withColumn(
        "label_name",
        F.when(folder_lower == "blotch_apple", F.lit("Blotch_Apple"))
        .when(folder_lower == "normal_apple", F.lit("Normal_Apple"))
        .when(folder_lower == "rot_apple", F.lit("Rot_Apple"))
        .when(folder_lower == "scab_apple", F.lit("Scab_Apple")),
    )

    # Validate the actual file contents with PIL. This keeps every readable
    # image format supported by PIL, even if its extension is unusual.
    return (
        df.withColumn("is_image", is_image_udf(F.col("path")))
        .filter(F.col("is_image") == F.lit(True))
        .select("path", "label_name")
    )


def show_image_counts(df, split_name: str):
    """Print total image count and count for every class."""
    print(f"\n{split_name} image counts:")
    (
        df.groupBy("label_name")
        .count()
        .orderBy("label_name")
        .show(truncate=False)
    )
    print(f"Total {split_name} images: {df.count()}")


# -----------------------------------------------------------------------------
# 2) Augmentation operations
# -----------------------------------------------------------------------------
# Mapping requested by the user:
# rotate              -> RandomRotation
# skew                -> RandomPerspective
# gaussian distortion -> ElasticTransform (Gaussian-smoothed displacement field)
# shear               -> RandomAffine(shear=...)
# crop random         -> RandomResizedCrop
# flip random         -> RandomHorizontalFlip / RandomVerticalFlip
AUGMENT_PIPELINE = transforms.Compose(
    [
        transforms.RandomRotation(degrees=30),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.75),
        transforms.ElasticTransform(alpha=45.0, sigma=5.0),
        transforms.RandomAffine(degrees=0, shear=(-15, 15, -10, 10)),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.90, 1.10),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.20),
    ]
)


def augment_partition(rows):
    """Spark worker function: create AUG_PER_IMAGE augmented images per input."""
    # Different seed per Python worker/process.
    worker_seed = (os.getpid() * 9973 + SEED) & 0xFFFFFFFF
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    for row in rows:
        src = Path(uri_to_local_path(row.path))
        label = row.label_name

        if not src.exists():
            yield (str(src), label, "missing", "")
            continue

        if not is_image(str(src)):
            yield (str(src), label, "not_image", "")
            continue

        dst_dir = AUG_ROOT / "Train" / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src) as im:
                image = im.convert("RGB")

                # Keep original image in augmented dataset.
                original_dst = dst_dir / src.name
                if not original_dst.exists():
                    image.save(original_dst, quality=95)
                yield (str(original_dst), label, "original", str(src))

                # Produce new augmented copies.
                for k in range(AUG_PER_IMAGE):
                    aug = AUGMENT_PIPELINE(image)
                    out = dst_dir / f"{src.stem}__aug_{k:02d}.jpg"
                    aug.save(out, format="JPEG", quality=95)
                    yield (str(out), label, "augmented", str(src))
        except Exception as exc:
            yield (str(src), label, f"error:{type(exc).__name__}", str(exc))


def copy_split(split: str):
    """Copy Val/Test unchanged so output keeps the same directory structure."""
    src_root = DATA_ROOT / split
    dst_root = AUG_ROOT / split
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def run_spark_augmentation(spark: SparkSession):
    AUG_ROOT.mkdir(parents=True, exist_ok=True)

    train_df = image_manifest_df(spark, DATA_ROOT / "Train")
    print("Original Train images:", train_df.count())
    show_image_counts(train_df, "Train")

    # mapPartitions is important: transforms are initialized once in each worker,
    # not once for every single image.
    result_rdd = train_df.repartition(max(spark.sparkContext.defaultParallelism, 1)).rdd.mapPartitions(
        augment_partition
    )
    result_df = spark.createDataFrame(
        result_rdd,
        schema="path string, label_name string, kind string, source string",
    ).cache()

    # Trigger Spark execution.
    print("Augmentation result:")
    result_df.groupBy("kind").count().show(truncate=False)
    result_df.groupBy("label_name", "kind").count().orderBy("label_name", "kind").show(truncate=False)

    # Evaluation data must remain untouched.
    copy_split("Val")
    copy_split("Test")

    return result_df

def get_augmented_images(spark):
    train_df = image_manifest_df(spark, AUG_ROOT / 'Train' )
    return train_df

# -----------------------------------------------------------------------------
# 3) Spark -> lightweight path manifest -> PyTorch Dataset
# -----------------------------------------------------------------------------
def collect_manifest(spark: SparkSession, split: str):
    df = image_manifest_df(spark, AUG_ROOT / split)
    rows = df.select("path", "label_name").collect()
    samples = [
        (uri_to_local_path(r.path), CLASS_TO_IDX[r.label_name])
        for r in rows
    ]
    random.Random(SEED).shuffle(samples)
    print(f"{split}: {len(samples)} images")
    return samples


class AppleDiseaseDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# Augmentation has already been materialized by Spark, so training transforms
# only resize/normalize. This avoids accidentally applying a second heavy augmentation.
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def make_loaders(spark: SparkSession):
    train_ds = AppleDiseaseDataset(collect_manifest(spark, "Train"), train_transform)
    val_ds = AppleDiseaseDataset(collect_manifest(spark, "Val"), eval_transform)
    test_ds = AppleDiseaseDataset(collect_manifest(spark, "Test"), eval_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# 4) Train LocalGlobalCrossAttentionNet
# -----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_n = targets.size(0)
            total_loss += loss.item() * batch_n
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_items += batch_n

    return total_loss / total_items, total_correct / total_items


def train_model(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    best_val_acc = -1.0
    best_path = "best_local_global_cross_attention.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Saved:", best_path)
    return model, best_path


def test_model(model, checkpoint_path, test_loader):
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}, test accuracy={test_acc:.4f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    spark = (
        SparkSession.builder
        .appName("AppleDisease-Augmentation-Training")
        .master("spark://192.168.0.100:7077")  # remove/change this when running on a real Spark cluster
        # .master("local[8]")  # remove/change this when running on a real Spark cluster
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        
        run_spark_augmentation(spark)
        train_loader, val_loader, test_loader = make_loaders(spark)
        s = time.time()
        model, checkpoint_path = train_model(train_loader, val_loader)
        e = time.time()
        print('Time:', e-s)
        # test_model(model, checkpoint_path, test_loader)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

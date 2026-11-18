# MONAI — Medical Open Network for AI

Nov 18, 2026 · @Sera

## What is MONAI?

MONAI (Medical Open Network for AI) is an open-source deep-learning framework purpose-built for medical imaging. It was initiated by NVIDIA and King's College London in 2019 and is now governed by a consortium that includes academic medical centres, national labs and imaging vendors. The project sits under the Linux Foundation. Source code, issues and roadmap are on [GitHub](https://github.com/Project-MONAI/MONAI); the latest stable release as of late 2026 is **1.6**. Licence: Apache 2.0 — free for research and commercial use.

MONAI is built on top of PyTorch and follows its conventions: tensors, `nn.Module` networks, `Dataset` / `DataLoader` pipelines. The key addition is a comprehensive layer of medical-imaging-specific building blocks that sit above raw PyTorch and below task-specific application code.

```
pip install "monai[nibabel,tqdm]==1.6.1"
```

This single command installs MONAI with NIfTI file support (`nibabel`) and progress bars (`tqdm`). Optional extras add ITK, OpenCV, PyTorch Ignite, and others.

## Core components

### Transforms (`monai.transforms`)

Transforms are the most-used part of MONAI. They work on dictionaries keyed by modality (`"image"`, `"label"`), making it easy to apply the same geometric augmentation to both simultaneously. They compose, they are deterministic when seeded, and they can run on CPU or GPU.

```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged,
    RandCropByPosNegLabeld,
)

train_tf = Compose([
    LoadImaged(keys=["image", "label"]),          # NIfTI → tensor
    EnsureChannelFirstd(keys=["image", "label"]), # add channel dim
    Orientationd(keys=["image", "label"], axcodes="RAS"),  # canonical axes
    Spacingd(keys=["image", "label"],             # resample to 1.5×1.5×2 mm
             pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys="image",            # HU window → [0, 1]
                         a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
    RandCropByPosNegLabeld(keys=["image", "label"],  # balanced patch sampling
                           label_key="label", spatial_size=(96, 96, 96),
                           pos=1, neg=1, num_samples=4),
])
```

### Datasets (`monai.data`)

`CacheDataset` runs the deterministic part of the transform pipeline once and stores every scan in memory. The random augmentations (crop, flip, noise) run on-the-fly during training. On a 40-scan training set this gives a 5–10× speedup over re-loading from disk at every epoch.

```python
from monai.data import CacheDataset, DataLoader

ds = CacheDataset(files, train_tf, cache_rate=1.0, num_workers=4)
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4)
```

### Networks (`monai.networks.nets`)

MONAI ships production-quality implementations of the standard medical segmentation architectures. All support 2D and 3D and accept `spatial_dims` as an argument.

| Class | Architecture | Typical use |
| --- | --- | --- |
| `UNet` | 3D U-Net with optional residual units | General segmentation baseline |
| `UNETR` | ViT encoder + CNN decoder | Large-dataset whole-body |
| `SwinUNETR` | Swin Transformer encoder | Multi-organ, multi-modal |
| `DynUNet` | nnU-Net re-implementation | Self-configuring baseline |
| `SegResNet` | Residual encoder-decoder | Brain tumour |

```python
from monai.networks.nets import UNet

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch",
).to(device)
```

### Losses (`monai.losses`)

`DiceLoss`, `DiceCELoss` (Dice + cross-entropy), `FocalLoss`, `TverskyLoss` and several others. All handle the one-hot / softmax conversion internally when `to_onehot_y=True` and `softmax=True` are set.

### Metrics (`monai.metrics`)

`DiceMetric`, `compute_hausdorff_distance`, `compute_surface_dice`, `ConfusionMatrixMetric`. Metric objects accumulate over batches and return the final value with `.aggregate()`.

### Inferers (`monai.inferers`)

`sliding_window_inference` tiles a fixed-size patch across a volume too large for GPU memory and stitches the results with Gaussian blending. It is the standard inference method for 3D medical segmentation.

```python
from monai.inferers import sliding_window_inference

with torch.no_grad():
    logits = sliding_window_inference(
        input_tensor,       # shape [B, C, X, Y, Z]
        roi_size=(96,96,96),
        sw_batch_size=4,
        predictor=model,
        overlap=0.5,        # 50 % overlap between adjacent patches
    )
```

## Why MONAI instead of plain PyTorch?

Plain PyTorch is a general deep-learning library. It has no concept of a Hounsfield unit, a NIfTI file, a voxel spacing, an axial orientation, or a Hausdorff distance. Building a medical segmentation pipeline from scratch requires solving each of these before writing a single training loop.

MONAI solves them once, correctly, and makes the solutions composable.

| Problem | Plain PyTorch approach | MONAI approach |
| --- | --- | --- |
| Reading NIfTI / DICOM | write a nibabel / pydicom loader | `LoadImaged` — handles metadata, spacing, affine |
| Resampling to isotropic voxels | write a scipy / ITK call | `Spacingd` — resamples image and label consistently |
| Canonical orientation | write axis-swap logic per dataset | `Orientationd(axcodes="RAS")` |
| Balanced patch sampling | write a positive/negative sampler | `RandCropByPosNegLabeld` |
| Caching preprocessed scans | write a custom on-disk cache | `CacheDataset(cache_rate=1.0)` |
| Sliding-window inference | write a tiling + blending loop | `sliding_window_inference` |
| Dice + HD95 per patient | implement from scratch | `DiceMetric`, `compute_hausdorff_distance` |

Beyond convenience, MONAI's transforms respect the **physical geometry** of medical images. `Spacingd` reads the voxel spacing from the NIfTI affine and resamples accordingly. `Orientationd` reads the direction cosines. A rotation augmentation in `RandRotated` rotates by a physical angle in degrees, not by a number of voxels. This matters: a dataset where patients are stored in LPS versus RAS orientation will produce mirror-image predictions without orientation normalisation.

## The MONAI ecosystem

MONAI core (`pip install monai`) is the training library covered in this document. The wider project includes several companion tools.

**[MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo)** — a curated registry of pretrained bundles (model weights + config + transforms) for common tasks: spleen CT segmentation, whole-body CT segmentation, brain MRI segmentation, pathology classification and others. Each bundle is self-contained: the same config drives training, evaluation and inference. Download with `monai.bundle.download(name="spleen_ct_segmentation")`.

**[MONAI Label](https://github.com/Project-MONAI/MONAILabel)** — an active-learning annotation server. A radiologist annotates a few cases in 3D Slicer or OHIF; MONAI Label trains a model on them, suggests pre-labels for the next case, and iterates. It eliminates the cold-start problem for custom datasets.

**[MONAI Deploy](https://github.com/Project-MONAI/monai-deploy)** — an application SDK and packaging standard for deploying MONAI models in clinical informatics environments (DICOM RT, HL7, cloud inference services). Bridges the gap between a trained `.pt` file and a production pipeline.

**[MONAI Federated Learning](https://github.com/Project-MONAI/FL)** — integration with NVIDIA FLARE and OpenFL for training across hospitals without sharing patient data. Used in the FeTS brain tumour federated challenge.

## Quick-start: from NIfTI to prediction in 20 lines

The snippet below is a minimal but complete pipeline. It downloads the Medical Segmentation Decathlon spleen dataset, defines a transform pipeline, trains for one epoch, runs sliding-window inference on one test case, and prints the Dice score.

```python
import glob, torch
from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
    Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged,
    Spacingd, AsDiscrete,
)

# 1. Download data (Medical Segmentation Decathlon, Task 09 Spleen, CC-BY-SA 4.0)
download_and_extract(
    url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    filepath="Task09_Spleen.tar", output_dir=".",
    hash_val="410d4a301da4e5b2f6f86ec3ddba524e", hash_type="md5",
)
images = sorted(glob.glob("Task09_Spleen/imagesTr/*.nii.gz"))
labels = sorted(glob.glob("Task09_Spleen/labelsTr/*.nii.gz"))
files = [{"image": i, "label": l} for i, l in zip(images, labels)]
train_files, val_files = files[:-9], files[-9:]

# 2. Transform pipeline
common = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
             mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys="image", a_min=-57, a_max=164,
                         b_min=0.0, b_max=1.0, clip=True),
]
train_tf = Compose(common + [
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                           spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
])
val_tf = Compose(common)

# 3. Data loaders with caching
train_ds = CacheDataset(train_files, train_tf, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# 4. Model, loss, optimiser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(spatial_dims=3, in_channels=1, out_channels=2,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
             num_res_units=2).to(device)
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 5. One training epoch
model.train()
for batch in train_loader:
    x, y = batch["image"].to(device), batch["label"].to(device)
    optimiser.zero_grad()
    loss_fn(model(x), y).backward()
    optimiser.step()

# 6. Validation: sliding-window inference + Dice
val_ds = CacheDataset(val_files, val_tf, cache_rate=1.0)
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

model.eval()
with torch.no_grad():
    for batch in DataLoader(val_ds, batch_size=1):
        pred = sliding_window_inference(
            batch["image"].to(device), (96, 96, 96), 4, model)
        preds  = [post_pred(i)  for i in decollate_batch(pred)]
        labels = [post_label(i) for i in decollate_batch(batch["label"].to(device))]
        dice_metric(y_pred=preds, y=labels)

print("Mean Dice after one epoch:", dice_metric.aggregate().item())
```

## Limitations

**MONAI is a training and research library, not a clinical product.** The documentation explicitly states that it is not a medical device and is not intended for clinical use. Models trained with MONAI require independent clinical validation before any patient-facing deployment.

**It assumes PyTorch familiarity.** MONAI does not abstract away the training loop, the GPU device management, or the checkpoint strategy. A user who is not comfortable with PyTorch tensors, `nn.Module`, and `DataLoader` will find the learning curve steep. The [MedNIST tutorial](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb) is a gentler entry point.

**CacheDataset requires RAM.** Caching a 50-scan 3D CT dataset at full resolution can require 30–60 GB of RAM. On machines with less memory, `cache_rate` must be reduced or `PersistentDataset` (disk cache) used instead, at the cost of slower epoch times.

**No DICOM-RT pipeline out of the box.** MONAI reads and writes NIfTI natively. Reading DICOM RT Structure Sets into masks, and writing predicted masks back to RT Structure Sets, requires additional tooling (`rt-utils`, `dicompyler`, or MONAI Deploy's RT extensions).

**Benchmarking against nnU-Net.** For a new dataset without a strong prior on architecture, [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) often outperforms a hand-configured MONAI pipeline because it auto-configures all hyperparameters from dataset fingerprints. MONAI is the right choice when you need fine-grained control, custom architectures, or integration with the broader MONAI ecosystem. For a quick baseline, run nnU-Net first.

---

**Key links**

- Documentation: [docs.monai.io](https://docs.monai.io)
- Tutorials: [github.com/Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials)
- Model Zoo: [github.com/Project-MONAI/model-zoo](https://github.com/Project-MONAI/model-zoo)
- Paper: Cardoso et al. 2022, *MONAI: An open-source framework for deep learning in healthcare*, arXiv 2211.02701

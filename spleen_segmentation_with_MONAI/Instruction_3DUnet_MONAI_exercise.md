# Train it, then commission it
### Practical session on deep learning for medical image segmentation
#### One-week school on AI for Medical Physicists

---

This 80-minute session uses MONAI to train a 3D U-Net to auto-contour the spleen on CT, then shifts focus to what a medical physicist needs to do with any AI-based auto-contouring tool: **evaluate it, and find out where it breaks**.

> **This is a teaching exercise only. Nothing produced here is a medical device or is intended for clinical use.**

---

## Files in this repository

| File | Purpose |
|---|---|
| `spleen_segmentation_MONAI_student.ipynb` | The notebook you work in — three lines are left as TODOs |
| `spleen_segmentation_MONAI_solution.ipynb` | Instructor copy with every cell complete |
| `README.md` | This file |

---

## Before the session

**1. No local installation required.** The notebook runs on Google Colab. You need a Google account to run a Colab notebook for free.

**2. Switch the runtime to GPU.** In Colab: `Runtime → Change runtime type → T4 GPU`. Without a GPU the training step will be far too slow (you can still follow the evaluation and stress-test parts).

**3. Open the student notebook.**
   - Option A (recommended): click the badge below.
   - Option B: download `spleen_segmentation_MONAI_student.ipynb`, go to [colab.research.google.com](https://colab.research.google.com), choose `File → Upload notebook`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CHANGE_ME/REPO_NAME/blob/main/spleen_segmentation_MONAI_student.ipynb)

> Replace `CHANGE_ME/REPO_NAME` in the badge URL above with your actual GitHub username and repository name.

---

## Session structure

Work in pairs. Run the cells in order. Cells marked ✏️ contain a discussion question — pause and talk through it before moving on. Those questions are where most of the learning happens.

| Part | Time | What you do |
|---|---|---|
| 0. Setup | — | Install MONAI, import libraries, download the dataset (~1.5 GB, ~2 min) |
| 1. Look at the data | 0–10 min | Inspect one CT scan: HU range, voxel size, foreground fraction, organ boundary |
| 2. Train a 3D U-Net | 10–30 min | Complete three TODOs, train for 10 minutes, observe the learning curves |
| 3. Evaluate like a physicist | 30–50 min | Compute Dice, HD95, Surface Dice and volume error per patient; find the worst case |
| 4. Stress test | 50–70 min | Probe the model with noise, HU offset, thick slices and mirrored anatomy |
| 5. Debrief | 70–80 min | Each pair reports one failure and one QA check that would catch it |

---

## Your three TODOs

The session teaches you the pipeline by asking you to fill in three single-line gaps. Each has an assertion that stops the notebook if the line is missing or wrong.

**TODO 1 — intensity scaling (Part 2, preprocessing)**

```python
# TODO 1: add a ScaleIntensityRanged transform that maps [HU_MIN, HU_MAX] to [0, 1] and clips values outside
```

Hint: look up `monai.transforms.ScaleIntensityRanged`. The parameters `a_min`, `a_max`, `b_min`, `b_max` and `clip` map directly onto `HU_MIN`, `HU_MAX` and the target range.

**TODO 2 — loss function (Part 2, model)**

```python
loss_fn = None  # TODO 2: create a DiceCELoss (labels are integer maps, the network outputs logits)
```

Hint: look up `monai.losses.DiceCELoss`. The network produces logits (not probabilities), and the labels are integer class maps (not one-hot). The two arguments that handle this are `softmax` and `to_onehot_y`.

**TODO 3 — sliding-window inference (Part 2, predict function)**

```python
logits = None  # TODO 3: call sliding_window_inference with ROI, SW_BATCH and the model
```

Hint: look up `monai.inferers.sliding_window_inference`. A whole CT scan does not fit in GPU memory at once, so the inferer tiles a patch (`roi_size`) across the volume and stitches the results.

---

## Dataset

Medical Segmentation Decathlon, Task 09 (Spleen CT).

- 61 CT volumes from Memorial Sloan Kettering Cancer Center
- One binary label per scan (spleen / background)
- Modality: portal-venous-phase CT
- Licence: **CC-BY-SA 4.0** — open for any use with attribution and share-alike
- Citation: Simpson et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*, arXiv 1902.09063

The notebook downloads the data automatically on first run. If your environment has no internet access, download the archive from the link in the notebook and place it at `ROOT/Task09_Spleen.tar` before running the download cell.

---

## Software

| Package | Version pinned | Why |
|---|---|---|
| MONAI | 1.6.1 | Medical imaging transforms, 3D U-Net, metrics |
| PyTorch | installed by Colab | Deep learning backend |
| nibabel | via MONAI extras | NIfTI file I/O |
| matplotlib / numpy / pandas | via MONAI or Colab | Plotting and tables |

The setup cell runs:
```
pip install "monai[nibabel,tqdm]==1.6.1"
```
This takes about 30 seconds on Colab. Everything else is already available in the Colab environment.

---

## Expected results

With a T4 GPU and 10 minutes of training you should see a validation Dice around **0.75–0.90**. The reference model (trained for 800 epochs) reaches **Dice ≈ 0.96** on the test set, as reported by the MONAI Model Zoo.

The stress test produces the most informative numbers. As a rough guide:

| Perturbation | What typically happens |
|---|---|
| Noise ≤ 50 HU | Dice drops < 0.05 — robust |
| HU offset +100 HU | Dice can drop to < 0.3 — the HU window assumption is fragile |
| Thick slices (×4) | Dice drops 0.05–0.15 depending on slice thickness |
| Left-right mirror | Varies; the model may generalise or fail completely |

Your own 10-minute model is likely more fragile than the reference model on all perturbations.

---

## Metric definitions

The notebook reports four numbers per patient. The two most important ones are defined here.

### Dice Similarity Coefficient (Dice)

Given a predicted segmentation mask **P** and a reference mask **R** (both binary), Dice measures the overlap as a fraction of the total number of positive voxels in both masks:

```
           2 · |P ∩ R|
Dice = ——————————————————
            |P| + |R|
```

where `|·|` denotes the number of voxels in the set. Dice ranges from 0 (no overlap) to 1 (perfect overlap). It is equivalent to the F1 score computed at the voxel level.

A few properties worth knowing:

- **Size sensitivity.** Dice is volume-weighted. A 2 mm boundary error costs a small structure far more Dice than the same error on a large organ, so a single Dice threshold is not appropriate across different organs.
- **Blindness to position within the volume.** A disconnected false-positive island far from the organ can dominate the denominator without touching the boundary near the organ. HD95 catches this; Dice does not.
- **Undefined when both masks are empty.** The convention used here is Dice = 1 when both P and R are empty (perfect agreement) and Dice = 0 when only one is empty.

### 95th-percentile Hausdorff Distance (HD95)

The standard Hausdorff distance between two surfaces S(P) and S(R) is the maximum over all one-sided directed distances:

```
HD(P, R) = max( h(S(P), S(R)),  h(S(R), S(P)) )

where  h(A, B) = max_{a ∈ A} min_{b ∈ B} d(a, b)
```

`d(a, b)` is the Euclidean distance in millimetres between surface points `a` and `b`. The directed distance `h(A, B)` finds the point on surface A that is furthest from any point on surface B.

The full Hausdorff distance is dominated by a single outlier (a stray voxel, an artefact). HD95 replaces the outer `max` with the **95th percentile** of the pooled set of directed distances:

```
HD95(P, R) = 95th percentile of
             { min_{b ∈ S(R)} d(a, b) : a ∈ S(P) }
           ∪ { min_{a ∈ S(P)} d(b, a) : b ∈ S(R) }
```

HD95 is reported in **millimetres**. A lower value is better; 0 mm means the surfaces coincide. Unlike Dice, HD95 is insensitive to organ size and directly answers the question: *how far off is the worst part of the boundary, ignoring the top 5 % of outliers?*

A practical rule of thumb for radiotherapy contouring: an HD95 below the typical inter-observer variability for that structure (~2–5 mm for large organs, ~1–2 mm for small ones) is often used as an acceptance criterion.

> **Why both?** Dice can remain above 0.9 while HD95 is tens of millimetres, if the error is a narrow peninsula or a disconnected island. Conversely, HD95 can be large while Dice is high for a large organ, because a small absolute surface displacement barely changes the volume overlap. Reporting both together catches complementary failure modes.

---

## Discussion questions overview

Part 1 — data
- Why do we resample every scan to the same voxel size before training?
- Why is voxel-level accuracy a useless metric for a rare foreground class?
- Why do we clip HU values to a soft-tissue window instead of feeding the raw range?

Part 2 — training
- What does the model know about where in the body a patch comes from?
- How much would you trust a 0.02 mean Dice difference estimated on five patients?

Part 3 — evaluation
- How can Dice be high while HD95 is poor? What kind of error causes that?
- Dice is organ-size-dependent. Is a single acceptance threshold appropriate for all organs?
- Is the volume error a bias or scatter? Which matters more in clinical practice?

Part 4 — stress test
- Which perturbation causes the sharpest degradation, and why?
- Surface Dice degrades before Dice — which one would you monitor in a QA programme?
- What does the mirroring failure imply for a patient with an unusual anatomy or a large tumour?
- Does the model give any sign that its prediction is unreliable?

---

## Going further

If you finish early, the notebook suggests three extensions:

1. **Compare to your own model:** replace `stress_net = models["reference"]` with `models["mine"]` and re-run Part 4. Is a short-trained model more or less fragile?
2. **Add a perturbation:** add a metal insert, a cropped field of view, or a wider HU window to the `conditions` dictionary.
3. **Add data augmentation:** add `RandFlipd` on the left-right axis, retrain, and check whether the mirror stress test improves.

For deeper reading after the school:

- **nnU-Net** — Isensee et al. 2021, *Nature Methods*: a self-configuring baseline that often matches or beats hand-tuned models.
- **TotalSegmentator** — Wasserthal et al. 2023, *Radiology AI*: a pretrained whole-body segmentation tool (research use only; free licence for academic work).
- **Metrics Reloaded** — Reinke et al. 2024, *Nature Methods*: a framework for choosing the right validation metrics, with an interactive online tool.
- **Clinical acceptability of auto-contouring** — Rhee et al. 2021, *Radiotherapy and Oncology*: surface metrics predict clinical acceptability better than volumetric Dice.

---

## Instructor notes

**Checkpoint for the reference model.** The notebook tries two URLs for the pretrained MONAI spleen model. If both are unreachable (network restrictions at the school venue), provide a local copy and add its path to `REFERENCE_URLS` as a `file://` path, or change the download cell to copy from a USB drive.

**Data download.** On a typical venue Wi-Fi, downloading 1.5 GB takes 2–5 minutes. Pre-seed the Google Drive of participating accounts, or host a local mirror and replace the S3 URL with yours.

**Time budget.** `TRAIN_MINUTES`, `N_TEST`, `N_VAL` and `N_STRESS` are all in one config cell (cell 4 after the imports). Increase `TRAIN_MINUTES` to 20 if the session is 90 minutes or if participants arrive with the data already cached.

**Colab disconnects.** If Colab disconnects during training, the best checkpoint is saved to `ROOT/my_model.pt`. Re-running the training cell resumes from epoch 1, but the model load at the end of the cell picks up the best weights. Data caching restarts.

**Skill spread.** Three levels of engagement are possible without changing the notebook:
- *Run only*: skip all TODOs and load the solution notebook.
- *Fill blanks*: complete the three TODO lines.
- *Extend*: add a perturbation, change the architecture, or add augmentation.

**Solution.** The solution notebook (`spleen_segmentation_MONAI_solution.ipynb`) is identical except that all three TODO lines are filled in. Keep it out of the student repository or in a branch participants cannot see before the session.

---

## Licence

The notebooks in this repository are released under the **MIT licence**. The dataset is under **CC-BY-SA 4.0** (see [medicaldecathlon.com](http://medicaldecathlon.com)).

---

*Prepared for the one-week school on AI for Medical Physicists — November 2026.*

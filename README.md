#  Smart Waste Sorting System
### Transfer Learning for Garbage Classification | EfficientNetV2S · MobileNetV2 · ResNet50

A deep learning pipeline that classifies waste images into 10 categories and maps each prediction to an actionable disposal category (Recycle / Landfill / Special Disposal) using a hybrid CNN + rule-based system.

---

## Project Overview

The project benchmarks three transfer-learning models on the Kaggle Garbage Classification V2 dataset, then fine-tunes the best performer for maximum accuracy. The final system acts as a **smart waste sorter**: given an image of waste, it predicts the material class and tells you exactly how to dispose of it.

---

## Results

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| EfficientNetV2S | 87.88% | 0.8775 | Frozen head only |
| MobileNetV2 | 89.46% | 0.8935 | Frozen head only |
| ResNet50 | 92.17% | 0.9207 | Frozen head — **best baseline** |
| **ResNet50 Fine-Tuned** | **See notebook** | **See notebook** | All layers unfrozen, 20 epochs |

> ResNet50 was selected for fine-tuning based on its superior baseline performance. Fine-tuning unfreezes all layers and trains for 20 epochs at a reduced learning rate (1e-4).

---

## Dataset

**Garbage Classification V2** by [sumn2u on Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

- **12,259 images** across **10 classes**
- Split: **70% train / 15% val / 15% test**

| Class | Disposal Category |
|---|---|
| cardboard, glass, metal, paper, plastic | Recycle |
| biological, clothes, shoes, trash | Landfill |
| battery | Special Disposal |

---

## Architecture & Approach

### Transfer Learning Strategy
All three models use the same approach:
1. Load **pretrained ImageNet weights**
2. **Freeze** all convolutional layers (preserve learned visual features)
3. Replace the final classification head with a **10-class output layer** + Dropout(0.3)
4. Train only the new head for faster convergence, less overfitting on a ~12k dataset

### Why These 3 Models?
- **EfficientNetV2S** : State-of-the-art, literature-proven for waste classification tasks
- **MobileNetV2** : Lightweight architecture representing edge/IoT deployment scenarios
- **ResNet50** : Classic, well-documented academic baseline

### Hybrid AI Layer
Predictions feed into a **rule-based recyclability mapping layer** (`RECYCLABILITY_MAP`) that converts any class prediction into a human-readable disposal instruction.

---

##  Running on Kaggle (Recommended)

This notebook is designed to run end-to-end on Kaggle with zero local setup. Follow these steps exactly to reproduce all results.

### Step 1 — Import the Notebook

1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Click **Create → New Notebook**, then select **File → Import Notebook**.
3. Upload `garbagesorting-aiap-at3.ipynb` from this repository.

### Step 2 — Add the Dataset

The notebook downloads the dataset automatically via the Kaggle CLI in Cell 2, but you can also attach it manually for faster startup:

1. In your notebook, click the **Data** tab in the right-hand panel → **Add Data**.
2. Search for **Garbage Classification V2** (owner: `sumn2u`).
3. Click **Add** and it mounts at `/kaggle/input/garbage-classification-v2/`.

> If you skip this step, Cell 2 will download and extract it to `/kaggle/working/garbage_classification/original/` automatically (requires your Kaggle API key in Secrets).

**To set up the Kaggle API key for auto-download:**
1. Go to **kaggle.com → Your Profile → Settings → API → Create New Token** — Copy the new token.
2. In your notebook, go to **Add-ons → Secrets** and add two secrets:
   - `KAGGLE_USERNAME` - your Kaggle username
   - `KAGGLE_KEY` - your API token

### Step 3 — Enable GPU Acceleration

Training on CPU is extremely slow (~30–60× slower). Always enable GPU:

1. In the notebook editor, go to **Settings** (right panel) → **Accelerator**.
2. Select **GPU T4 x2** or **GPU P100**, either works.
3. Click **Save**. The session will restart with GPU available.

Cell 2 prints the active device on run:
```
Device : cuda
```
If it prints `cpu`, the GPU was not enabled so go back to Settings and verify.

### Step 4 — Run All Cells

With the dataset attached and GPU enabled:

1. Click **Run All** (from the **Run** menu, or `Shift+Enter` through each cell).
2. Expected total runtime: **~45–90 minutes** (baseline training + fine-tuning).

Cell order matters. Do not skip or reorder cells. Each cell depends on variables and DataLoaders defined in prior cells.

### Step 5 — Locate Outputs

All outputs are written to `/kaggle/working/`. After the notebook finishes, find them in the **Output** tab of the right panel, or access them at these paths:

**Charts**
```
/kaggle/working/sample_images.png
/kaggle/working/class_distribution.png
/kaggle/working/training_history.png
/kaggle/working/confusion_matrices.png
/kaggle/working/model_comparison.png
/kaggle/working/finetuned_history.png
/kaggle/working/finetuned_confusion_matrix.png
/kaggle/working/all_predictions.png
```

**Model Checkpoints**
```
/kaggle/working/EfficientNetV2S_weights.pth
/kaggle/working/MobileNetV2_weights.pth
/kaggle/working/ResNet50_weights.pth
/kaggle/working/ResNet50_FineTuned_weights.pth
/kaggle/working/SmartWasteSorter_Final.pth   ← final model bundle
```

**Inference Examples**
The `all_predictions.png` chart is generated in the final cell using five test images (cardboard, glass, paper, plastic, clothes). Each image displays the predicted class, disposal category, and confidence score. To run inference on your own images, upload them via **Data → Add Data → Upload** and update the `IMAGE_PATHS` list in the last cell.

### Reproducing the Exact Train/Val/Test Split

The split is deterministic. These two lines in Cell 6 are the only requirements:

```python
RANDOM_SEED = 42          # defined in Cell 2
torch.manual_seed(RANDOM_SEED)
train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size]
)
```

As long as `RANDOM_SEED = 42` is unchanged and Cell 2 runs before Cell 6, every run produces the identical split: **8,581 train / 1,838 val / 1,840 test** images.

> **Do not change `RANDOM_SEED`** if you want to reproduce the reported accuracy numbers. Any other seed produces a different split and different test-set metrics.

---

## 🔧 Local / Colab Setup (Optional)

For running outside Kaggle:

```bash
pip install torch torchvision matplotlib scikit-learn seaborn tqdm pillow
```

Download the dataset via the Kaggle CLI:

```bash
kaggle datasets download -d sumn2u/garbage-classification-v2
unzip garbage-classification-v2.zip -d garbage_classification/original
```

Then update `DATASET_PATH` in Cell 2 to point to your local extraction path.

---

## ⚙️ Configuration

All key hyperparameters are defined in Cell 2:

```python
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 10        # 20 for fine-tuning
LEARNING_RATE = 0.001     # 1e-4 for fine-tuning
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42
```

---

##  Data Augmentation

Training uses augmentation to improve generalisation on imbalanced classes. Validation and test sets use only resize + normalise (no augmentation) to prevent data leakage.

| Split | Transforms |
|---|---|
| Train | Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°), ColorJitter, RandomGrayscale, Normalise |
| Val / Test | Resize, Normalise only |

ImageNet mean/std normalisation is applied to all splits since all three models were pretrained on ImageNet.

---

## Notebook Structure

| Cell | Description |
|------|---|
| 1    | Install & import libraries |
| 2    | Configuration, dataset download & verification |
| 3    | Visualise sample images per class |
| 4    | Class distribution bar chart |
| 5    | Data transforms & augmentation |
| 6    | Load dataset & create train/val/test DataLoaders |
| 7    | Model definitions (EfficientNetV2S, MobileNetV2, ResNet50) |
| 8    | Training loop with StepLR scheduler & best-weight tracking |
| 9    | Evaluation function + `predict_with_recyclability()` hybrid inference |
| 10   | Train all three models & save weights |
| 11   | Training curves (accuracy & loss) |
| 12   | Confusion matrices for all three models |
| 13   | Model comparison bar chart (Accuracy vs F1) |
| 14   | Final summary table |
| 15   | Demo: per-class recyclability predictions |
| 16   | Fine-tune ResNet50 (all layers unfrozen, 20 epochs) |
| 17   | Baseline vs fine-tuned comparison chart |
| 18   | Live batch demo with top-3 uncertainty breakdown |
| 19   | Fine-tuned confusion matrix |
| 20   | Save final model (`SmartWasteSorter_Final.pth`) + report checklist |
| 21   | Inference on custom images from Kaggle input |

---
##  Inference

Load the final model and run a prediction on any image:

```python
import torch
from torchvision import transforms, models
from PIL import Image

# Load
checkpoint = torch.load('SmartWasteSorter_Final.pth', map_location='cpu')
CLASS_NAMES       = checkpoint['class_names']
RECYCLABILITY_MAP = checkpoint['recyclability_map']

model = build_resnet50(len(CLASS_NAMES))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img = Image.open('your_image.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)
    confidence, idx = probs.max(1)
    predicted_class = CLASS_NAMES[idx.item()]
    recyclability, icon = RECYCLABILITY_MAP[predicted_class]

print(f"Class      : {predicted_class}")
print(f"Confidence : {confidence.item():.2%}")
print(f"Disposal   : {icon} {recyclability}")
```

---

## Tech Stack

- **PyTorch** : model training & inference
- **torchvision** : pretrained models & transforms
- **scikit-learn** : classification metrics
- **matplotlib / seaborn** : visualisation
- **Pillow** : image loading
- **tqdm** : progress bars
- **Kaggle** : compute environment & dataset hosting

---

## Repository Structure

```
├── Demo images/          # Input images used for inference demonstrations
├── Generated Figures/    # All charts and plots produced during training
├── Models/               # Saved model artefacts (.pth checkpoints)
├── Notebook/             # The main Jupyter notebook
└── README.md
```

---
## Notes

- The 70/15/15 split is applied via `random_split` with `RANDOM_SEED = 42` for reproducibility.
- Val/test datasets use a `copy.deepcopy` of the full dataset with the augmentation-free transform to avoid transform leakage.
- The StepLR scheduler halves the learning rate every 5 epochs during baseline training.
- Fine-tuning uses a lower learning rate (1e-4) with all layers unfrozen for 20 epochs.

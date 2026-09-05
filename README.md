# Fruit and Vegetable Image Classification using Transfer Learning

## Overview

A deep learning project for classifying images into **50 fruit and vegetable categories** using **PyTorch and a pretrained GoogLeNet model**.

The project demonstrates a complete transfer-learning workflow including dataset analysis, class-imbalance handling, data augmentation, fine-tuning, model evaluation, and custom image prediction.

---

## Key Results

| Metric                   |                 Result |
| ------------------------ | ---------------------: |
| Classes                  |                     50 |
| Training Images          |                 46,191 |
| Validation Images        |                 12,545 |
| Test Images              |                 13,353 |
| Model                    |   Pretrained GoogLeNet |
| Input Size               |              224 × 224 |
| Batch Size               |                     32 |
| Optimizer                |                   Adam |
| Learning Rate            |                 0.0001 |
| Loss                     | Weighted Cross-Entropy |
| Epochs                   |                     15 |
| Best Validation Accuracy |             **92.87%** |
| Test Accuracy            |             **90.44%** |
| Macro F1-Score           |             **90.67%** |

---

## Dataset

The dataset contains **72,089 RGB images** across 50 classes and is divided into:

```text
fruit_vegetables/
├── train/
├── validation/
└── test/
```

Classes include:

`Apple`, `Avocado`, `Banana`, `Beetroot`, `Blackberry`, `Blueberry`, `Broccoli`, `Cabbage`, `Capsicum`, `Carrot`, `Cauliflower`, `Chilli Peper`, `Corn`, `Cucumber`, `Dates`, `Dragonfruit`, `Eggplant`, `Fig`, `Garlic`, `Ginger`, `Grapes`, `Guava`, `Jalepeno`, `Kiwi`, `Lemon`, `Lettuce`, `Mango`, `Mushroom`, `Okra`, `Olive`, `Onion`, `Orange`, `Paprika`, `Peanuts`, `Pear`, `Peas`, `Pineapple`, `Pomegranate`, `Potato`, `Pumpkin`, `Raddish`, `Rambutan`, `Soy Beans`, `Spinach`, `Strawberry`, `Sweetcorn`, `Sweetpotato`, `Tomato`, `Turnip`, `Watermelon`.

---

## Data Preprocessing

Images are resized to **224 × 224** and normalized using ImageNet statistics.

### Training

* Resize
* Random Horizontal Flip
* Random Rotation (±10°)
* Convert to Tensor
* ImageNet normalization

### Validation / Test

* Resize
* Convert to Tensor
* ImageNet normalization

Data augmentation is applied only during training to improve generalization.

---

## Handling Class Imbalance

The dataset contains significant differences in the number of images per class.

Instead of oversampling, **class-weighted Cross-Entropy Loss** was used:

```python
weight_c = N / (C * N_c)
```

The weights are calculated from the training-set class distribution and passed to:

```python
nn.CrossEntropyLoss(weight=class_weights)
```

This gives greater importance to underrepresented classes during training.

---

## Transfer Learning

A pretrained GoogLeNet model was loaded using Torchvision:

```python
googlenet = models.googlenet(weights="DEFAULT")
```

The original GoogLeNet classifier contains:

```text
in_features = 1024
out_features = 1000
```

Since our dataset contains 50 classes, the classifier was replaced:

```python
googlenet.fc = nn.Linear(
    googlenet.fc.in_features,
    50
)
```

All pretrained parameters were allowed to update, making this a **fine-tuning** approach rather than simple feature extraction.

```python
for param in googlenet.parameters():
    param.requires_grad = True
```

---

## Training

### Configuration

```text
Model          : GoogLeNet
Pretrained     : Yes
Training Mode  : Fine-Tuning
Classes        : 50
Batch Size     : 32
Epochs         : 15
Optimizer      : Adam
Learning Rate  : 0.0001
Loss           : Weighted Cross-Entropy
Hardware       : Google Colab NVIDIA T4
```

The training process follows the standard PyTorch cycle:

```text
Data → Forward Pass → Loss
     → Backpropagation → Optimizer Update
```

Validation was performed after every epoch using `model.eval()` and `torch.no_grad()`.

---

## Training Performance

| Epoch | Train Acc. |   Val Acc. |
| ----: | ---------: | ---------: |
|     1 |     69.51% |     86.38% |
|     5 |     92.10% |     91.29% |
|    10 |     95.64% |     92.82% |
|    13 |     96.61% | **92.87%** |
|    15 |     97.01% |     92.34% |

The model showed **mild overfitting** toward the later epochs, as training accuracy continued increasing while validation performance fluctuated.

---

## Test Evaluation

The final model achieved:

```text
Test Loss     : 0.3971
Test Accuracy : 90.44%
Macro F1      : 0.9067
Weighted F1   : 0.9046
```

### Strong-performing classes

* Turnip — 98.15% F1
* Rambutan — 97.69%
* Okra — 97.03%
* Mushroom — 96.97%
* Strawberry — 96.85%

### More challenging classes

* Potato — 75.16%
* Olive — 76.19%
* Sweetcorn — 78.43%
* Corn — 78.97%
* Lemon — 79.08%

A normalized confusion matrix was also used to analyze class-level errors.

---

## Custom Image Prediction

The trained model was tested on a user-provided image.

Prediction pipeline:

```text
Image
 ↓
Resize + Normalize
 ↓
GoogLeNet
 ↓
Class Logits
 ↓
Softmax
 ↓
Predicted Class + Confidence
```

A custom **apple image** was correctly classified as:

```text
Predicted Class : Apple
Confidence      : 99.98%
```

---

## Project Workflow

```text
Dataset
   ↓
Class Analysis
   ↓
Class Mapping & Weights
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
ImageFolder + DataLoader
   ↓
Pretrained GoogLeNet
   ↓
Replace Classifier
   ↓
Fine-Tuning
   ↓
Validation
   ↓
Test Evaluation
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
Custom Image Prediction
```

---

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Pillow
* Google Colab
* CUDA / NVIDIA T4 GPU

---

## Concepts Covered

* Convolutional Neural Networks
* Transfer Learning
* Fine-Tuning
* Pretrained Models
* Data Augmentation
* Class Imbalance
* Weighted Cross-Entropy Loss
* Adam Optimizer
* Forward & Backward Propagation
* Training vs Validation
* Overfitting
* Precision, Recall & F1-score
* Confusion Matrix
* Model Inference

---

## Project Structure

```text
fruit-vegetable-transfer-learning/
│
├── notebooks/
│   └── fruit_vegetable_googlenet_transfer_learning.ipynb
│
├── src/
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset, Kaggle credentials, and other sensitive or large files should **not** be committed to the repository.

---

## Future Improvements

* Save the best validation checkpoint
* Add learning-rate scheduling
* Experiment with stronger augmentation
* Compare feature extraction vs fine-tuning
* Tune dropout and other hyperparameters
* Compare GoogLeNet with ResNet, EfficientNet, or ConvNeXt
* Build a standalone inference application

---

## Conclusion

This project demonstrates a complete **multi-class image-classification pipeline using transfer learning with PyTorch**.

A pretrained GoogLeNet model was fine-tuned for 50 fruit and vegetable classes and achieved:

**92.87% best validation accuracy** and **90.44% test accuracy**, with a **90.67% macro F1-score**.

The project provides practical experience with transfer learning, class imbalance, data augmentation, model training, evaluation, and real-world image inference.

---

## Author

**ASEEM LAIS T P**

# ResNet-18 CIFAR-10 Classifier in PyTorch

A clean, modular, and performant implementation of the **ResNet-18** architecture built from scratch in PyTorch, customized specifically for the **CIFAR-10** dataset ($32 \times 32$ images). 

The model reaches **~94.25% Test Accuracy** within 50 epochs without using pre-trained weights, leveraging custom residual connections, modern regularization techniques, and learning rate scheduling.

---

## 📈 Performance & Results

* **Test Accuracy:** `94.25%`
* **Epochs:** `50`
* **Optimizer:** SGD with Momentum ($0.9$), Weight Decay ($5 \times 10^{-4}$)
* **Scheduler:** `CosineAnnealingLR` ($T_{max} = 50$)
* **Batch Size:** `128` (Train), `100` (Test)

---

## 🏗️ Project Structure

```text
resnet18-cifar10-pytorch/
├── model.py             # Modular PyTorch ResNet-18 architecture definition
├── train.py             # Training loop, evaluation, and weights exporter
├── predict.py           # Inference script with Top-K probability visualizer
├── resnet18_cifar10.pth # Trained model weights state dict
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore configurations
└── README.md            # Project documentation
```

---

## ⚡ Key Architectural Features

1. **Adapted First Conv Layer:** Replaced the standard $7 \times 7$ (stride 2) convolution with a $3 \times 3$ (stride 1) filter to prevent rapid reduction of spatial dimensions on small $32 \times 32$ input resolution.
2. **Residual Blocks (`BasicBlock`):** Implemented skip connections ($y = F(x) + x$) to solve the vanishing gradient problem and stabilize deep feature extraction.
3. **Data Augmentation:** Applied `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` during training for strong generalization.

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install required packages:


### 2. Train from Scratch

To start training the ResNet-18 model on CIFAR-10:

```bash
python train.py
```
*The script will automatically download the CIFAR-10 dataset if not present, execute 50 epochs, and save the weights as `resnet18_cifar10.pth`.*

### 3. Inference / Prediction

Place any target image named `test_image.jpg` in the project root directory and run:

```bash
python predict.py
```
## 📥 Pre-trained Weights
You can download the pre-trained weights (`resnet18_cifar10.pth`) directly from the (https://github.com/MoBa-create/resnet18-cifar10-pytorch/releases/tag/v1.0.0)
and place it in the project root directory.

It will output the predicted class, confidence percentage, and plot the top probability distribution.

## 🌐 Web Interface (Gradio)
You can launch the interactive web interface to test custom images:
```bash
python app.py

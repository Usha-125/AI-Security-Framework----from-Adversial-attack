# FGSM Adversarial Dataset Generator

## Project Overview

This project focuses on **AI Security** by generating **adversarial examples** using the **Fast Gradient Sign Method (FGSM)** attack on the MNIST dataset.

The system trains a Convolutional Neural Network (CNN) on MNIST and then creates adversarial images using different **epsilon values** to analyze how small perturbations affect model predictions.

This project demonstrates how machine learning models can be vulnerable to adversarial attacks and helps build datasets for **robust AI and adversarial defense research**.

---

# Project Workflow

The project is divided into several stages:

### 1. Model Training

A **Simple Convolutional Neural Network (CNN)** is trained on the MNIST handwritten digits dataset.

Architecture:

* Conv2D Layer (1 → 32)
* Conv2D Layer (32 → 64)
* MaxPooling
* Dropout
* Fully Connected Layer
* Output Layer (10 classes)

Dataset:

* MNIST Training Images
* MNIST Test Images

Output:

```
mnist_cnn.pth
```

This file stores the trained model weights.

---

# 2. FGSM Adversarial Attack

After training the model, adversarial examples are generated using the **Fast Gradient Sign Method (FGSM)**.

FGSM perturbs the input image slightly using the gradient of the loss with respect to the input image.

Formula:

x_adv = x + ε * sign(∇x J(θ, x, y))

Where:

* x = original image
* ε (epsilon) = attack strength
* ∇xJ = gradient of loss
* sign() = sign of gradient

---

# 3. Multi-Epsilon Attack Generation

Adversarial datasets are generated for multiple attack strengths.

Epsilon values used:

```
0.10
0.15
0.20
0.25
```

For each epsilon:

* 1000 clean images are saved
* 1000 adversarial images are generated

Total dataset generated:

```
4000 Clean Images
4000 Adversarial Images
```

---

# 4. Dataset Structure

The generated dataset is organized as follows:

```
adversarial_dataset/

    eps_0.1/
        images/
            clean_0.pt
            adv_0.pt
            ...

    eps_0.15/
        images/
            clean_0.pt
            adv_0.pt
            ...

    eps_0.2/
        images/
            clean_0.pt
            adv_0.pt
            ...

    eps_0.25/
        images/
            clean_0.pt
            adv_0.pt
            ...
```

Each folder contains the clean image and its adversarial counterpart.

---

# 5. Technologies Used

* Python
* PyTorch
* Torchvision
* MNIST Dataset
* FGSM Adversarial Attack

---

# 6. How to Run

Step 1 — Install dependencies

```
pip install torch torchvision
```

Step 2 — Run dataset builder

```
python dataset_builder.py
```

The script will:

1. Train the CNN model
2. Save the model
3. Generate adversarial datasets for all epsilon values

---

# Current Project Status

Completed:

* CNN model training on MNIST
* FGSM attack implementation
* Multi-epsilon adversarial dataset generation
* Clean and adversarial dataset creation

Generated Dataset:

```
8000 total samples
(4000 clean + 4000 adversarial)
```

---

# Next Steps (Planned)

Future improvements include:

* Visualizing adversarial examples
* Training an adversarial detector (Clean vs Adversarial classifier)
* Evaluating model robustness
* Testing additional attacks such as:

  * PGD
  * BIM
  * DeepFool

---

# Research Goal

This project aims to study:

* How neural networks fail under adversarial attacks
* The relationship between epsilon and attack success
* Methods for improving **AI robustness and security**

---

# Author

Usha S Gowda
AI Security & Machine Learning Project

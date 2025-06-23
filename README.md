# Neural_Bonus_Assignment
# Home & Bonus Assignment – Transformers and Conditional GANs

**Student Name**: Sai Sravan Chintala  
**Student ID**: 700773836  
**Course**: Summer 2025 Neural Network & Deep Learning (CS-5720-0) 
**Submission Date**: 06/18/2025

---

## 📌 Overview

This submission includes the following:
1. A Question Answering (QA) system using Hugging Face transformers.
2. A Conditional GAN (cGAN) implementation for digit-controlled MNIST generation.
3. Short theoretical responses comparing GAN types and image-to-image learning.

---

## 🧠 Question 1: Question Answering with Transformers

### 🎯 Objective

Build a simple QA system using Hugging Face’s Transformers library to answer questions based on context.

### ✅ Tasks Completed

- Used Hugging Face's `pipeline` to load default QA model.
- Switched to `deepset/roberta-base-squad2` as a custom model.
- Created a custom context and asked two questions with correct answers and scores > 0.70.

### 📎 Files

- `home_assignment_Bonus_solution.py`: Full QA code with 3 tasks.
- All answers returned with correct `answer`, `score`, `start`, and `end`.

---

## 🤖 Question 2: Digit-Class Controlled Image Generation with Conditional GAN

### 🎯 Objective

Implement a cGAN that generates MNIST digits (0–9) based on class labels, to show how conditioning improves control over generated images.

### ✅ Tasks Completed

- Generator and discriminator modified to accept label input.
- Trained on MNIST dataset with labels.
- Visualized generated digits by class (0–9).

### 📎 Files

- `conditional_gan_mnist.py`: Full code.
- `generated_digits_by_label.png`: Output image showing digits 0–9.

---

## ✍️ Bonus: Short Answer Questions

### 🔹 Q1: How does a Conditional GAN differ from a vanilla GAN?

> A vanilla GAN generates data from noise without context.  
> A Conditional GAN (cGAN) adds **conditional input** (e.g., label or text) to both the generator and discriminator, enabling controlled generation.

✅ **Example**: Text-to-image generation using a caption like “A red car driving on a mountain road”.

---

### 🔹 Q2: What does the discriminator learn in an image-to-image GAN? Why is pairing important?

> The discriminator learns not just realism, but also whether the **output matches the input mapping** (e.g., sketch → photo).

✅ **Pairing is important** so the generator learns a meaningful, direct translation between inputs and outputs. Without pairs, learning becomes unstructured.

---


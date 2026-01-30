# Elevate_Task10

This project implements a Handwritten Digit Classification system using the K-Nearest Neighbors (KNN) algorithm. The model classifies digits (0–9) from image data based on distance similarity. The project demonstrates the importance of feature scaling, hyperparameter tuning, and model evaluation in distance-based machine learning algorithms.

# Objectives

Implement KNN for digit classification

Understand distance-based learning

Analyze the impact of different K values

Evaluate model performance using accuracy and confusion matrix

# Dataset
Primary Dataset

Sklearn Digits Dataset (load_digits())

Total samples: 1797

Image size: 8 × 8 pixels

Number of classes: 10 (digits 0–9)

Features per image: 64

# Tools & Technologies

Python

Scikit-learn

NumPy

Matplotlib

Jupyter Notebook

# Project Workflow

Load the digits dataset

Visualize sample digit images

Split dataset into training and testing sets

Apply feature scaling using StandardScaler

Train KNN model with K = 3

Evaluate accuracy

Tune hyperparameter by testing K = 3, 5, 7, 9

Plot Accuracy vs K

Generate confusion matrix

Display final predictions visually

# Results

Achieved accuracy: ~97–99%

Best performance obtained after tuning K


# Output

<img width="721" height="369" alt="Image" src="https://github.com/user-attachments/assets/d50c42a9-6df9-4c80-9efc-2699e2f724a0" />

<img width="741" height="388" alt="Image" src="https://github.com/user-attachments/assets/ceb86077-c583-446e-863c-780d96e43809" />

<img width="628" height="626" alt="Image" src="https://github.com/user-attachments/assets/1a580cdb-e737-4988-a451-2d1267857c25" />

<img width="734" height="588" alt="Image" src="https://github.com/user-attachments/assets/276115d1-e28c-43f2-8257-3a23f3443f75" />

Confusion matrix shows minimal misclassification

Strong performance on unseen test data

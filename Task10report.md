**1. Introduction**



Handwritten digit classification is a classic problem in machine learning where the objective is to correctly identify digits (0–9) from image data. This task is widely used to demonstrate image-based classification techniques and distance-based learning algorithms.



In this project, the K-Nearest Neighbors (KNN) algorithm is used to classify handwritten digits using the Sklearn Digits dataset. The task helps in understanding how distance metrics, feature scaling, and hyperparameter tuning influence model performance.



**2. Objective**



The main objectives of this project are:



To understand and implement the KNN algorithm



To classify handwritten digit images accurately



To analyze the effect of different K values



To evaluate the model using accuracy and confusion matrix



**3. Dataset Description**



Dataset Used:



Sklearn Digits Dataset (load\_digits())



Dataset Characteristics:



Total samples: 1797



Image size: 8 × 8 pixels



Features per image: 64



Target classes: Digits 0 to 9



Each image is converted into a 1D array of 64 pixel intensity values, and each array is associated with a corresponding digit label.



**4. Tools \& Technologies Used**



Programming Language: Python



Libraries:



Scikit-learn



NumPy



Matplotlib



Development Environment: Jupyter Notebook



**5. Methodology**



**Step 1:** Data Loading



The digits dataset is loaded using Scikit-learn. The feature matrix (X) contains pixel values, and the target vector (y) contains digit labels.



**Step 2**: Data Visualization



Sample digit images are visualized to verify the correctness of the dataset and to understand the visual structure of the digits.



**Step 3**: Train-Test Split



The dataset is split into training and testing sets using an 80:20 ratio to evaluate the model on unseen data.



**Step 4**: Feature Scaling



StandardScaler is applied to normalize feature values. Since KNN relies on distance calculations, scaling ensures that all features contribute equally.



**Step 5:** Model Training



A KNN classifier is trained with an initial value of K = 3. The model predicts the class of test samples based on majority voting among nearest neighbors.



**Step 6**: Hyperparameter Tuning



The model is trained using multiple K values (3, 5, 7, 9). Accuracy is calculated for each value to determine the optimal K.



**Step 7:** Model Evaluation



The performance of the model is evaluated using:



Accuracy score



Accuracy vs K plot



Confusion matrix



Visual inspection of predicted digits





**6. Results**



The KNN model achieved an accuracy of approximately 97–99%.



The Accuracy vs K graph helped identify the optimal value of K.



The confusion matrix showed minimal misclassification, mostly between visually similar digits.



**7. Evaluation Metrics**



Accuracy



Measures the percentage of correctly classified digits.



Confusion Matrix



Provides a detailed view of true vs predicted labels and helps identify misclassified digit pairs.





**8. Advantages of KNN**



Simple and easy to understand



No training phase required



Effective for small datasets





**9. Limitations of KNN**



Computationally expensive for large datasets



High memory usage



Sensitive to noise and irrelevant features



Performance depends heavily on choice of K and scaling



**10. Conclusion**



This project successfully demonstrates handwritten digit classification using the K-Nearest Neighbors algorithm. The task highlights the importance of feature scaling, distance-based learning, and hyperparameter tuning. KNN proves to be an effective algorithm for small, well-structured datasets like handwritten digits.


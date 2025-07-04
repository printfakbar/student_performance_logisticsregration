# student_performance_logisticsregration
Sure! Here’s the **cleaned-up README** without the extra emojis:

---

# Student Performance Prediction using Logistic Regression

## Project Title

**Student Performance Prediction using Logistic Regression**

---

## Week 2 – Mini Project (AI & ML)

This project aims to build a simple machine learning model to predict whether a student will **pass or fail**.

The prediction is based on two key factors: **hours studied** and **attendance percentage**.

The model is created using **Logistic Regression**, which is a supervised classification algorithm.

---

## Objective

The main objective is to understand and implement a basic machine learning classification algorithm — **Logistic Regression** — using a small dataset.

The model should be able to learn from data, predict student outcomes (pass/fail), and evaluate its prediction accuracy.

This project is beginner-friendly and focuses on understanding core ML concepts like data handling, model training, testing, and making predictions.

---

## Dataset Description

The dataset was created manually based on the assignment example.

It has three columns:

| Feature Name    | Description                       |
| --------------- | --------------------------------- |
| `Hours_Studied` | Number of hours a student studied |
| `Attendance`    | Student's attendance percentage   |
| `Pass_Fail`     | Target label: 1 = Pass, 0 = Fail  |

---

### Sample Data (From Assignment)

| Hours\_Studied | Attendance | Pass\_Fail |
| -------------- | ---------- | ---------- |
| 5              | 85         | 1          |
| 2              | 60         | 0          |
| 4              | 75         | 1          |
| 1              | 50         | 0          |

This is a very small dataset, intended purely for learning and experimentation.

---

## Tools and Libraries Used

This project was implemented in **Google Colab**, using Python.

The main libraries used are:

| Library   | Purpose                              |
| --------- | ------------------------------------ |
| `pandas`  | Create and handle the dataset        |
| `sklearn` | Build, train, and evaluate the model |

---

### Example Imports

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## Machine Learning Workflow

### 1. Dataset Creation

The dataset is hardcoded using Python dictionaries and converted into a DataFrame with `pandas`.

---

### 2. Data Splitting

The dataset is split into:

* **Features (`X`)**: `Hours_Studied` and `Attendance`
* **Label (`y`)**: `Pass_Fail`

We use `train_test_split()` to divide data into training and testing sets.

---

### 3. Model Training

A `LogisticRegression()` model is trained using the training data (`X_train`, `y_train`).

---

### 4. Model Testing

The model predicts on the test data (`X_test`), and the accuracy is checked using `accuracy_score()`.

---

### 5. Prediction on New Data

The trained model is used to predict for a **new student**, for example, 3 hours studied and 70% attendance.

---

## Results

### Model Accuracy

The model gives accuracy depending on the small data split.

---

### Example Prediction

```python
new_data = pd.DataFrame([[3, 70]], columns=['Hours_Studied', 'Attendance'])
result = model.predict(new_data)
print("Pass" if result[0] == 1 else "Fail")
```

**Output:**

```
Prediction for new student: Pass
```

---

## Project Files

| File Name                                       | Description             |
| ----------------------------------------------- | ----------------------- |
| `student_performance_logistic_regression.ipynb` | Main notebook with code |
| `README.md`                                     | Project documentation   |

---

## How to Run This Project

1. Open [jupyter notebook]
2. Upload the `.ipynb` file
3. Run each cell step-by-step
4. Test predictions by changing new data values

---

## Important Notes

* This project is based on a **very small dataset** and is only for learning.
* Using `stratify=y` in `train_test_split()` helps avoid imbalanced splits.
* Logistic Regression is a great way to start learning about ML classification.

---

## Author

* **Name:** *Akbar Ali*

---

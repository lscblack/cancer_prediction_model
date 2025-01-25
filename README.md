# Cancer Prediction Model

## Overview
This project demonstrates a simple machine learning model using logistic regression to predict whether a tumor is malignant or benign based on input features. The model is trained using the Scikit-learn library in Python.
## Link To Data
- **kaggle Breast cancer**: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data]

### Backend Api To Access Model Built
FastApi For Backend (service.py)
---

## Features
- **Binary Classification**: Predicts tumor malignancy (malignant or benign).
- **Model Used**: Logistic Regression.
- **Libraries**: Scikit-learn, Joblib, Pandas, NumPy, Matplotlib, and Seaborn.
- **Frameworks**: FASTAPI

---

## Dataset
The model is trained on a cancer-related dataset containing features extracted from tumor samples. Each sample is labeled as malignant or benign. The dataset includes features such as:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness

### Target Variable:
- `1`: Malignant
- `0`: Benign

---

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install scikit-learn joblib pandas numpy matplotlib seaborn
```

---

## Steps to Run the Project

1. **Train the Model**:
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from joblib import dump

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    # Save the model
    dump(lr, "cancer_prediction_model.joblib")
    print("Model saved as cancer_prediction_model.joblib")
    ```

2. **Test the Model**:
    ```python
    from joblib import load

    # Load the saved model
    model = load("cancer_prediction_model.joblib")

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    ```

3. **Visualize Results**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    ```

---

## Model File
- **Filename**: `cancer_prediction_model.joblib`
- **Format**: Serialized model saved using Joblib.

### How to Load the Model
```python
from joblib import load
model = load("cancer_prediction_model.joblib")
```

---

## Results
Evaluate the performance of the model using metrics such as:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

Sample metrics might include:
- Accuracy: 98%
- Precision, Recall, and F1-score for both classes.

---

## Notes
- Ensure the dataset is preprocessed (e.g., handling missing values, scaling features) before training the model.
- Hyperparameters for Logistic Regression can be tuned to improve performance.

---

## Author
**Loue Sauveur Christian**

If you have any questions or need further help, feel free to reach out!


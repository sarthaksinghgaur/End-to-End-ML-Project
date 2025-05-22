# Diabetes Prediction - ML Deployment on AWS Elastic Beanstalk

This project is an **end-to-end machine learning pipeline** built to predict whether a person is diabetic or not based on various medical attributes. It includes **data preprocessing, feature engineering, model training with ensembling**, and **deployment on AWS Elastic Beanstalk** using a Flask web interface.

---

## 🚀 Demo

Live deployment: _[Hosted on AWS Elastic Beanstalk](http://endtoendmlproject-diabetespredic-env.eba-4z8i5sjw.eu-north-1.elasticbeanstalk.com)_  
Open it (use HTTP for now)

---

## 📂 Project Structure

```
.
├── application.py              # Flask app entrypoint for AWS Elastic Beanstalk
├── artifact/                   # Contains trained model, preprocessor, and data
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── *.csv
├── catboost_info/             # Training logs for CatBoost
├── notebook/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── src/                       # Source code for ML pipeline
│   ├── components/            # Data ingestion, transformation, training modules
│   ├── pipeline/              # Prediction pipeline
│   └── utils.py, logger.py    # Utility functions and logging
├── templates/                 # HTML templates for Flask frontend
│   ├── home.html
│   └── index.html
├── requirements.txt           # Python dependencies
├── setup.py
└── README.md
```

---

## 📊 Input Features

The model expects the following input features:

| Feature             | Description                        |
|---------------------|----------------------------------|
| `gender`            | Male/Female                      |
| `age`               | Age in years                     |
| `hypertension`      | 0 = No, 1 = Yes                  |
| `heart_disease`     | 0 = No, 1 = Yes                  |
| `smoking_history`   | Smoking status (e.g. never, current)|
| `bmi`               | Body Mass Index                  |
| `HbA1c_level`       | Hemoglobin A1C level             |
| `blood_glucose_level` | Blood Glucose Level            |

📥 Sample input:

```csv
gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
Female,44,0,0,never,19.31,6.5,200
```

📤 Output:

```
Diabetic / Not Diabetic
```

---

## 🔬 Feature Engineering

- **BMI Category**: Underweight / Normal / Overweight / Obese
- **Age Category**: Child / Young Adult / Middle Aged / Senior

---

## 🧠 Model Training

- Applied **SMOTE** to balance the imbalanced dataset.
- Evaluated multiple classifiers:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Logistic Regression
  - XGBoost
  - CatBoost
  - AdaBoost
  - Voting Classifier
  - Stacking Classifier
- Best model selected based on **F1 Score**, with additional metrics:
  - ROC AUC
  - Precision
  - Recall

---

## ⚙️ Deployment

- Deployed on **AWS Elastic Beanstalk**
- WSGI entry point configured via `application.py`

```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python application.py
```

---

## 🧪 Run Jupyter Notebook (EDA)

```bash
cd notebook
jupyter notebook
```

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

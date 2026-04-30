# IBM Project
### Software and tools requirement
1. [Github account](https://github.com)
2. [VS code account](https://code.visualstudio.com)

# 🎓 Breast_Cancer_Detection (End-To-End ML Project)

## 📌 Overview

This project aims to predict the survival outcome of breast cancer patients using machine learning models based on clinical and diagnostic features. By analysing historical medical data, the system classifies whether a patient is likely to survive or not.

The project follows a **modular machine learning pipeline architecture**, including data ingestion, transformation, model training, and evaluation.

---

## 🚀 Project Objectives

* Develop an accurate predictive model
* Estimate breast cancer survival outcomes
* Enable early decision-making
* Improve treatment planning

---

## 🧠 Problem Statement

Predicting breast cancer survival is complex due to multiple clinical factors, and existing methods may lack accuracy. Therefore, an efficient machine learning model is needed to accurately predict survival outcomes and support better medical decision-making.

##### Target Variable: Patient_Status

---

## 🗂️ Project Structure

```
Student-Performance-Prediction/
│
├── artifacts/                # Generated files (datasets, models, preprocessor)
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── notebooks/               # Jupyter notebooks (EDA & experiments)
│   ├── data/
│   │    └── stud.csv
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
│
├── src/
│   ├── components/
│   │    ├── data_ingestion.py
│   │    ├── data_transformation.py
│   │    └── model_trainer.py
│   │
│   ├── pipeline/
│   │    └── training_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── venv/                    # Virtual environment
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Tech Stack

* **Python 3.10**
* **Pandas, NumPy**
* **Scikit-learn**
* **Matplotlib, Seaborn**
* **Pickle (Model Serialization)**

---

## 🔄 ML Pipeline Workflow

### 1. Data Ingestion

* Reads dataset from source
* Splits into train & test datasets
* Saves raw and processed files

### 2. Data Transformation

* Handles missing values
* Applies encoding to categorical features
* Scales numerical features
* Saves preprocessing pipeline (`preprocessor.pkl`)

### 3. Model Training

* Trains regression models
* Evaluates performance
* Saves best model (`model.pkl`)

---

## 📊 Features Used

### Numerical Features

* 'Age',
* 'Protein1',
* 'Protein2',
* 'Protein3',
* 'Protein4'

### Categorical Features
* 'Gender',
* 'Tumour_Stage',
* 'Histology',
* 'ER status',
* 'PR status',
* 'HER2 status',
* 'Surgery_type'


---

## 🧪 How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-repo-link>
cd Student-Performance-Prediction
```

### Step 2: Create & activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the pipeline

```bash
python -m src.components.data_ingestion
```

---

## 📈 Sample Output

After running the pipeline, the following files are generated:

```
artifacts/
 ├── data.csv
 ├── train.csv
 ├── test.csv
 ├── preprocessor.pkl
 └── model.pkl
```

---

## 📌 Key Highlights

* Modular and scalable ML architecture
* Production-level folder structure
* Custom logging and exception handling
* Reusable preprocessing pipeline
* Clean separation of concerns

---

## ⚠️ Common Issues & Fixes

| Issue                | Solution                         |
| -------------------- | -------------------------------- |
| File not found error | Ensure correct working directory |
| Kernel crash         | Install `ipykernel` in venv      |
| Model not saving     | Check artifacts path             |
| Import errors        | Run using `python -m`            |

---

## 📚 Future Improvements

* Add Flask/FastAPI deployment
* Integrate CI/CD pipeline
* Add model monitoring
* Hyperparameter tuning
* Docker containerization

---

## 👨‍💻 Author

**Priyanshi Vishwakarma**

---

## ⭐ Acknowledgements

* Scikit-learn documentation
* Kaggle dataset inspiration
* ML pipeline best practices

---

## 📬 Contact

Feel free to connect for collaboration or queries.

---

# IBM Project

### Software and tools requirements
1. [Github account](https://github.com)
2. [VS code account](https://code.visualstudio.com)

# Breast_Cancer_Detection

## 📌Overview
This project aims to predict the survival outcome of breast cancer patients using machine learning models based on clinical and diagnostic features. By analysing historical medical data, the system classifies whether a patient is likely to survive or not.

The project follows a **modular machine learning pipeline architecture**, including data ingestion, transformation, model training, and evaluation.

---

## 🚀Project Objective

* Develop an accurate predictive model
* Estimate breast cancer survival outcomes
* Enable early decision-making
* Improve treatment planning

---

## 🧠Problem Statement

Predicting breast cancer survival is complex due to multiple clinical factors, and existing methods may lack accuracy. Therefore, an efficient machine learning model is needed to accurately predict survival outcomes and support better medical decision-making.

---

## 🗂️Project Structure

```
Breast_Cancer_Detection/
│
├── ├── artifacts/                # Generated files (datasets, models, preprocessor)
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
|   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   └── breast_cancer_best_model.pkl
│
├── notebooks/               # Jupyter notebooks (EDA & experiments)
│   ├── data/
│   │    └── data.csv
│   ├──  EDA.ipynb
│   └──  model_trainer.ipynb
│
├── src/
│   ├── components/
│   │    ├── data_ingestion.py
|   |    ├── __init__.py
│   │    ├── data_transformation.py
│   │    └── model_trainer.py
|   |          
│   │
│   ├── pipeline/
│   │    ├── training_pipeline.py
|   |    ├── predict_pipeline.py
│   │    └── __init__.py
|   |
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
|   └── __init__.py
│
├── venv/                    # Virtual environment
├── requirements.txt
├── setup.py
└── README.md
```
---

## ⚙️Tech Stack

* **Python 3.10**
* **Pandas, NumPy**
* **Scikit-learn**
* **Matplotlib, Seaborn**
* **Pickle (Model Serialization)**

---

## 🔄ML Pipeline Workflow

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

## 📊Features used

### Numerical Features

* Age
* Protein1
* Protein2
*  Protein3
* Protein4
* Followup_Days


### Categorical Features

* Gender
* Tumour_Stage 
* Histology
* ER status
* PR status
* HER2 status
* Surgery_type

---

## 🧪How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-repo-link>
cd Breast_Cancer_Detection
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

## 📈Sample Output

After running the pipeline, the following files are generated:

```

├── artifacts/                # Generated files (datasets, models, preprocessor)
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
|   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   └── breast_cancer_best_model.pkl
```

---

## 📌Key Highlights

* Uses machine learning to predict breast cancer survival
* Analyzes clinical and diagnostic patient data
* Identifies important factors influencing survival
* Compares multiple models for best accuracy
* Provides reliable predictions to support medical decisions
* Helps in early risk assessment and treatment planning

---
## ⚠️Common Issues & Fixes

| **Issue**          | **Fix**                             |
| ------------------ | ----------------------------------- |
| Missing values     | Fill or remove missing data         |
| Imbalanced dataset | Use SMOTE or class weighting        |
| Overfitting        | Use cross-validation/regularization |
| Low accuracy       | Tune model and select features      |
| Encoding issues    | Use Label or One-Hot Encoding       |
| Scaling problems   | Apply StandardScaler                |
| NaN in data        | Clean and preprocess dataset        |
| Import errors      | Install and import libraries        |

---

## 📚Future Improvements


* Use advanced ML/DL models to improve accuracy
* Train on larger and more diverse datasets
* Deploy as a web or mobile application
* Add explainable AI for better understanding
* Include more patient and genetic data
* Automate the complete prediction pipeline
* Update model with new data regularly

---

## 👨‍💻Author

**Sakshi Khandu**

---

##  ⭐Acknowledgements

* Scikit-learn documentation
* Kaggle dataset inspiration
* ML pipeline best practices

---
##  📬Contact

Feel free to connect for collaboration or queries.

---
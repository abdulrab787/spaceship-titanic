🌌 Spaceship Titanic — Machine Learning Classification Project
A complete end‑to‑end ML pipeline with feature engineering, modeling, and Kaggle submission.

## Objective
Predict whether passengers were transported to an alternate dimension.

## Dataset
Kaggle Spaceship Titanic Competition

## Approach
- EDA and visualization
- Feature engineering
- Classification models
- Pipeline-based training


🏷️ Badges
https://img.shields.io/badge/Python-3.10-blue https://img.shields.io/badge/ScikitLearn-ML-yellow https://img.shields.io/badge/XGBoost-GradientBoosting-orange https://img.shields.io/badge/Kaggle-Competition-blue https://img.shields.io/badge/Status-Active-success https://img.shields.io/badge/License-MIT-green


🚀 Project Overview
The Spaceship Titanic dataset challenges you to predict whether passengers were transported to another dimension after a mysterious cosmic accident.
This project demonstrates a full professional ML workflow, including:
- Data cleaning & preprocessing
- Feature engineering
- Model training & evaluation
- XGBoost optimization
- Kaggle submission generation
- Reproducible project structure

🧠 Key Features & Engineering
🔧 Engineered Features
- Deck, Side, CabinNum extracted from Cabin
- TotalSpend (sum of all spending categories)
- GroupSize (passenger group count)
- Age imputation
- Boolean cleanup for VIP and CryoSleep
- Removal of non‑predictive identifiers

📦 Preprocessing Pipeline
- Numeric imputation (median)
- Categorical imputation (most frequent)
- One‑hot encoding
- Integrated into a Scikit‑Learn Pipeline

📊 Model Performance
|Model              |Validation Accuracy | 

|Random Forest      | 0.7999             | 
|XGBoost            | 0.8177             | 
|XGBoost + GroupSize| 0.8177             | 


XGBoost consistently delivered the strongest performance.

📁 Project Structure
spaceship-titanic/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│
├── submissions/
│   ├── submission.csv
│
└── README.md



🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit‑Learn
- XGBoost
- Matplotlib / Seaborn
- VS Code
- Kaggle

📥 How to Run the Project
1. Install dependencies
pip install -r requirements.txt


2. Train the model
python src/train.py


3. Generate Kaggle submission
python src/train.py --submit



📤 Kaggle Submission
The final model generates a submission.csv file with:
- PassengerId
- Transported (True/False)
This file is ready to upload directly to Kaggle.





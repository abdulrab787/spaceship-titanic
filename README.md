🚀 Spaceship Titanic — Kaggle Machine Learning Project

Predicting passenger transportation to an alternate dimension using supervised machine learning

📌 Project Overview

The Spaceship Titanic competition is a binary classification problem hosted on Kaggle.
The goal is to predict whether a passenger was transported to an alternate dimension after a spacetime anomaly, using personal and travel-related records recovered from the ship’s computer system.

This project demonstrates a full end-to-end machine learning pipeline, including data exploration, feature engineering, model comparison, hyperparameter tuning, and final Kaggle submission — all implemented with reproducibility and professional project structure in mind.

🧠 Problem Statement

Given structured tabular data describing passengers (demographics, cabin information, spending behavior, and travel groups), predict the target variable:

Transported ∈ {True, False}

Evaluation Metric

Accuracy (percentage of correctly predicted labels)

🗂️ Project Structure
spaceship-titanic/
│
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── models/
│   └── xgb_final_model.pkl
│
├── submission.csv
├── requirements.txt
└── README.md

🔍 Exploratory Data Analysis (EDA)

Key insights from EDA:

Passengers often travel in groups/families, encoded in PassengerId

Spending behavior (RoomService, Spa, VRDeck, etc.) strongly correlates with transport outcome

Missing values are common but structured

Certain categorical variables (e.g., CryoSleep, Deck) provide strong signal

EDA was used to guide feature engineering and model selection, not just visualization.

🛠️ Feature Engineering & Preprocessing
Key preprocessing steps:

Missing value imputation (numerical + categorical)

Cabin feature extraction:

Deck

Side

GroupSize feature extracted from PassengerId

Separation of:

Numerical features → scaled where appropriate

Categorical features → one-hot encoded

Modular preprocessing implemented in src/preprocessing.py

This design ensures reusability across notebooks, scripts, and Kaggle.

🤖 Model Development & Comparison

Multiple models were evaluated using a validation split:

Model	Validation Accuracy
Logistic Regression (baseline)	~0.79
Random Forest	~0.73
LightGBM	~0.816
🏆 Tuned XGBoost	0.8223
Final Model: XGBoost

Captures non-linear interactions in spending & cabin features

Handles mixed feature types effectively

Demonstrated best generalization performance

⚙️ Final Model Configuration (XGBoost)

Key parameters:

n_estimators = 800

max_depth = 5

learning_rate = 0.03

subsample = 0.85

colsample_bytree = 0.85

Regularization to reduce overfitting

The final model was trained on the full training dataset before submission.

📈 Results

Validation Accuracy: 0.8223

Competitive Kaggle leaderboard placement

Clean, reproducible ML pipeline suitable for real-world projects

📤 Kaggle Submission

The final submission file follows the required format:

PassengerId,Transported
0013_01,False
0018_01,True
...


The model was retrained on the full dataset before generating predictions for the test set.

🧪 Reproducibility

This project emphasizes:

Modular code (src/)

Clear separation of data, models, and notebooks

Environment consistency between VS Code and Kaggle

Explicit path handling to avoid import issues

🧠 Key Takeaways

Feature engineering can outperform complex ensembling

Group-level information significantly improves performance

Boosted tree models are highly effective for structured tabular data

Validation-driven decision making is essential in Kaggle competitions

🧩 Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

XGBoost, LightGBM

Kaggle Notebooks

Git & GitHub

VS Code

📬 Contact

If you’d like to discuss this project or collaborate:

Abdurrab Nizamuddeen
📧 GitHub: (https://github.com/abdulrab787)
📊 Kaggle: (https://www.kaggle.com/abdurrabnizamuddeen)

⭐ If you found this project useful, feel free to star the repository!
Customer Churn Prediction — Artificial Neural Network

A deployed, end-to-end churn prediction system built with TensorFlow/Keras, trained on the classic bank customer churn dataset, and served through an interactive Streamlit app.

🔗 Live demo: add your Streamlit Cloud link here after deployment

Problem

Predict whether a bank customer will churn (leave the bank) based on their profile — credit score, geography, age, balance, product usage, and activity status — so the business can proactively target at-risk customers.

Dataset
10,000 customer records, 11 features after dropping identifier columns (RowNumber, CustomerId, Surname)
Target: Exited (1 = churned, 0 = stayed)
Class distribution: ~20.4% churn rate (imbalanced)
Model

A 3-layer feedforward neural network:

Dense(64, activation='relu', input_shape=(12,))
Dense(32, activation='relu')
Dense(1, activation='sigmoid')
Optimizer: Adam (lr=0.01)
Loss: Binary Crossentropy
Callbacks: Early Stopping (patience=10, monitors val_loss, restores best weights) and TensorBoard logging
Hyperparameter search: hidden layer sizes and neuron counts tuned via GridSearchCV/Scikeras before settling on this architecture
Training stopped at epoch 14 (of 100 max); best weights restored from epoch 4
Results (held-out test set, 2,000 rows never used in training)
Metric	Score
Accuracy	86.35%
Precision (churn class)	72.6%
Recall (churn class)	49.1%
F1 (churn class)	0.586
ROC-AUC	0.857

Note on class imbalance: with only ~20% of customers churning, accuracy alone is a misleading headline number. Recall on the churn class (49%) shows the model currently misses roughly half of actual churners — a natural next step would be class weighting, threshold tuning, or resampling (e.g. SMOTE) to push recall up, likely at some cost to precision.

Tech stack

Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Streamlit, TensorBoard

Project structure
├── app.py                      # Streamlit app
├── model.h5                    # Trained Keras model
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for Geography
├── requirements.txt
└── notebooks/
    ├── experiments.ipynb       # Data prep, model training, hyperparameter tuning
    └── prediction.ipynb        # Standalone inference walkthrough
Run locally
bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
Author

Shushant Kumar — LinkedIn · GitHub

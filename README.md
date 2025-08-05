# Relative-Sports-Prediction-using-Machine-Learning
This project uses Machine Learning to predict the outcome of 1v1 competitive sports matches by analyzing relative differences between two players' stats — such as wins, losses, height, reach, and age.
# 🧠 AI-Based Relative Sports Outcome Predictor

This project uses supervised Machine Learning to predict the outcome of one-on-one sports matches based on **relative differences** in player statistics. Instead of evaluating players in isolation, it compares their attributes directly — such as wins, losses, height, reach, and age — making it adaptable to sports like boxing, tennis, esports, and more.

## 📁 Project Structure


## ⚙️ Setup

1. Install dependencies (recommended: Python 3.9+)
    ```bash
    pip install -r requirements.txt
    ```

2. Ensure your dataset is placed inside the `data/` folder.

3. Run the processing and training steps:
    ```bash
    python read.py         # Prepares data and saves features
    python train_rf.py     # Trains and saves the Random Forest model
    ```

4. Predict random fights:
    ```bash
    python predict.py
    ```

5. Predict unseen matchups:
    ```bash
    python predict_new.py
    ```

## 📌 Notes

- The model uses **relative feature engineering**: differences between two opponents’ stats.
- Predictions include **confidence scores** from the model.
- You can extend this framework using XGBoost, add more features, or apply it to other sports.

---

*Built as a hands-on application of AI in sports analytics.*

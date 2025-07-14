
# 🎭 Sentiment Analysis on IMDB Movie Reviews

This project performs **binary sentiment classification** (positive or negative) on the IMDB movie review dataset using machine learning models.

## 📚 Dataset

- **Source**: [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size**: Originally 50,000 reviews (25k pos / 25k neg)
- **After preprocessing**: 49,582 samples (after dropping duplicates)
- **Split**: 80% training (~39,665), 20% testing (~9,917)
- **Labels**: `0 = Negative`, `1 = Positive`

## ⚙️ Preprocessing

- Removed HTML tags and special characters
- Converted text to lowercase
- Tokenized and vectorized using `TF-IDF` (`max_features=10000`)
- Preserved raw structure without stemming/lemmatization

## 🧠 Base Models

- **Decision Tree** (`criterion='entropy'`)
- **Random Forest** (`default params`)

| Model           | Accuracy | Time (s) |
|----------------|----------|----------|
| Decision Tree  | **71.29%** | 40.45    |
| Random Forest  | **83.89%** | 85.60    |

## 📈 Extended Comparison (Optimized & Boosting Models)

Additional models were added and optimized in the extended section:

- **Decision Tree** (`max_depth=20`)
- **Random Forest** (`n_estimators=200`)
- **AdaBoost**, **Gradient Boosting**, **XGBoost**

| Model              | Accuracy | Time (s) |
|--------------------|----------|----------|
| Decision Tree      | 72.56%   | 16.06    |
| Random Forest      | 83.28%   | 28.35    |
| AdaBoost           | 74.77%   | 53.20    |
| Gradient Boosting  | 79.40%   | 101.28   |
| XGBoost            | **80.08%** | 18.70    |

> 🏆 **XGBoost** achieves the best trade-off between accuracy and training time.

## 📊 Visualization

The notebook includes:
- Side-by-side bar plots for **Accuracy** and **Training Time**
- Accuracy labels displayed on each bar for clarity

## 🚀 How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```

2. Open the notebook
   ```
   Sentiment_Analysis_IMDB.ipynb
   ```

3. Run all cells (tested on Python 3.10, scikit-learn, xgboost)

## 📌 Notes

- All models were evaluated on a 20% held-out test set.
- Consider further improvements using deep learning (e.g. LSTM, BERT), more text preprocessing, or hyperparameter tuning via `GridSearchCV`.

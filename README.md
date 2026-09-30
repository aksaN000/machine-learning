# Machine Learning Notebooks

Eight Jupyter notebooks from my machine learning coursework (CSE427, BRAC University). They move from exploratory data analysis to classic algorithms, several of which are implemented from scratch alongside the scikit-learn version, and end with an introduction to deep learning.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 1 | `part1_titanic_survival_prediction` | EDA and preprocessing on Titanic; decision tree and random forest with scikit-learn **and from scratch** |
| 2 | `part2_classification_algorithms` | EDA, preprocessing and AdaBoost with scikit-learn |
| 3 | `part3_iris_knn_classification` | K-nearest neighbours **from scratch** on Iris |
| 4 | `part4_data_preprocessing` | Logistic regression **from scratch** |
| 5 | `part5_advanced_ml_techniques` | A small multilayer perceptron with sigmoid activations and hand-derived backpropagation in NumPy |
| 6 | `part6_real_estate_price_prediction` | Real-estate price regression: EDA, feature engineering, model comparison, feature importance |
| 7 | `part7_ml_algorithms_comparison` | Side-by-side comparison of classifiers and regressors on shared datasets, with timing |
| 8 | `part8_deep_learning_neural_networks` | Intro to deep learning in TensorFlow: CNNs, transfer learning, recurrent networks |

`ml_utilities.py` is a standalone helper module (`DataPreprocessor`, `MLEvaluator`, sample-dataset generators) written while consolidating the labs. `MACHINE_LEARNING_GUIDE.md` and `PROJECT_REPORT.md` are my study notes.

## Tech stack

Python · NumPy · pandas · scikit-learn · Matplotlib · seaborn · SciPy · TensorFlow (notebook 8 only) · Jupyter

## Running locally

```bash
git clone https://github.com/aksaN000/machine-learning.git
cd machine-learning
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Most notebooks use scikit-learn's built-in datasets (Iris, Wine, Breast Cancer, Diabetes) or Keras datasets. Notebooks that read a CSV load it from a URL set in their first cells. `dataset.csv` is a small plant-growth dataset used for practice.

# SMS Spam Detection

Simple machine learning project to detect spam in SMS messages. The solution follows a basic data science workflow: exploratory data analysis, leakage-aware train/test split, TF-IDF text representation, Logistic Regression, model evaluation, and a Streamlit web app for inference.


## Project Structure

- `spam.csv`: SMS Spam Collection dataset.
- `app/`: Python source code for training, inference, and the Streamlit app.
- `notebook/`: Jupyter notebook with the step-by-step data science process.
- `models/`: trained model artifact used by the app.
- `reports/`: saved evaluation metrics.
- `tests/`: functional tests for inference.
- `requirements.txt`: Python packages and pinned versions.
- `Dockerfile`: container definition for the Streamlit app.

## Data Science Process

The dataset contains SMS messages labeled as `ham` or `spam`. During cleaning, only the useful columns are kept, empty messages are removed, labels are normalized, and duplicated messages are removed before splitting to reduce leakage risk.

The dataset is imbalanced after cleaning:

- Ham: 4,516 messages, 87.55%.
- Spam: 642 messages, 12.45%.
- Total rows after cleaning: 5,158.

Because spam is the minority class, the project uses a stratified split and `class_weight="balanced"` in Logistic Regression. The final model is a Scikit-learn `Pipeline` with `TfidfVectorizer` and `LogisticRegression`, so TF-IDF is fitted only on training folds and not on the full dataset.

Overfitting and data leakage controls:

- Train/test split is performed before TF-IDF fitting.
- TF-IDF and classifier are wrapped in a single pipeline.
- Hyperparameters are selected with `GridSearchCV` only on the training set.
- Cross-validation uses stratified folds.
- TF-IDF uses document-frequency limits and validated n-gram ranges.
- Logistic Regression uses regularization and class balancing.

## Model Metrics

Final holdout test metrics:

- Accuracy: 0.9884.
- Spam precision: 0.9603.
- Spam recall: 0.9453.
- Spam F1-score: 0.9528.
- ROC-AUC: 0.9972.
- Average precision: 0.9841.

Confusion matrix on the test set:

- Ham correctly predicted as ham: 899.
- Ham incorrectly predicted as spam: 5.
- Spam incorrectly predicted as ham: 7.
- Spam correctly predicted as spam: 121.

These metrics are appropriate for the dataset because the classes are imbalanced. Spam precision and recall show how well the model catches spam while limiting false spam alerts; average precision and ROC-AUC evaluate ranking quality from predicted probabilities.

## Local Execution

Environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model training:

```bash
python -m app.train
```

Streamlit application execution:

```bash
streamlit run app/streamlit_app.py
```

Functional test execution:

```bash
pytest
```

## Docker Execution

Image build after model training:

```bash
docker build -t spam-detector .
```

Container execution:

```bash
docker run --rm -p 8501:8501 spam-detector
```

The application is available at `http://localhost:8501`.
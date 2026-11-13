# MindScope

MindScope is an educational machine-learning project that predicts student mental-health risk levels from structured student data. It includes a Streamlit dashboard for exploring the dataset and generating predictions, plus an Apache Airflow pipeline for training and evaluating models.

> **Important:** MindScope is a learning prototype, not a medical diagnostic tool. Its predictions should not be used to make clinical or high-stakes decisions.

## What it includes

- **Streamlit dashboard** with dataset statistics, model metrics, and a confusion matrix.
- **CSV prediction upload** with downloadable prediction results.
- **Model comparison** between Logistic Regression, K-Nearest Neighbors, and Random Forest using cross-validation.
- **Airflow pipeline** with preparation, training, and evaluation tasks.
- Saved model and evaluation reports in `models/` and `reports/`.

## Technology

Python 3.11, Pandas, NumPy, scikit-learn, Joblib, Streamlit, Apache Airflow, Matplotlib, Seaborn, Jupyter, and Docker Compose.

## Machine-learning workflow

The target column is `Mental_Health_Risk`. `Student_ID` is treated as an identifier and is excluded from training. The remaining columns describe academic, lifestyle, social, and personal factors, including age, year of study, sleep, study time, GPA, screen time, physical activity, financial stress, family support, attendance, diet, and previous mental-health history.

### Data preparation

- The data is split into **80% training** and **20% testing** sets.
- The split uses `random_state=42` and stratification so target-class proportions are preserved.
- Numerical columns use median imputation followed by `StandardScaler`.
- Categorical columns use constant-value imputation (`missing`) followed by `OneHotEncoder`.
- Transformations are inside each scikit-learn `Pipeline`, preventing preprocessing information from leaking from validation or test data into training.

### Models compared

Three classification pipelines are trained:

| Model | Tuned hyperparameters |
|---|---|
| Logistic Regression | `C`: 0.1, 1, 10; `max_iter=1000` |
| K-Nearest Neighbors | `n_neighbors`: 3, 5, 7, 9; `weights`: `uniform`, `distance` |
| Random Forest | `n_estimators`: 100, 200, 300; `max_depth`: `None`, 10, 20; `min_samples_split`: 2, 5, 10 |

### Optimization and selection

Hyperparameters are selected with scikit-learn `GridSearchCV`. Each configuration is evaluated with **StratifiedKFold**, using 5 shuffled folds and `random_state=42`. The optimization objective is **macro F1-score**, which gives equal importance to each risk class. The model with the highest cross-validation macro F1-score is saved as the final model.

The current result selects **Logistic Regression**, with a cross-validation macro F1-score of approximately **0.7500**.

## Evaluation

The final model is evaluated on the held-out test set using:

- **Accuracy**: overall proportion of correct predictions.
- **Macro precision**: average precision across risk classes.
- **Macro recall**: average recall across risk classes.
- **Macro F1-score**: balanced summary of precision and recall.
- **Confusion matrix**: counts of correct and incorrect predictions for each class.

Current saved test metrics:

| Metric | Score |
|---|---:|
| Accuracy | 75.25% |
| Macro precision | 75.19% |
| Macro recall | 75.18% |
| Macro F1-score | 75.18% |

## Project structure

```text
app/                 Streamlit application
dags/                Airflow DAG
src/                 Training, preprocessing, and evaluation code
data/raw/            Original dataset
data/processed/      Train/test datasets
models/              Saved trained model
reports/             Metrics, selected model, and confusion matrix
notebooks/            Exploratory analysis and training notebooks
docker/               Dockerfiles for Streamlit and Airflow
```

## Run locally

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   ```

2. Install the application dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```bash
   streamlit run app/app.py
   ```

4. Open [http://localhost:8501](http://localhost:8501).

## Run with Docker Compose

Start the Streamlit app and Airflow:

```bash
docker compose up --build
```

- Streamlit: [http://localhost:8501](http://localhost:8501)
- Airflow: [http://localhost:8080](http://localhost:8080)

To stop the services:

```bash
docker compose down
```

## Generate predictions

In the **Prediction** page, upload a CSV containing the same input feature columns used by the trained model. `Student_ID` may be included, but the target column `Mental_Health_Risk` is not required for new data. The app displays predictions and provides a `mindscope_predictions.csv` download.

Example input columns:

```text
Student_ID,Age,Gender,Year_of_Study,Sleep_Hours,Study_Hours_Per_Week,
GPA,Social_Media_Hours,Screen_Time_Hours,Physical_Activity,
Financial_Stress,Family_Support,Attendance_Rate,Part_Time_Job,
Diet_Quality,Previous_Mental_Health_History
```

The uploaded values must use the same column names and compatible value formats as the training data. Extra columns are preserved in the output but are not used by the model.

## Train and evaluate

The Airflow DAG is named `mindscope_pipeline`. It runs these tasks in order:

1. Load and inspect the processed train/test data.
2. Train and compare the configured models.
3. Save the best model to `models/mindscope_model.joblib`.
4. Save model information and evaluation results to `reports/`.

The current saved evaluation reports show Logistic Regression as the best model, with an accuracy of approximately **75.25%** on the test split.

## Outputs

- `models/mindscope_model.joblib`: serialized best preprocessing-and-model pipeline.
- `reports/best_model.json`: selected model and cross-validation score.
- `reports/metrics.json`: test-set accuracy, precision, recall, and F1-score.
- `reports/confusion_matrix.csv`: test-set confusion matrix.

## Notebooks

The `notebooks/` directory contains the project analysis workflow:

- `EDA.ipynb`: exploratory data analysis and visualizations.
- `Preprcessing.ipynb`: cleaning, feature preparation, and train/test split.
- `Model-Training.ipynb`: model comparison, hyperparameter tuning, and evaluation.

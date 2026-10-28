# SmartPCA – PCA-Based Machine-Failure Prediction (Experiment 8)

AI4I 2020 Predictive Maintenance Dataset · Jupyter Notebook (backend) · Streamlit GUI (frontend)

Final model: 9 features → PCA (6 components, 99.56 % variance) → SVM  ·  **Test accuracy 98.35 %**, 5-fold CV 98.46 %

## Files

| File | What it is |
|---|---|
| `PCA_Machine_Failure.ipynb` | Jupyter Notebook – Part A baseline (same as report pages 3–14) and Part B improved model |
| `app.py` | Streamlit GUI – 5 pages, including **Predict Failure** and **Model Accuracy** |
| `ai4i2020.csv` | The dataset (used automatically if the UCI download does not work) |
| `artifacts/` | Trained model and results – created by the notebook, read by the GUI |
| `requirements.txt` | Libraries to install |

## How to run on a Mac (Anaconda already installed)

1. Unzip the folder (for example onto the Desktop).
2. Open **Terminal**, type `cd ` (with a space), drag the folder into the window, press Return.
3. Install the libraries (once):
   ```
   pip install -r requirements.txt
   ```
4. Run the notebook:
   ```
   jupyter notebook
   ```
   Open `PCA_Machine_Failure.ipynb` → **Kernel → Restart & Run All** (about 1 minute).
5. Open a second Terminal window (`Cmd + N`), `cd` into the folder again, and start the GUI:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501. Stop it with `Control + C`.

   Always start the GUI from **inside the project folder**: the hidden folder `.streamlit` there keeps the
   dashboard in its light theme even when the Mac is in dark mode. (Finder hides folders starting with a dot –
   don't delete it.)

`requirements.txt` pins scikit-learn 1.9.1 – the version used for the report screenshots – so you get exactly the
same numbers. With another scikit-learn version the **improved model gives the same 98.35 %**, but the
baseline Random Forest numbers can change slightly (e.g. 98.4 % instead of 98.2 %).

## Checklist – what you should see

**Notebook (no red error boxes anywhere):**

| Section | Expected output |
|---|---|
| Dataset loading | `Dataset shape: (10000, 14)` (the "Online download failed" message is fine – the local file is used) |
| Split | `Training records: 8000`, `Testing records: 2000` |
| PCA (baseline) | `Components selected: 3`, `Variance retained: 95.01 %` |
| Baseline evaluation | RF with PCA features: Accuracy 0.968, Recall 0.4412 |
| Why accuracy is misleading | `Accuracy of always predicting 'No Failure': 96.6 %` |
| Step 6 – PCA improved | `Components selected: 6 of 9`, `Variance retained: 99.56 %` |
| Step 7 – grid search | `Best parameters : {'model__C': 300, 'model__gamma': 0.05}`, `Best CV accuracy (%) : 98.46` |
| Step 8 – evaluation | `Accuracy : 0.9835`, `Recall : 0.6765`, `Accuracy requirement (>= 94 %): MET` |
| Save | `Model saved to : artifacts/pca_failure_model.joblib` |
| Test new readings | Normal machine → No Failure; the other three → Failure |

**GUI:**

| Page | Test | Expected |
|---|---|---|
| Sidebar | look | "Test accuracy: 98.35 %" |
| Project Overview | open | 10,000 rows, 5 numerical features |
| PCA Analysis | slider to 3 | Variance retained 95.02 % (the GUI uses all 10,000 rows; the notebook's 95.01 % uses the 8,000 training rows) |
| Predict Failure | load each example and click Predict Failure | Normal → ✅ No Failure; Heat dissipation / Power / Overstrain → ⚠️ Failure |
| Model Accuracy | open | Accuracy 98.35 %, Recall 67.65 %, confusion matrix 1921 / 11 / 22 / 46 |

## Common problems

| Problem | Fix |
|---|---|
| GUI says "Trained model not found" | Run the notebook once (step 4), then refresh the browser |
| `command not found: streamlit` | `python -m streamlit run app.py` |
| `ModuleNotFoundError` | `pip install` the name shown |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

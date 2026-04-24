# Architecture Overview 

Build an **“AutoBias”** prototype that takes a trained model and a dataset (CSV) as input, then computes fairness metrics, counterfactual tests, and root‑cause analysis, and presents results in a web UI. The system has two parts:

- **Backend (Python)** – A FastAPI service (or Flask) that (a) loads the data and model; (b) adapts them into a common format; (c) calls AIF360’s APIs to compute fairness metrics; (d) runs counterfactual tests and feature‐importance analyses; and (e) returns a structured JSON report.  
- **Frontend (React)** – A single-page app with forms and charts. The user uploads the CSV and model file, specifies the target column and sensitive attribute(s), and clicks “Analyze.” The app then calls the backend and displays the results (e.g. metrics, bar charts, tables of top features, suggestions).  

This document details each component, including data formats, required AIF360 calls, and what is/ isn’t provided by AIF360. The goal is a **developer‐friendly prototype** (no training, just analysis).  

# Backend Components 

## 1. Data and Model Input 

- **Dataset (CSV)**: The user’s CSV is loaded into a Pandas DataFrame (`df`). It must contain a target column (`y`) and one or more *sensitive attributes* (e.g. `gender`, `race`). For the prototype, we *ask the user* to specify which column(s) are sensitive and which column is the target. (Automating detection via an LLM can be added later.)  
- **Model file**: Accept a serialized model (e.g. a pickled scikit-learn model). To limit scope, support **scikit-learn models** (LogisticRegression, RandomForest, XGBoost, etc.). The backend should load the model (e.g. via `joblib.load(path)`) into a `sklearn` estimator object. If needed later, you could add support for ONNX or PyTorch, but for prototyping, one common interface suffices.

### Model Adapter Interface 

Since we may accept different model types, define a simple wrapper class. For example:

```python
class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        # Convert X (DataFrame) to numpy if needed, then return model.predict(X)
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

In practice, the code checks: if the model has `predict_proba`, use it for additional score analysis; otherwise use `predict`.  The wrapper normalizes input (Pandas → NumPy) so all models behave the same. 

> *Note:* We do **not** re-train the model here. We assume it is pre-trained and simply call `model.predict(X)` on the input dataset. The backend can validate compatibility (e.g. the model’s expected features match the CSV columns).

## 2. Preparing Data for AIF360 

AIF360 expects data in specific formats for its metrics and datasets. The key formats are:

- **Arrays/Series for metrics**: Many AIF360 functions can accept raw arrays or Pandas Series for `y_true`, `y_pred`, and a protected-attribute array. For example, `aif360.sklearn.metrics.statistical_parity_difference(y_true, y_pred, prot_attr=prot_array, priv_group=...)` expects `y_true` and `y_pred` arrays and a `prot_attr` array of sensitive values (e.g. 0/1 or strings).  
- **StructuredDataset / BinaryLabelDataset**: A more “data-centric” approach is to use AIF360’s `StandardDataset` (a subclass of `BinaryLabelDataset`) to wrap the DataFrame. For instance, `StandardDataset(df, label_name='outcome', favorable_classes=[1], protected_attribute_names=['gender'], privileged_classes=[[1]])`. This automatically encodes the data: it one-hot-encodes categoricals, maps protected attributes to 0/1 (privileged=1, unprivileged=0) and maps labels to 0/1【19†L64-L72】【19†L81-L89】. Once wrapped, you can use `BinaryLabelDatasetMetric` or `ClassificationMetric` to compute group-fairness metrics. This is often simpler to handle multiple protected attributes and thresholds【22†L13-L22】【29†L219-L225】.  

For our pipeline, either approach works. A straightforward plan is:

1. Extract `X` (features) and `y` (target) from the DataFrame (dropping the target and sensitive columns from X).  
2. Extract `prot_attr` array from the sensitive column (or columns). For binary-sensitive (e.g. Male/Female), map to 0/1 if needed. Determine `privileged` vs `unprivileged` label (for example, user indicates that “Male”=1 is privileged).  
3. Use the model to get `y_pred = model.predict(X)` (and probabilities if desired).  

### AIF360 Metrics 

Once we have `y_pred` and `prot_attr`, we can compute fairness metrics. AIF360 provides both high-level classes and sklearn-style functions:

- **Group-fairness (Binary outcome)** – We can use the `BinaryLabelDatasetMetric` class. E.g.:
   ```python
   from aif360.datasets import StandardDataset
   from aif360.metrics import BinaryLabelDatasetMetric
   # Wrap dataset
   dataset = StandardDataset(df, label_name=target, favorable_classes=[positive_class],
                             protected_attribute_names=[sensitive_col], 
                             privileged_classes=[[privileged_value]])
   # Create a copy with predictions as labels
   dataset_pred = dataset.copy()
   dataset_pred.labels = y_pred
   # Define groups for metric
   priv = [{sensitive_col: privileged_value}]
   unpriv = [{sensitive_col: unprivileged_value}]
   # Compute metrics
   metric = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
   spd = metric.statistical_parity_difference()    # fairness gap
   di = metric.disparate_impact()                 # ratio
   eqopp = metric.equal_opportunity_difference()  # TPR gap
   ```
   As shown in [22] (StackOverflow), calling `metric_pred.statistical_parity_difference()` yields numbers (e.g. SPD = –0.667) and `metric_pred.disparate_impact()` yields a ratio【22†L13-L22】. The result dictionary in [22] demonstrates example outputs. This method requires specifying privileged/unprivileged groups (as shown above).

- **Individual fairness** – AIF360’s sklearn API has `consistency_score(X, y_pred)` which measures if similar individuals get similar predictions【29†L198-L204】. We can call:
   ```python
   from aif360.sklearn.metrics import consistency_score
   cons = consistency_score(X, y_pred)
   ```
   (Similarity is typically defined via nearest neighbors in feature space).

- **Distributional fairness** – The generalized entropy index (`generalized_entropy_error`) measures inequality in model “benefits”【29†L216-L224】. Compute:
   ```python
   from aif360.sklearn.metrics import generalized_entropy_error
   ge = generalized_entropy_error(y_true, y_pred)
   ```
   (Here we need the true labels for “benefit” distribution. If not available, you could use predictions as proxy.)

- **Other metrics** – AIF360 also offers `statistical_parity_difference`, `disparate_impact_ratio` etc in the sklearn API. For example:
   ```python
   from aif360.sklearn.metrics import statistical_parity_difference, disparate_impact_ratio
   spd_func = statistical_parity_difference(y_true, y_pred, prot_attr=prot_array, priv_group=priv_label)
   di_func  = disparate_impact_ratio(y_true, y_pred, prot_attr=prot_array, priv_group=priv_label)
   ```
   As documented, these return SPD and DI directly【23†L1-L4】. (In practice, note the sklearn version uses only `y_pred` for DI/SPD if provided, ignoring `y_true`).

- **Bias detectors** – AIF360 includes a `bias_scan` function to find high-bias subgroups【29†L219-L225】. One can call:
   ```python
   from aif360.sklearn.detectors import bias_scan
   subset, score = bias_scan(X, y_true, y_pred)
   ```
   This returns a subset definition and score of worst bias. It’s advanced and can be included optionally.

In summary, **AIF360 can calculate many metrics** if we supply the right inputs. The toolkit expects:
- Data as Pandas DataFrames or NumPy arrays.
- Protected attributes explicitly provided (e.g. `prot_attr=...` or as part of `StandardDataset`).
- For `StandardDataset`, you must specify which values are privileged. The class documentation says it will map protected attributes to 0/1 privileged values【19†L64-L72】【19†L81-L89】. For metrics like `BinaryLabelDatasetMetric`, you supply the privileged/unprivileged group definitions. 

We should note what *AIF360 cannot do*: it does **not** automatically pick privileged groups or root causes, and it doesn’t generate counterfactual examples by itself. We will have to write that logic. AIF360 also focuses on tabular data only【29†L219-L224】 and won’t suggest fixes automatically.

## 3. Counterfactual Generation 

A key custom feature is to test **“if we change the sensitive attribute, does the prediction change?”** For each row in the dataset, create one or more counterfactual inputs:

- **Binary sensitive attribute**: If the sensitive column is binary (e.g. Gender = {Male, Female}), flip it for each record. (E.g., if row has Male, set it to Female; and vice versa.) Then call `model.predict()` on the modified row.  
- **Multi-category sensitive attribute**: If more than two values, you can iterate through all other possible values as alternatives.

Algorithm for binary case:
```text
changed = 0
for each row i in original dataset:
    original_pred = model.predict(row)
    for each alternative value v in sensitive_values excluding row[sens]:
        cf_row = row.copy()
        cf_row[sensitive] = v
        new_pred = model.predict(cf_row)
        if new_pred != original_pred:
            changed += 1
# Percentage of flipped decisions = changed/total_rows
```
(If multiple alternatives, you might count a row “flipped” if ANY alternative causes a change, or measure flip-rate across all alternatives.)

The output to report: e.g. *“X% of records changed classification when the sensitive attribute was toggled.”* This matches the *counterfactual fairness* idea【27†L456-L464】. AIF360 does not provide this out of the box, so we must implement it.  

For large datasets, doing this per-row can be slow; for the prototype, use a sample (or vectorize where possible). But it’s straightforward code. We can cite [27] for context: the article notes that **“Counterfactual Testing ensures predictions remain consistent if sensitive attributes are changed”**【27†L456-L464】, which is exactly what we implement.

## 4. Root-Cause Analysis (Feature Proxy Detection) 

AIF360 won’t tell us why bias exists; we must analyze features. A simple approach:

- **Feature importance**: For a given model (if linear or tree-based), extract feature importances or coefficients:
  - For linear models (`LogisticRegression`), use `model.coef_[0]` (absolute value).  
  - For trees/random forests (`RandomForestClassifier`), use `model.feature_importances_`.  
  - For other models, you could use permutation importance (`sklearn.inspection.permutation_importance`) or a library like SHAP (out of scope for basic prototype).  
- **Correlation with sensitive attribute**: For each feature, compute its correlation or mutual information with the sensitive attribute (e.g., `sklearn.feature_selection.mutual_info_classif(X, prot)` treating `prot` as target). High correlation means the feature is a proxy.  

Combine these signals: features that are **both important for prediction and highly correlated with the sensitive attribute** are likely causing bias (proxy variables). For example, ZIP code often correlates with race and may have high influence. 

The report can list top features by importance, and highlight those with strong sensitive correlation. We might show a table like: 

| Feature        | Importance | Corr with Sensitive |
|----------------|-----------:|--------------------:|
| ZIP code       | 0.24       | 0.78               |
| Income         | 0.22       | 0.05               |
| Years experience | 0.10     | 0.01               |

This section explains why the model’s decisions differ. The Codewave guide on bias highlights using mutual information to detect proxies【27†L449-L458】. We will implement e.g. `mutual_info_classif` or simple Pearson correlation on numeric features.

## 5. Backend API Endpoints

Design a minimal REST API (using FastAPI or Flask):

- **POST /analyze**: Accepts multipart/form-data with `dataset.csv`, `model.pkl`, plus form fields: `target_column`, `sensitive_column`, `privileged_value` (which sensitive value is privileged), optionally `positive_label`. On call, it:
  1. Loads CSV to DataFrame.  
  2. Loads the model via joblib.  
  3. Extracts X, y, prot arrays.  
  4. Runs the ModelWrapper to get `y_pred`.  
  5. Computes AIF360 metrics (SPD, DI, EOD, etc.) and individual/distributional fairness.  
  6. Runs counterfactual test (computes flip% as described).  
  7. Runs root-cause (feature importances and correlations).  
  8. Builds a JSON response containing:  
     - Group fairness metrics (numbers for each metric).  
     - Individual/distributional fairness scores.  
     - Counterfactual flip rate.  
     - List of top-k features and their importance/corr.  
     - Any suggestions (e.g. “Consider removing feature X or reweighting data”).  
  Example JSON structure:
  ```json
  {
    "metrics": {
      "statistical_parity_difference": -0.45,
      "disparate_impact": 0.56,
      "equal_opportunity_difference": 0.10,
      "consistency_score": 0.92,
      "generalized_entropy": 0.35
    },
    "counterfactual_flip_rate": 0.12,
    "top_features": [
      {"feature": "zip_code", "importance": 0.25, "corr_with_sensitive": 0.78},
      {"feature": "income",   "importance": 0.20, "corr_with_sensitive": 0.05},
      ...
    ],
    "suggestions": ["Apply reweighing to balance classes", "Drop proxy feature 'zip_code'"]
  }
  ```
- **GET /status** (optional): Health check.
- **(Optional)** POST endpoints for data clearing or incremental analysis. But for MVP, one analyze call is enough.

Internally, the backend should log the findings (for audit) and return the JSON to the front-end.

# Frontend Components 

Because Streamlit was noted as slow, we’ll use a modern JavaScript framework such as **React** (e.g. Create React App or Vite). The React app will call the backend via HTTP. Key parts:

- **Page/Component: File Upload Form**
  - Two file inputs: “Dataset (CSV)” and “Model File (Pickle)”.  
  - Text inputs / dropdowns: target column name, sensitive column name(s), privileged value (e.g. “Male” or 1).  
  - Analyze button.

- **HTTP Call**: On clicking “Analyze,” the app packages the files and inputs into a `multipart/form-data` POST to `/api/analyze`.

- **Display Results**: Once the JSON response arrives, show:
  - **Fairness Metrics** – e.g. cards or a table listing SPD, DI, EOD, Consistency, etc. Use color coding: red flag if beyond threshold.  
  - **Counterfactual Summary** – e.g. “12% of outcomes changed under counterfactual edits.”  
  - **Feature Table** – table of feature importances and correlations (sorted by importance). Highlight any high-proxy.  
  - **Suggested Fixes** – text bullets (e.g. “Consider re-weighting data or removing feature X”).  
  - Charts: Optional bar charts showing selection rates per group (e.g. Male vs Female acceptance rates). Could use Chart.js or Recharts.

- **Frameworks/Libraries**: Use React with functional components. For UI, you can use Material-UI or Bootstrap for forms/charts. For HTTP, use `fetch()` or `axios`. For charts, a simple bar-chart library or create with HTML/CSS.

- **Workflow**: The user fills form → front end disables button and shows “Analyzing…” → gets response → renders the report. This covers the “user-friendly interface” requirement. 

# Data Flow and Formats 

Summarize the format expectations:

- **Frontend → Backend**: multipart/form-data with fields:
  - `file`: the CSV.
  - `model`: the .pkl file.
  - `target_column`: string.
  - `sensitive_column`: string.
  - `privileged_value`: string or number.
  - (Optionally) `positive_label` for classification threshold.

- **Backend Processing**:
  1. Read `df = pandas.read_csv(file)`.  
  2. `X = df.drop([target_col], axis=1)`, extract features.  
  3. Extract `y = df[target_col]`.  
  4. Extract `prot = df[sensitive_col]` (convert to 0/1 if not already). Use `privileged_value` to mark one group as privileged.  
  5. Load model (`model = joblib.load(model_file)`). Wrap in `ModelWrapper`.  
  6. Compute `y_pred = model.predict(X)`.  

  At this point, we have `y_true`, `y_pred`, and `prot`. These feed into AIF360 metrics and our analyses.

- **AIF360 Dataset (optional)**: If using StandardDataset, do:
  ```python
  from aif360.datasets import StandardDataset
  ds = StandardDataset(df, label_name=target_col, favorable_classes=[pos_label],
                       protected_attribute_names=[sensitive_col],
                       privileged_classes=[[privileged_value]])
  ds_pred = ds.copy()
  ds_pred.labels = y_pred
  ```
  Then pass `ds_pred` into `BinaryLabelDatasetMetric` as shown earlier.

- **JSON Response**: As outlined, contain numeric metrics and arrays of objects. This will be consumed by React to render tables/charts.

# AIF360 Capabilities and Limitations 

**What AIF360 can do:**

- Compute many fairness metrics on tabular data: group metrics (statistical parity, disparate impact, equal opportunity, average odds, etc.), individual metrics (consistency), and distributional metrics (generalized entropy)【29†L198-L204】【29†L216-L224】.  
- Offer `bias_scan` detectors to find worst subgroups【29†L219-L225】.  
- Provide **mitigation algorithms** (pre-, in-, post-processing) if we wanted to fix bias (e.g. reweighing, adversarial debiasing, reject option). These exist in AIF360 (like `Reweighing`, `ExponentiatedGradientReduction`, `RejectOptionClassifier`【13†L135-L143】【13†L150-L159】), but **for the prototype we only need measurement, not mitigation**. (We may simply *suggest* reweighing or feature removal in text.)

- Handle data via its `StandardDataset` class: takes a Pandas DataFrame and required parameters (label column, protected column, privileged class) and converts to a BinaryLabelDataset【19†L64-L72】【19†L81-L89】. This automates encoding and makes metric computation easy. 

**What AIF360 can’t do (out of the box):**

- **No automatic bias fixing**: It provides algorithms, but it won’t choose or apply them for you automatically. Our prototype will not re-train or fix the model; it only reports metrics and suggestions.  
- **No root-cause explanation**: AIF360 doesn’t highlight which feature is causing bias. We add that via importance/correlation as described above.  
- **No auto-protected attribute detection**: It assumes you tell it which column is sensitive. (In future, an LLM or heuristic could guess this.)  
- **Limited data modalities**: It focuses on tabular, binary-label tasks【29†L231-L235】. For multi-class or images, you’d need custom work.  
- **Performance**: Some AIF360 algorithms (like bias_scan or large pre-processors) may be slow on big data. For a prototype, restrict input size or sample.

By pointing out exactly what calls we use (with citations) and what we implement ourselves, we clarify the division of responsibilities. 

# Implementation Plan (Step-by-Step)

1. **Set up Python backend** with FastAPI (or Flask). Define routes and data model.  
2. **Install AIF360** (`pip install aif360[all]`) and dependencies.  
3. **Write data loaders**: functions to load DataFrame and model file.  
4. **Implement model adapter** (as above).  
5. **Compute predictions**: apply the model to the dataset.  
6. **Call AIF360**:  
   - Use `StandardDataset` and `BinaryLabelDatasetMetric` to get SPD, DI, etc. (Cite example usage【22†L13-L22】).  
   - Or use sklearn metrics functions for quick computation (cite examples【23†L1-L4】【29†L198-L204】).  
   - Compute consistency (`consistency_score`) and generalized entropy (`generalized_entropy_error`)【29†L198-L204】.  
7. **Counterfactual logic**: loop over rows, flip sensitive, measure change rate.  
8. **Root-cause logic**: get feature importance (e.g. `model.feature_importances_`), compute feature–sensitive correlations.  
9. **Package results in JSON**.  
10. **Build React app**:  
    - UI for upload and parameters.  
    - On submit, POST to backend.  
    - Display metrics and analyses (use citations style figures if needed, but mostly dynamic content).  
11. **Testing**: Use standard fairness datasets (like UCI Adult or synthetic) to verify metrics.  
12. **Demo**: Prepare a scenario where a bias is present and show the report. 

Throughout, ensure to comment code and handle errors (e.g. user selects non-existent column → return error message). 

# References 

We rely on AIF360’s documentation and examples. For instance, the `StandardDataset` class documentation explains how to feed a DataFrame and how it maps labels and protected attributes to 0/1【19†L64-L72】【19†L81-L89】. The `BinaryLabelDatasetMetric` docs (and examples like [22]) show how to compute SPD, disparate impact, etc. The “Getting Started” guide shows usage of metrics and detectors (e.g. `consistency_score`, `bias_scan`)【29†L198-L204】【29†L219-L225】. Finally, an external guide notes that “Counterfactual Testing ensures predictions remain consistent if sensitive attributes are changed”【27†L456-L464】, justifying our counterfactual generation step. These sources should be consulted during implementation. Each AIF360 function (like `statistical_parity_difference`) may have required parameters (we must supply `prot_attr` or privileged group labels) as shown in its docstring【23†L1-L4】【22†L13-L22】. 

Putting it all together, this design document should give a clear roadmap for Claude or any developer to implement the “Unbiased AI Decision” prototype: a **React+Python** app that uses AIF360 for detection (metrics) and custom code for counterfactual and root-cause analysis, producing an actionable bias report. 


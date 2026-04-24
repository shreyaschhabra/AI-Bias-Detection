# AutoBias: ML Observability & Remediation Platform

## 1. High-Level System Overview
AutoBias is a full-stack Machine Learning auditing and mitigation platform engineered to identify, explain, and automatically heal algorithmic biases within trained models. Built primarily for Data Scientists and Compliance Officers, the platform utilizes advanced mathematical libraries (IBM AIF360, SHAP) and LLM-assisted reporting (Google GenAI) to deliver a seamless fairness observation sequence.

**Core Capabilities:**
- **Auto-Discovery:** Programmatically detects sensitive/protected attributes (e.g., race, age, sex) directly from standard dataset headers via semantic profiling and statistical bounding.
- **Global Fairness Metrics:** Calculates mathematically rigorous metrics (Disparate Impact, Statistical Parity, Equal Opportunity Difference) across single and multiple intersectional demographics.
- **Local Explainability (SHAP):** Unpacks black-box models using SHAP (SHapley Additive exPlanations) to identify the precise feature attributions driving a specific individual's prediction.
- **The Counterfactual Lab:** Provides a sandboxed environment to manipulate an individual's demographic traits natively in the browser and observe changes to the model's live prediction scalar (e.g., "Would this person have been hired if they were male?").
- **Auto-Mitigation (AIF360 Reweighing):** Implements a pre-processing algorithm that transforms the dataset's base mathematical weights to offset historical biases. It re-trains the provided scikit-learn model automatically and generates a fair `.pkl` file for immediate extraction.
- **Automated Data Storytelling:** Summarizes raw statistical boundaries into simple, human-readable 3-paragraph executive statements utilizing Google Gemini 2.5 Flash.

**Core Technologies:**
- **Backend:** Python (3.9+), FastAPI, Pandas, scikit-learn, AIF360, SHAP, Joblib, Google GenAI SDK.
- **Frontend:** React, Vite, Framer Motion (for animations), Recharts (for SVG rendering), Lucide-React, PapaParse.
- **Aesthetics:** 100% Custom Vanilla CSS token architecture built around Glassmorphism and Dark Mode. 

---

## 2. Directory Structure & File Map

```text
ML_Bias/
├── backend/                  # FastAPI Application serving the Auditing & Mitigation Engine
│   ├── main.py               # Core REST API router housing the `/analyze`, `/mitigate`, and `/predict-counterfactual` endpoints.
│   ├── schemas.py            # Pydantic data contracts binding Frontend HTTP requests to Backend types ensuring schema compliance.
│   ├── auto_discovery.py     # Contains heuristic dictionaries and Mutual Information statistics to detect protected demographics without human input.
│   ├── fairness_metrics.py   # Wraps AIF360 equations to execute group-level metrics (DI, SPD) and cross-cuts data to find extreme intersectional boundaries.
│   ├── mitigation.py         # The remediation engine. Executes AIF360's Reweighing, clones the ML model, retrains it via `sample_weight`, and exports it.
│   ├── root_cause.py         # The proxy detection engine. Computes Pearson Correlation against demographic metrics to alert if a variable (e.g., `zip_code`) is secretly acting as a proxy for race.
│   ├── counterfactual.py     # Measures aggregate model fragility by deep-copying a dataframe, flipping all demographics explicitly, and tracking the flip-rate percentage.
│   ├── suggestions.py        # Maps static, heuristic-based textual advice payloads triggered when computed fairness boundaries exceed threshold minimums.
│   ├── llm_client.py         # Securely contacts the Google Gemini HTTP pipeline to construct strict 3-paragraph HR summaries using prompt engineering.
│   ├── model_wrapper.py      # Standardizer matrix. Normalizes different `scikit-learn` algorithms ensuring that `.predict_proba()` output bounds are standardized regardless of algorithmic origin.
│   └── requirements.txt      # Python dependencies encompassing `aif360`, `shap`, `fastapi`, and `scikit-learn`.
│
└── frontend/                 # Interactive React/Vite Multi-Step Wizard
    ├── package.json          # Node dependencies orchestrating building (`recharts`, `framer-motion`, `lucide-react`, `papaparse`).
    └── src/
        ├── main.jsx          # React DOM mounting initialization entrypoint.
        ├── App.jsx           # Master Orchestrator component. Contains the `useState` State Machine determining the exact sequential position of the UI Wizard sequence.
        ├── App.css           # Structural CSS grid definitions binding spacing logic, wizard step dots, and standard animation properties.
        ├── index.css         # Global aesthetics definition point. Assigns Dark Mode hex tokens (`--bg`, `--accent`), radial gradient styling, and the `.glass-panel` definitions.
        └── components/
            └── Wizard/       # The active architectural UI components (sequential path)
                ├── UploadStep.jsx         # Step 1: Implements a Drag-and-drop file configuration for CSVs and physical `.pkl` model drops. 
                ├── DiscoveryStep.jsx      # Step 2: Renders auto-discovered variables alongside the Google Gemini Narrative in a visual container.
                ├── GlobalMetricsStep.jsx  # Step 3: Graphical readout of Disparate Impact and Statistical differences via conditionally colored text vectors.
                ├── LocalExplorerStep.jsx  # Step 4: Contains the `Recharts` interactive bee-swarm scatterplot translating complex mathematical SHAP boundaries into visual click-nodes.
                ├── SidePanel.jsx          # Sub-Drawer inside Step 4. Unveils the individual's local trait attributions and invokes the Counterfactual prediction flip.
                └── MitigationStep.jsx     # Step 5: Before/After comparison dashboard visualizing metric progression alongside the final `.pkl` download hook.
```

---

## 3. Backend Architecture (FastAPI)

The entire server uses a stateless execution design, ensuring files and heavy models aren't unnecessarily cached globally, keeping the HTTP lifecycle swift.

### Core Pipelines
1. **`POST /api/analyze`**: 
   - Receives `UploadFile` (Pickled model and CSV dataframe).
   - Maps inputs via `ModelWrapper`. If the user leaves the `sensitive_column` parameter blank, it invokes `auto_discovery.py` to hunt for protected classes natively.
   - It converts the dataframe into an AIF360 compatible `StandardDataset`.
   - Fires `fairness_metrics.py` to calculate exact boundary metric differences (SPD, DI).
   - Fires `counterfactual.py` to physically invert the dataframe and assess prediction flips.
   - Executes SHAP (`TreeExplainer` for Random Forests, `KernelExplainer` for SVMs). It parses out a multi-dimensional attribution array mapping feature influence dynamically per row. It automatically trims multi-class SHAP dimensions `(N, F, C)` down to `(N, F)` focusing exclusively on the positive target class.

2. **`POST /api/mitigate`**:
   - Takes the exact same CSV and `.pkl` model logic over `multipart/form-data`.
   - Relays them to `mitigation.apply_reweighing_and_retrain()`.
   - The dataset converts into `StandardDataset`. AIF360 `Reweighing` assigns heavier mathematical value calculations to unprivileged individuals with positive outcomes and lighter weights to privileged individuals to establish algorithmic equality.
   - Deep-copies the `.pkl` model object and executes `.fit(X, y, sample_weight=weights)`.
   - Translates the new metric distribution and base64 encodes the repaired `.pkl` directly into the JSON response loop.

3. **`POST /api/predict-counterfactual`**:
   - Unpacks the model and takes a singular JSON `row_data` object passed down from the frontend representing a single individual's traits.
   - Converts the traits back into a DataFrame row natively, returning the isolated live `prediction` and exact `probability` (if `.predict_proba()` is supported).

### Mathematical Definitions
- **Statistical Parity Difference (SPD):** Measures if the favorable prediction rate is independent of the protected class. *Formula:* `Pr(Y^=1 | D=unprivileged) - Pr(Y^=1 | D=privileged)`. Ideal is `0.0`. Less than `-0.1` indicates bias.
- **Disparate Impact (DI):** Maps the ratio of favorable outcomes utilizing the EEOC 80% Rule logic. *Formula:* `Pr(Y^=1 | D=unprivileged) / Pr(Y^=1 | D=privileged)`. Ideal is `1.0`. A score below `0.80` implies concerning discrimination parameters.
- **Equal Opportunity Difference (EOD):** Evaluates if the True Positive Rate behaves identically across groupings. *Formula:* `TPR_unprivileged - TPR_privileged`. Ideal is `0.0`.
- **Consistency Score:** AIF360 calculation tracking identicality of a prediction output compared dynamically to its k-nearest neighbors in standard dimensional space.
- **Generalized Entropy Error (GEE):** Calculates general inequality tracking independent allocation vectors using information theory constructs to track individual-level mathematical unfairness limits.
- **Roots Cause / Proxy Feature Logic:** Uses `sklearn.feature_selection.mutual_info_classif` against the demographic array combined structurally with normalized SHAP impacts. A feature flagged as highly predictive of the final model outcome *and* highly correlated with a protected trait bounds a red flag (e.g. tracking `zip_code` behaving identically to `race`).

---

## 4. Data Contracts & API Schemas (`schemas.py`)

A full listing of cross-communication Pydantic schemas expected between Vite and FastAPI.

**`AnalyzeResponse` (Returned from `/api/analyze`)**
```json
{
  "metrics": {
    "statistical_parity_difference": -0.21,
    "disparate_impact": 0.76,
    "equal_opportunity_difference": -0.15,
    "consistency_score": 0.92,
    "generalized_entropy_error": 0.08
  },
  "counterfactual_flip_rate": 0.045,
  "top_features": [
    { "feature": "income", "importance": 0.45, "corr_with_sensitive": 0.02 },
    { "feature": "zip_code", "importance": 0.12, "corr_with_sensitive": 0.88 }
  ],
  "suggestions": ["Assess zip_code as a potential proxy variable..."],
  "group_selection_rates": { "male": 0.65, "female": 0.52 },
  "detected_sensitive_cols": ["sex", "race"],
  "intersectional_metrics": [
    { "subgroup": "White_Male", "disparate_impact": 1.15 }
  ],
  "shap_values": [
    // Array matching physical CSV row dimensionality. Maps feature string to float offset
    {"income": 0.15, "zip_code": -0.04, "sex": -0.01}, 
    {"income": 0.10, "zip_code": 0.02, "sex": 0.05}     
  ],
  "shap_truncated": false
}
```

**`MitigateResponse` (Returned from `/api/mitigate`)**
```json
{
  "before_metrics": { /* format matches fairness metrics above */ },
  "after_metrics": { /* format matches fairness metrics above */ },
  "mitigated_model_base64": "gASV... (A Base64 string of the joblib dumped sklearn .pkl)"
}
```

**`SummaryRequest` (Sent to `/api/generate-summary`)**
```json
{
  "metrics": { /* Strict payload of FairnessMetrics obj */ },
  "top_features": [ /* Array of top FeatureInfo objects obj */ ]
}
```

**Counterfactual Request Forms (`/api/predict-counterfactual`)**
Transmits as strict `multipart/form-data` natively since we must loop back the `.pkl` artifact.
Expected form fields:
- `model`: (Binary) The File Upload.
- `row_data`: (Stringified JSON) Exact dictionary map of the individual's data values.
- `feature_cols`: (Stringified JSON Array) Ordered list of variables bound to the matrix to guarantee input architecture aligns with model expectation mappings.

---

## 5. Frontend Architecture (React/Vite)

The core aesthetic focuses on Glassmorphism. Transparency overlays atop vivid radial color backdrops generate visual depth.

**Component Form:**
```
App.jsx (React Context/State Manager)
 ├── <nav className="navbar glass-panel" />
 └── <main className="wizard-container">
      └── <AnimatePresence> (Framer-Motion Transitions)
           ├── UploadStep.jsx
           ├── DiscoveryStep.jsx
           ├── GlobalMetricsStep.jsx
           ├── LocalExplorerStep.jsx (Interactive Node Wrapper)
           │    ├── <ScatterChart /> (Recharts Bee-Swarm Plot)
           │    └── SidePanel.jsx    (Drawer handling live API fetches)
           └── MitigationStep.jsx
```

**State Tracking (Session Data Packaging):**
- Data ingestion is encapsulated in `App.jsx` utilizing standard React hooks:
  ```javascript
  const [sessionData, setSessionData] = useState({
    file: null, // React File object referencing the raw physical CSV
    model: null, // React File object interfacing the exact .pkl
    target_column: "",
    sensitive_column: "", 
    privileged_value: "",
    positive_label: "1"
  });
  ```
- All interaction occurs via Fetch APIs routing to `http://localhost:8000/api/...`. Form packaging utilizes `new FormData()` loops seamlessly transforming the file uploads alongside textual inputs into valid stream blocks natively.

**The Local Explorer Functionality (`LocalExplorerStep.jsx`)**
- Upon unlocking the Local Explorer Step, the system utilizes `PapaParse` to actively consume, decrypt, and parse the raw `sessionData.file` natively in browser Javascript memory.
- It parses SHAP mappings passed down from the backend explicitly drawing X/Y coordinate vectors across `Recharts`.
- **Scaling Limit Hook:** To prevent browser DOM failure when scaling scatter dots against 50,000+ CSV parameters, a strict `.slice(0, 400)` filter bound natively caps maximum DOM representation size while ensuring visual interpretability isn't compromised.

---

## 6. Local Setup & Run Instructions

**Backend Setup & Architecture Boot:**
```bash
cd backend

# Initialize standard python environment mappings
python -m venv venv
source venv/bin/activate  # Windows users: \venv\Scripts\activate

pip install -r requirements.txt

# (OPTIONAL) Data Storytelling: 
# Create a .env file and specify your API key natively to prevent console warnings
# Example: GEMINI_API_KEY="AI..."

# Boot the FastApi environment
uvicorn main:app --reload --port 8000
```

**Frontend Boot:**
```bash
cd frontend

# Ingest all node modules (Lucide, Recharts, Framer-motion)
npm install

# Deploy Vite Hot Server
npm run dev
```
Navigate natively to `http://localhost:5173`. Wait for the console mapping to confirm success. 

---

## 7. Known Limitations & Constraints

Due to the fundamental mathematical frameworks wrapping AIF360 mapping limitations, the architecture imposes specific boundaries dictating operational success for Claude tracking.
1. **Binary Classification Prerequisite:** The mitigation logic arrays are mapped exclusively onto Binary target outputs (`1` or `0`, `True` or `False`). They will critically crash if asked to evaluate Continuous integer regressions. Ensure target schemas natively represent boolean outcomes.
2. **`sample_weight` Requirement for Mitigation:** The AIF360 `Reweighing` pipeline generates floating-point arrays explicitly to bias the fit calculation boundaries. To execute remediation, the uploaded `.pkl` structure **MUST** natively invoke algorithms supporting `.fit(X, y, sample_weight=weights)` bounds internally (e.g. Scikit-learn's `RandomForestClassifier`, `LogisticRegression`). Basic K-Nearest Neighbors implementations structurally denying `sample_weight` will forcibly crash Phase 2 Mitigation.
3. **Strict `.pkl` Boundaries:** Neural Network architectures deployed over `.h5` Tensorflow or PyTorch `.pt` variants are structurally incompatible with the current Backend `ModelWrapper`. It depends explicitly upon `scikit-learn` integration standards bound heavily via `joblib.load()`.
4. **Data Privacy Bounds (LLM Integration):** For Phase 3, standard metric statistical derivations (`disparate impact` calculations, generalized feature string names) are securely relayed onto the public Google Gemini Network. **Do not** utilize the proxy storytelling framework to track heavily restricted PII variables. 
```

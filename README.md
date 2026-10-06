# Phenotype-Guided Cardiovascular Classification in Type 2 Diabetes

A research prototype for phenotype-aware comparison of cardiovascular classification architectures. The application accompanies a study evaluating whether phenotype-related heterogeneity should be handled through hard specialist routing or continuous phenotype integration.

**This application estimates the probability of prevalent/concurrent cardiovascular disease. It does not predict future cardiovascular events. Research use only. Not intended for diagnosis or treatment decisions.**

## Scientific question and architectures

Does predictor-by-phenotype heterogeneity support splitting a prediction task into specialists, or adding continuous phenotype information to a pooled model?

|Architecture|Frozen implementation|
|---|---|
|Gatekeeper|8 predictors; observed ODKD target; used for routing, not CVD prediction|
|M0 Global|11 predictors; pooled CVD classifier|
|M1 Hard phenotype routing|Track A (8 predictors) / Track B (7 predictors), trained using observed ODKD; deployed using the Gatekeeper route|
|M2 Hard OOF-route-trained|The same Track A/B panels; specialists trained using Gatekeeper out-of-fold routes|
|M3 Soft integration|Global 11 plus continuous Gatekeeper probability; the additional input was OOF during training; no new feature selection|

The main interface shows M0 and routed M1. The expandable architecture comparison also shows M2 and M3. These individual outputs do not establish overall superiority or clinical utility.

## Inputs and units

|Input|Unit|Input|Unit|
|---|---|---|---|
|Age|years|BUN|mmol/L|
|RDW|%|SUA|μmol/L|
|HbA1c|%|Cl|mmol/L|
|A/G|ratio|NEU#|10^9/L|
|SCr|μmol/L|Non-HDL-C|mmol/L|
|PLT|10^9/L|ALT|U/L|
|K|mmol/L|MCV|fL|
|MON#|10^9/L|LYM#|10^9/L|

Enter canonical SI values directly. **There is no mg/dL-to-SI conversion in the application.** Age must be supplied and must be ≥20. Blank laboratory inputs use the model-specific frozen training medians; these are not newly estimated. The app rejects negative or infinite values. It does not establish that an entered value is clinically plausible.

Gatekeeper routing is fixed: probability ≥ **0.45518994331359863** selects A; otherwise B. The interface displays **T = 0.4552 / 45.52%**. There is no threshold control or CVD classification cutoff. Gatekeeper routing is not a substitute for UACR-based ODKD diagnosis.

## Installation and local use

Use Python **3.12** in a new virtual environment. From this repository's root:

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install -r requirements.txt
streamlit run app.py
```

Load Example 1 or Example 2 and select **Compare frozen architectures**. The two examples are the already-displayed Figure 6 illustrative cases. Unchanged example fields retain their full-precision inputs even though the form displays rounded values; editing a field uses the entered value. Results refer to the last submitted inputs.

Local explanations use `shap.TreeExplainer` with `data=None`, `feature_perturbation='tree_path_dependent'`, `model_output='raw'`, `approximate=False`, and additivity checked. Only M0 and the selected M1 specialist are explained. Bars represent contributions on the log-odds scale, not causal effects. Positive bars are dark gray; negative bars are light gray. No external background dataset is used.

## Streamlit Community Cloud deployment

1. Upload **all contents of this folder**, preserving the `src`, `models`, `examples`, and `.streamlit` subdirectories, to a GitHub repository. Include the dotfiles. Do not upload the ZIP as the sole repository file.
2. In Streamlit Community Cloud, choose **Create app**, select that repository and its branch, and set the main file to **app.py**.
3. In **Advanced settings**, select **Python 3.12**. `runtime.txt` documents the intended version; use the Cloud setting to select the interpreter.
4. Deploy. No secrets, API keys, external database or private data files are needed.
5. Check both examples and the expanded architecture comparison after deployment. This package is locally tested; it has not been deployed by its preparer.

Dependency pins are in `requirements.txt`. The JSON files are native XGBoost model exports, accompanied by exact feature orders and median values. Model and preprocessing hashes are verified at load time. No Python serialized estimator is required by the deployed application.

References: [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency management](https://docs.streamlit.io/deploy/concepts/dependencies), [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), [TreeExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

## Data availability and privacy

No cohort tables, individual training/validation records, stored cohort predictions, or hospital source files are distributed. **The only individual-input exception is the two Figure 6 illustrative cases**, distributed without identifiers, observed outcomes or saved predicted outputs. The app computes their outputs at runtime. Frozen model parameters and training median summaries are provided.

The app does not write submitted inputs or outputs to disk, analytics, a database or a download file. Inputs are processed in the active server session; the hosting platform's own network/security behavior is separate. Do not enter identifying information. There is no file-upload feature.

## Model card, licensing and citation

See `MODEL_CARD.md` for intended use, populations and limitations. See `LICENSE` for the repository's MIT license; it makes no claim to grant rights in underlying source datasets or third-party software. No proprietary font is redistributed. Times New Roman is preferred when available, with a serif fallback on hosts without it.

`CITATION.cff` cites this software prototype using the study author list and the current repository working title. The manuscript is unpublished; no DOI or journal publication is claimed. Update bibliographic details after publication.

The repository is intentionally limited to deployment files and requested public documentation. Local tests, export scripts, parity fixtures and QA screenshots are maintained separately and are not required for deployment.

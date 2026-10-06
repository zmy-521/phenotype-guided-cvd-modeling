# Model card

## Intended use

Research and methodological demonstration for adults aged ≥20 years with type 2 diabetes. This prototype compares frozen CVD classification architectures. It is not a medical device and is not intended for diagnosis, treatment decisions, triage or treatment recommendations.

The outcome is **prevalent/concurrent cardiovascular disease**, not future cardiovascular events or prospective risk. The Gatekeeper target is **observed occult diabetic kidney disease (ODKD)**. Its output informs routing; it does not replace UACR-based ODKD assessment.

## Development and evaluation

- Canonical derivation: NHANES adults with type 2 diabetes, **n=2197**.
- Independent pooled external evaluation: a two-hospital cohort, **n=1154**. Evaluation was pooled, not two separate center validations.
- Feature panels were derived during the original model-development stage and retained unchanged for the canonical age ≥20 reanalysis.
- The released seven estimators are the frozen full-derivation Gatekeeper and M0–M3 components from Stage02. Exporting JSON did not retrain or recalibrate them.
- Saved training medians and exact feature orders are supplied in `models/preprocessing.json`. Models receive canonical SI units without conversion.

## Architecture details

Gatekeeper: Age, BUN, RDW, SUA, HbA1c, Cl, A/G, NEU#.

M0 Global: Age, SCr, SUA, Non-HDL-C, RDW, K, PLT, MCV, ALT, MON#, LYM#.

M1 Track A: Age, SCr, HbA1c, Non-HDL-C, RDW, PLT, ALT, Cl. M1 Track B: Age, SCr, Non-HDL-C, RDW, SUA, K, MCV. M1 training strata use observed ODKD; deployment routes use the Gatekeeper.

M2 uses identical Track A/B feature panels, with specialists trained by Gatekeeper OOF routes. M3 adds continuous Gatekeeper probability to Global 11; this probability was OOF in training. At deployment, M3 receives the frozen full Gatekeeper's probability under the original feature name. It is continuous integration, not an average of specialist outputs.

Exact frozen routing threshold: **0.45518994331359863**. Route A when probability ≥ threshold; otherwise B. The threshold is displayed as 0.4552 and cannot be changed through the interface. No independent CVD decision threshold is applied.

## Evidence boundaries and limitations

- Retrospective/cross-sectional outcome; no future-risk interpretation.
- External cohort shift limits generalization.
- Hard specialist routing showed lower external performance than Global in the frozen evaluation.
- Soft integration showed only a modest AP improvement signal over Global; this does not establish overall superiority or clinical benefit.
- Predictor-by-phenotype heterogeneity does not itself demonstrate specialization benefit.
- No clinical utility claim, diagnostic substitution, subgroup guarantee, or center-specific validation is made.
- Historical preprocessing and feature discovery were not independently repeated by this export. Imputation from frozen medians does not recreate raw pre-imputation measurements or validate arbitrary missing-input patterns.
- The two fixed illustrative cases are demonstrations, not evidence of aggregate performance.

## Local explanations

M0 and the routed M1 specialist use exact Tree SHAP: `tree_path_dependent`, raw log-odds, no external background, `approximate=False`, additivity checked. SHAP 0.52.0 and XGBoost 3.1.2 are pinned. A signed contribution describes model behavior, not a protective factor, causal effect or treatment target.

## Release and data scope

XGBoost JSON exports contain trees/objectives and model parameters; preprocessing contains aggregate frozen median values. No training/validation records or stored cohort probabilities are included. The two already-displayed Figure 6 input examples are the only permitted individual-input exception; identifiers and observed outcomes are omitted.

This release has no live learning, threshold selection, user-data persistence, remote model service, or automatic recalibration. The local release checks compare native JSON inference with frozen estimator predictions, including missing-input cases. Successful local testing does not establish clinical validity or guarantee hosting availability.

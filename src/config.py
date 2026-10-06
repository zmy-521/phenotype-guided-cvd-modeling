from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
EXAMPLE_FILE = ROOT / 'examples' / 'illustrative_cases.json'
TITLE = 'Phenotype-Guided Cardiovascular Classification in Type 2 Diabetes'
GROUPS = {
    'Demographics': ['Age'],
    'Metabolic / biochemical': ['BUN', 'SUA', 'HbA1c', 'Cl', 'A/G', 'SCr', 'Non-HDL-C', 'ALT', 'K'],
    'Hematologic': ['RDW', 'NEU#', 'PLT', 'MCV', 'MON#', 'LYM#'],
}

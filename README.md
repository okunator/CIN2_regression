# CIN2_regression

This repository contains pipelines for predicting CIN2 regression potential through interpretable features.

## Installation (uv)

1. Clone the repository:
```
git clone https://github.com/okunator/CIN2_regression.git
cd CIN2_regression
```

2. Install uv (if needed):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and sync the environment:
```
uv venv
uv sync
```

4. Activate the environment:
```
source .venv/bin/activate
```

## Optional dependencies

Feature interaction analysis uses FACET and sklearndf. Install them if you plan to run [src/feature_interactions.py](src/feature_interactions.py).
```
uv add gamma-facet sklearndf
```

## Expected inputs

### Feature extraction inputs
- WSI folder: slide images (any extension). Files are matched by stem.
- Nuclei annotations: files in a separate folder with matching stems.
- Tissue annotations: files in a separate folder with matching stems.

### Aggregated feature table (classification)
The classification step expects a single CSV/Parquet table with:
- patient/sample identifier
- outcome: binary label (default: reg vs non-reg)
- Feature columns


## Analysis steps

1. Run feature extraction for a folder of WSIs using [src/extract_feats.py](src/extract_feats.py).
2. Aggregate sample-level features using [src/aggregate_feats.py](src/aggregate_feats.py).
3. Run the AutoGluon pipeline on aggregated features using [src/classify.py](src/classify.py).
4. Select features from the results using [src/select_features.py](src/select_features.py).
5. Run the AutoGluon pipeline again using the selected features from [src/select_features.py](src/select_features.py) until you reach the top 10 features.
6. Run feature interaction analysis using [src/feature_interactions.py](src/feature_interactions.py).

## Command-line examples

1. Feature extraction
```
python -m src.extract_feats \
	--wsi_folder /path/to/wsi_folder \
	--nuc_folder /path/to/nuc_annotations \
	--tis_folder /path/to/tis_annotations \
	--result_folder /path/to/features_output
```

2. Aggregate sample-level features
```
python -m src.aggregate_feats --result_folder /path/to/features_output --output_csv /path/to/extracted_sample_feats_agg.csv
```

3. AutoGluon pipeline on aggregated features
```
python -m src.classify --input-data /path/to/extracted_sample_feats_agg.csv --output-dir /path/to/classify_output
```

4. Feature selection
```
python -m src.select_features --input-pkl /path/to/classify_output/results.pkl --output-csv /path/to/selected_features.csv
```

5. AutoGluon pipeline with selected features
```
python -m src.classify --input-data /path/to/selected_feature_dataset.csv --output-dir /path/to/classify_output_refined
```

6. Feature interaction analysis
```
python -m src.feature_interactions --input-data /path/to/extracted_sample_feats_agg.csv --features-file /path/to/selected_features.txt --output-dir /path/to/interaction_output
```

## Outputs

- Feature extraction: per-sample CSVs in the result folder.
- Aggregation: a patient-level CSV (default: extracted_sample_feats_agg.csv).
- Classification: metrics, plots, and results.pkl in the output directory.
- Feature selection: selected_features.csv (or stdout).
- Feature interactions: synergy_matrix.csv, redundancy_matrix.csv, top_synergies.csv.

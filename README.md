# HARP-Med

HARP-Med is a history-aware residual personalized medication recommendation model for predicting appropriate medication combinations based on longitudinal electronic health records.

## Installation

1. Create a new Conda environment:

```bash
conda create -n HARPmed python=3.8
```

2. Activate the environment:

```bash
conda activate HARPmed
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Download the Data

1. You must have obtained access to the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) database before running the code.

2. Download the MIMIC-III dataset and place the required files in the `data/` directory.

Specifically, the following files are required:

`DIAGNOSES_ICD.csv`, `PROCEDURES_ICD.csv`, and `PRESCRIPTIONS.csv`.

3. Download the DrugBank drug information and DDI-related files, and place them in the `data/` directory.

The directory can be organized as:

```text
data/
├── DIAGNOSES_ICD.csv
├── PROCEDURES_ICD.csv
├── PRESCRIPTIONS.csv
└── ...
```

## Process the Data

Run the following command to preprocess the data:

```bash
python process.py
```

The processed files will be saved in the corresponding data directory and used for model training and evaluation.

## Run the Model

The main implementation files are located in `src/`.

```text
HARP-Med/
├── data/
├── src/
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```

Train HARP-Med:

```bash
python src/train.py
```

Test the model:

```bash
python src/train.py --test
```

The evaluation includes commonly used medication recommendation metrics, including Jaccard, PRAUC, F1-score, and DDI Rate.

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{harpmed2026,
  title={HARP-Med: History-Aware Residual Personalized Medication Representation Learning for Drug Recommendation},
  author={},
  journal={},
  year={2026}
}
```

Citation information will be updated after publication.

## Acknowledgements

This implementation is developed based on previous open-source medication recommendation frameworks.

We thank the authors of these projects for making their code publicly available.

# HARP-Med: History-Aware Residual Personalized Medication Recommendation

HARP-Med is a history-aware personalized medication recommendation model.
It leverages patients' longitudinal EHR information to construct personalized
medication representations for medication combination prediction.

## Installation

1. Create a new conda environment:

```bash
conda create -n HARPmed python=3.8
2. Activate the environment:
conda activate HARPmed

3. Install the required packages:
pip install -r requirements.txt

Download Data
1. Download the MIMIC-III dataset from PhysioNet.
2. Place the required MIMIC-III files into the data/ directory.

3.The main files include:
DIAGNOSES_ICD.csv
PROCEDURES_ICD.csv
PRESCRIPTIONS.csv

4. Download the drug information and DDI-related files following the preprocessing settings of previous medication recommendation studies.

The directory can be organized as:

data/
├── DIAGNOSES_ICD.csv
├── PROCEDURES_ICD.csv
├── PRESCRIPTIONS.csv
└── ...
Data Processing

Run the preprocessing script:

python process.py

The processed data will be saved in the corresponding data directory and used
for model training and evaluation.

Run HARP-Med

The main implementation files are located in src/.

HARP-Med/
├── data/
├── src/
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md

Train HARP-Med:

python src/train.py

Test the model:

python src/train.py --test

The evaluation includes commonly used medication recommendation metrics,
including Jaccard, PRAUC, F1-score, and DDI Rate.

Citation

If you find this repository useful, please consider citing our work:

@article{harpmed2026,
  title={HARP-Med: History-Aware Residual Personalized Medication Representation Learning for Drug Recommendation},
  author={},
  journal={},
  year={2026}
}

Citation information will be updated after publication.

Acknowledgements

This implementation is developed based on previous open-source medication
recommendation frameworks.

We thank the authors of these projects for making their code publicly available.

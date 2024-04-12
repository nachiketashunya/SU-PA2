# SU-PA2-Q1

## Speech Assignment Question 1 Repository

Welcome to the repository for Question 1 of the Speech Assignment!

### Instructions for Performing Tasks

Follow these steps to perform various tasks:

Selected Models = Wavlm Base+, Unispeech-sat-base, Ecapa-TDNN

1. **Compute Equal Error Rate (EER) on Voxceleb1-H Dataset:**
   - Run the `Speech-PA2/evaluation/vox_eval.py`

2. **Compute Equal Error Rate (EER) on KathBath Dataset:**
   - Execute the `Speech-PA2/evaluation/kb_eval.py` script.

3. **Fine Tune on KathBath Dataset:**
   - Run `SU-PA2-Q2/evaluation/evaluate.py`.

    *Evaluation Metrics:*
    - Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Signal Distortion Ratio (SDR) are computed for each separated source in the mixture.
    - These metrics quantify the quality of the separated sources compared to the ground truth.

    *Logging and Reporting:*
    - WandB (Weights & Biases) is used for experiment tracking.
    - Intermediate results (SNR and SDR for each source) are logged during evaluation.
    - Final average SNR and SDR values are calculated and reported.

Speech-PA2
==============================

This is an assignment of Speech Understanding

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── docs                                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── Speech-PA2           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                                <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Scripts to train models
    │   │
    │   ├── validate                            <- Scripts to validate models
    │   │
    │   └── visualization                       <- Scripts to create exploratory and results oriented visualizations

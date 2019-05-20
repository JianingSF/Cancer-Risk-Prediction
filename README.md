# Cancer risk prediction

By leveraging the combination of clinical, laboratory, and sequencing data, create algorithms / classifiers that help diagnose disease and better understand risk, or aggressiveness of a patient’s cancer. 
Difination of disease-associated risk: The clinical determination of risk is carried out through the combination of cancer stage, grade, overall survival in days following diagnosis, and vital status (alive/dead). These are presented for each patient ID in the patient_data.tsv file.

Inputs:
* Clinical data: patient_data.csv
* Tumor Mutation Sequencing data: seq_data.csv
* Tumor mRNA gene expression data: mrna_data.csv

Outputs:
* Predict cancer stage
* Predict grade
* Predict vital status
* Predict survival in days

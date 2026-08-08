# Ethics and Data Statement

This repository contains data gathered through a multimodal transportation survey conducted among students. 

## Data Anonymization
All publicly released datasets (`data/survey_results/`) have been rigorously anonymized to protect the privacy of the participants, in compliance with standard ethical guidelines for human-subject research:
- **Direct Identifiers**: All respondent names and direct identifiers were stripped prior to publication. Respondents are identified strictly via a stable pseudonym (`STU_XXXX`).
- **Demographic Masking**: Exact ages have been binned into 5-year intervals. Distances and financial budgets have been discretized or rounded to prevent re-identification through exact geographical matching.
- **Categorical Generalization**: Fields such as the exact campus location and specific student status have been generalized to generic labels (e.g. `University`, `Student`) as they are not required to reproduce the numerical results.

## Informed Consent
Participants were informed of the purpose of the study (modeling multimodal transportation preferences) and gave explicit consent for their anonymized responses to be used for scientific research and published as part of the reproducibility package for the corresponding manuscript.

## Reuse
The dataset is intended solely for reproducing the experiments described in the manuscript and for further non-commercial academic research in transportation engineering and multi-objective optimization.

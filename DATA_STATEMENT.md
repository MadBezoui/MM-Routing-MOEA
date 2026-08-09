# Ethics and Data Statement

This repository contains data gathered through a multimodal transportation survey conducted among students. 

## Data Anonymization
All publicly released datasets (`data/survey_results/`) have been rigorously anonymized to protect the privacy of the participants, in compliance with standard ethical guidelines for human-subject research:
- **Direct Identifiers**: All respondent names and direct identifiers were stripped prior to publication. Respondents are identified strictly via a stable pseudonym (`STU_XXXX`).
- **Demographic Masking**: Exact ages have been entirely removed and binned into 5-year intervals (`age_group`). Distances and financial budgets have been discretized or rounded to prevent re-identification through exact geographical matching.
- **Categorical Generalization**: Fields such as the exact campus location and specific student status have been generalized to generic labels (e.g. `University`, `Student`) as they are not required to reproduce the numerical results.

## Informed Consent and Ethical Review
Participants were informed of the purpose of the study (modeling multimodal transportation preferences) and gave explicit consent for their anonymized responses to be used for scientific research and published as part of the reproducibility package for the corresponding manuscript.
The study design, consent procedure, and the release of this pseudonymized dataset have been reviewed and approved by the institutional Data Protection Officer (DPO) and the relevant ethical committee.

## Licensing and Usage Constraints

This repository employs a dual-licensing structure to respect the origin of its components:

- **Source Code**: MIT License (see `LICENSE`), permitting open reuse and modification.
- **OpenStreetMap Data**: Open Data Commons Open Database License (ODbL) v1.0. Any derived network artifacts must carry the same ODbL terms.
- **GTFS Timetables**: Subject to the original open data license of the transit authority (Compagnie des Transports Strasbourgeois).
- **Survey Data (`data/survey_results/`)**: Released exclusively for **academic, non-commercial research** regarding multimodal routing and user preferences. The MIT license of the software codebase does *not* grant commercial usage rights to this human-subjects dataset.

# Dataset notes

The notebook expects labelled images under directories named `all` and `hem`, where `all` contains ALL-positive images and `hem` contains HEM/normal images. It searches recursively under the configured Kaggle input root.

Before publishing results, record:
- Original dataset name, authors, version, and source URL
- Dataset licence and redistribution restrictions
- Inclusion/exclusion rules and any image preprocessing
- How patient identifiers are parsed from filenames
- Patient-level split assignment and counts
- Whether near-duplicates or multiple images per slide/patient exist

Do not commit dataset images or private/identifiable health data. The Kaggle notebook includes a fallback grouping rule, but inspect and validate the patient-ID parser against actual filenames before relying on leakage checks. If identifiers cannot be parsed reliably, stop and define a verified mapping rather than silently using image-level splits.

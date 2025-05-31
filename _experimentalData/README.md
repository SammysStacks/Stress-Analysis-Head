<h1 align="center">
  📁 Experimental Data Directory
</h1>

## 🧠 Overview

This directory contains structured subfolders used for organizing experimental datasets utilized in training and evaluating emotion recognition models. All data has been formatted for compatibility with the pipeline used in `Stress-Analysis-Head`, enabling streamlined access and processing.

Please note that no identifiable participant data is shared here. To reproduce or extend experiments, users may need to acquire access to raw data directly from the original dataset providers (see links in the top-level [README](../README.md)).

---

## 📦 Subfolder Structure

### `./_compiledData/`

A compressed and preprocessed version of all datasets, designed for direct use in training and evaluation. These files are generated via the model compilation pipeline and stored in `.pkl.gz` format to:
- Reduce file size for easy sharing
- Package aligned sensor, label, and metadata information

---

### `./_empatchDataset/`

Contains proprietary experimental data collected through the EMPATCH platform:
- Emotion recognition training data
- Behavioral therapy session data

---

### `./_metaDatasets/`

Includes dataset-specific files. These include:
- AMIGOS, DAPPER, EMOGNITION, CASE, and WESAD datasets

Note: Most original raw data is not included due to licensing and privacy requirements. You must contact the respective dataset authors to obtain full access and agree to their terms (e.g., EULA forms for AMIGOS and EMOGNITION).

---

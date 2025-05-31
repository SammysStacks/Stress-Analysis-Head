<h1 align="center">
  📁 EMPATCH Experimental Data Directory
</h1>

## 🧠 Overview

This directory contains data collected from the EMPATCH platform — a wearable-based system developed to support emotion recognition, state anxiety modeling, and adaptive therapy interventions. The dataset includes multimodal physiological signals, user-reported affective labels, and session-specific metadata from both controlled training protocols and real-world behavioral therapy deployments.

All files have been de-identified and formatted to be directly compatible with the `Stress-Analysis-Head` modeling pipeline. Data was collected under IRB-approved protocols, and its use is subject to institutional data-sharing agreements.

---

## 📦 Subfolder Structure

Each session-specific folder within this directory contains:

### 🗂 `*.xlsx` Files
Raw physiological data exported from the EMPATCH wearable system. Each `.xlsx` file includes synchronized multichannel time-series data for various biosignals (EOG, EEG, EDA, Temp) recorded during a single session.

- File naming convention typically includes subject/session identifiers
- Sheets include both sensor data and auxiliary metadata

### 📁 `saveFeatures/`
This subdirectory contains preprocessed and feature-extracted versions of the corresponding raw session data.

---

To regenerate these features from raw `.xlsx` files, use the [mainControl.py](https://github.com/SammysStacks/Stress-Analysis-Head/blob/main/mainControl.py) script located in the main repository, which parses the input files, extracts features, and saves them in the appropriate format.
This file interfaces with [trainingProtocols.py](https://github.com/SammysStacks/Stress-Analysis-Head/blob/main/helperFiles/machineLearning/trainingProtocols.py) to ensure consistent feature extraction across all sessions.
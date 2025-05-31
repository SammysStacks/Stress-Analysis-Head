<h1 align="center">
  Wearable Affective General Intelligence
</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/SammysStacks/Stress-Analysis-Head/commits/main"><img src="https://img.shields.io/github/last-commit/SammysStacks/Stress-Analysis-Head.svg" alt="Last Commit"></a>
  <a href="https://github.com/SammysStacks/Stress-Analysis-Head/issues"><img src="https://img.shields.io/github/issues/SammysStacks/Stress-Analysis-Head.svg" alt="GitHub Issues"></a>
</p>

---

## 🧠 Overview

**Wearable Affective General Intelligence** is a research platform for modeling affective states using physiological data collected from diverse wearable sensors. Built around a Lie manifold neural architecture, it enables real-time and offline emotion recognition, missing data reconstruction, and cross-dataset generalization.

The system was trained and validated across five public biosensing datasets (e.g., WESAD, AMIGOS, EMOGNITION), integrating signals from 15 platforms across five body locations. By embedding these inputs into a shared low-dimensional affective space, the model supports generalizable emotion decoding, even under signal dropout or novel conditions.

---

## ✨ Contents

- [Overview](#🧠-overview)
- [System Requirements](#💻-system-requirements)
- [Installation Guide](#🔧-installation-guide)
- [Demo](#🎥-demo)
- [Referenced Datasets](#📂-referenced-datasets)
- [License](#📜-license)

---

## 💻 System Requirements

- Python 3.8+
- 4GB+ RAM (recommended 8GB+ for meta-training)
- GPU optional (helpful for deep learning extensions)
- Compatible wearable device for streaming (e.g., Empatica E4)

All dependencies are listed in `requirements.txt`. Tested on Windows and macOS.

---

## 🔧 Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/SammysStacks/Stress-Analysis-Head.git

# 2. Move into the project directory
cd Stress-Analysis-Head

# 3. (Optional) Create a virtual environment
python -m venv env
source env/bin/activate  # or "env\\Scripts\\activate" on Windows

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🎥 Demo

### `mainControl.py`

Control script for data preprocessing, single-session analysis, model training, and real-time inference.

Set only one of the following flags to `True`:

```python
readDataFromExcel = False  # Analyze a specific .xlsx session file
trainModel = True          # Train on all files in collectedDataFolder
streamData = False         # Stream from connected wearable sensor
```

- For single-session evaluation, specify:
<pre><code>currentFilename = "yourfile.xlsx"
testSheetNum = 0
</code></pre>

### `metaTrainingControl.py`

Script for cross-dataset meta-learning and affective generalization. Automatically trains on multiple datasets and evaluates generalization. Outputs model accuracy and affective reconstruction metrics.

<pre><code>python metaTrainingControl.py
</code></pre>

This reproduces results presented in our manuscript.

---

## 📂 Referenced Datasets

> Please obtain and cite original datasets. Some require signing a license or EULA.

- **WESAD**  
  [Paper](https://dl.acm.org/doi/10.1145/3242969.3242985) • [Dataset](https://ubicomp.eti.uni-siegen.de/home/datasets)

- **AMIGOS**  
  [Paper](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/Paper_TAC.pdf) • [Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/index.html)

- **EMOGNITION**  
  [Paper](https://doi.org/10.1038/s41597-022-01262-0) • [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/R9WAF4)

- **DAPPER**  
  [Paper](https://doi.org/10.1038/s41597-021-00945-4) • [Dataset](https://www.synapse.org/Synapse:syn22418021/wiki/605529)

- **CASE**  
  [Paper](https://doi.org/10.1038/s41597-019-0209-0) • [Dataset](https://springernature.figshare.com/articles/dataset/CASE_Dataset-full/8869157)

---


## 📜 License

This repository is licensed under the MIT License. For research use only. Please cite our work if used in academic publications.

> © 2025 Samuel Aaron Solomon. All rights reserved.

---

Enjoy building emotionally intelligent systems! 🧠✨

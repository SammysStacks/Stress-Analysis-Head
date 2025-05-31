<h1 align="center">
  Organized Data for Training and Evaluation
</h1>

## 📦 `compiledMetaTrainingInfo` Files

The `compiledMetaTrainingInfo_{datasetName}_fullAnalysisParams.pkl.gz` files contain de-identified, preprocessed metadata from their respective datasets. These files enable efficient downstream training, evaluation, and visualization without requiring manual organization of the original raw data.

While all metadata has been de-identified to preserve participant privacy, please **cite the original dataset papers** if you use this data in your research. In particular, for the AMIGOS and EMOGNITION datasets, please complete the required EULA agreements before use:

- **EMOGNITION**: [Paper](https://doi.org/10.1038/s41597-022-01262-0) | [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/R9WAF4)
- **DAPPER**: [Paper](https://doi.org/10.1038/s41597-021-00945-4) | [Dataset](https://www.synapse.org/Synapse:syn22418021/wiki/605529)
- **AMIGOS**: [Paper](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/Paper_TAC.pdf) | [Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/index.html)
- **WESAD**: [Paper](https://dl.acm.org/doi/10.1145/3242969.3242985) | [Dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
- **CASE**: [Paper](https://doi.org/10.1038/s41597-019-0209-0) | [Dataset](https://springernature.figshare.com/articles/dataset/CASE_Dataset-full/8869157?file=16260497)

---

### 📁 File Structure: `compiledMetaTrainingInfo_{datasetName}_fullAnalysisParams.pkl.gz`

Each file is a serialized dictionary with one key per dataset:

```python
{
  'compiledMetaTrainingInfo_{datasetName}': [
    # Raw Feature Data
    'allRawFeatureTimesHolders': (batchSize, numBiomarkers, numTimePoints),
    'allRawFeatureHolders': (batchSize, numBiomarkers, numTimePoints, numBiomarkerFeatures),
    'allRawFeatureIntervalTimes': (batchSize, numBiomarkers, finalDistributionLength),
    'allRawFeatureIntervals': (batchSize, numBiomarkers, finalDistributionLength, numBiomarkerFeatures),

    # Compiled Feature Data
    'allCompiledFeatureIntervalTimes': List[np.ndarray], shape = (finalDistributionLength,),
    'allCompiledFeatureIntervals': List[np.ndarray], shape = (finalDistributionLength, totalFeatures),
    # where totalFeatures = numBiomarkers × numBiomarkerFeatures

    # Metadata
    'subjectOrder': List[str], length = batchSize,
    'experimentalOrder': List[str], length = batchSize,
    'allFinalLabels': List[np.ndarray], shape = (labelDim,),
    'featureLabelTypes': List[str],
    'surveyQuestions': List[str],
    'surveyAnswersList': List[List[str]], shape = (numQuestions, numOptions),
    'surveyAnswerTimes': List[np.ndarray],
    'activityNames': List[str],
    'activityLabels': List[np.ndarray], shape = (finalDistributionLength,),
    'featureNames': List[str], length = totalFeatures,
    'numQuestionOptions': int
  ]
}
```

### 🧮 Dimensional Definitions

```python
- batchSize – Number of samples (typically subject-experiment pairs) in the dataset.
- numBiomarkers – Number of distinct biosignals (e.g., EDA, ACC, TEMP, etc.).
- numTimePoints – Number of raw time points for each biomarker trace before resampling.
- numBiomarkerFeatures – Number of extracted features per biomarker signal (e.g., mean, slope, std). 
- finalDistributionLength – Standardized sequence length after time normalization/resampling. 
- totalFeatures – Flattened feature vector size per time point, defined as sum(numBiomarkerFeatures) for all biomarkers. 
- numQuestionOptions – Number of multiple-choice answer options for each survey question.
```

### 📁 File Structure: `compiledMetaTrainingInfo_{datasetName}.pkl.gzl`

This version contains a compressed subset of the full file, which is used when training the model, including:

```python
[
  allRawFeatureIntervalTimes,
  allRawFeatureIntervals,
  subjectOrder,
  featureNames,
  surveyQuestions,
  surveyAnswerTimes,
  surveyAnswersList,
  activityNames,
  activityLabels,
  numQuestionOptions
]
```

<hr>

Here the provide code that loads in the files: [compileModelDataHelpers.py](https://github.com/SammysStacks/Stress-Analysis-Head/blob/main/helperFiles/machineLearning/dataInterface/compileModelDataHelpers.py#L52).

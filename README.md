# Diabetes Prediction Using Machine Learning

**Course:** CSE570 – Machine Learning with Python  
**Programme:** M.Tech – Computer Science and Engineering  
**Specialization:** Data Science and Analytics  
**Project Type:** Academic Machine Learning Project  
**Language:** Python  
**Platform:** Google Colab

---

## 1. About the Project

As part of my first-semester M.Tech coursework, I selected diabetes prediction as a practical problem to understand how machine-learning algorithms can be applied to health-related tabular data.

I worked with the Diabetes Binary Health Indicators dataset, which contains information related to individuals' health conditions, lifestyle habits and demographic characteristics. My main focus was to explore whether these indicators contain useful patterns for distinguishing individuals with diabetes from those without diabetes.

Rather than directly applying classification algorithms, I first examined the dataset, checked its quality, studied the distributions and relationships between variables, and then developed different models to compare their performance.

This project helped me understand the complete process of a machine-learning task, from preparing raw data to interpreting model evaluation results.

## 2. Problem Formulation

**Main question:**

Can demographic, lifestyle and health-related indicators be used by machine-learning classification algorithms to identify patterns associated with diabetes status?

I explored this question through the following sub-questions:

- Does BMI show a difference between the diabetes and non-diabetes groups?
- How are age, general health, physical health and other indicators associated with diabetes status?
- Which features contribute more to the Decision Tree model's predictions?
- How do Logistic Regression, Decision Tree and KNN compare when the target variable is imbalanced?
- Why can a model with high accuracy still perform poorly in identifying diabetes cases?

The target variable used was `Diabetes_binary`, where `0` represents no diabetes and `1` represents diabetes.

## 3. Dataset Information

**Dataset name:** Diabetes Binary Health Indicators – BRFSS 2015

The dataset is based on health-related survey information from the CDC's Behavioral Risk Factor Surveillance System (BRFSS).

| Description | Details |
|---|---|
| Original records | 253,680 |
| Predictor variables | 21 |
| Target variable | Diabetes_binary |
| Total columns | 22 |
| Records after duplicate removal | 229,474 |
| Problem type | Binary classification |

The predictor variables include BMI, age, high blood pressure, high cholesterol, physical activity, smoking, general health, mental health, income and education.

**Dataset references:**

- [UCI Machine Learning Repository – CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [CDC – 2015 BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html)

## 4. Data Preparation

I started by examining the dataset's dimensions, column names, data types and basic statistical information.

The following checks and operations were performed:

- Checked for missing values.
- Identified exact duplicate records and removed them.
- Examined the distribution of the BMI variable.
- Used boxplots and histograms to inspect possible outliers.
- Checked the target variable distribution.
- Separated predictor variables and the target variable.
- Applied feature scaling for models that require it.

One important observation was that the target classes were imbalanced. The number of non-diabetes records was considerably higher than the number of diabetes records. This became an important consideration during model evaluation.

## 5. Exploratory Data Analysis

I used visualizations to understand the data before training the models.

Some of the main analysis performed included:

1. **Target distribution:** Compared the number of records in the diabetes and non-diabetes groups.
2. **BMI distribution:** Examined BMI values using histograms and boxplots.
3. **Group comparison:** Compared BMI across the two diabetes-status groups.
4. **Correlation analysis:** Used a correlation heatmap to examine relationships among the health indicators.
5. **Feature relationships:** Examined selected demographic and health indicators in relation to diabetes status.

I also used statistical tests, including chi-square and independent-samples t-tests, to examine selected relationships in the dataset.

## 6. Machine Learning Implementation

I implemented and compared four models as part of the coursework.

| Model | Purpose |
|---|---|
| Linear Regression | Additional comparison experiment; continuous predictions were converted into classes using a 0.5 threshold |
| Logistic Regression | Binary classification |
| Decision Tree Classifier | Classification using decision rules |
| K-Nearest Neighbors (KNN) | Classification based on nearby observations |

A majority-class baseline was also included to understand how much improvement the models provided over simply predicting the most frequent class.

### Training and Testing

- Used an 80:20 training and testing split.
- Applied stratification to maintain the class proportions.
- Used feature scaling for Logistic Regression and KNN.
- Used an unscaled feature set for the Decision Tree.
- Tested different K values for KNN using F1-score.
- Compared Decision Tree depth settings using validation performance.

## 7. Model Evaluation

I evaluated the models using accuracy, precision, recall, F1-score, confusion matrices and ROC-AUC.

The initial model comparison recorded in my notebook was:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Linear Regression (thresholded) | 84.91% | 59.92% | 4.09% | 7.66% |
| Logistic Regression | 85.05% | 53.98% | 15.37% | 23.93% |
| Decision Tree | 77.75% | 29.27% | 32.11% | 30.63% |
| KNN (K=5) | 83.24% | 40.76% | 21.07% | 27.78% |

### ROC-AUC Comparison

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.8102 |
| KNN | 0.6696 |
| Decision Tree | 0.5904 |

### What I observed

Logistic Regression produced the highest accuracy and ROC-AUC among the three classification models in this comparison.

However, its recall was relatively low. This means it failed to identify a considerable proportion of actual diabetes cases.

The Decision Tree achieved the highest recall and F1-score in this initial comparison. This showed why it is important to consider more than accuracy when working with an imbalanced dataset.

The majority-class baseline itself achieved approximately 84.71% accuracy. Therefore, a high accuracy value alone does not necessarily indicate that a model is useful for identifying the minority class.

*Note: The KNN values above refer to the initial K=5 model. The notebook also contains a separate K-value tuning experiment.*

## 8. Decision Tree Feature Importance

I also examined the feature importance values obtained from the Decision Tree model.

The most influential features in that model included:

- BMI
- Age
- Income
- Physical health
- High blood pressure
- Education
- Mental health
- General health

BMI and age received the highest importance values in the displayed feature-importance output.

These values indicate how the Decision Tree used the features when making predictions. They should not be interpreted as proof that a feature causes diabetes.

## 9. Learning Outcomes

Through this implementation, I gained practical experience in:

- Working with a real-world tabular health dataset.
- Performing data cleaning and exploratory data analysis.
- Understanding class imbalance and its effect on model performance.
- Applying classification algorithms using Scikit-learn.
- Understanding the difference between regression and classification.
- Evaluating models using confusion matrices, precision, recall, F1-score and ROC-AUC.
- Interpreting feature importance and model limitations.

One of my main takeaways was that selecting a model based only on accuracy can lead to misleading conclusions, particularly when the target classes are imbalanced.

## 10. Limitations and Scope for Improvement

This implementation has some limitations:

- The dataset is based on self-reported survey information.
- The target classes are imbalanced.
- The models may behave differently on data collected from other populations.
- The current model results do not establish clinical reliability.
- The classification threshold can affect precision and recall.

Further work could explore class-balancing techniques, threshold selection, additional validation strategies and more detailed model interpretation.

## 11. Conclusion

This project gave me an opportunity to apply the machine-learning concepts covered in my first semester to a practical classification problem.

The analysis showed that Logistic Regression performed best in terms of accuracy and ROC-AUC in the initial comparison, whereas Decision Tree provided better recall and F1-score.

More importantly, I understood that model evaluation should depend on the actual problem being solved. In a diabetes-related classification task, identifying positive cases is important, so recall and F1-score deserve attention alongside accuracy.

This work represents my initial practical exploration of machine learning for health-indicator data as an M.Tech student.

---

## 12. Repository Structure

```text
Diabetes_Prediction_Machine_Learning/
│
├── Diabetes_Prediction_Using_Machine_Learning.ipynb
├── requirements.txt
├── README.md
│
└── visualizations/
    ├── diabetes_distribution.png
    ├── bmi_distribution.png
    ├── bmi_boxplot.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curve_comparison.png
    └── decision_tree_feature_importance.png
```

## 13. Running the Notebook

1. Clone or download this repository.
2. Install the libraries listed in `requirements.txt`.
3. Download the dataset and upload it to Google Colab.
4. Update the dataset path if necessary.
5. Run the notebook cells in sequence.


**GitHub:** https://github.com/karthikloki666/Diabetes_Prediction_Machine_Learning


<p align="center"><img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/prediction_rob.png" width="600" height="400"></p>


   # Stroke Prediction Model

## Comprehensive Stroke Risk Analysis with Machine Learning

### Overview
This project develops a machine learning model to predict stroke occurrences using health-related data. The process involves data preprocessing, model building, hyperparameter tuning, and evaluation.



## Table of Contents
- [Data Loading](#data-loading)
- [Data Analysis](#data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Final Predictions and Submission](#final-predictions-and-submission)
- [Conclusion](#conclusion)

<a name="data-loading"></a>
## Data Loading
The dataset for stroke prediction is loaded using Pandas, a powerful data manipulation library in Python.


```python
import pandas as pd
train_df = pd.read_csv('/path/to/stroke_train_set.csv')
test_df = pd.read_csv('/path/to/stroke_test_set_nogt.csv')
```

This step involves importing the necessary libraries and reading the training and testing datasets using Pandas.

<a name="data-analysis"></a>


 Utilizing a rich dataset spanning various demographics, health indicators, and lifestyle choices, we endeavor to uncover patterns and correlations that may lead to a more profound understanding of stroke risks. Our goal is to leverage machine learning models to predict the likelihood of stroke events accurately.

## Data Insight and Exploration

The journey begins with an exploratory data analysis (EDA), which serves as the cornerstone for our predictive modeling. By scrutinizing the dataset, we extract meaningful insights and set the stage for advanced analytical techniques.

### Demographic Dynamics: Age and Its Implications

Our dataset encapsulates a broad spectrum of ages, capturing a demographic mosaic from toddlers to the elderly. The distribution of age is particularly significant, as it directly influences the stroke risk profile.


<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/Distribution%20of%20age.png" width="400" height="300">

*Insight:* The age distribution curve suggests a bimodal trend, hinting at varied stroke risk across different life stages.

### Metabolic Markers: The Tale of Glucose

Glucose levels are a pivotal metabolic marker, with their distribution shedding light on the metabolic health landscape of our dataset.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/Distribution%20of%20average%20glucose%20levels.png" width="400" height="300">

*Insight:* A right-skewed glucose level distribution signals that while elevated glucose levels are less prevalent, their impact on stroke risk might be disproportionate, warranting a deeper dive into their role.

### The Weight of Health: BMI's Story

Body Mass Index (BMI) serves as a proxy for assessing the weight-related health status of individuals, with its distribution offering clues to the population's overall health.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/Distribution%20of%20BMI.png" width="400" height="300">

*Insight:* The right skewness in BMI distribution echoes a concerning trend towards overweight and obesity, known risk factors for stroke.

### The Silent Afflictions: Heart Disease and Hypertension

Heart disease and hypertension are often silent yet significant stroke risk factors. Their distribution in our dataset is critical for understanding their prevalence and impact.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/distribution%20of%20heart%20disease.png" width="400" height="300"><img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/Distribution%20of%20hypertension.png" width="400" height="300">

*Insight:* Although less common, the presence of heart disease and hypertension could be instrumental in predicting stroke occurrences.

### Stroke Occurrences: A Glimpse into the Data's Heart

The stroke occurrence distribution offers an unvarnished look at the dataset's balance and the stark contrast between stroke and non-stroke instances.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/Distribution%20of%20stroke%20cases.png" width="400" height="300">

*Insight:* The dataset presents a clear imbalance with a smaller proportion of stroke cases, challenging our model to learn from limited positive instances.

### Integrated Analysis: Interweaving Age and Glucose

A multidimensional analysis combining age and average glucose levels elucidates the potential interplay between these factors and stroke risk.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/combined.png" width="500" height="500">

*Insight:* The scatter plot unveils a potential clustering of stroke cases among older individuals with higher glucose levels, suggesting a compound risk effect.

## Statistical Summary: The Numerical Narrative

A quantitative summary breathes life into the raw numbers, painting a picture of the underlying data characteristics.

| Feature                | Range           | Mean  |
|------------------------|-----------------|-------|
| Age                    | 0.08 - 82 years | ~43   |
| Average Glucose Level  | 55.12 - 267.76  | ~106  |
| BMI                    | 10.3 - 97.6     | ~28.9 |

*Interpretation:*
- The age range confirms the dataset's inclusivity, highlighting the universal nature of stroke risk.
- The glucose level extends into hyperglycemic territory, emphasizing the need to address high glucose levels as a stroke risk factor.
- The BMI range spans from underweight to severe obesity, underscoring the diverse body weight profiles within the dataset.

## Conclusive Insights and Future Directions

Our exploratory journey reveals:
- A diverse age range necessitates age-specific stroke risk stratification.
- Glucose levels and BMI distributions signify the metabolic health's role in stroke risk.
- The subtle yet profound presence of heart disease and hypertension accentuates their contribution to stroke risk.

With these insights, we pivot to predictive modeling, harnessing machine learning algorithms to forecast stroke likelihood. Our analytical odyssey continues with data preprocessing, model training, and evaluation, aiming to distill a reliable prediction tool from the complex tapestry of stroke-related data.

## Repository Contents

- `data/` - The datasets used for analysis and modeling.
- `notebooks/` - Jupyter notebooks detailing

 the analysis process.
- `scripts/` - Python scripts for preprocessing, model training, and evaluation.
- `models/` - Serialized versions of the trained machine learning models.
- `visuals/` - Generated visualizations from the EDA.


<a name="Feature _Engineering"></a>
## Feature Engeneering 

To gain deeper insights and uncover relationships within this data, we can use more sophisticated visualization techniques. Below are those we are gonna dive into:

### Correlation Heatmap: 
This will help in visualizing the correlation between numerical features like age, avg_glucose_level, and bmi. It's useful for identifying features that might be strongly correlated with the target variable (stroke).

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/correlation%20heat%20map.png" width="600" height="400">

### Insights: 
This heatmap provides a visual representation of how closely related different numerical features are to each other. Strong correlations (either positive or negative) between two features suggest a significant relationship.
### Feature Engineering Implications:
Redundancy Check: Highly correlated features can introduce redundancy. For instance, if two features are highly correlated,  might consider dropping one to reduce overfitting.
### Feature Interaction: Moderate correlations can be a cue to create interaction features, potentially capturing more complex relationships (e.g., a product or ratio of two moderately correlated features).

### Pair Plot: 
This is a great way to see both distribution of single variables and relationships between two variables. Pair plots can help identify trends and patterns that might be useful for classification.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/PairPlot.png" width="600" height="400">

### Insights: 
Shows both the distribution of single variables and the pairwise relationships between them. It's particularly useful for identifying linear or non-linear relationships and potential outliers.
### Feature Engineering Implications:
Non-linear Transformations: 
If a non-linear relationship is observed, applying transformations (like squaring, log, or square root) to these features might better capture their relationship with the target.
### Outlier Treatment: 
Identification of outliers can lead to either their removal or the creation of new features indicating these outliers.

### Box Plot for Categorical Data: 
This can be used to see the distribution of numerical data across different categories like gender, work_type, and smoking_status.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/Boxplot%20for%20categorical%20data.png" width="600" height="400">

### Insights: 
Reveals the distribution of BMI across different work types, showing median, quartiles, and outliers.
### Feature Engineering Implications:
Group-specific Features: Different work types might have unique characteristics affecting heart health. Creating group-specific statistics (like mean, median, or custom aggregation of BMI within each work type) could enhance model performance.
### One-hot Encoding vs. Target Encoding: 
For categorical variables like work type, this plot can help decide whether to use one-hot encoding or more sophisticated methods like target encoding.

### Violin Plot: 
Similar to box plots, but also shows the probability density of the data at different values. This is useful for comparing the distribution of numerical variables across different categories.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/violin%20plot.png" width="600" height="400">

### Insights:
Shows the distribution of average glucose levels for different smoking statuses, providing both the density estimation and the box plot.
### Feature Engineering Implications:
### Custom Grouping: 
If certain categories show similar patterns, they can be grouped together, reducing the number of categories and potentially improving model robustness.
### Interaction with Numerical Features: 
The relationship between categorical features like smoking status and numerical features like glucose levels can inform creating interaction terms.

### Scatter Plot with Hue for Categorical Data: 
This plot can be used to visualize relationships between two numerical variables while also segmenting points by a categorical feature (showing avg_glucose_level vs. bmi, segmented by stroke).

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/scatterplot.png" width="600" height="400">

### Insights: 
Demonstrates how two numerical features relate to each other, with an additional categorical dimension (stroke occurrence).
Feature Engineering Implications:
### Segmentation: 
Age and BMI could be binned into categorical variables (like age groups or BMI ranges) to capture non-linear effects better.
### Polynomial Features:
If a complex relationship is observed, polynomial features (like age^2, BMI^2, age*BMI) might capture these nuances more effectively.

### Facet Grid: 
Facet grid can create a grid of plots based on a categorical feature. This is useful for comparing distributions across different categories.

<img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/facetgrid.png" width="600" height="400">

### Insights: 
Allows comparison of a single distribution across different sub-categories.
### Feature Engineering Implications:
### Category-specific Distributions: 
Understanding how distributions vary across categories can lead to creating features that are normalized or standardized within each group.
### Custom Labels for Categories: 
Based on the distributions, custom thresholds for binning continuous variables within each category can be established.

<a name="Comprehensive-Feature-Engineering-and-Model-Optimization-Plan-for-Stroke-Prediction-Analysis"></a>

## 1. Preprocessing Steps:
### Numerical Transformer Pipeline:
### SimpleImputer (median):
Imputes missing values in numerical features using the median, which is robust to outliers.
### StandardScaler: 
Scales features to have a mean of 0 and a standard deviation of 1, which is important for models sensitive to feature magnitude (like SVMs).
### PolynomialFeatures: 
Generates polynomial and interaction features (degree=2), allowing the model to capture non-linear relationships.
### Categorical Transformer Pipeline:
### SimpleImputer (most frequent): 
Imputes missing values in categorical features using the most frequent value.
### OneHotEncoder: 
Encodes categorical variables into a one-hot numeric array, making them suitable for machine learning algorithms.

## 2. Column Transformer:
Combines numerical and categorical transformers, applying appropriate preprocessing to each type of feature.

## 3. Model Definitions and Pipelines:
### Machine Learning Models: 
Use of a diverse set of models (Random Forest, Gradient Boosting, Extra Trees, Logistic Regression, SVC, XGBoost) to capture different aspects of the data.
### bImbalanced Learning with SMOTE: 
Addresses class imbalance by oversampling the minority class in the training data, which is crucial for datasets with skewed class distributions.

## 4. Hyperparameter Tuning (Grid Search):
GridSearchCV: Systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance.

## 5. Advanced Ensemble Techniques:
Voting Classifier: Combines predictions from multiple models (soft voting) to improve accuracy and robustness.
Stacking Classifier: Stacks the output of individual models and uses a Logistic Regression as the final estimator to predict the target variable.

## 6. Feature Selection:
Although not explicitly stated in the code, the use of SelectFromModel in the context suggests an intention for feature selection based on model importance, which helps in reducing dimensionality and potentially improving model performance.

## 7. Model Evaluation:
Evaluation metrics like F1 Score and classification report provide a comprehensive understanding of model performance, especially precision, recall, and F1 score for each class, which are crucial in the context of imbalanced datasets.

## Implications for Feature Engineering:
Polynomial Features: Capturing complex, non-linear relationships that might not be apparent with linear models.
One-Hot Encoding for Categorical Variables: Ensures that categorical variables are properly incorporated into the model.
SMOTE for Imbalanced Datasets: Enhances the model’s ability to identify the minority class, which is often the class of interest in medical datasets like stroke prediction.




<a name="data-preprocessing"></a>
## Data Preprocessing
The data preprocessing step includes splitting the data into features (`X`) and target (`y`), followed by identifying numerical and categorical columns for further processing.

```python
X = train_df.drop('stroke', axis=1)
y = train_df['stroke']
```

Here, we separate the feature set and the target variable from the training dataset. The feature set (`X`) includes all columns except 'stroke', and the target (`y`) is the 'stroke' column.

<a name="model-building"></a>
## Model Building
We define various machine learning models from scikit-learn and XGBoost libraries for stroke prediction.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
```

Each model serves a specific purpose and has its strengths in handling different types of data and patterns.

<a name="hyperparameter-tuning"></a>
## Hyperparameter Tuning
GridSearchCV is employed for hyperparameter tuning to find the best parameters for each model.

```python
from sklearn.model_selection import GridSearchCV
# Example for RandomForestClassifier
param_grid_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [10, 15, 20, None],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
}
```

This step involves defining a range of potential values for the model's parameters and using cross-validation to identify the most effective combination.


## Hyperparameter Tuning with GridSearchCV

Hyperparameter tuning is a critical step in the machine learning pipeline that can significantly enhance the performance of a model. It involves searching through a predefined range of hyperparameters to find the combination that produces the best results based on a chosen evaluation metric.

### What is GridSearchCV?

`GridSearchCV` is a powerful tool provided by the `scikit-learn` library, which automates the process of hyperparameter tuning. It systematically works through multiple combinations of parameter tunes, cross-validates each to determine which one gives the best performance.

### How does Hyperparameter Tuning Affect the Model?

Each machine learning algorithm comes with a set of hyperparameters that we can adjust to optimize its performance. These hyperparameters are not learned from the data; instead, they are set prior to the training process and remain constant during it. Adjusting these parameters impacts the learning process and the model's ability to generalize from the training data to unseen data.

### Implementing GridSearchCV for RandomForestClassifier

The `RandomForestClassifier` is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) of the individual trees. The following hyperparameters are commonly tuned:

- `n_estimators`: The number of trees in the forest.
- `max_depth`: The maximum depth of the trees.
- `min_samples_split`: The minimum number of samples required to split an internal node.

#### Example of GridSearchCV in Action

For our stroke prediction model, we use `GridSearchCV` to find the best hyperparameters for the `RandomForestClassifier`. Here's how we set it up:

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for RandomForestClassifier
param_grid_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [10, 15, 20, None],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
}

# Initialize the GridSearchCV object
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
```

In the above code, we define a grid of hyperparameters to search through, initialize `GridSearchCV` with our model pipeline, and specify the number of folds for cross-validation (`cv=5`) and the scoring metric (`scoring='f1'`). The `n_jobs=-1` parameter tells the program to use all available cores on the processor to speed up the search.

### The Outcome of Hyperparameter Tuning

By running `grid_search_rf.fit(X, y)`, we fit the GridSearchCV object to the data. After the search is complete, `grid_search_rf` will contain the best model, which can be accessed with `grid_search_rf.best_estimator_`. This model is expected to have the optimal balance between bias and variance, making it adept at making predictions on new, unseen data.

The hyperparameter tuning through `GridSearchCV` essentially fine-tunes the model to ensure that it not only learns well from the training data but also has a good generalization capability. This process plays a vital role in building a robust machine learning model capable of making accurate predictions.



<a name="model-evaluation"></a>
## Model Evaluation
The models are evaluated based on their F1 scores, precision, recall, and accuracy to determine their effectiveness in stroke prediction.

```python
from sklearn.metrics import f1_score, classification_report
y_pred = voting_clf.predict(X)
print("F1 Score:", f1_score(y, y_pred))
print(classification_report(y, y_pred))
```

This section shows how the final model is assessed using various metrics to gauge its performance on the training data.

<a name="final-predictions-and-submission"></a>
## Final Predictions and Submission
The trained model is used to make predictions on the test dataset, which are then prepared for submission.

```python
final_predictions = voting_clf.predict(test_df)
submission_df = pd.DataFrame({'ID': range(0, len(test_df)), 'stroke': final_predictions})
submission_df.to_csv('final_submission.csv', index=False)
```

The predictions from the test dataset are stored in a DataFrame and exported as a CSV file for submission.

<a name="conclusion"></a>
## Conclusion
This project demonstrates the application of various machine learning techniques in predicting stroke occurrences. Through rigorous preprocessing, model selection, and hyperparameter tuning, we achieved meaningful insights and predictions from the data.


After implementing and evaluating the stroke prediction model, we've derived key insights from its performance metrics. The table below summarizes the model's effectiveness based on the classification report:

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 0.83   | 0.91     | 3888    |
| 1     | 0.20      | 0.83   | 0.33     | 200     |

*Overall Accuracy: 0.83*

### Detailed Interpretation

- **Precision (Class 0: No Stroke)**: The model exhibits excellent precision for non-stroke predictions, with a value of 0.99. This implies that when it predicts a patient does not have a stroke, it is correct 99% of the time.

- **Recall (Class 0: No Stroke)**: The recall for non-stroke predictions stands at 0.83, indicating the model successfully identifies 83% of all true non-stroke instances.

- **F1 Score (Class 0: No Stroke)**: Combining precision and recall, the F1 score for non-stroke predictions is 0.91, reflecting a strong harmonic mean between the two metrics.

- **Precision (Class 1: Stroke)**: Precision for stroke predictions is significantly lower at 0.20, indicating a high false positive rate; the model incorrectly labels many non-stroke instances as strokes.

- **Recall (Class 1: Stroke)**: The recall for stroke cases is 0.83, which is quite high. It means the model is capable of capturing a majority of the true stroke instances.

- **F1 Score (Class 1: Stroke)**: The F1 score for stroke predictions is 0.33, which is relatively low due to the poor precision. This score is crucial because it balances the trade-off between precision and recall.

### Model Insights and Improvement Strategies

The current model is more reliable in predicting non-stroke cases than strokes. While it's good at identifying patients who have had a stroke (high recall), it also misclassifies many healthy patients as having a stroke (low precision). This could lead to unnecessary anxiety and medical costs for those patients, as well as potentially overburdening healthcare systems.

To further enhance the model's performance, particularly for stroke predictions (Class 1), we could explore the following strategies:

- **Data Collection**: More data, especially for stroke instances, could improve the learning process.

- **Feature Engineering**: Creating new features or transforming existing ones might provide new insights for the model.

- **Algorithm Tuning**: Further hyperparameter optimization might yield better results, particularly for handling imbalanced classes.

- **Advanced Algorithms**: Employing more sophisticated machine learning algorithms or deep learning approaches could enhance predictive power.

- **Cost-sensitive Training**: Given the imbalance and the high cost of false negatives (failing to predict a stroke), implementing cost-sensitive learning could be beneficial.

- **Ensemble Methods**: Stacking different models may lead to better generalization by combining the strengths of individual models.

In conclusion, while the model shows promise, it requires fine-tuning and potentially more sophisticated approaches to achieve a balanced performance for both classes. The ultimate goal is to increase the precision for stroke predictions without compromising the recall, ensuring reliable and actionable insights for medical interventions.

---

---






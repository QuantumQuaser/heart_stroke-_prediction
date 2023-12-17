
<p align="center"><img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/prediction_rob.png" width="600" height="400"></p>


   # Stroke Prediction Model

## Comprehensive Stroke Risk Analysis with Machine Learning

### Overview
This project develops a machine learning model to predict stroke occurrences using health-related data. The process involves data preprocessing, model building, hyperparameter tuning, and evaluation.



## Table of Contents

- [Data Loading](#data-loading)
- [Data Analysis](#data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Comprehensive Feature Engineering and Model Optimization Plan for Stroke Prediction Analysi](#Comprehensive-Feature-Engineering-and-Model-Optimization-Plan-for-Stroke-Prediction-Analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Final Predictions and Submission](#final-predictions-and-submission)
- [Conclusion](#conclusion)
- [Explore the Evolution of Stroke Prediction Models](#Eplore-The-Evolution-Of-Stroke-Prediction-Model)

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


<p align="center"><img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/data_exploration.png" width="600" height="400"></p>


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


  <p align="center"><img src="https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/visuals/feature%20engineering.png" width="600" height="400"></p>


<a name="Feature _Engineering"></a>
## Feature Engeneering 

To gain deeper insights and uncover relationships within this data, we can use more sophisticated visualization charts. Below are those we are gonna dive into:

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
### Implications:
Polynomial Features: Capturing complex, non-linear relationships that might not be apparent with linear models.
One-Hot Encoding for Categorical Variables: Ensures that categorical variables are properly incorporated into the model.
SMOTE for Imbalanced Datasets: Enhances the model’s ability to identify the minority class, which is often the class of interest in medical datasets like stroke prediction.

# Comprehensive-Feature-Engineering-and-Model-Optimization-Plan-for-Stroke-Prediction-Analysis:

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
## Model Performance Metrics: A Story in Numbers

#### Below is a comprehensive analysis of our model's performance and a contemplation of paths for its enhancement.



Imagine a garden where each plant represents a patient. Our model is the gardener, tasked with identifying which plants are thriving and which are at risk of withering (strokes). Here's how well our gardener performs:

| Aspect | Class 0 (Flourishing) | Class 1 (At Risk) |
|--------|-----------------------|-------------------|
| **Precision** | 1.00 (Perfect Eye) | 0.62 (Needs Glasses) |
| **Recall** | 0.97 (Vigilant) | 0.94 (Attentive) |
| **F1 Score** | 0.98 (Harmonious) | 0.75 (Balanced, but Improvable) |
| **Support** | 3888 (Abundant) | 200 (Scarce) |

**Overall Garden's Health (Accuracy):** 0.97

### Interpretation Through the Gardener's Lens

- **Precision (Class 0):** Our gardener rarely mistakes a flourishing plant for a withering one. Precision is perfect (1.00), but is there over-caution?

- **Recall (Class 0):** Out of every 100 flourishing plants, our gardener correctly identifies 97. Three might not receive the accolades they deserve.

- **F1 Score (Class 0):** Harmony is the theme here, with a score of 0.98 indicating a balanced approach to recognizing plant health.

- **Precision (Class 1):** In identifying at-risk plants, our gardener is somewhat myopic, mistaking quite a few healthy plants for withering ones.

- **Recall (Class 1):** However, vigilance pays off. Out of 100 at-risk plants, 94 are correctly identified, ensuring timely care.

- **F1 Score (Class 1):** The balance of precision and recall in at-risk plant care is decent (0.75), but there's room for improvement.

## Cultivating a Better Garden: Strategies for Model Enhancement

As our gardener gains experience, here are some strategies to enhance the craft:

1. **Richer Soil (Data Collection):** Our garden's diversity is skewed; more variety, especially in at-risk plants, can lead to better care.

2. **New Gardening Tools (Feature Engineering):** Exploring new tools and techniques could unveil patterns unseen to the naked eye.

3. **Refined Gardening Techniques (Algorithm Tuning):** Our gardener's methods are good but tweaking them could lead to better identification of at-risk plants.

4. **Consulting Sage Gardeners (Advanced Algorithms):** Sometimes, the wisdom of seasoned gardeners (advanced ML algorithms) can provide new insights.

5. **Weighing the Stakes (Cost-sensitive Training):** Each plant's life is precious. Adjusting our care depending on the plant's condition could lead to a more balanced garden.

6. **Community Gardening (Ensemble Methods):** Bringing in a team of gardeners with different strengths (stacking models) might improve overall garden health.

7. **Gardener's Reflection (Post-Modeling Analysis):** After a day's work, our gardener reflects on the mistakes made to learn and grow.

## Concluding Thoughts

In our garden of stroke prediction, the journey has been enlightening. While our gardener excels in nurturing the flourishing, the care for the at-risk needs refinement. Our future paths involve richer soils, better tools, and collective wisdom. Here's to growing a healthier, more vibrant garden!


<a name="Eplore-The-Evolution-Of-Stroke-Prediction-Model"></a>
# Explore the Evolution of Stroke Prediction Models
- [Evolution Enhancements Notebook](https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/Stroke_Prediction_Model_Evolution/Analysis%20of%20Stroke%20Prediction%20Model%20Enhancements.ipynb)
- [Model 1: Basic Processing and SMOTE](https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/Stroke_Prediction_Model_Evolution/Model_1_Basic_Preprocessing_and_SMOTE.ipynb)
- [Model 2: Advanced Ensemble Techniques](https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/Stroke_Prediction_Model_Evolution/Model_2_Advanced_Ensemble_Techniques.ipynb)
- [Model 3: Optimized Hyperparameter Tuning](https://github.com/QuantumQuaser/heart_stroke-_prediction/blob/main/Stroke_Prediction_Model_Evolution/Model_3_Optimized_Hyperparameter_Tuning.ipynb)


#### Model 1: Initial Foundation (Early Explorations)
In "Model 1: " I started my journey in stroke prediction modeling. As a novice, my focus was on grasping the basics, which is evident from the simpler model architecture and preprocessing steps. I employed basic machine learning algorithms like Logistic Regression and SVC, and handled class imbalance with SMOTE. This model represents the first step in my  journey, where I was learning to navigate the complexities of feature preprocessing and model selection, using insights from the initial data analysis and chart correlations.

#### Model 2: Intermediate Refinement (Developing Complexity)
"Model 2: " marks a progression in my understanding and application of machine learning concepts. Here, I experimented with a wider array of algorithms, including Gradient Boosting and Extra Trees classifiers, and refined my feature preprocessing techniques. This model illustrates a deeper understanding of the stroke prediction problem, where I started to incorporate more sophisticated strategies like feature engineering and ensemble methods. I also explored different hyperparameter tuning techniques to optimize the models further. This stage was about building upon the foundational skills developed in Model 1 and pushing towards more complex solutions.

#### Model 3: Advanced Synthesis (Advanced Integration)
In "Model 3:" I reached a culmination point in my machine learning journey. This model showcases an integrated approach, combining advanced algorithms like XGBoost with nuanced preprocessing techniques. I implemented feature selection methods and tuned the models using GridSearchCV, which demonstrated a sophisticated understanding of the algorithms and their interaction with our stroke dataset. The use of voting and stacking classifiers in this model highlights my ability to synthesize different machine learning techniques into a cohesive and effective solution.

Each of these models represents a distinct phase in my learning curve. The progression from Model 1 to Model 3 illustrates a journey from basic understanding to advanced application .

**Note:** Despite the advancements in Model 3, I acknowledge that there's always more to learn and ways to improve and I'm eager to continue exploring and enhancing these models. Your feedback and insights are always appreciated and will be instrumental in my continued learning journey.
---






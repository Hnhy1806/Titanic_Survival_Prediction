# Titanic Survival Prediction

## Project Overview

This project uses the Titanic dataset to build and compare machine learning models that predict whether a passenger survived or did not survive. The target variable is binary:

- `1` = Survived
- `0` = Did not survive

The main goal of the project is to compare different classification models, evaluate their performance using meaningful metrics, and select the most reliable model for predicting survival.

## Dataset Understanding

The dataset contains passenger-level information from the Titanic, including demographic, socioeconomic, and family-related variables.

Main features used in the analysis include:

- `Age`
- `Sex`
- `Pclass`
- `Fare`
- `SibSp`
- `Parch`
- `Embarked`

During the exploratory data analysis, missing values were found in `Age`, `Cabin`, and `Embarked`. The analysis also showed strong relationships between survival and key features such as gender and passenger class. Female passengers and passengers in higher classes had higher survival rates.

## Data Preprocessing

Several preprocessing steps were applied before model training:

### Handling Missing Values

- Missing values in `Age` were filled using the median because it is less affected by outliers.
- Missing values in `Embarked` were filled using the mode to preserve the most common category.
- `Cabin` was removed because it contained too many missing values and had limited predictive value.

### Feature Selection

The following columns were removed because they were not useful for model training or had limited predictive value:

- `Name`
- `PassengerId`
- `Ticket`
- `Cabin`

### Encoding Categorical Variables

Machine learning models require numerical inputs, so categorical variables were encoded:

- `Sex` and `Embarked` were encoded using OneHotEncoder.
- `Pclass` was encoded using OrdinalEncoder because passenger class has a meaningful order.

### Data Splitting and Scaling

The dataset was split into:

- 70% training set
- 15% validation set
- 15% testing set

Min-Max scaling was applied to numerical features that were not one-hot encoded.

## Models Developed

The project trained and evaluated multiple models:

1. K-Nearest Neighbors (KNN)
2. Decision Tree
3. Linear Regression as a baseline classification model
4. Naïve Bayes
5. Logistic Regression
6. Support Vector Machine (SVM)

The main evaluation metric was the F1 score for class `1` because the project focused on correctly identifying passengers who survived. F1 score was selected because it balances precision and recall.

## Model Performance

| Model | Key Tuning / Method | F1 Score for Survived Class |
|---|---|---:|
| K-Nearest Neighbors | Best K = 3 | 0.727 |
| Decision Tree | Best max depth = 3 | 0.785 |
| Linear Regression | Threshold = 0.5 | 0.748 |
| Naïve Bayes | Optimized threshold = 0.462 | 0.724 |
| Logistic Regression | Best C = 10, optimized threshold = 0.588 | 0.750 |
| Support Vector Machine | C = 0.1, RBF kernel, optimized threshold = 0.175 | 0.730 |

## Final Model Selection

The Decision Tree model achieved the highest F1 score of **0.785**, making it the best-performing model in this project.

This model was selected as the final model because it provided the strongest balance between precision and recall for identifying survivors. It also captured important relationships between features such as gender, fare, passenger class, age, and family-related variables.

## Key Insights

The analysis showed that passenger survival was strongly influenced by both demographic and socioeconomic factors.

Important findings include:

- Female passengers had a much higher survival rate than male passengers.
- Passengers in higher classes had a higher chance of survival.
- Fare and passenger class were meaningful indicators of socioeconomic status.
- Age and family-related features such as `SibSp` and `Parch` also helped the models distinguish between survivors and non-survivors.

Overall, the project demonstrated that machine learning models can identify meaningful survival patterns from historical passenger data.

## Conclusion

This project successfully built and compared several machine learning models for Titanic survival prediction. After preprocessing the data, encoding categorical variables, scaling numerical features, and tuning model parameters, the Decision Tree model performed the best based on F1 score.

The final result suggests that a simple Decision Tree can effectively capture important survival patterns in the Titanic dataset while remaining easier to interpret than more complex models.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Possible Future Improvements

Future improvements could include:

- Performing more feature engineering, such as extracting titles from passenger names.
- Comparing additional ensemble models such as Random Forest or Gradient Boosting.
- Improving threshold tuning for all classification models.
- Using cross-validation more extensively to test model stability.
- Creating visual explanations for the final Decision Tree model.

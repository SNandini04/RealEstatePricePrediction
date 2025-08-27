# 🏡 Real Estate Price Prediction
This project predicts real estate prices in Gurgaon using machine learning techniques. The dataset is collected from 99acres.com, including data for flats, independent houses, and apartments. The project covers data cleaning, feature engineering, exploratory data analysis (EDA), and model building for price prediction.

---

# 📊 Dataset
The following datasets were collected and used for analysis:
- [Flats Data](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/data/flats.csv) – Raw dataset for flats.
- [Houses Data](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/data/houses.csv) – Raw dataset for independent houses.
- [Apartment Data](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/data/appartments.csv) – Raw dataset for apartments.
- [Merged Flats_Houses Data](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/data/gurgaon_properties.csv) – Combined cleaned dataset for analysis.

---

# Data Cleaning
The following notebooks document the steps taken to clean and standardize the datasets:
- [Flats_Data_EDA](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/Flats_eda.ipynb) – Initial cleaning and exploration of flats data.
- [Houses_Data_EDA](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/houses_eda.ipynb) – Initial cleaning and exploration of houses data.
- [Cleaned_Data](https://github.com/SNandini04/RealEstatePricePrediction/tree/main/notebook/cleaned_data) – Final cleaned datasets after merging flats and houses.

---

## Data Preprocessing
This notebook contains the steps to preprocess data for machine learning models:
- [Data_Preprocessing](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/data_preprocessing.ipynb) – Handling missing values, scaling, and encoding features.

---

## Data Visualization
Visualizations to understand data distributions and relationships:
- [Univariate_Analysis](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/univariate_analysis.ipynb) – Analyzing single features.
- [Multivariate_Analysis](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/multivariate_analysis.ipynb) – Exploring relationships between multiple features.

---

## Feature Engineering
Notebooks for creating and selecting meaningful features:
- [Feature_Engineering](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/feature_eng1.ipynb) – Domain-specific feature creation.
- [Feature_Engineering2](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/feature_selection_feature_engineering.ipynb) – Feature selection and refinement.

---

## Outlier Correction
Notebook to detect and correct outliers in numerical features:
- [Outlier_Correction](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/Outlier_correction.ipynb)

---

## Missing Value Imputation
Notebook for imputing missing values in the dataset:
- [Imputation](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/Missing_Value_Imputation.ipynb)

---

## Feature Selection
Notebook detailing the selection of the most relevant features for modeling:
- [Feature_Selection](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/feature_selection.ipynb)

---

## Model Selection
Notebooks for building baseline and tuned machine learning models:
- [Baseline_Model](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/baseline%20model.ipynb) – Initial baseline models.
- [Model_Selection](https://github.com/SNandini04/RealEstatePricePrediction/blob/main/notebook/EDA_Notebook/model_selection.ipynb) – Model tuning and selection.

---

## 🚀 Model Performance Highlights
- Achieved **90.2% R² score** using RandomForestRegressor with GridSearchCV hyperparameter tuning.
- Engineered **specific features** and applied **robust feature transformation** techniques to ensure effective preprocessing and maximize model accuracy.
- Optimized the model to reach **0.46 MAE** using cross-validation, improving prediction accuracy by **15% over baseline**.

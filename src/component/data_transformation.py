
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataTransformer:
    """
    Class to handle all data transformation steps:
    - Encoding categorical features
    - Scaling numerical features
    - Imputation of missing values
    - Custom feature extraction
    """

    def __init__(self, df):
        self.df = df.copy()
        self.pipeline = None

    def extract_numeric_area(self, col='areaWithType'):
        """
        Extract numeric value from area column like '1200 sqft'
        """
        self.df['area_sqft'] = self.df[col].str.extract('(\d+\.?\d*)').astype(float)
        return self.df

    def encode_and_scale(self, cat_cols, num_cols):
        """
        Encode categorical features and scale numerical features
        """
        # Define transformers
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False))
        ])

        # Combine transformers
        self.pipeline = ColumnTransformer(transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

        # Fit and transform
        transformed_array = self.pipeline.fit_transform(self.df)
        # Combine with column names
        cat_feature_names = self.pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
        all_columns = num_cols + list(cat_feature_names)
        self.df = pd.DataFrame(transformed_array, columns=all_columns)
        return self.df

    def transform_new_data(self, new_df):
        """
        Transform new/unseen data using the fitted pipeline
        """
        transformed_array = self.pipeline.transform(new_df)
        cat_cols = [name for name in new_df.select_dtypes(include='object').columns]
        num_cols = [col for col in new_df.columns if col not in cat_cols]
        cat_feature_names = self.pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
        all_columns = num_cols + list(cat_feature_names)
        return pd.DataFrame(transformed_array, columns=all_columns)

    def get_pipeline(self):
        """
        Return the pipeline object for saving/loading
        """
        return self.pipeline

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
from src.preprocess import get_average_vector


# Function to drop rater traits
def drop_rater_traits(df):
    columns_to_drop = [col for col in df.columns if 'trait' in col]
    return df.drop(columns=columns_to_drop)

# Function to impute null domain scores
def impute_domain_scores(df, strategy='mean'):
    if strategy == 'mean':
        # Compute the mean of the rater domain scores
        df['domain1_score'] = df['domain1_score'].fillna(df[['rater1_domain1', 'rater2_domain1', 'rater3_domain1']].mean(axis=1))
        df['domain2_score'] = df['domain2_score'].fillna(df[['rater1_domain2', 'rater2_domain2']].mean(axis=1))
        # drop the rater domain score columns
        df = df.drop(columns=['rater1_domain1', 'rater2_domain1', 'rater3_domain1', 'rater1_domain2', 'rater2_domain2'])
    elif strategy == 'pca':
        # Apply PCA and use the first principal component to impute missing values
        pca = PCA(n_components=1)
        domain1_scores = pca.fit_transform(df[['rater1_domain1', 'rater2_domain1', 'rater3_domain1']].dropna())
        df['domain1_score'] = df['domain1_score'].fillna(np.squeeze(domain1_scores))
        
        domain2_scores = pca.fit_transform(df[['rater1_domain2', 'rater2_domain2']].dropna())
        df['domain2_score'] = df['domain2_score'].fillna(np.squeeze(domain2_scores))
        # drop the rater domain score columns
        df = df.drop(columns=['rater1_domain1', 'rater2_domain1', 'rater3_domain1', 'rater1_domain2', 'rater2_domain2'])
    return df

# Main training pipeline
def training_pipeline(df, get_average_vector, domain_score_imputer_strategy='mean'):
    # Drop rater trait columns
    df = drop_rater_traits(df)
    
    # Impute domain scores
    df = impute_domain_scores(df, strategy=domain_score_imputer_strategy)
    
    # Preprocess essays using the get_average_vector function
    df['essay'] = df['essay'].apply(lambda text: get_average_vector(text, 100))
    
    # df['essay'] = essay_preprocessor(df['essay'])
    
    return df


def validation_pipeline(df, essay_preprocessor):
    df['essay'] = essay_preprocessor(df['essay'])
    return df

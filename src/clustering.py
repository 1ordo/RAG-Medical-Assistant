"""
Medical Patient Clustering Module

This module provides functions for analyzing patient data using K-means clustering
to determine diagnosis severity, estimated cost, and treatment priority.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path="data/Hospital_Inpatient_Discharges.csv"):
    """
    Load hospital discharge data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing hospital discharge data
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def preprocess_data(df):
    """
    Preprocess the hospital discharge data for analysis.
    
    Args:
        df: Input DataFrame with hospital discharge data
        
    Returns:
        tuple: (preprocessed DataFrame, diagnosis encoder)
    """
    # Drop rows with missing critical values
    df = df.dropna(subset=['CCS Diagnosis Description', 'APR Severity of Illness Description', 'APR MDC Description','Total Costs'])

    # Map severity descriptions to numeric values
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Major': 3}
    df['severity_encoded'] = df['APR Severity of Illness Description'].map(severity_mapping)
    df['Total Costs'] = df['Total Costs'].str.replace(',', '').astype(float)
    
    # Drop rows where severity couldn't be mapped
    df = df.dropna(subset=['severity_encoded'])

    # Find the most common severity for each diagnosis
    most_common_severity = df.groupby('CCS Diagnosis Description')['APR Severity of Illness Description'].agg(
        lambda x: x.mode()[0]
    )

    # Filter df to keep only rows with the most common severity for each diagnosis
    df = df.merge(
        most_common_severity.reset_index(),
        on=['CCS Diagnosis Description', 'APR Severity of Illness Description']
    )

    # Encode the diagnosis descriptions
    diagnosis_encoder = LabelEncoder()
    df['diagnosis_encoded'] = diagnosis_encoder.fit_transform(df['CCS Diagnosis Description'])

    return df, diagnosis_encoder


def find_closest_diagnosis(input_diagnosis, diagnosis_list):
    """
    Find the closest matching diagnosis using fuzzy string matching.
    
    Args:
        input_diagnosis: User input diagnosis text
        diagnosis_list: List of valid diagnoses
        
    Returns:
        str or None: The closest matching diagnosis if similarity score is above threshold
    """
    input_diagnosis = input_diagnosis.strip().lower()
    match = process.extractOne(input_diagnosis, diagnosis_list)
    if match:
        diagnosis_match, score, _ = match
        return diagnosis_match if score >= 75 else None  # Use a threshold of 75 for similarity
    return None


def train_kmeans(df, n_clusters=3):
    """
    Train a K-Means clustering model on the preprocessed data.
    
    Args:
        df: Preprocessed DataFrame
        n_clusters: Number of clusters to create
        
    Returns:
        KMeans: Trained clustering model
    """
    # Use diagnosis and severity as features for clustering
    features = df[['diagnosis_encoded', 'severity_encoded']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans


def predict_priority(kmeans, diagnosis_encoder, df, diagnosis):
    """
    Predict the priority level for a patient based on their diagnosis.
    
    Args:
        kmeans: Trained KMeans model
        diagnosis_encoder: LabelEncoder for diagnosis
        df: Preprocessed DataFrame
        diagnosis: Patient diagnosis
        
    Returns:
        tuple: (severity, priority, estimated_cost, apr_mdc)
    """
    # Encode the input diagnosis
    try:
        diagnosis_encoded = diagnosis_encoder.transform([diagnosis])[0]
    except ValueError:
        logger.warning(f"Diagnosis '{diagnosis}' not found in the dataset. Using default cluster.")
        diagnosis_encoded = 0  # Default to the first cluster if diagnosis not found

    # Prepare data for prediction (use default severity of Moderate if missing)
    severity_encoded = 2  # Moderate by default
    user_data = pd.DataFrame({'diagnosis_encoded': [diagnosis_encoded], 'severity_encoded': [severity_encoded]})
    df['cluster'] = kmeans.labels_
    
    # Predict the cluster
    cluster = kmeans.predict(user_data)[0]
    
    # Calculate estimated cost based on cluster
    cluster_costs = df.groupby('cluster')['Total Costs'].mean()
    estimated_cost = cluster_costs[cluster] if cluster in cluster_costs.index else 0.0

    # Get the APR MDC Description for the diagnosis
    apr_mdc = df[df['CCS Diagnosis Description'] == diagnosis]['APR MDC Description'].iloc[0] if diagnosis in df['CCS Diagnosis Description'].values else 'Unknown'

    # Analyze the clusters in the dataset to assign severity and priority dynamically
    cluster_severity = df.groupby('cluster')['severity_encoded'].mean()

    # Map clusters to severity (Minor, Moderate, Major) based on their average severity
    cluster_severity_sorted = cluster_severity.sort_values()
    cluster_to_severity = {
        cluster_severity_sorted.index[0]: 'Minor',
        cluster_severity_sorted.index[1]: 'Moderate',
        cluster_severity_sorted.index[2]: 'Major'
    }

    # Map severity to priority
    severity_to_priority = {'Minor': 'Low Priority', 'Moderate': 'Medium Priority', 'Major': 'High Priority'}
    
    # Get severity and priority for the current cluster
    severity = cluster_to_severity.get(cluster, 'Moderate')  # Default to Moderate if cluster not found
    priority = severity_to_priority.get(severity, 'Medium Priority')  # Default to Medium if severity not found

    return severity, priority, estimated_cost, apr_mdc


def get_patient_state(name, diagnosed_with):
    """
    Analyze patient data using medical clustering algorithms to determine condition severity,
    estimated cost, APR MDC Description and Priority Admission to Intensive Care.
    
    Args:
        name: Patient's name
        diagnosed_with: Patient's diagnosis
        
    Returns:
        tuple: (name, diagnosed_with, severity, estimated_cost, apr_mdc, priority)
    """
    df = load_data()
    df, diagnosis_encoder = preprocess_data(df)
    kmeans = train_kmeans(df)
    severity, priority, estimated_cost, apr_mdc = predict_priority(kmeans, diagnosis_encoder, df, diagnosed_with)
    
    logger.info(f"Patient Analysis Results - Name: {name}, Diagnosis: {diagnosed_with}, "
                f"Severity: {severity}, Estimated Cost: ${estimated_cost:.2f}, "
                f"APR MDC: {apr_mdc}, Priority: {priority}")
    
    return name, diagnosed_with, severity, estimated_cost, apr_mdc, priority

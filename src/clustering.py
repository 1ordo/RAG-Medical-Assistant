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
        # Load with low_memory=False to handle mixed types and specify dtype for problematic columns
        return pd.read_csv(file_path, low_memory=False, dtype={'Total Costs': 'str'})
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
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['CCS Diagnosis Description', 'APR Severity of Illness Description', 'APR MDC Description','Total Costs'])

    # Map severity descriptions to numeric values
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Major': 3}
    df.loc[:, 'severity_encoded'] = df['APR Severity of Illness Description'].map(severity_mapping)
    
    # Handle Total Costs - convert to string first if not already, then clean and convert to float
    if df['Total Costs'].dtype != 'object':
        df.loc[:, 'Total Costs'] = df['Total Costs'].astype(str)
    
    # Clean and convert Total Costs
    df.loc[:, 'Total Costs'] = df['Total Costs'].str.replace(',', '').str.replace('$', '').str.strip()
    
    # Convert to numeric, handling any non-numeric values
    df.loc[:, 'Total Costs'] = pd.to_numeric(df['Total Costs'], errors='coerce')
    
    # Drop rows where severity couldn't be mapped or costs couldn't be converted
    df = df.dropna(subset=['severity_encoded', 'Total Costs'])

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
    df.loc[:, 'diagnosis_encoded'] = diagnosis_encoder.fit_transform(df['CCS Diagnosis Description'])

    return df, diagnosis_encoder


def find_closest_diagnosis(input_diagnosis, diagnosis_list):
    """
    Find the closest matching diagnosis using improved fuzzy string matching and keyword analysis.
    
    Args:
        input_diagnosis: User input diagnosis text or symptoms
        diagnosis_list: List of valid diagnoses
        
    Returns:
        str or None: The closest matching diagnosis if similarity score is above threshold
    """
    if not input_diagnosis or not diagnosis_list:
        return None
        
    input_diagnosis = input_diagnosis.strip().lower()
    
    # Define symptom-to-diagnosis keyword mappings for better matching
    symptom_keywords = {
        'respiratory': ['cough', 'fever', 'throat', 'sore throat', 'respiratory', 'breathing', 'shortness of breath', 'chest'],
        'viral': ['fever', 'tired', 'fatigue', 'ache', 'loss of smell', 'loss of taste'],
        'infection': ['fever', 'sore throat', 'swollen', 'pain'],
        'pneumonia': ['cough', 'fever', 'chest pain', 'breathing', 'shortness'],
        'bronchitis': ['cough', 'chest', 'mucus', 'wheezing'],
        'influenza': ['fever', 'ache', 'tired', 'fatigue', 'cough', 'throat'],
        'upper respiratory': ['cough', 'throat', 'sore throat', 'runny nose', 'congestion'],
        'covid': ['fever', 'cough', 'loss of smell', 'loss of taste', 'tired', 'fatigue', 'throat']
    }
    
    # Extract keywords from input
    input_words = input_diagnosis.split()
    
    # Score diagnoses based on keyword relevance
    scored_diagnoses = []
    for diagnosis in diagnosis_list:
        diagnosis_lower = diagnosis.lower()
        base_score = 0
        
        # Check for direct keyword matches in diagnosis
        for category, keywords in symptom_keywords.items():
            if any(keyword in input_diagnosis for keyword in keywords):
                if category in diagnosis_lower or any(kw in diagnosis_lower for kw in keywords):
                    base_score += 20
        
        # Additional scoring for common medical terms
        medical_matches = 0
        if 'fever' in input_diagnosis and ('fever' in diagnosis_lower or 'viral' in diagnosis_lower or 'infection' in diagnosis_lower):
            medical_matches += 15
        if 'cough' in input_diagnosis and ('respiratory' in diagnosis_lower or 'pneumonia' in diagnosis_lower or 'bronch' in diagnosis_lower):
            medical_matches += 15
        if 'throat' in input_diagnosis and ('throat' in diagnosis_lower or 'upper respiratory' in diagnosis_lower):
            medical_matches += 15
        if 'smell' in input_diagnosis and ('viral' in diagnosis_lower or 'respiratory' in diagnosis_lower):
            medical_matches += 10
        if 'tired' in input_diagnosis or 'fatigue' in input_diagnosis:
            if 'viral' in diagnosis_lower or 'infection' in diagnosis_lower:
                medical_matches += 10
        
        base_score += medical_matches
        
        # Use fuzzy matching as additional factor
        fuzzy_match = process.extractOne(input_diagnosis, [diagnosis_lower])
        if fuzzy_match:
            fuzzy_score = fuzzy_match[1]
            # Combine scores: prioritize keyword matching but include fuzzy matching
            combined_score = base_score + (fuzzy_score * 0.3)
            scored_diagnoses.append((diagnosis, combined_score, fuzzy_score))
    
    # Sort by combined score
    scored_diagnoses.sort(key=lambda x: x[1], reverse=True)
    
    if scored_diagnoses:
        best_diagnosis, combined_score, fuzzy_score = scored_diagnoses[0]
        
        # Log the actual scores
        logger.info(f"Best match: '{best_diagnosis}' with combined score {combined_score:.1f} (fuzzy: {fuzzy_score:.1f})")
        
        # Require a minimum combined score for acceptance
        if combined_score >= 25:  # Adjusted threshold for keyword-based matching
            return best_diagnosis
        else:
            # If no good keyword match, fall back to pure fuzzy matching with higher threshold
            pure_fuzzy = process.extractOne(input_diagnosis, [d.lower() for d in diagnosis_list])
            if pure_fuzzy and pure_fuzzy[1] >= 80:  # Higher threshold for pure fuzzy
                original_index = [d.lower() for d in diagnosis_list].index(pure_fuzzy[0])
                logger.info(f"Fallback fuzzy match: '{diagnosis_list[original_index]}' with score {pure_fuzzy[1]}")
                return diagnosis_list[original_index]
    
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
    # Add cluster labels to the dataframe if not already present
    if 'cluster' not in df.columns:
        df['cluster'] = kmeans.labels_
    
    # Try to find exact match first, then fuzzy match
    matched_diagnosis = diagnosis
    if diagnosis not in diagnosis_encoder.classes_:
        # Try fuzzy matching
        available_diagnoses = list(diagnosis_encoder.classes_)
        matched_diagnosis = find_closest_diagnosis(diagnosis, available_diagnoses)
        
        if matched_diagnosis:
            logger.info(f"Using fuzzy matched diagnosis: '{matched_diagnosis}' for input: '{diagnosis}'")
        else:
            logger.warning(f"No suitable match found for diagnosis: '{diagnosis}'. Using most common diagnosis.")
            # Use the most common diagnosis as fallback
            most_common_diagnosis = df['CCS Diagnosis Description'].mode()[0]
            matched_diagnosis = most_common_diagnosis
    
    # Encode the matched diagnosis
    try:
        diagnosis_encoded = diagnosis_encoder.transform([matched_diagnosis])[0]
    except ValueError:
        logger.error(f"Failed to encode diagnosis: '{matched_diagnosis}'. Using default.")
        diagnosis_encoded = 0

    # Get the most common severity for this diagnosis from the dataset
    diagnosis_data = df[df['CCS Diagnosis Description'] == matched_diagnosis]
    if not diagnosis_data.empty:
        # Use the actual severity from the dataset
        severity_encoded = diagnosis_data['severity_encoded'].mode()[0]
        # Get APR MDC Description
        apr_mdc = diagnosis_data['APR MDC Description'].iloc[0]
    else:
        # Default values if diagnosis not found
        severity_encoded = 2  # Moderate
        apr_mdc = 'Unknown'

    # Prepare data for prediction
    user_data = pd.DataFrame({'diagnosis_encoded': [diagnosis_encoded], 'severity_encoded': [severity_encoded]})
    
    # Predict the cluster
    cluster = kmeans.predict(user_data)[0]
    
    # Calculate estimated cost based on cluster
    cluster_costs = df.groupby('cluster')['Total Costs'].mean()
    estimated_cost = cluster_costs[cluster] if cluster in cluster_costs.index else df['Total Costs'].mean()

    # Analyze the clusters to assign severity and priority dynamically
    cluster_severity = df.groupby('cluster')['severity_encoded'].mean()
    
    # Ensure we have enough clusters
    if len(cluster_severity) >= 3:
        cluster_severity_sorted = cluster_severity.sort_values()
        cluster_to_severity = {
            cluster_severity_sorted.index[0]: 'Minor',
            cluster_severity_sorted.index[1]: 'Moderate',
            cluster_severity_sorted.index[2]: 'Major'
        }
    else:
        # Fallback mapping for fewer clusters
        severity_map = {0: 'Minor', 1: 'Moderate', 2: 'Major'}
        cluster_to_severity = {i: severity_map.get(i, 'Moderate') for i in range(len(cluster_severity))}

    # Map severity to priority
    severity_to_priority = {'Minor': 'Low Priority', 'Moderate': 'Medium Priority', 'Major': 'High Priority'}
    
    # Get severity and priority for the current cluster
    predicted_severity = cluster_to_severity.get(cluster, 'Moderate')
    priority = severity_to_priority.get(predicted_severity, 'Medium Priority')

    return predicted_severity, priority, estimated_cost, apr_mdc


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

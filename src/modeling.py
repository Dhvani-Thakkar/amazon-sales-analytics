"""
Machine Learning Module
Contains predictive models for sales forecasting and clustering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
import joblib
from pathlib import Path

class AmazonMLModels:
    """Class for machine learning models"""
    
    def __init__(self, df, model_dir='models'):
        self.df = df
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
    
    def prepare_features(self):
        """Prepare features for modeling"""
        # Select numeric features
        feature_cols = ['actual_price', 'discount_percentage', 'rating', 'rating_count']
        
        # Create dummy variables for categories
        category_dummies = pd.get_dummies(self.df['main_category'], prefix='category')
        
        # Combine features
        self.X = pd.concat([self.df[feature_cols], category_dummies], axis=1)
        self.X = self.X.fillna(0)
        
        return self.X
    
    def train_sales_predictor(self):
        """Train regression model to predict rating_count (sales proxy)"""
        print("\n--- Training Sales Prediction Model ---")
        
        X = self.prepare_features()
        y = np.log1p(self.df['rating_count'])  # Log transform for better distribution
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save model
        joblib.dump(model, self.model_dir / 'sales_predictor.pkl')
        print("✓ Model saved: sales_predictor.pkl")
        
        self.models['sales_predictor'] = model
        
        return model, rmse, r2
    
    def train_rating_classifier(self):
        """Train classifier to predict high-rating products"""
        print("\n--- Training Rating Classification Model ---")
        
        X = self.prepare_features()
        y = (self.df['rating'] >= 4.0).astype(int)  # Binary: high rating or not
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Rating', 'High Rating']))
        
        # Save model
        joblib.dump(model, self.model_dir / 'rating_classifier.pkl')
        print("✓ Model saved: rating_classifier.pkl")
        
        self.models['rating_classifier'] = model
        
        return model
    
    def perform_product_clustering(self, n_clusters=5):
        """Cluster products based on features"""
        print(f"\n--- Performing K-Means Clustering (k={n_clusters}) ---")
        
        # Select features for clustering
        cluster_features = ['discounted_price', 'rating', 'discount_percentage', 'rating_count']
        X_cluster = self.df[cluster_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        self.df['cluster'] = clusters
        
        # Cluster analysis
        print("\nCluster Characteristics:")
        for i in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_data)} products):")
            print(f"  Avg Price: ₹{cluster_data['discounted_price'].mean():.2f}")
            print(f"  Avg Rating: {cluster_data['rating'].mean():.2f}")
            print(f"  Avg Discount: {cluster_data['discount_percentage'].mean():.1f}%")
            print(f"  Avg Review Count: {cluster_data['rating_count'].mean():.0f}")
        
        # Save model
        joblib.dump(kmeans, self.model_dir / 'product_clusters.pkl')
        joblib.dump(scaler, self.model_dir / 'cluster_scaler.pkl')
        print("\n✓ Models saved: product_clusters.pkl, cluster_scaler.pkl")
        
        self.models['kmeans'] = kmeans
        self.models['scaler'] = scaler
        
        return kmeans, clusters
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "="*50)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*50)
        
        self.train_sales_predictor()
        self.train_rating_classifier()
        self.perform_product_clustering()
        
        print("\n" + "="*50)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*50)
        
        return self.models


if __name__ == "__main__":
    # Load cleaned data and train models
    df = pd.read_csv('data/processed/amazon_cleaned.csv')
    ml_models = AmazonMLModels(df)
    models = ml_models.train_all_models()
    
    # Save updated dataframe with clusters
    df.to_csv('data/processed/amazon_with_clusters.csv', index=False)
    print("\n✓ Data with clusters saved")

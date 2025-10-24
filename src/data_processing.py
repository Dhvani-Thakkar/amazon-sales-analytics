"""
Data Processing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

class AmazonDataProcessor:
    """Class to handle all data processing tasks"""
    
    def __init__(self, raw_data_path='/data/amazon.csv'):
        self.raw_data_path = raw_data_path
        self.df = None
        
    def load_data(self):
        """Load raw data from CSV"""
        print(f"Loading data from {self.raw_data_path}...")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def clean_price_columns(self):
        """Clean and convert price columns to numeric"""
        # Remove currency symbols and convert to float
        price_columns = ['discounted_price', 'actual_price']
        
        for col in price_columns:
            if col in self.df.columns:
                # Remove ₹ symbol and commas, convert to float
                self.df[col] = self.df[col].astype(str).str.replace('₹', '').str.replace(',', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print("✓ Price columns cleaned")
        return self
    
    def clean_discount_column(self):
        """Clean discount percentage column"""
        if 'discount_percentage' in self.df.columns:
            # Remove % symbol and convert to float
            self.df['discount_percentage'] = self.df['discount_percentage'].astype(str).str.replace('%', '')
            self.df['discount_percentage'] = pd.to_numeric(self.df['discount_percentage'], errors='coerce')
        
        print("✓ Discount column cleaned")
        return self
    
    def extract_main_category(self):
        """Extract main category from category hierarchy"""
        if 'category' in self.df.columns:
            # Category is in format: "Main|Sub|SubSub"
            # Extract the last part (most specific category)
            self.df['main_category'] = self.df['category'].str.split('|').str[-1]
            self.df['main_category'] = self.df['main_category'].str.strip()
        
        print("✓ Main category extracted")
        return self
    
    def clean_rating_columns(self):
        """Clean rating and rating_count columns"""
        # Clean rating
        if 'rating' in self.df.columns:
            self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        
        # Clean rating_count (remove commas)
        if 'rating_count' in self.df.columns:
            self.df['rating_count'] = self.df['rating_count'].astype(str).str.replace(',', '')
            self.df['rating_count'] = pd.to_numeric(self.df['rating_count'], errors='coerce')
        
        print("✓ Rating columns cleaned")
        return self
    
    def handle_missing_values(self):
        """Handle missing values appropriately"""
        # Fill numeric columns with median
        numeric_cols = ['discounted_price', 'actual_price', 'rating', 'rating_count']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill discount_percentage with 0 if missing
        if 'discount_percentage' in self.df.columns:
            self.df['discount_percentage'].fillna(0, inplace=True)
        
        # Drop rows with critical missing values
        critical_cols = ['product_name', 'main_category']
        self.df.dropna(subset=critical_cols, inplace=True)
        
        print(f"✓ Missing values handled. Remaining records: {len(self.df)}")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate records"""
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)
        removed = initial_count - len(self.df)
        print(f"✓ Removed {removed} duplicate records")
        return self
    
    def engineer_features(self):
        """Create new features for analysis"""
        # Calculate discount amount
        self.df['discount_amount'] = self.df['actual_price'] - self.df['discounted_price']
        
        # Create price bins
        self.df['price_range'] = pd.cut(
            self.df['discounted_price'],
            bins=[0, 500, 1000, 2000, 5000, float('inf')],
            labels=['Budget (<500)', 'Low (500-1K)', 'Mid (1K-2K)', 'High (2K-5K)', 'Premium (>5K)']
        )
        
        # Create rating categories
        self.df['rating_category'] = pd.cut(
            self.df['rating'],
            bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0],
            labels=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent']
        )
        
        # Create discount bins
        self.df['discount_level'] = pd.cut(
            self.df['discount_percentage'],
            bins=[0, 20, 40, 60, 100],
            labels=['Low (<20%)', 'Medium (20-40%)', 'High (40-60%)', 'Very High (>60%)']
        )
        
        # Calculate estimated revenue (proxy using rating_count as sales proxy)
        self.df['estimated_revenue'] = self.df['discounted_price'] * self.df['rating_count']
        
        print("✓ New features engineered")
        return self
    
    def get_data_summary(self):
        """Get summary statistics of the dataset"""
        summary = {
            'total_records': len(self.df),
            'total_categories': self.df['main_category'].nunique(),
            'avg_price': self.df['discounted_price'].mean(),
            'avg_rating': self.df['rating'].mean(),
            'avg_discount': self.df['discount_percentage'].mean(),
            'total_products': self.df['product_id'].nunique()
        }
        return summary
    
    def save_cleaned_data(self, output_path='data/processed/amazon_cleaned.csv'):
        """Save cleaned data to CSV"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✓ Cleaned data saved to {output_path}")
        return self
    
    def run_full_pipeline(self):
        """Execute complete data processing pipeline"""
        print("\n" + "="*50)
        print("STARTING DATA PROCESSING PIPELINE")
        print("="*50 + "\n")
        
        self.load_data()
        self.clean_price_columns()
        self.clean_discount_column()
        self.extract_main_category()
        self.clean_rating_columns()
        self.handle_missing_values()
        self.remove_duplicates()
        self.engineer_features()
        
        summary = self.get_data_summary()
        print("\n" + "="*50)
        print("DATA PROCESSING COMPLETE")
        print("="*50)
        print(f"Total Records: {summary['total_records']}")
        print(f"Total Categories: {summary['total_categories']}")
        print(f"Average Price: ₹{summary['avg_price']:.2f}")
        print(f"Average Rating: {summary['avg_rating']:.2f}")
        print(f"Average Discount: {summary['avg_discount']:.1f}%")
        
        self.save_cleaned_data()
        return self.df


if __name__ == "__main__":
    # Run the pipeline
    processor = AmazonDataProcessor()
    df_clean = processor.run_full_pipeline()

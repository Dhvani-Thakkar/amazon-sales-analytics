"""
Visualization Module
Contains all plotting functions for analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class AmazonVisualizer:
    """Class for creating visualizations"""
    
    def __init__(self, df, output_dir='reports/figures'):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_category_distribution(self, save=True):
        """Plot distribution of products across categories"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        category_counts = self.df['main_category'].value_counts().head(15)
        
        sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis', ax=ax)
        ax.set_xlabel('Number of Products', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.set_title('Top 15 Product Categories by Volume', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(category_counts.values):
            ax.text(v + 5, i, str(v), va='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: category_distribution.png")
        
        plt.show()
    
    def plot_price_distribution(self, save=True):
        """Plot distribution of discounted prices"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(self.df['discounted_price'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Discounted Price (₹)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Product Prices', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['discounted_price'].median(), color='red', linestyle='--', 
                       label=f'Median: ₹{self.df["discounted_price"].median():.0f}')
        axes[0].legend()
        
        # Box plot
        sns.boxplot(y=self.df['discounted_price'], ax=axes[1], color='lightgreen')
        axes[1].set_ylabel('Discounted Price (₹)', fontsize=12)
        axes[1].set_title('Price Distribution (Box Plot)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: price_distribution.png")
        
        plt.show()
    
    def plot_rating_analysis(self, save=True):
        """Analyze rating patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Rating distribution
        axes[0, 0].hist(self.df['rating'], bins=20, color='gold', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Rating', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Distribution of Product Ratings', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(self.df['rating'].mean(), color='red', linestyle='--',
                          label=f'Mean: {self.df["rating"].mean():.2f}')
        axes[0, 0].legend()
        
        # Rating count distribution (log scale)
        axes[0, 1].hist(np.log10(self.df['rating_count'] + 1), bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Log10(Rating Count)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of Review Counts (Log Scale)', fontsize=14, fontweight='bold')
        
        # Price vs Rating scatter
        axes[1, 0].scatter(self.df['discounted_price'], self.df['rating'], alpha=0.3, s=10)
        axes[1, 0].set_xlabel('Discounted Price (₹)', fontsize=12)
        axes[1, 0].set_ylabel('Rating', fontsize=12)
        axes[1, 0].set_title('Price vs Rating Correlation', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlim(0, 10000)
        
        # Rating category counts
        rating_cat_counts = self.df['rating_category'].value_counts()
        axes[1, 1].pie(rating_cat_counts.values, labels=rating_cat_counts.index, autopct='%1.1f%%',
                      colors=sns.color_palette('Set2'), startangle=90)
        axes[1, 1].set_title('Rating Category Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'rating_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: rating_analysis.png")
        
        plt.show()
    
    def plot_discount_effectiveness(self, save=True):
        """Analyze discount impact on sales proxy"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Discount distribution
        axes[0].hist(self.df['discount_percentage'], bins=30, color='purple', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Discount Percentage (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Discount Percentages', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['discount_percentage'].mean(), color='red', linestyle='--',
                       label=f'Mean: {self.df["discount_percentage"].mean():.1f}%')
        axes[0].legend()
        
        # Discount vs Rating Count (sales proxy)
        discount_bins = self.df.groupby('discount_level')['rating_count'].mean().sort_values(ascending=False)
        sns.barplot(x=discount_bins.index, y=discount_bins.values, palette='coolwarm', ax=axes[1])
        axes[1].set_xlabel('Discount Level', fontsize=12)
        axes[1].set_ylabel('Average Rating Count (Sales Proxy)', fontsize=12)
        axes[1].set_title('Discount Impact on Product Popularity', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'discount_effectiveness.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: discount_effectiveness.png")
        
        plt.show()
    
    def plot_correlation_heatmap(self, save=True):
        """Create correlation matrix heatmap"""
        numeric_cols = ['discounted_price', 'actual_price', 'discount_percentage', 
                       'rating', 'rating_count', 'discount_amount', 'estimated_revenue']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: correlation_heatmap.png")
        
        plt.show()
    
    def plot_top_categories_revenue(self, save=True):
        """Plot top categories by estimated revenue"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        category_revenue = self.df.groupby('main_category')['estimated_revenue'].sum().sort_values(ascending=False).head(15)
        
        sns.barplot(x=category_revenue.values/1e6, y=category_revenue.index, palette='plasma', ax=ax)
        ax.set_xlabel('Estimated Revenue (Millions ₹)', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.set_title('Top 15 Categories by Estimated Revenue', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'category_revenue.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: category_revenue.png")
        
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Price vs Rating scatter with category color
        fig = px.scatter(
            self.df.sample(min(1000, len(self.df))),
            x='discounted_price',
            y='rating',
            color='main_category',
            size='rating_count',
            hover_data=['product_name', 'discount_percentage'],
            title='Interactive: Price vs Rating by Category',
            labels={'discounted_price': 'Price (₹)', 'rating': 'Rating'}
        )
        
        fig.update_layout(height=600)
        fig.write_html(self.output_dir / 'interactive_dashboard.html')
        print("✓ Saved: interactive_dashboard.html")
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations at once"""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50 + "\n")
        
        self.plot_category_distribution()
        self.plot_price_distribution()
        self.plot_rating_analysis()
        self.plot_discount_effectiveness()
        self.plot_correlation_heatmap()
        self.plot_top_categories_revenue()
        self.create_interactive_dashboard()
        
        print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    # Load cleaned data and create visualizations
    df = pd.read_csv('data/processed/amazon_cleaned.csv')
    visualizer = AmazonVisualizer(df)
    visualizer.generate_all_visualizations()

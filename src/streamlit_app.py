"""
Interactive Streamlit Dashboard for Amazon Sales Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Amazon Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/amazon_cleaned.csv')
    return df

df = load_data()

# Title
st.title("📊 Amazon Sales Analytics Dashboard")
st.markdown("### Interactive Data Exploration & Insights")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")

# Category filter
categories = ['All'] + sorted(df['main_category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Category", categories)

# Price range filter
price_range = st.sidebar.slider(
    "Price Range (₹)",
    min_value=float(df['discounted_price'].min()),
    max_value=float(df['discounted_price'].max()),
    value=(float(df['discounted_price'].min()), float(df['discounted_price'].max()))
)

# Rating filter
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)

# Apply filters
filtered_df = df.copy()
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['main_category'] == selected_category]
filtered_df = filtered_df[
    (filtered_df['discounted_price'] >= price_range[0]) &
    (filtered_df['discounted_price'] <= price_range[1]) &
    (filtered_df['rating'] >= min_rating)
]

# KPI Metrics
st.header("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Products", f"{len(filtered_df):,}")
with col2:
    st.metric("Avg Price", f"₹{filtered_df['discounted_price'].mean():.2f}")
with col3:
    st.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}⭐")
with col4:
    st.metric("Avg Discount", f"{filtered_df['discount_percentage'].mean():.1f}%")

st.markdown("---")

# Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Category Distribution")
    category_counts = filtered_df['main_category'].value_counts().head(10)
    fig1 = px.bar(
        x=category_counts.values,
        y=category_counts.index,
        orientation='h',
        labels={'x': 'Number of Products', 'y': 'Category'},
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("💰 Price Distribution")
    fig2 = px.histogram(
        filtered_df,
        x='discounted_price',
        nbins=30,
        labels={'discounted_price': 'Price (₹)'},
        color_discrete_sequence=['#636EFA']
    )
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Second row
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("⭐ Rating vs Price")
    fig3 = px.scatter(
        filtered_df.sample(min(500, len(filtered_df))),
        x='discounted_price',
        y='rating',
        color='main_category',
        size='rating_count',
        hover_data=['product_name'],
        labels={'discounted_price': 'Price (₹)', 'rating': 'Rating'}
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col_right2:
    st.subheader("🎯 Discount Effectiveness")
    discount_impact = filtered_df.groupby('discount_level')['rating_count'].mean().sort_values(ascending=False)
    fig4 = px.bar(
        x=discount_impact.index,
        y=discount_impact.values,
        labels={'x': 'Discount Level', 'y': 'Avg Review Count'},
        color=discount_impact.values,
        color_continuous_scale='RdYlGn'
    )
    fig4.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Data table
st.subheader("📋 Filtered Product Data")
st.dataframe(
    filtered_df[['product_name', 'main_category', 'discounted_price', 'rating', 'discount_percentage']].head(100),
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("**Data Source**: Kaggle Amazon Sales Dataset | **Built with**: Streamlit & Plotly")

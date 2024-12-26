import streamlit as st
import pandas as pd
import plotly.express as px
import sklearn
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Rename columns for easier handling
df.columns = [
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
]

# Streamlit app
st.title("Iris Dataset Dashboard")

# Sidebar selection
st.sidebar.header("Visualization Settings")
feature = st.sidebar.selectbox(
    "Select Feature to Visualize:",
    options=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    index=0
)

# Boxplot visualization
st.header(f"Distribution of {feature.replace('_', ' ').title()} by Species")
boxplot_fig = px.box(
    df, x='species', y=feature, color='species',
    title=f"{feature.replace('_', ' ').title()} by Species",
    labels={feature: feature.replace('_', ' ').title()}
)
st.plotly_chart(boxplot_fig)

# Scatterplot for petal length and petal width
st.header("Petal Length vs Petal Width by Species")
scatter_fig = px.scatter(
    df, x='petal_length', y='petal_width',
    color='species', symbol='species',
    title="Petal Length vs Petal Width",
    labels={'petal_length': 'Petal Length', 'petal_width': 'Petal Width'}
)
st.plotly_chart(scatter_fig)

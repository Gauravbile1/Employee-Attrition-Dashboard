import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data_path = r"Preprocessed_Data.csv"
data = pd.read_csv(data_path)

# Title and Custom Styling
st.set_page_config(layout='wide', page_title="Employee Attrition Analysis")

st.title("📊 Employee Attrition Dashboard")
st.markdown("<style>h1{color: #4A90E2;}</style>", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔧 Customize Analysis")

dept_filter = st.sidebar.selectbox("🏢 Select Department", ['All'] + list(data['Department'].unique()))
attrition_filter = st.sidebar.selectbox("💼 Attrition Status", ['All', 'Yes', 'No'])

# Apply Filters
if dept_filter != 'All':
    data = data[data['Department'] == dept_filter]
if attrition_filter != 'All':
    data = data[data['Attrition'] == attrition_filter]

# Data Overview Section
st.subheader("📌 Data Overview")
with st.expander("🔍 View Dataset"):
    st.dataframe(data.head())

# Key Metrics
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(data))
col2.metric("Average Age", round(data['Age'].mean(), 1))
col3.metric("Average Monthly Income", f"₹{int(data['MonthlyIncome'].mean())}")

# Visualization Section
st.subheader("📌 Attrition Breakdown")
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(data=data, x='Attrition', palette='pastel', ax=ax1)
ax1.set_title("Attrition Breakdown", fontsize=14, fontweight='bold')
st.pyplot(fig1)

st.subheader("💰 Income Distribution by Department")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data, x='Department', y='MonthlyIncome', hue='Attrition', palette='Set2', ax=ax2)
ax2.set_title("Income Distribution by Department", fontsize=14, fontweight='bold')
st.pyplot(fig2)

# Heatmap for Correlation Analysis
st.subheader("🔥 Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
heatmap_data = data.select_dtypes(include='number')
sns.heatmap(
    heatmap_data.corr(), 
    annot=True, 
    cmap='coolwarm', 
    linewidths=0.5, 
    ax=ax3, 
    annot_kws={'size': 8}, 
    fmt='.2f'
)
ax3.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
st.pyplot(fig3)

# Insights and Conclusions
st.success("✨ Key Insights: High attrition in specific departments with lower income & satisfaction levels. Retention strategies are essential!")

#Streamlit App – OfferFlow™ by EY (Updated UI)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from typing import List, Optional
import re
import random
import base64
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, ValidationError
import os
import io
import contextlib
import traceback
import google.generativeai as genai
import certifi
import requests
from requests.adapters import HTTPAdapter
from streamlit.components.v1 import html
from urllib3.poolmanager import PoolManager
import ssl



# -------------------------
# Load Offer History Data
# -------------------------
data_path = 'Corrected_Offer_Data_With_Variation.csv'
df = pd.read_csv(data_path, parse_dates=[
    "Offer_Send_Date", "Offer_Start_Date", "Offer_End_Date",
    "Offer_Open_Date", "Offer_Activation_Date", "Offer_Redeem_Date"
])

df['Offer_Send_Date'] = pd.to_datetime(df['Offer_Send_Date'], errors='coerce')
df['Offer_Send_Date_DateOnly'] = df['Offer_Send_Date'].dt.date

min_date = df['Offer_Send_Date_DateOnly'].min()
max_date = df['Offer_Send_Date_DateOnly'].max()

# Derived Fields
df['Time_to_Respond'] = (df['Offer_Redeem_Date'] - df['Offer_Start_Date']).dt.days
df['Redeem_DayOfWeek'] = df['Offer_Redeem_Date'].dt.day_name()
df['Offer_Month'] = df['Offer_Send_Date'].dt.to_period('M').astype(str)
df['Reward_Utilized'] = df['Redeemed'] * df['Reward_Value_USD']



combinations = df[['Offer_Title', 'SubCategory2']].drop_duplicates()

# Simulate Ratings
def simulate_rating(subcat):
    subcat = str(subcat).lower()
    if any(x in subcat for x in ['Hot Coffee', 'Sweet Snacks', 'Energy Drinks','Salty Snacks']):
        return np.random.choice(["Very Good", "Good"], p=[0.6, 0.4])
    elif any(x in subcat for x in ['Soft Drinks', 'Car Care']):
        return np.random.choice(["Average", "Good"], p=[0.5, 0.5])
    elif any(x in subcat for x in ['Printer Services', 'Frozen Foods','Grocery']):
        return np.random.choice(["Poor", "Below Average"], p=[0.3, 0.7])
    else:
        return np.random.choice(["Very Good", "Good", "Average", "Below Average", "Poor"],
                                p=[0.15, 0.2, 0.3, 0.2, 0.15])

combinations['Rating'] = combinations['SubCategory2'].apply(simulate_rating)
combinations.rename(columns={'Offer_Title': 'Offer Title', 'SubCategory2':'Subcategory'}, inplace=True)


# ========== GEMINI API KEY ==========
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("gemini-2.0-flash")


# SIMULATED DATA GENERATION
# -------------------------
def simulate_qc_data():
    date_range = pd.date_range(start="2024-01-01", end="2024-06-30", freq='D')
    journeys = ['Early Life', 'In-Life', 'Lapsed']
    data = []

    for date in date_range:
        for journey in journeys:
            metrics = {}
            if journey == "In-Life":
                metrics = {
                    'QC_Generosity_Utilization': np.random.uniform(88, 98) if random.random() < 0.8 else np.random.uniform(50, 85),
                    'QC_Timing_Compliance': np.random.uniform(90, 99) if random.random() < 0.8 else np.random.uniform(60, 85),
                    'QC_Eligibility_Adherence': np.random.uniform(85, 97),
                    'QC_Reactivation_Correctness': np.random.uniform(90, 98),
                    'QC_Consistency_Weekly': np.random.uniform(88, 97),
                    'QC_Coverage_Saturation': np.random.uniform(85, 96),
                    'QC_Journey_Mismatch_Error': np.random.uniform(2, 10),
                    'QC_Collision_Resolved': np.random.uniform(90, 98)
                }
            elif journey == "Early Life":
                metrics = {
                    'QC_Generosity_Utilization': np.random.uniform(80, 96),
                    'QC_Timing_Compliance': np.random.uniform(70, 95),
                    'QC_Eligibility_Adherence': np.random.uniform(60, 90),
                    'QC_Reactivation_Correctness': np.random.uniform(70, 95),
                    'QC_Consistency_Weekly': np.random.uniform(60, 90),
                    'QC_Coverage_Saturation': np.random.uniform(70, 95),
                    'QC_Journey_Mismatch_Error': np.random.uniform(5, 15),
                    'QC_Collision_Resolved': np.random.uniform(75, 95)
                }
            else:  # Lapsed
                metrics = {
                    'QC_Generosity_Utilization': np.random.uniform(60, 95),
                    'QC_Timing_Compliance': np.random.uniform(50, 90),
                    'QC_Eligibility_Adherence': np.random.uniform(40, 85),
                    'QC_Reactivation_Correctness': np.random.uniform(60, 90),
                    'QC_Consistency_Weekly': np.random.uniform(50, 85),
                    'QC_Coverage_Saturation': np.random.uniform(55, 88),
                    'QC_Journey_Mismatch_Error': np.random.uniform(10, 20),
                    'QC_Collision_Resolved': np.random.uniform(65, 90)
                }
            data.append({
                'Date': date,
                'Journey': journey,
                'Offers_Sent': np.random.randint(5000, 10000),
                'Unique_Customers': np.random.randint(3000, 8000),
                **metrics
            })

    return pd.DataFrame(data)

qc_df = simulate_qc_data()

# Helper: Simulation Logic
# -------------------------
def simulate_campaign(base_rate, generosity_pct, campaign_pct, reward_value, rev_per_redemption=6.0, elasticity=0.04, max_increase=0.2):
    sim_rate = base_rate + (max_increase * (1 - np.exp(-generosity_pct * elasticity)))
    penalty = 0.1 if campaign_pct > 80 and generosity_pct > 30 else 0.0
    sim_rate = max(0.01, min(sim_rate - penalty, 1.0))
    target_cust = cust_count * campaign_pct / 100
    redemptions = sim_rate * target_cust
    revenue = redemptions * rev_per_redemption
    cost = redemptions * reward_value
    incremental = revenue - (base_rate * target_cust * rev_per_redemption)
    profit = revenue - cost
    roi = (incremental / cost) if cost != 0 else np.nan
    return sim_rate, redemptions, revenue, cost, incremental, roi, profit

# Simulated Data Enhancements (added in-code)
# -------------------------
np.random.seed(42)
df['Promo_Sales'] = df['Reward_Value_USD'] * np.random.uniform(10, 15, len(df)).round(2)
df['Base_Sales'] = df['Promo_Sales'] - df['Incremental_Revenue']
df['Total_Sales'] = df['Promo_Sales'] + df['Base_Sales']
df['Baseline_Impact'] = df['Base_Sales'] * np.random.uniform(0.7, 1.0, len(df))
df['Promo_Lift'] = df['Incremental_Revenue']
df['Cannibalization'] = -1 * df['Promo_Sales'] * np.random.uniform(0.05, 0.15, len(df))
df['Distribution_Impact'] = df['Promo_Sales'] * np.random.uniform(0.05, 0.2, len(df))
df['Channel_Impact'] = df['Promo_Sales'] * np.random.uniform(0.03, 0.1, len(df))
df['Markdown'] = df['Promo_Sales'] * np.random.uniform(0.05, 0.15, len(df))
df['Promo_Profit'] = df['Promo_Sales'] - df['Markdown']
df['Sales_Uplift_Per_Markdown'] = df['Promo_Lift'] / (df['Markdown'] + 1e-6)
df['Offer_Rating'] = np.select(
    [
        df['Sales_Uplift_Per_Markdown'] >= 5,
        df['Sales_Uplift_Per_Markdown'] >= 3,
        df['Sales_Uplift_Per_Markdown'] >= 1.5
    ],
    ['VERY GOOD', 'GOOD', 'QUESTION'],
    default='REVIEW'
)
df['Forecast_Sales_Uplift'] = df['Promo_Sales'] * np.random.uniform(0.05, 0.25, len(df))
df['Forecast_Profit_Uplift'] = df['Promo_Profit'] * np.random.uniform(0.05, 0.25, len(df))
df['Forecast_Volume_Uplift'] = df['Reward_Value_USD'] * np.random.uniform(0.2, 0.8, len(df))

#Define Offer Constraints Table
# -------------------------

scenario_constraints_df = pd.read_csv('Prepared_Scenario_Constraint_Table.csv')




# Aggregates
# funnel_by_type = df.groupby('Offer_Type').agg(Sent=('Offer_ID', 'count'), Redeemed=('Redeemed', 'sum')).reset_index()
# funnel_by_region = df.groupby('Region').agg(Sent=('Offer_ID', 'count'), Redeemed=('Redeemed', 'sum')).reset_index()
# funnel_by_freq = df.groupby('Duration_Type').agg(Sent=('Offer_ID', 'count'), Redeemed=('Redeemed', 'sum')).reset_index()
# response_by_loyalty = df.groupby('Loyalty_Tier')['Achievement_Rate'].mean().reset_index()
# response_by_segment = df.groupby('Segment')['Achievement_Rate'].mean().reset_index()
# region_perf = df.groupby('Region')[['Redeemed', 'Achievement_Rate']].mean().reset_index()
#segment_offer = df.groupby(['Segment', 'Offer_Type'])['Achievement_Rate'].mean().reset_index()
# top_subcats = df.groupby('SubCategory2').agg(Redemption_Rate=('Redeemed', 'mean'),
#     Achievement_Rate=('Achievement_Rate', 'mean'), Redemptions=('Redeemed', 'sum')).reset_index().sort_values(by='Achievement_Rate', ascending=False).head(10)
# daily_trend = df.groupby(df['Offer_Send_Date'].dt.date).agg(Activation=('Activated', 'sum'), Redemption=('Redeemed', 'sum')).reset_index()
# weekly_trend = df.groupby(df['Offer_Send_Date'].dt.to_period('W').astype(str)).agg(Redemption=('Redeemed', 'sum')).reset_index()
# offer_summary = df.groupby('Offer_Title').agg(SubCategory=('SubCategory2', 'first'), Redemptions=('Redeemed', 'sum'), Achievement_Rate=('Achievement_Rate', 'mean')).reset_index()
# top_offers = offer_summary.sort_values(by='Achievement_Rate', ascending=False).head(5)
# bottom_offers = offer_summary.sort_values(by='Achievement_Rate').head(5)
# repeat_customers = df[df['Redeemed'] == 1].groupby('Segment')['Customer_Journey'].value_counts(normalize=True).unstack().fillna(0).reset_index()
# utilized_ratio = df['Reward_Utilized'].sum() / df['Reward_Value_USD'].sum()

# # Distribution of Unique Offers and Customers by Duration Type
offers_by_type = df.groupby('Duration_Type')['Offer_ID'].nunique().reset_index(name='Unique_Offers')
customers_by_type = df.groupby('Duration_Type')['Customer_ID'].nunique().reset_index(name='Unique_Customers')
offer_summary_freq = pd.merge(offers_by_type, customers_by_type, on='Duration_Type')
total_customers = df['Customer_ID'].nunique()
offer_summary_freq['Frequency_Rate'] = (offer_summary_freq['Unique_Customers'] / total_customers) * 100

# Calculate Reward Utilization
total_reward_value = df['Reward_Value_USD'].sum()
total_utilized_value = df[df['Redeemed'] == 1]['Reward_Value_USD'].sum()

# Avoid division by zero
reward_utilization_pct = (total_utilized_value / total_reward_value * 100) if total_reward_value > 0 else 0


# App Config
st.set_page_config(page_title="Offer Dashboard & Simulator", layout="wide")

st.markdown("""
    <style>
    /* Hide default Streamlit top padding & loading block */
    .block-container {
        padding-top: 0rem !important;
    }

    header, .stDeployButton {
        visibility: hidden;
        height: 0px;
    }

    /* Remove margin from main app wrapper */
    .main {
        padding-top: 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Read your image and encode it
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded

image_path = "app_logo.png"
encoded_image = get_base64_image(image_path)

# -------------------------------------------------
# Refined Banner Code
# -------------------------------------------------

# --- BANNER HEADER AREA ---

st.markdown("""
<style>
/* Clean padding for main container */
.block-container {
    padding-top: 0rem !important;
}

html, body {
    background-color: #0b0b0b !important;
}

header, .stDeployButton, #MainMenu, footer {
    visibility: hidden;
}

/* Optional font improvements */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

div.banner-container {
    background: linear-gradient(90deg, #0a0a0a, #111111);
    border: 1px solid #333;
    padding: 22px 28px;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Inter', sans-serif;
}

div.banner-title {
    font-size: 38px;
    font-weight: 700;
    color: #FFD700;
}

div.banner-subtext {
    font-size: 17px;
    font-weight: 400;
    margin-top: 6px;
    color: #DDDDDD;
    line-height: 1.6;
}

div.banner-subtext span {
    color: #FFD700;
    font-weight: 600;
}

img.banner-logo {
    height: 60px;
}
</style>
""", unsafe_allow_html=True)

# Injected Header with Encoded Logo
st.markdown(f"""
    <div class='banner-container'>
        <div>
            <div class='banner-title'>OfferFlow™ Analytics Hub</div>
            <div class='banner-subtext'>
                A centralized <span>AI intelligence layer</span> to analyze, simulate, and refine promotional strategy — 
                powered by <span>Generative AI insights</span> and built-in <span>quality governance</span>.
            </div>
        </div>
        <img src="data:image/png;base64,{encoded_image}" class="banner-logo">
    </div>
""", unsafe_allow_html=True)



st.markdown("""
    <style>
        /* Hide Streamlit's default white header block */
        header {visibility: hidden;}
        /* Optional: Darken the root html/body area */
        html, body {
            background-color: #1A1A1A !important;
        }
        /* Optional: Hide hamburger menu and settings */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Styles
st.markdown("""
    <style>
    /* App background */
    body {
        background-color: #0a0a0a;
        color: #f5f5f5;
    }
    
    /* Main area */
    .stApp {
        background-color: #0a0a0a;
        color: #f5f5f5;
    }

        
    /* Header */
    h1, h2, h3, h4 {
        color: #ffd700;
    }

    <style>
       <style>
            .kpi-container {
                margin-top: 10px;
                margin-bottom: 25px;
            }
            
            .kpi-card {
                background-color: #2c2c2c;
                border-radius: 10px;
                padding: 14px 18px;
                box-shadow: 0 2px 8px rgba(255, 215, 0, 0.25);
                transition: all 0.2s ease-in-out;
                min-height: 80px;
                max-width: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
            }
            
            .kpi-card h4 {
                font-size: 16px; /* increased from 13px */
                color: #FFD700;
                margin: 0 0 6px 0;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .kpi-card h2 {
                font-size: 26px; /* increased from 20px */
                color: #FFFFFF;
                margin: 0;
                font-weight: 800;
            }
            </style>
        """, unsafe_allow_html=True)


# Inject custom CSS to center the tabs, enlarge, and beautify them
st.markdown("""
<style>
/* Center tab headers */
.stTabs [role="tablist"] {
    justify-content: center;
    margin-bottom: 0.2rem;
}

/* Base tab appearance */
.stTabs [role="tab"] {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: white !important;
    background-color: #1a1a1a !important;
    border: none !important;
    border-radius: 12px 12px 0 0 !important;
    margin-right: 6px !important;
    padding: 10px 18px !important;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(255, 215, 0, 0.15);
}

/* Hover effect */
.stTabs [role="tab"]:hover {
    color: #FFD700 !important;
    background-color: #2a2a2a !important;
    border-bottom: 2px solid #FFD700 !important;
}

/* Selected tab */
.stTabs [role="tab"][aria-selected="true"] {
    color: #FFD700 !important;
    background-color: #111111 !important;
    border-bottom: 3px solid #FFD700 !important;
}

/* Reduce top and bottom spacing */
h1 {
    margin-top: 5px;
    margin-bottom: 0px;
}
h2, h3 {
    margin-top: 0px;
    margin-bottom: 10px;
}
.block-container > div:nth-child(2) {
    margin-bottom: 5px !important;
}
.stTabs {
    margin-top: -5px !important;
    margin-bottom: 0px !important;
}
.css-18e3th9 {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)



# Define tabs
tabs = st.tabs([
    "📈 Effectiveness Dashboard",
    "🛠️ Quality Control Dashboard",
    "🎛️ Offer Scenario Simulator",
    "🧠 AI Insights & Queries"
])


# ========== Tab 1: Effectiveness Dashboard ========== #
with tabs[0]:
    # st.markdown("""
    # <style>
    # label {
    #     color: white !important;
    #     font-weight: 500;
    #     font-size: 22px;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    
    # st.markdown("""
    #     <h2 style='text-align: center; color: #FFD700;'>📊 Overall Offer Effectiveness</h2>
    # """, unsafe_allow_html=True)

    st.markdown("""
<div style='text-align: left; font-weight: 600; font-size: 19px; margin-bottom: 15px;'>
    📊 Analyze end-to-end offer performance across campaigns — decompose value into baseline and uplift, assess achievement and ROI, evaluate forecast vs actual incremental impact, and spotlight top-performing offers by segment, subcategory ,frequency etc.
</div>
""", unsafe_allow_html=True)

    # 🎯 Filter Section – Compact + Stylish
# -------------------------

    # Header
    st.markdown("""<h4 style='color:#FFD700;'>🔍 Filter by Offer Attributes</h4>""", unsafe_allow_html=True)
    
    # Step 1: Unified Date Range Slider
    min_date = df["Offer_Send_Date"].min().date()
    max_date = df["Offer_Send_Date"].max().date()
    
    date_range = st.slider(
        "Select Offer Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter by date range
    filtered_df = df[
        (df["Offer_Send_Date"].dt.date >= date_range[0]) &
        (df["Offer_Send_Date"].dt.date <= date_range[1])
    ]
    
    # Step 2: Power BI-style slicers (aligned in one row)
    def render_slicer(label, column_name):
        options = ['All'] + sorted(df[column_name].dropna().unique().tolist())
        return st.selectbox(label, options, key=label)
    
    # 6 Columns horizontally aligned
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        sel_offer_type = render_slicer("Offer Type", "Offer_Type")
    with col2:
        sel_region = render_slicer("Region", "Region")
    with col3:
        sel_subcategory = render_slicer("SubCategory", "SubCategory2")
    with col4:
        sel_segment = render_slicer("Customer Segment", "Segment")
    with col5:
        sel_loyalty = render_slicer("Loyalty Tier", "Loyalty_Tier")
    with col6:
        sel_journey = render_slicer("Customer Lifecycle", "Customer_Journey")
    
    # Step 3: Apply filters
    if sel_offer_type != 'All':
        filtered_df = filtered_df[filtered_df['Offer_Type'] == sel_offer_type]
    if sel_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == sel_region]
    if sel_subcategory != 'All':
        filtered_df = filtered_df[filtered_df['SubCategory2'] == sel_subcategory]
    if sel_segment != 'All':
        filtered_df = filtered_df[filtered_df['Segment'] == sel_segment]
    if sel_loyalty != 'All':
        filtered_df = filtered_df[filtered_df['Loyalty_Tier'] == sel_loyalty]
    if sel_journey != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Journey'] == sel_journey]


    offer_period_visits = filtered_df['Offer_Period_Visited'].sum()
    unique_customers = filtered_df['Customer_ID'].nunique()
    offer_period_visit_rate = (offer_period_visits / unique_customers * 100) if unique_customers > 0 else 0
    avg_time_to_redeem = filtered_df[filtered_df['Redeemed'] == 1]['Time_to_Respond'].mean()    
        
    # KPIs
    #st.markdown("### \U0001F4CA Key Metrics")
    st.markdown("""<h4 style='color:#FFD700;'>\U0001F4CA Key Metrics</h4>""", unsafe_allow_html=True)


    # KPI Layout
    k1, k2, k3, k4 = st.columns(4)
    k5, k6, k7, k8 = st.columns(4)
    
    k1.markdown(f"<div class='kpi-card'><h4>Total Offers Sent</h4><h2>{filtered_df.shape[0]:,}</h2></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><h4>Offers Activated</h4><h2>{filtered_df['Activated'].sum():,}</h2></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><h4>Offers Redeemed</h4><h2>{filtered_df['Redeemed'].sum():,}</h2></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'><h4>Activation Rate</h4><h2>{filtered_df['Activated'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
    
    k5.markdown(f"<div class='kpi-card'><h4>Redemption Rate</h4><h2>{filtered_df['Redeemed'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
    k6.markdown(f"<div class='kpi-card'><h4>Offer Period Visit Rate</h4><h2>{offer_period_visit_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    k7.markdown(f"<div class='kpi-card'><h4>Avg Time to Redeem</h4><h2>{avg_time_to_redeem:.1f} days</h2></div>", unsafe_allow_html=True)
    k8.markdown(f"<div class='kpi-card'><h4>Incremental Revenue</h4><h2>${filtered_df['Incremental_Revenue'].sum():,.0f}</h2></div>", unsafe_allow_html=True)

    
    # ---------- Demand Decomposition Chart ----------
    decomp_cols = ["Baseline_Impact", "Promo_Lift", "Cannibalization", "Distribution_Impact", "Channel_Impact"]
    decomp_df = filtered_df[decomp_cols].sum().reset_index()
    decomp_df.columns = ['Component', 'Value']
    
    fig1 = px.bar(
        decomp_df,
        x='Component',
        y='Value',
        color='Component',
        title="📉 Offer Value Decomposition Chart",
        text='Value',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig1.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        marker_line=dict(width=0.5, color='white')
    )
    
    fig1.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='white'),
        title_font=dict(color="#FFD700", size=18),
        xaxis=dict(title='', tickfont=dict(color='white'), showgrid=False, zeroline=False),
        yaxis=dict(title='Value', tickfont=dict(color='white'), gridcolor='gray'),
        margin=dict(t=40, b=40, l=20, r=20),
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)

    # ---------- TARGET VS FORECAST INCREMENTAL ----------
    filtered_df['Month'] = pd.to_datetime(filtered_df['Offer_Send_Date']).dt.to_period('M').astype(str)
    
    # Simulate realistic incremental and forecast data
    monthly_actuals = filtered_df.groupby('Month')['Incremental_Revenue'].sum().reset_index()
    monthly_actuals['Forecasted_Incremental'] = monthly_actuals['Incremental_Revenue'] * 0.9  # Base forecast
    
    # Simulate variation (few overperforming months)
    if len(monthly_actuals) >= 3:
        monthly_actuals.loc[1, 'Forecasted_Incremental'] = monthly_actuals.loc[1, 'Incremental_Revenue'] * 1.2
        monthly_actuals.loc[3, 'Forecasted_Incremental'] = monthly_actuals.loc[3, 'Incremental_Revenue'] * 1.1
    
    # Calculate index %
    monthly_actuals["Index %"] = (
        (monthly_actuals["Incremental_Revenue"] / monthly_actuals["Forecasted_Incremental"]) - 1
    ) * 100
    monthly_actuals["Index_Label"] = monthly_actuals["Index %"].apply(lambda x: f"{x:+.1f}%")
    
    # Create grouped bar chart
    fig_monthly_trend = go.Figure()
    
    # Actual bar
    fig_monthly_trend.add_trace(go.Bar(
        x=monthly_actuals['Month'],
        y=monthly_actuals['Incremental_Revenue'],
        name='Actual Incremental',
        marker_color='lightblue'
    ))
    
    # Forecast bar
    fig_monthly_trend.add_trace(go.Bar(
        x=monthly_actuals['Month'],
        y=monthly_actuals['Forecasted_Incremental'],
        name='Forecasted Incremental',
        marker_color='orange'
    ))
    
    # Annotate index % above the higher bar for each month
    for i, row in monthly_actuals.iterrows():
        y_max = max(row['Incremental_Revenue'], row['Forecasted_Incremental'])
        label_color = "#00FF7F" if row["Index %"] >= 0 else "#FF4C4C"  # Green for positive, Red for negative
        fig_monthly_trend.add_annotation(
            x=row['Month'],
            y=y_max * 1.05,
            text=row['Index_Label'],
            showarrow=False,
            font=dict(color=label_color, size=12),
            align="center"
        )
    
    # Layout and theme styling
    fig_monthly_trend.update_layout(
        title='📈 Monthly Incremental Value vs Forecast',
        barmode='group',
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF"),
        title_font=dict(color="#FFD700", size=18),
        legend=dict(font=dict(color="#FFFFFF")),
        xaxis=dict(title='Month', title_font=dict(color="#FFFFFF"), tickfont=dict(color="#FFFFFF")),
        yaxis=dict(title='Incremental Value ($)', title_font=dict(color="#FFFFFF"), tickfont=dict(color="#FFFFFF"))
    )
    
    st.plotly_chart(fig_monthly_trend, use_container_width=True)
    
#------------------------------------------------------------------------------------------    

    def styled_chart(chart):
        st.markdown("""
            <div class="chart-container">
        """, unsafe_allow_html=True)
        st.plotly_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # def styled_chart(chart):
    #     st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    #     st.plotly_chart(chart, use_container_width=True)
    #     st.markdown("</div>", unsafe_allow_html=True)


    funnel_data = pd.DataFrame({
    'Stage': ['Sent', 'Opened', 'Activated', 'Redeemed'],
    'Count': [
        filtered_df.shape[0],
        filtered_df["Opened"].sum(),
        filtered_df["Activated"].sum(),
        filtered_df["Redeemed"].sum()
    ]
    })
    funnel_data['Percent_of_Sent'] = (funnel_data['Count'] / funnel_data['Count'][0] * 100).round(1)
    funnel_data['Formatted_Count'] = funnel_data['Count'].apply(lambda x: f"{int(x):,}")
    funnel_data['Label'] = funnel_data.apply(
        lambda row: f"{row['Stage']}– {row['Percent_of_Sent']}%", axis=1)

    # Color coding
    stage_colors = {
        "Sent": "#4C78A8",
        "Opened": "#72B7B2",
        "Activated": "#54A24B",
        "Redeemed": "#E45756"
    }
    funnel_data['Color'] = funnel_data['Stage'].map(stage_colors)
    
    # Funnel chart
    fig_funnel = px.funnel(
        funnel_data,
        x="Count",
        y="Label",
        title="Offer Funnel with Stage % of Total",
        color="Stage",
        color_discrete_map=stage_colors
    )
    fig_funnel.update_traces(textposition='inside', texttemplate='%{value:,}')
    
    fig_funnel.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )

    # ✅ Show inside styled chart box
    styled_chart(fig_funnel)

    # Weekly & Monthly Aggregation
    weekly_data = filtered_df.set_index("Offer_Send_Date").resample('W').agg({
        "Offer_ID": "count",
        "Activated": "sum",
        "Redeemed": "sum"
    }).reset_index()
    weekly_data["Activation Rate (%)"] = (weekly_data["Activated"] / weekly_data["Offer_ID"]) * 100
    weekly_data["Redemption Rate (%)"] = (weekly_data["Redeemed"] / weekly_data["Offer_ID"]) * 100
    weekly_data["Drop-Off (%)"] = weekly_data["Activation Rate (%)"] - weekly_data["Redemption Rate (%)"]

    monthly_data = filtered_df.set_index("Offer_Send_Date").resample('M').agg({
        "Offer_ID": "count",
        "Activated": "sum",
        "Redeemed": "sum"
    }).reset_index()
    monthly_data["Activation Rate (%)"] = (monthly_data["Activated"] / monthly_data["Offer_ID"]) * 100
    monthly_data["Redemption Rate (%)"] = (monthly_data["Redeemed"] / monthly_data["Offer_ID"]) * 100
    monthly_data["Drop-Off (%)"] = monthly_data["Activation Rate (%)"] - monthly_data["Redemption Rate (%)"]

    # Trend Chart with Drop-Off Lines Between Points
    #st.markdown("### \U0001F4C9 Activation vs Redemption Trend + Drop-Off Lines")
    st.markdown("""<h5 style='color:#FFD700;'>\U0001F4C9 Activation vs Redemption Trend + Drop-Off Lines</54>""", unsafe_allow_html=True)
    
    trend_view = st.radio("Select Trend View", ["Weekly", "Monthly"], horizontal=True)

    if trend_view == "Weekly":
        trend_data = weekly_data
        x_axis = "Offer_Send_Date"
    else:
        trend_data = monthly_data
        x_axis = "Offer_Send_Date"

    fig = go.Figure()

    # Activation and Redemption Lines
    fig.add_trace(go.Scatter(
        x=trend_data[x_axis],
        y=trend_data["Activation Rate (%)"],
        mode='lines+markers',
        name='Activation Rate (%)',
        line=dict(color='#00CC96')
    ))
    fig.add_trace(go.Scatter(
        x=trend_data[x_axis],
        y=trend_data["Redemption Rate (%)"],
        mode='lines+markers',
        name='Redemption Rate (%)',
        line=dict(color='#FFA15A')
    ))

    # Drop-off Connectors with Labels toward top-center
    for i in range(len(trend_data)):
        x_val = trend_data[x_axis][i]
        act_y = trend_data["Activation Rate (%)"][i]
        red_y = trend_data["Redemption Rate (%)"][i]
        drop_val = trend_data['Drop-Off (%)'][i]
        mid_y = max(act_y, red_y) + 1.5  # move label just above the higher point

        fig.add_trace(go.Scatter(
            x=[x_val, x_val],
            y=[act_y, red_y],
            mode='lines+text',
            line=dict(color='red', dash='dot', width=1),
            showlegend=False,
            text=[None, f"{drop_val:.1f}%"],
            textposition="top center",
            textfont=dict(size=10),
            hoverinfo='text'
        ))

    fig.update_layout(
        #title="Activation vs Redemption Rates with Drop-Off Connectors",
        xaxis=dict(title="Date",title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")),
        yaxis=dict(title="Rate (%)",title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")),
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF"),
        #title_font=dict(color="#FFD700", size=18),
        legend=dict(font=dict(color="#FFFFFF")),
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

#------------------------------------Offer Period Visit v/s Redemptions----------------------------------------------------------------------    

    filtered_df["Week_Start"] = pd.to_datetime(filtered_df["Offer_Send_Date"])
    filtered_df["Week_Label"] = filtered_df["Week_Start"].dt.to_period("W").apply(lambda r: r.start_time.strftime("%b %d"))
    
    # Step 2: Redeemed status flag
    #filtered_df["Is_Redeemed"] = filtered_df["Offer_Status"].apply(lambda x: 1 if str(x).strip().lower() == "redeemed" else 0)
    
    # Step 3: Weekly aggregation
    weekly_trend = filtered_df.groupby("Week_Label").agg({
        "Offer_Period_Visited": "sum",
        "Offer_ID": "nunique",
        "Redeemed": "sum"
    }).reset_index()
    
    # Step 4: Metric calculations
    weekly_trend["Visit Rate (%)"] = (weekly_trend["Offer_Period_Visited"] / weekly_trend["Offer_ID"]) * 100
    weekly_trend["Achievement Rate (%)"] = (weekly_trend["Redeemed"] / weekly_trend["Offer_ID"]) * 100
    
    # Step 5: Plot
    fig = go.Figure()
    
    # Visit Rate trace
    fig.add_trace(go.Scatter(
        x=weekly_trend["Week_Label"],
        y=weekly_trend["Visit Rate (%)"],
        mode="lines+markers+text",
        name="Visit Rate (%)",
        line=dict(color="#F8766D", width=3),
        marker=dict(size=6),
        text=[f"{x:.0f}%" for x in weekly_trend["Visit Rate (%)"]],
        textposition="top center",
        hovertemplate="%{x}<br>Visit Rate: %{y:.1f}%"
    ))
    
    # Achievement Rate trace
    fig.add_trace(go.Scatter(
        x=weekly_trend["Week_Label"],
        y=weekly_trend["Achievement Rate (%)"],
        mode="lines+markers+text",
        name="Achievement Rate (%)",
        line=dict(color="#00BA38", width=3, dash='dot'),
        marker=dict(size=6),
        text=[f"{x:.0f}%" for x in weekly_trend["Achievement Rate (%)"]],
        textposition="top center",
        hovertemplate="%{x}<br>Achievement Rate: %{y:.1f}%"
    ))
    
    # Layout
    fig.update_layout(
        title="📈 Weekly Trend: Visit Rate vs Achievement Rate",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF"),
        title_font=dict(color="#FFD700", size=18),
        legend=dict(font=dict(color="#FFFFFF")),
        xaxis=dict(
            title="Week (by Month)",
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            tickangle=45
        ),
        yaxis=dict(
            title="Rate (%)",
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")
        ),
        height=480
    )
    
    # Show chart
    st.plotly_chart(fig, use_container_width=True)



    # Step 1: Create Reward Bands
    filtered_df["Reward_Band"] = pd.cut(
        filtered_df["Reward_Value_USD"],
        bins=[0, 1, 2, 3, 4,  float('inf')],
        labels=["$0-1", "$1-2", "$2-3", "$3-5", "$4+"],
        include_lowest=True
    )
    
    # Step 2: Aggregate achievement rate by reward band
    reward_summary = filtered_df.groupby("Reward_Band").agg({
        "Offer_ID": "count",
        "Redeemed": "sum"
    }).reset_index()
    
    reward_summary["Achievement Rate (%)"] = (reward_summary["Redeemed"] / reward_summary["Offer_ID"]) * 100
    
    # Step 3: Plot bar chart
    fig_reward_ach = go.Figure()
    fig_reward_ach.add_trace(go.Bar(
        x=reward_summary["Reward_Band"],
        y=reward_summary["Achievement Rate (%)"],
        name="Achievement Rate (%)",
        marker_color="#6baed6",
        text=[f"{val:.1f}%" for val in reward_summary["Achievement Rate (%)"]],
        textposition="outside"
    ))
    
    # Step 4: Layout customization
    fig_reward_ach.update_layout(
        title="🎯 Achievement Rate by Reward Value",
        xaxis=dict(title="Reward Value (USD Band)",title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")),
        yaxis=dict(title="Achievement Rate (%)",title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")),
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),
        title_font=dict(color="#FFD700", size=18),
        legend=dict(font=dict(color="#FFFFFF")),
        height=450
    )
    
    # Step 5: Show in Streamlit
    #st.plotly_chart(fig_reward_ach, use_container_width=True)

    # Step 1: Preprocessing
    df['Offer_Send_Date'] = pd.to_datetime(df['Offer_Send_Date'])
    df['DayOfWeek'] = df['Offer_Send_Date'].dt.day_name()
    
    # Filter only offers with final status (Redeemed / Expired)
    #valid_statuses = ['Redeemed', 'Expired']
    filtered = df #[df['Offer_Status'].isin(valid_statuses)]
    
    # Step 2: Group by day of the week
    dow_summary = (
        filtered.groupby('DayOfWeek')
        .agg(
            Offers_Sent=('Offer_ID', 'nunique'),
            Achieved=('Redeemed', 'sum')
        )
        .reset_index()
    )
    
    # Step 3: Calculate achievement rate
    dow_summary['Achievement_Rate (%)'] = round((dow_summary['Achieved'] / dow_summary['Offers_Sent']) * 100, 2)
    
    # Sort by weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_summary['DayOfWeek'] = pd.Categorical(dow_summary['DayOfWeek'], categories=weekday_order, ordered=True)
    dow_summary = dow_summary.sort_values('DayOfWeek')
    
    # Step 4: Plot combo chart
    fig_combo = go.Figure()
    
    # Bar - Offers Sent
    fig_combo.add_trace(go.Bar(
        x=dow_summary['DayOfWeek'],
        y=dow_summary['Offers_Sent'],
        name='Offers Sent',
        marker_color='#6BAED6',
        yaxis='y1'
    ))
    
    # Line - Achievement Rate
    fig_combo.add_trace(go.Scatter(
        x=dow_summary['DayOfWeek'],
        y=dow_summary['Achievement_Rate (%)'],
        name='Achievement Rate (%)',
        mode='lines+markers+text',
        text=[f"{v}%" for v in dow_summary['Achievement_Rate (%)']],
        textposition='top center',
        yaxis='y2',
        line=dict(width=3, color='#FFD700'),
        marker=dict(size=8)
    ))
    
    # Step 5: Layout styling
    fig_combo.update_layout(
        title='Offer Sent vs Achievement Rate by Day of Week',
        xaxis=dict(title='Day of Week',title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")),
        yaxis=dict(title='Offers Sent', showgrid=False,title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF")),
        yaxis2=dict(title='Achievement Rate (%)', overlaying='y', side='right'),
        barmode='group',
        height=450,
        plot_bgcolor='#1A1A1A',
        paper_bgcolor='#1A1A1A',
        font=dict(color='#FFFFFF'),
        title_font=dict(color="#FFD700", size=18),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    
    #st.plotly_chart(fig_combo, use_container_width=True)

     #styled_chart(fig_region)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_reward_ach, use_container_width=True) # Offer Type
    with c2:
        st.plotly_chart(fig_combo, use_container_width=True)



    
    
    # Step 1: Aggregate data
    offer_perf = filtered_df.groupby('Offer_Type').agg(
        Offers_Sent=('Offer_ID', 'count'),
        Redeemed_offers=('Redeemed', 'sum')
    ).reset_index()

    
    offer_perf["Achievement_Rate"] = (offer_perf["Redeemed_offers"] / offer_perf["Offers_Sent"]) * 100
    
    # Step 2: Format labels
    offer_perf['Offer Type'] = offer_perf['Offer_Type'].str.replace('_', ' ')
    
    # Step 3: Create dual-axis figure
    fig_combo = go.Figure()
    
    # Bar trace – Offers Sent
    fig_combo.add_trace(go.Bar(
        x=offer_perf['Offer Type'],
        y=offer_perf['Offers_Sent'],
        name='Offers Sent',
        marker=dict(color='#6BAED6'),
        yaxis='y1'
    ))
    
    # Line trace – Achievement Rate
    fig_combo.add_trace(go.Scatter(
        x=offer_perf['Offer Type'],
        y=offer_perf['Achievement_Rate'] * 100,
        name='Achievement Rate (%)',
        mode='lines+markers+text',
        text=[f"{val:.1f}%" for val in offer_perf['Achievement_Rate']],
        textposition='top center',
        marker=dict(color='#F58518', size=8),
        line=dict(width=3),
        yaxis='y2'
    ))
    
    # Step 4: Define layout with dual axes
    fig_combo.update_layout(
        title='Performance by Offer Type',
        xaxis=dict(title='Offer Type'),
        yaxis=dict(
            title='Offers Sent',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title='Achievement Rate (%)',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
        margin=dict(t=40, b=40),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#ffffff',
        font=dict(size=12),
        height=450
    )

    fig_combo.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )

    
    # Step 5: Display in styled chart box
    #styled_chart(fig_combo)

    # Step 1: Aggregate data
    region_perf = filtered_df.groupby('Region').agg(
        Offers_Sent=('Offer_ID', 'count'),
        Redeemed_offers=('Redeemed', 'sum')
    ).reset_index()

    region_perf["Achievement_Rate"] = (region_perf["Redeemed_offers"] / region_perf["Offers_Sent"]) * 100
    
    # Step 2: Format labels
    region_perf['Region'] = region_perf['Region'].str.replace('_', ' ')
    
    # Step 3: Create dual-axis chart
    fig_region = go.Figure()
    
    # Bar – Offers Sent
    fig_region.add_trace(go.Bar(
        x=region_perf['Region'],
        y=region_perf['Offers_Sent'],
        name='Offers Sent',
        marker=dict(color='#6BAED6'),
        yaxis='y1'
    ))
    
    # Line – Achievement Rate
    fig_region.add_trace(go.Scatter(
        x=region_perf['Region'],
        y=region_perf['Achievement_Rate'] * 100,
        name='Achievement Rate (%)',
        mode='lines+markers+text',
        text=[f"{val:.1f}%" for val in region_perf['Achievement_Rate']],
        textposition='top center',
        marker=dict(color='#FD8D3C', size=8),
        line=dict(width=3),
        yaxis='y2'
    ))
    
    # Step 4: Layout styling
    fig_region.update_layout(
        title='Performance by Region',
        xaxis=dict(title='Region'),
        yaxis=dict(
            title='Offers Sent',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title='Achievement Rate (%)',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
        margin=dict(t=40, b=40),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#ffffff',
        font=dict(size=12),
        height=450
    )

    fig_region.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )

    
    # Step 5: Render chart in styled container
    
    #styled_chart(fig_region)
    c1, c2 = st.columns(2)
    with c1:
        styled_chart(fig_combo)  # Offer Type
    with c2:
        styled_chart(fig_region)

    # Combo Chart: Unique Offers vs. Customers by Frequency
    fig_combo = go.Figure()
    fig_combo.add_trace(go.Bar(
        x=offer_summary_freq['Duration_Type'],
        y=offer_summary_freq['Unique_Offers'],
        name='Unique Offers',
        marker_color='#6BAED6',
        yaxis='y1'
    ))
    fig_combo.add_trace(go.Scatter(
        x=offer_summary_freq['Duration_Type'],
        y=offer_summary_freq['Unique_Customers'],
        name='Unique Customers',
        yaxis='y2',
        mode='lines+markers+text',
        text=[f"{v:,}" for v in offer_summary_freq['Unique_Customers']],
        textposition='top center',
        marker=dict(color='#e6550d', size=8),
        line=dict(width=3)
    ))
    fig_combo.update_layout(
        title='Offer Assignment by Frequency Type',
        xaxis=dict(title='Offer Frequency'),
        yaxis=dict(title='Unique Offers', showgrid=False),
        yaxis2=dict(title='Unique Customers', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#ffffff',
        height=450
    )

    fig_combo.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=18),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )

    

    #styled_chart(fig_combo)
    
    # Pie/Donut Chart for Frequency Rate
    fig_pie = px.pie(
        offer_summary_freq,
        names='Duration_Type',
        values='Frequency_Rate',
        title='Frequency Rate Distribution among Customers',
        hole=0.4
    )

    fig_pie.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )
    
    #styled_chart(fig_pie)

    c1, c2 = st.columns(2)
    with c1:
        styled_chart(fig_pie)  # Offer Type
    with c2:
        styled_chart(fig_combo)

        # Step 1: Aggregate data
    freq_perf = filtered_df.groupby('Duration_Type').agg(
        Offers_Sent=('Offer_ID', 'count'),
        Redeemed_offers=('Redeemed', 'sum')
    ).reset_index()

    freq_perf["Achievement_Rate"] = (freq_perf["Redeemed_offers"] / freq_perf["Offers_Sent"]) * 100
    
    # Step 2: Format labels (optional cleanup if needed)
    freq_perf['Duration_Type'] = freq_perf['Duration_Type'].str.replace('_', ' ')
    
    # Step 3: Create dual-axis chart
    fig_freq = go.Figure()
    
    # Bar – Offers Sent
    fig_freq.add_trace(go.Bar(
        x=freq_perf['Duration_Type'],
        y=freq_perf['Offers_Sent'],
        name='Offers Sent',
        marker=dict(color='#74C476'),
        yaxis='y1'
    ))
    
    # Line – Achievement Rate
    fig_freq.add_trace(go.Scatter(
        x=freq_perf['Duration_Type'],
        y=freq_perf['Achievement_Rate'] * 100,
        name='Achievement Rate (%)',
        mode='lines+markers+text',
        text=[f"{val:.1f}%" for val in freq_perf['Achievement_Rate']],
        textposition='top center',
        marker=dict(color='#E6550D', size=8),
        line=dict(width=3),
        yaxis='y2'
    ))

    # Step 4: Layout styling
    fig_freq.update_layout(
        title=dict(
            text='Performance by Frequency',
            font=dict(color='#FFD700', size=18)
        ),
        xaxis=dict(
            title=dict(text='Offer Frequency', font=dict(color='#FFFFFF')),
            tickfont=dict(color='#FFFFFF'),
            gridcolor='#333333',
            zerolinecolor='#444444'
        ),
        yaxis=dict(
            title=dict(text='Offers Sent', font=dict(color='#FFFFFF')),
            tickfont=dict(color='#FFFFFF'),
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text='Achievement Rate (%)', font=dict(color='#FFFFFF')),
            tickfont=dict(color='#FFFFFF'),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='right',
            x=1,
            font=dict(color='#FFFFFF')
        ),
        margin=dict(t=40, b=40),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(size=12, color='#FFFFFF'),
        height=450
    )

   
    # Step 5: Render chart in styled container
    #styled_chart(fig_freq)

    # Step 1: Aggregate data
    journey_perf = filtered_df.groupby('Customer_Journey').agg(
        Offers_Sent=('Offer_ID', 'count'),
        Redeemed_offers=('Redeemed', 'sum')
    ).reset_index()

    journey_perf["Achievement_Rate"] = (journey_perf["Redeemed_offers"] / journey_perf["Offers_Sent"]) * 100

    # Step 2: Clean labels (optional if underscores exist)
    journey_perf['Customer_Journey'] = journey_perf['Customer_Journey'].str.replace('_', ' ')
    
    # Step 3: Create dual-axis combo chart
    fig_journey = go.Figure()
    
    # Bar – Offers Sent
    fig_journey.add_trace(go.Bar(
        x=journey_perf['Customer_Journey'],
        y=journey_perf['Offers_Sent'],
        name='Offers Sent',
        marker=dict(color='#9ECAE1'),
        yaxis='y1'
    ))
    
    # Line – Achievement Rate
    fig_journey.add_trace(go.Scatter(
        x=journey_perf['Customer_Journey'],
        y=journey_perf['Achievement_Rate'] * 100,
        name='Achievement Rate (%)',
        mode='lines+markers+text',
        text=[f"{val:.1f}%" for val in journey_perf['Achievement_Rate'] * 100],
        textposition='top center',
        marker=dict(color='#F16913', size=8),
        line=dict(width=3),
        yaxis='y2'
    ))
    
    # Step 4: Layout formatting
    fig_journey.update_layout(
        title='Performance by Customer Lifecycle',
        xaxis=dict(title='Customer Journey'),
        yaxis=dict(
            title='Offers Sent',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title='Achievement Rate (%)',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
        margin=dict(t=40, b=40),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#ffffff',
        font=dict(size=12),
        height=450
    )

    fig_journey.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )

    
    # Step 5: Render chart
    #styled_chart(fig_journey)

    c1, c2 = st.columns(2)
    with c1:
        styled_chart(fig_freq) 
    with c2:
        styled_chart(fig_journey)




      # Step 1: Aggregate data
    segment_offer = filtered_df.groupby(['Segment','Offer_Type']).agg(
        Offers_Sent=('Offer_ID', 'count'),
        Redeemed_offers=('Redeemed', 'sum')
    ).reset_index()

    segment_offer["Achievement_Rate"] = (segment_offer["Redeemed_offers"] / segment_offer["Offers_Sent"]) * 100

    # Step 2: Clean labels (optional if underscores exist)
    segment_offer['Customer_Journey'] = journey_perf['Customer_Journey'].str.replace('_', ' ')
    
    
    # Step 2: Create heatmap
    fig_segment_offer = px.density_heatmap(
        segment_offer,
        x="Offer_Type",
        y="Segment",
        z="Achievement_Rate",
        color_continuous_scale="YlOrRd",
        title="Segment-Offer Type Heatmap",
        labels={
        "Offer_Type": "Offer Type",
        "Segment": "Customer Segment",
        "Achievement_Rate_Pct": "Achievement Rate (%)"
    }
    )
    
    # Step 3: Layout and legend styling
    fig_segment_offer.update_layout(
        font=dict(size=12),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#ffffff",
        margin=dict(t=40, l=20, r=20, b=20),
        coloraxis_colorbar=dict(
            title="Achievement Rate (%)",
            ticksuffix="%",
            showticksuffix="all"
        )
    )

    fig_segment_offer.update_layout(
        plot_bgcolor="#1A1A1A",       # Match your black theme
        paper_bgcolor="#1A1A1A",
        font=dict(color="#FFFFFF", size=13),  # Brighter white text
        title_font=dict(color="#FFD700", size=18),  # Golden title
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        yaxis=dict(
            title_font=dict(color="#FFFFFF"),
            tickfont=dict(color="#FFFFFF"),
            gridcolor="#333333",
            zerolinecolor="#444444"
    ),
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="#1A1A1A"
        )
    )

    
    # Step 4: Render inside styled card
    styled_chart(fig_segment_offer)

    # Reuse the prepared data from earlier
    subcat_perf = filtered_df.groupby("SubCategory2").agg({
        "Offer_ID": "nunique",
        "Achievement_Rate": "sum"
    }).reset_index()
    subcat_perf["Achievement Rate (%)"] = (subcat_perf["Achievement_Rate"] / subcat_perf["Offer_ID"]) * 100
    # subcat_perf = subcat_perf.sort_values("Achievement Rate (%)", ascending=True)
    
    # Plotly Horizontal Bar with Gradient Fill
    fig = px.bar(
        subcat_perf,
        x="Achievement Rate (%)",
        y="SubCategory2",
        orientation="h",
        text=subcat_perf["Achievement Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        color="Achievement Rate (%)",  # Gradient based on value
        color_continuous_scale=px.colors.sequential.YlOrBr  # EY-styled gradient
    )
    
    # Styling
    fig.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="#222", width=1))
    )
    
    fig.update_layout(
        title="🎯 Achievement Rate by Subcategory",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="white"),
        title_font=dict(color="#FFD700", size=18),
        coloraxis_showscale=False,
        xaxis=dict(title="Achievement Rate (%)", title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(title="Subcategory", title_font=dict(color="white"), tickfont=dict(color="white")),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("""<h4 style='color:#FFD700;'>🩺 Offer Diagnostics</h4>""", unsafe_allow_html=True)
    # Filter for ratings
   
    rating_filter = st.multiselect("🎯 Select Ratings to View",
                                   options=["Very Good", "Good", "Average", "Below Average", "Poor"],
                                   default=["Very Good", "Good", "Average", "Below Average", "Poor"])
    
    filtered_df_ = combinations[combinations['Rating'].isin(rating_filter)]
    
    
    # Layout: Donut + Table in one row
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = px.pie(
            filtered_df_,
            names="Rating",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.YlOrBr,
            title="Offer Distribution by Rating"
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        fig.update_layout(
            paper_bgcolor='#111111',
            plot_bgcolor='#111111',
            font=dict(color="white"),
            title_font=dict(size=16, color="#f2f2f2"),
            legend=dict(font=dict(color="#f2f2f2"))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        #st.markdown("##### 📋 Offer Ratings Table")
        st.markdown("""<h5 style='color:#FFD700;'>📋 Offer Ratings Table</h5>""", unsafe_allow_html=True)
        styled_df = filtered_df_[['Offer Title', 'Subcategory', 'Rating']].style\
            .set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#222222'), ('color', 'white')]},
                {'selector': 'tbody tr:nth-child(even) td', 'props': [('background-color', '#1c1c1c'), ('color', 'white')]},
                {'selector': 'tbody tr:nth-child(odd) td', 'props': [('background-color', '#2a2a2a'), ('color', 'white')]},
                {'selector': 'tbody td', 'props': [('border', '1px solid #444')]}
            ])\
            .hide(axis='index')
        st.dataframe(styled_df, use_container_width=True)

    
    
        # Recalculate top/bottom offers with all required fields
    offer_summary = filtered_df.groupby('Offer_Title').agg(
        SubCategory=('SubCategory2', 'first'),
        Offers_Sent=('Offer_ID', 'count'),
        Offers_Redeemed=('Redeemed', 'sum'),
        Offers_Activated=('Activated', 'sum'),
        Incremental_Value=('Incremental_Revenue', 'sum')
        # or sum if you want total cost
    ).reset_index()
    
    
    # Format percentage columns
    offer_summary["Activation_Rate"] = (offer_summary["Offers_Activated"] / offer_summary["Offers_Sent"]) * 100
    offer_summary["Achievement_Rate"] = (offer_summary["Offers_Redeemed"] / offer_summary["Offers_Sent"]) * 100

    
    # Top and bottom 5 offers
    top_offers = offer_summary.sort_values(by='Achievement_Rate', ascending=False).head(10)
    bottom_offers = offer_summary.sort_values(by='Achievement_Rate').head(10)
    
    # Clean column names
    col_map = {
        "Offer_Title": "Offer Title",
        "SubCategory": "SubCategory",
        "Offers_Sent": "# Offers Sent",
        "Offers_Activated": "# Activated",
        "Offers_Redeemed": "# Redeemed",
        "Activation_Rate": "Activation Rate (%)",
        "Achievement_Rate": "Achievement Rate (%)",
        "Incremental_Value": "Incremental Value ($)"
    }
    
    top_offers.rename(columns=col_map, inplace=True)
    bottom_offers.rename(columns=col_map, inplace=True)
    

    # Define custom styling for dark theme tables
    table_style = lambda df: df.style.set_properties(**{
        'color': '#FFFFFF',
        'background-color': '#1e1e1e',
        'border-color': '#444444',
        'text-align': 'left',
        'padding': '8px'
    }).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#FFD700'),
                ('background-color', '#FFD700'),  # Yellow header
                ('color', '#000000'),
                ('font-weight', 'bold'),
                ('padding', '10px'),
                ('border', '1px solid #444')
            ]
        },
        {
            'selector': 'td',
            'props': [('border', '1px solid #333')]
        }
    ]).apply(
        lambda _: ['background-color: #2a2a2a' if i % 2 == 0 else 'background-color: #1a1a1a' for i in range(len(df))],
        axis=0
    )
    
    # Display updated tables
    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    
    #st.markdown("#### 🏆 Top Performing Offers")
    st.markdown("""<h5 style='color:#FFD700;'>🏆 Top Performing Offers</h5>""", unsafe_allow_html=True)
    st.dataframe(table_style(top_offers), use_container_width=True)
    
    #st.markdown("#### ❌ Bottom Performing Offers")
    st.markdown("""<h5 style='color:#FFD700;'>❌ Bottom Performing Offers</h5>""", unsafe_allow_html=True)
    st.dataframe(table_style(bottom_offers), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)



    
#--------------------------QUALITY CONTROL DASHBOARD CODE-----------------------------------------------------------------------------------

with tabs[1]:
    st.markdown("""
            <div style='text-align: left; font-weight: 600; font-size: 19px; margin-bottom: 15px;'>
                ⚙️ Highlights whether the engine is running as intended by surfacing breaches in key operational thresholds. It provides a diagnostic layer that checks if offers were sent to the right segments, in the right time windows, with the right generosity levels — and flags any inconsistencies.
            </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Set white text for labels */
        label, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1y4p8pa {
            color: white !important;
        }
    
        /* Set input box background and text color */
        .stSelectbox, .stMultiselect, .stDateInput {
            background-color: #1A1A1A !important;
            color: white !important;
        }
    
        /* Force dropdown text color */
        .st-emotion-cache-1n76uvr, .st-emotion-cache-1dj0hjr {
            color: white !important;
        }
    
        /* Optional: fix streamlit scrollbar/thumb color for black background */
        ::-webkit-scrollbar-thumb {
            background-color: #FFD700;
        }
    
        </style>
    """, unsafe_allow_html=True)


    # Simulated Migration Data
    # ------------------------
    migration_df = pd.DataFrame({
        "Month": pd.date_range(start="2024-01-01", end="2024-06-01", freq="MS"),
        "New_Acquisition": [19.2, 25.4, 16.8, 27.1, 20.3, 22.6],     # Early → In-Life
        "Reactivation": [10.5, 13.2, 12.1, 15.8, 14.0, 11.9],        # Lapsed → In-Life
        "Lapsed": [9.1, 8.7, 10.9, 9.8, 7.4, 11.2]                   # In-Life → Lapsed
    })
    migration_df["Month_Str"] = migration_df["Month"].dt.strftime("%b")
    
    # ------------------------
    # Compute KPIs
    # ------------------------
    avg_new = migration_df["New_Acquisition"].mean()
    avg_reactivate = migration_df["Reactivation"].mean()
    avg_lapsed = migration_df["Lapsed"].mean()

    st.markdown("<h4 style='color:#FFD700;'>🔁 Customer Journey Migration Trends</h4>", unsafe_allow_html=True)
    kpi_col, chart_col = st.columns([1, 2])
    
    # Styled KPI Cards
    with kpi_col:
        def render_kpi(label, value, color):
            st.markdown(f"""
                <div style='background-color:#1A1A1A;padding:15px 10px;border-radius:12px;text-align:center;margin-bottom:15px;box-shadow:0 0 8px {color};'>
                    <h4 style='color:white;margin-bottom:5px'>{label}</h4>
                    <h2 style='color:{color};font-size:36px;font-weight:700;'>{value:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
    
        render_kpi("New Acquisitions", avg_new, "#00FFAA")
        render_kpi("Reactivated Customers", avg_reactivate, "#00AAFF")
        render_kpi("Lapsed Customers", avg_lapsed, "#FF4C4C")
    
    # Line Chart with All Labels in White
    with chart_col:
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=migration_df["Month_Str"], y=migration_df["New_Acquisition"],
            mode="lines+markers+text", name="New Acquisitions",
            line=dict(color="#00FFAA", width=3),
            marker=dict(color="#00FFAA", size=8),
            text=[f"{v:.1f}%" for v in migration_df["New_Acquisition"]],
            textfont=dict(color="white", size=12),
            textposition="top center"
        ))
        fig.add_trace(go.Scatter(
            x=migration_df["Month_Str"], y=migration_df["Reactivation"],
            mode="lines+markers+text", name="Reactivated",
            line=dict(color="#00AAFF", width=3),
            marker=dict(color="#00AAFF", size=8),
            text=[f"{v:.1f}%" for v in migration_df["Reactivation"]],
            textfont=dict(color="white", size=12),
            textposition="top center"
        ))
        fig.add_trace(go.Scatter(
            x=migration_df["Month_Str"], y=migration_df["Lapsed"],
            mode="lines+markers+text", name="Lapsed",
            line=dict(color="#FF4C4C", width=3),
            marker=dict(color="#FF4C4C", size=8),
            text=[f"{v:.1f}%" for v in migration_df["Lapsed"]],
            textfont=dict(color="white", size=12),
            textposition="top center"
        ))
    
        fig.update_layout(
            title="Monthly Migration Trends (%)",
            title_font=dict(color="#FFD700", size=18),
            plot_bgcolor="#1A1A1A",
            paper_bgcolor="#1A1A1A",
            font=dict(color="white"),
            legend=dict(orientation="h", y=-0.25, font=dict(color="white", size=12)),
            height=520,
            margin=dict(t=40, b=20),
            xaxis=dict(title="Month", tickfont=dict(color="white", size=12), title_font=dict(color="white")),
            yaxis=dict(title="%", tickfont=dict(color="white", size=12), title_font=dict(color="white"))
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
   
    
    st.markdown("""<h4 style='color:#FFD700;'>📋Quality Metrics by Customer LifeCycle</h4>""", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        journey_filter = st.multiselect("Customer LifeCycle", options=qc_df['Journey'].unique(), default=qc_df['Journey'].unique())
    with col2:
        date_range = st.date_input("Select Date Range", value=[qc_df['Date'].min(), qc_df['Date'].max()])
        
        # date_range = st.slider(
        #     "Select Date Range",
        #     min_value=qc_df['Date'].min(),
        #     max_value=qc_df['Date'].max(),
        #     value=(qc_df['Date'].min(), qc_df['Date'].max()),
        # )
    
    filtered_df = qc_df[(qc_df['Journey'].isin(journey_filter)) &
                        (qc_df['Date'] >= pd.to_datetime(date_range[0])) &
                        (qc_df['Date'] <= pd.to_datetime(date_range[1]))]

    
       
    # KPI CARDS
    
    kpi_cols = st.columns(3)
    journeys = ['Early Life', 'In-Life', 'Lapsed']
    
    for i, journey in enumerate(journeys):
        if journey in journey_filter:
            avg_score = filtered_df[filtered_df['Journey'] == journey][[col for col in qc_df.columns if col.startswith('QC_')]].mean().mean()
            if avg_score > 90:
                insight = "✅ Working Well"
            elif avg_score > 70:
                insight = "⚠ Can Be Reviewed"
            else:
                insight = "❌ Needs Attention"
            kpi_cols[i].markdown(f"""
                <div style='background-color:#2c2c2c;padding:10px;border-radius:7px;height: 170px;text-align:center;'>
                    <h4 style='color:white;'>{journey}</h4>
                    <h2 style='color:white;'>{avg_score:.1f}%</h2>
                    <p style='color:#FFD700;text-align:center;'>{insight}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            kpi_cols[i].markdown(f"""
                <div style='background-color:#2c2c2c;padding:10px;border-radius:10px;text-align:center;'>
                    <h4 style='color:white;'>{journey}</h4>
                    <h2 style='color:white;'>-</h2>
                    <p style='color:#FFD700;'>No Data</p>
                </div>
            """, unsafe_allow_html=True)
    
    # TREND CHARTS
    st.markdown("""<h4 style='color:#FFD700;'>📊 Offers Sent & Customers (Weekly/Monthly)</h4>""", unsafe_allow_html=True)
    resample_type = st.radio("Resample By", options=["Weekly", "Monthly"], horizontal=True)
    
    if resample_type == "Weekly":
        trend_df = filtered_df.copy()
        trend_df['Period'] = trend_df['Date'].dt.to_period('W').dt.start_time
    else:
        trend_df = filtered_df.copy()
        trend_df['Period'] = trend_df['Date'].dt.to_period('M').dt.to_timestamp()
    
    agg_trend = trend_df.groupby('Period').agg({
        'Offers_Sent': 'sum',
        'Unique_Customers': 'sum'
    }).reset_index()
    
    col3, col4 = st.columns(2)
    
    fig_offers = go.Figure()
    fig_offers.add_trace(go.Scatter(
        x=agg_trend['Period'],
        y=agg_trend['Offers_Sent'],
        mode='lines+markers+text',
        name="Offers Sent",
        text=[f"{x/1000:.0f}K" for x in agg_trend['Offers_Sent']],
        textposition="top center"
    ))
    fig_offers.update_layout(
        title="Total Offers Sent",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="white"),
        title_font=dict(color="#FFD700"),
        xaxis=dict(tickformat="%b %Y"),
        height=350
    )
    col3.plotly_chart(fig_offers, use_container_width=True)
    
    fig_cust = go.Figure()
    fig_cust.add_trace(go.Scatter(
        x=agg_trend['Period'],
        y=agg_trend['Unique_Customers'],
        mode='lines+markers+text',
        name="Unique Customers",
        text=[f"{x/1000:.0f}K" for x in agg_trend['Unique_Customers']],
        textposition="top center"
    ))
    fig_cust.update_layout(
        title="Unique Customers Targeted",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="white"),
        title_font=dict(color="#FFD700"),
        xaxis=dict(tickformat="%b %Y"),
        height=350
    )
    col4.plotly_chart(fig_cust, use_container_width=True)
    
    # GAUGE CHARTS
    from plotly.subplots import make_subplots
    metric_map = {
        'QC_Generosity_Utilization': "% of Offer Cost Utilized",
        'QC_Timing_Compliance': "% Offers Sent on Time",
        'QC_Eligibility_Adherence': "% Offers to Eligible Customers",
        'QC_Reactivation_Correctness': "% Lapsed Re-activation Correct",
        'QC_Consistency_Weekly': "% Weekly Offer Consistency",
        'QC_Coverage_Saturation': "% Customer Base Covered",
        'QC_Journey_Mismatch_Error': "% Offers Misaligned",
        'QC_Collision_Resolved': "% Conflicts Resolved"
    }
    
    st.markdown("""<h4 style='color:#FFD700;'>🔍 Metric Quality Gauges</h4>""", unsafe_allow_html=True)
    gauge_cols = st.columns(2)
    
    for i, (metric, label) in enumerate(metric_map.items()):
        val = filtered_df[metric].mean()
        color = "#28a745" if val > 90 else ("#ffc107" if val > 70 else "#dc3545")
        if val > 90:
            insight = "✅ On Target"
        elif val > 70:
            insight = "⚠ Review Suggested"
        else:
            insight = "❌ Urgent Attention"
    
        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': f"<b>{label}</b>", 'font': {'size': 14, 'color': 'white'}},
        number={
            'suffix': "%",
            'font': {'color': 'white', 'size': 22},  # Smaller size
            'valueformat': '.1f'
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickwidth': 1, 'ticklen': 6},
            'bar': {'color': color, 'thickness': 0.2},
            'bgcolor': "#1A1A1A",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': "#8B0000"},
                {'range': [70, 90], 'color': "#FFD700"},
                {'range': [90, 100], 'color': "#28a745"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': val
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}
        ))
    
        fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        height=280,
        font=dict(color="white"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
    )
        with gauge_cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<p style='color:#FFD700;text-align:center;'>{insight}</p>", unsafe_allow_html=True)

  
    



# ========== Tab 2: Offer Simulator ========== #
with tabs[2]:
    st.markdown("""
            <div style='text-align: left; font-weight: 600; font-size: 19px; margin-bottom: 15px;'>
                🔮 Design, configure, and test offer strategies in real time with our simulation engine — forecast performance metrics, understand cost-benefit tradeoffs, and make informed campaign decisions using data-driven projections.
            </div>
    """, unsafe_allow_html=True)

    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []

    st.markdown("""<h4 style='color:#FFD700;'>⚙️ Configure & Setup Offer Attributes</h4>""", unsafe_allow_html=True)
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("Start Date", df['Offer_Send_Date'].min().date(), key='start_sim')
    with col_date2:
        end_date = st.date_input("End Date", df['Offer_Send_Date'].max().date(), key='end_sim')

    df_sim_base = df[(df['Offer_Send_Date'].dt.date >= start_date) & (df['Offer_Send_Date'].dt.date <= end_date)]

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_subcat = st.selectbox("SubCategory", sorted(df['SubCategory2'].dropna().unique()))
    with col2:
        sel_segment = st.selectbox("Customer Segment", sorted(df['Segment'].dropna().unique()))
    with col3:
        sel_offer_type = st.selectbox("Offer Type", sorted(df['Offer_Type'].dropna().unique()))

    col4, col5 = st.columns(2)
    with col4:
        generosity = st.slider("Generosity (%)", 5, 100, 30)
    with col5:
        coverage_pct = st.slider("Customer Coverage (%)", 5, 100, 30)

    df_filtered = df_sim_base[(df_sim_base['SubCategory2'] == sel_subcat) &
                              (df_sim_base['Segment'] == sel_segment) &
                              (df_sim_base['Offer_Type'] == sel_offer_type)]

    base_rate = df_filtered['Redeemed'].mean()
    avg_reward = df_filtered['Reward_Value_USD'].mean()
    cust_count = df_filtered['Customer_ID'].nunique()
    incr_rev = df_filtered['Incremental_Revenue'].mean()
    base_redemptions = base_rate * cust_count 
    base_revenue =  incr_rev * base_redemptions * 6
    base_cost = base_redemptions * avg_reward
    base_incremental = base_revenue - (base_rate * cust_count * 6)
    base_profit = base_revenue - base_cost
    base_roi = base_incremental / base_cost if base_cost else 0

    subcat_to_constraint = {
        "Car Care": "Car Care",
        "Energy Drinks": "Energy Drinks",
        "Frozen Food": "Frozen Food",
        "Grocery": "Grocery",
        "Hot Coffee & Tea": "Hot Coffee & Tea",
        "Printing Services": "Printing Services",
        "Salty Snacks": "Salty Snacks",
        "Soft Drinks": "Soft Drinks",
        "Sweet Snacks": "Sweet Snacks"
    }

    sel_subcat = subcat_to_constraint.get(sel_subcat, None)

    def simulate(base_rate, generosity, coverage, reward_val, rev_per_redemption):
        generosity_rounded = int(round(generosity / 5.0) * 5)

        match_row = scenario_constraints_df[
            (scenario_constraints_df['SubCategory2'] == sel_subcat) &
            (scenario_constraints_df['Segment'] == sel_segment) &
            (scenario_constraints_df['Offer_Type'] == sel_offer_type)
        ]

        base_users = cust_count * coverage / 100
        base_incremental_value = base_rate * base_users * rev_per_redemption

        elasticity = 0.04
        max_inc = 0.2
        base_sim_rate = base_rate + (max_inc * (1 - np.exp(-generosity * elasticity)))

        rate = base_sim_rate
        penalty = 0.0
        boost = 0.0
        insight = "No specific scenario insight."

        if not match_row.empty:
            all_gen_levels = match_row['Generosity_Numeric'].dropna().unique()
            nearest_gen = sorted(all_gen_levels, key=lambda x: abs(x - generosity_rounded))[0]
            row = match_row[match_row['Generosity_Numeric'] == nearest_gen]
            if not row.empty:
                insight = row['Scenario_Insight'].values[0]

                if any(keyword in insight.lower() for keyword in ["not recommended", "avoid", "discouraged"]):
                    penalty += 0.2
                elif any(keyword in insight.lower() for keyword in ["too low", "minimal uplift"]):
                    penalty += 0.1
                elif any(keyword in insight.lower() for keyword in ["good", "effective", "optimal"]):
                    boost += 0.05

                ideal_range_text = insight.split('%')
                try:
                    ideal_parts = [int(s.replace('%', '').strip()) for s in ideal_range_text[0].split('–') if s.strip().isdigit()]
                    if len(ideal_parts) == 2:
                        ideal_low, ideal_high = ideal_parts
                        if generosity < ideal_low:
                            insight += " | ⬇️ Below optimal generosity range. Consider increasing."
                            penalty += 0.05
                        elif generosity > ideal_high:
                            insight += " | ⬆️ Above optimal generosity range. Cost may outweigh returns."
                            penalty += 0.05
                except:
                    pass

        rate = min(max(rate + boost - penalty, 0.01), 1.0)

        redemptions = rate * base_users
        revenue = redemptions * rev_per_redemption
        cost = redemptions * reward_val * (1 + penalty)
        incremental = revenue - base_incremental_value
        profit = revenue - cost
        roi = incremental / cost if cost else 0

        return rate, redemptions, revenue, cost, incremental, profit, roi, insight
        

    
        
    with st.spinner("⏳ Generating scenario..."):
        if st.button("🚀 Generate & Save Scenario", key="generate_scenario"):
            rate, redemptions, revenue, cost, incremental, profit, roi, insight = simulate(base_rate, generosity, coverage_pct, avg_reward, 6.0)
            scenario = {
                "label": f"{sel_segment} | {sel_subcat} | {sel_offer_type} | {generosity}% G | {coverage_pct}% C",
                "data": {
                    "Achievement Rate": rate * 100,
                    "Incremental Revenue": incremental,
                    "Incremental Volume": redemptions,
                    "ROI": roi * 100,
                    "Insight": insight
                }
            }
            if len(st.session_state.saved_scenarios) < 3:
                st.session_state.saved_scenarios.append(scenario)


    # 🧹 Reset button (added just below)
    if st.button("🧹 Reset Scenarios"):
        st.session_state.saved_scenarios = []
        st.success("✅ All saved scenarios have been reset. Please generate a new scenario.")
    

    st.markdown("""<h4 style='color:#FFD700;'>📋 Choose Scenario below to view Simulation Report</h4>""", unsafe_allow_html=True)
    if st.session_state.saved_scenarios:
        scenario_labels = [f"Scenario {i+1}" for i in range(3)]
        st.markdown("""
            <style>
                div[role="radiogroup"] > label {
                    color: #ffffff !important;
                    font-size: 16px;
                    padding: 4px 8px;
                    margin-bottom: 4px;
                }
        
                div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + div {
                    background-color: #FFD700 !important;
                    color: black !important;
                    font-weight: bold;
                    border-radius: 5px;
                    padding: 3px 10px;
                }
            </style>
        """, unsafe_allow_html=True)
        selected_index = st.radio("Select Scenario", options=list(range(len(st.session_state.saved_scenarios))), format_func=lambda i: st.session_state.saved_scenarios[i]['label'])
        if selected_index is not None:
            scenario = st.session_state.saved_scenarios[selected_index]
            sim_data = scenario['data']
            st.markdown("""
                <style>
                .kpi-card {
                    background-color: #1A1A1A;
                    border: 1px solid #FFD700;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 10px 0;
                    box-shadow: 0px 2px 8px rgba(255, 215, 0, 0.2);
                    text-align: center;
                }
                .kpi-title { font-size: 18px; color: #FFD700; margin-bottom: 4px; }
                .kpi-value { font-size: 16px; font-weight: bold; color: white; }
                </style>
            """, unsafe_allow_html=True)
            col1, col2, col3, col4,col5 = st.columns(5)
            col1.markdown(f"<div class='kpi-card'><div class='kpi-title'>🎯 Achievement Rate</div><div class='kpi-value'>{sim_data['Achievement Rate']:.1f}%</div></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='kpi-card'><div class='kpi-title'>📈 Incremental Value</div><div class='kpi-value'>${sim_data['Incremental Revenue']:.0f}</div></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='kpi-card'><div class='kpi-title'>📉 ROI (%)</div><div class='kpi-value'>{sim_data['ROI']:.1f}%</div></div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='kpi-card'><div class='kpi-title'>📦 Incremental Redemptions</div><div class='kpi-value'>{sim_data['Incremental Volume']:.0f}</div></div>", unsafe_allow_html=True)
            base_cost = base_redemptions * avg_reward
            simulated_cost = sim_data["Incremental Volume"] * avg_reward
            col5.markdown(f"<div class='kpi-card'><div class='kpi-title'>💸 Total Markdown (Offer Cost)</div><div class='kpi-value'>${simulated_cost:,.0f}</div></div>", unsafe_allow_html=True)



        # Base vs Simulated KPIs
        labels = ["Achievement Rate", "Incremental Revenue", "ROI", "Incremental Volume"]
        base_values = [base_rate * 100, base_incremental, base_roi * 100, base_redemptions]
        sim_values = [
            sim_data["Achievement Rate"],
            sim_data["Incremental Revenue"],
            sim_data["ROI"],
            sim_data["Incremental Volume"]
        ]
        
        delta = [s - b for s, b in zip(sim_values, base_values)]
        arrows = ['⬆️' if d > 0 else '⬇️' for d in delta]
        colors = ['green' if d > 0 else 'red' for d in delta]
        
        # Radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=base_values,
            theta=labels,
            fill='toself',
            name='Base Scenario',
            line=dict(color='#00BFC4', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=sim_values,
            theta=labels,
            fill='toself',
            name='Simulated Scenario',
            line=dict(color='#FFD700', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=dict(
                text="📊 Base vs Simulated Offer KPIs",
                font=dict(size=18, color='#FFD700'),
                x=0.0,
                xanchor='left'
            ),
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, max(max(base_values), max(sim_values)) * 1.2],  # Max value + 20% buffer
                showline=True,
                linewidth=1.5,
                gridcolor='#888',
                gridwidth=1,
                color='white',
                tickfont=dict(size=13, color='white')
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, color='white')
                ),
                bgcolor='#111111'
            ),
            font=dict(color='white'),
            legend=dict(
                font=dict(color='white'),
                bgcolor='#1A1A1A',
                bordercolor='#FFD700',
                borderwidth=1,
                orientation="h",
                yanchor="bottom", y=-0.25
            ),
            paper_bgcolor='#111111',
            plot_bgcolor='#111111',
            height=500
        )

        # Use red/green arrows only
        custom_arrows = ['🟢' if d > 0 else '🔻' for d in delta]
        
        # Build styled insight block
        insight_text = f"""
        <div style='
            background-color:#1A1A1A;
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 15px 20px;
            margin-top: 110px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0px 4px 8px rgba(255, 215, 0, 0.1);
        '>
            <h5 style='color:#FFD700; font-size:20px; margin-bottom: 12px; text-align: center;'>🧠 Simulation Insights</h5>
            <ul style='color:white; list-style:none; padding-left:0; font-size:16px; text-align:left;'>
        """
        
        for i in range(len(labels)):
            insight_text += f"<li><b>{labels[i]}:</b> {custom_arrows[i]} {delta[i]:+.1f}</li>"
        
        insight_text += """
            </ul>
        </div>
        """
        
       
        


        
        # insight_text = "<div style='color:white; font-size:18px; text-align:center;'>"
        # insight_text += "<h5 style='color:#FFD700;'>🧠 Simulation Insights</h5><ul style='list-style:none; padding-left:0;'>"
        # for i in range(len(labels)):
        #     insight_text += f"<li><b>{labels[i]}:</b> <span style='color:{colors[i]}'>{arrows[i]} {delta[i]:+.1f}</span></li>"
        # insight_text += "</ul></div>"
        
        # Render Chart + Insight Panel
        col_chart, col_text = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(fig, use_container_width=True)
        with col_text:
            #html(insight_text, height=240) # Render as Streamlit markdown
            st.markdown(insight_text, unsafe_allow_html=True)

        # Cost Flow Decomposition Chart
        # =========================
        
        cf_data = {
            "Scenario": ["Base", "Base", "Base", "Simulated", "Simulated", "Simulated"],
            "Stage": [
                "Gross Incremental Revenue", "Minus Offer Cost", "Net Incremental Value",
                "Gross Incremental Revenue", "Minus Offer Cost", "Net Incremental Value"
            ],
            "Value": [
                base_revenue, -base_cost, base_incremental,
                sim_data["Incremental Revenue"] + sim_data["Incremental Volume"] * avg_reward,
                -sim_data["Incremental Volume"] * avg_reward,
                sim_data["Incremental Revenue"]
            ]
        }
        cf_df = pd.DataFrame(cf_data)
        
        colors = {
            "Gross Incremental Revenue": "#FFD700",
            "Minus Offer Cost": "#FF5733",
            "Net Incremental Value": "#00BFC4"
        }
        
        fig_cf = go.Figure()

        base_color = "#00BFC4"       # Cyan for Base
        sim_color = "#FFD700"        # Yellow for Simulated
        
        for scenario in ["Base", "Simulated"]:
            scenario_df = cf_df[cf_df["Scenario"] == scenario]
            fig_cf.add_trace(go.Bar(
                x=scenario_df["Stage"],
                y=scenario_df["Value"],
                name=scenario,
                text=[f"${v:,.0f}" for v in scenario_df["Value"]],
                textposition="outside",
                marker=dict(
                    color=base_color if scenario == "Base" else sim_color
                )
        ))
        
        # for scenario in ["Base", "Simulated"]:
        #     scenario_df = cf_df[cf_df["Scenario"] == scenario]
        #     fig_cf.add_trace(go.Bar(
        #         x=scenario_df["Stage"],
        #         y=scenario_df["Value"],
        #         name=scenario,
        #         text=[f"${v:,.0f}" for v in scenario_df["Value"]],
        #         textposition="outside",
        #         marker_color=[colors[stage] for stage in scenario_df["Stage"]],
        #     ))
        
        fig_cf.update_layout(
            barmode="group",
            title=dict(
                text="💰 Cost Flow Decomposition by Scenario",
                font=dict(size=18, color="#ffd700"),
                x=0.0,
                xanchor="left"
            ),
            xaxis=dict(
                title="",
                tickfont=dict(color="white")
            ),
            yaxis=dict(
                title=dict(text="USD ($)", font=dict(color="white")),
                tickfont=dict(color="white")
            ),
            legend=dict(
                font=dict(color="white"),
                bgcolor="#1A1A1A",
                bordercolor="#FFD700",
                borderwidth=1
            ),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="white"),
            height=500
        )
        
        st.plotly_chart(fig_cf, use_container_width=True)

        incremental = sim_data["Incremental Revenue"]
        cost = sim_data["Incremental Volume"] * avg_reward
        net_incremental = incremental #- cost
        
        if net_incremental > 0:
            if cost > 1.2 * base_cost:
                insight_msg = "🟢 Positive ROI achieved with increased cost — still profitable. Strategy is acceptable but monitor margin impact."
            else:
                insight_msg = "✅ High ROI with controlled cost. Strong offer strategy."
        else:
            if incremental > base_incremental and cost > 1.2 * base_cost:
                insight_msg = "⚠️ Uplift achieved but cost too high — net gains eroded. Strategy is over-invested. Try reducing generosity."
            else:
                insight_msg = "❌ No significant uplift or ROI. Offer strategy ineffective. Recalibrate generosity or audience."

        # Cost-Benefit Insight Box
        st.markdown(f"""
        <div style="
            background-color:#1A1A1A;
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 15px 20px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(255, 215, 0, 0.1);
        ">
            <h5 style="color:#FFD700; font-size:18px; margin-bottom: 10px;">🧾 Cost-Benefit Analysis Insight</h5>
            <p style="color:white; font-size:15px; margin:0;">{insight_msg}</p>
            <ul style='color:white;list-style:none;padding-left:0;'>
                <li><b>Incremental Revenue:</b> ${incremental:,.0f}</li>
                <li><b>Offer Cost:</b> ${cost:,.0f}</li>
                <li><b>Net Incremental:</b> <span style="color:{'lime' if net_incremental > 0 else 'red'}">${net_incremental:,.0f}</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
 

        # Diminishing Returns Chart
        # =========================
        gen_levels = np.arange(5, 105, 5)
        ach_rates = []
        incr_values = []
        
        for gen in gen_levels:
            rate, redemptions, revenue, cost, incr, profit, roi, insight = simulate(base_rate, gen, coverage_pct, avg_reward, 6.0)
            ach_rates.append(rate * 100)
            incr_values.append(incr)
        
        returns_df = pd.DataFrame({
            "Generosity (%)": gen_levels,
            "Achievement Rate (%)": ach_rates,
            "Incremental Revenue": incr_values
        })
        
        # Calculate inflection point
        slope = np.gradient(returns_df["Achievement Rate (%)"], returns_df["Generosity (%)"])
        flatten_idx = np.where(slope < 0.2)[0][0]
        inflect_gen = returns_df["Generosity (%)"].iloc[flatten_idx]
        inflect_ach = returns_df["Achievement Rate (%)"].iloc[flatten_idx]
        
        # Chart
        fig2 = go.Figure()
        
        # Achievement Rate Line
        fig2.add_trace(go.Scatter(
            x=returns_df["Generosity (%)"],
            y=returns_df["Achievement Rate (%)"],
            mode="lines+markers+text",
            name="Achievement Rate",
            text=[f"{val:.1f}%" for val in returns_df["Achievement Rate (%)"]],
            textposition="top center",
            line=dict(color="#FFD700", width=3),
            marker=dict(color="#FFD700")
        ))
        
        # Incremental Revenue Line
        fig2.add_trace(go.Scatter(
            x=returns_df["Generosity (%)"],
            y=returns_df["Incremental Revenue"],
            mode="lines+markers+text",
            name="Incremental Revenue",
            text=[f"${val:,.0f}" for val in returns_df["Incremental Revenue"]],
            textposition="bottom center",
            yaxis="y2",
            line=dict(color="#00BFC4", width=3, dash="dot"),
            marker=dict(color="#00BFC4")
        ))
        
        # Add inflection vertical line
        fig2.add_vline(
            x=inflect_gen,
            line=dict(color="white", width=1.5, dash="dot"),
            annotation=dict(
                text=f"⬆️ Inflection Point ({inflect_gen}%)",
                font=dict(color="white", size=12),
                showarrow=False,
                xanchor="left"
            )
        )
        
        # Layout
        fig2.update_layout(
            title=dict(
                text="📉 Diminishing Returns: Generosity vs. Outcome",
                font=dict(size=18, color="#FFD700"),
                x=0,
                xanchor="left"
            ),
            xaxis=dict(
                title="Generosity (%)",
                color="white",
                gridcolor="#333",
                tickfont=dict(color="white")
            ),
            yaxis=dict(
                title="Achievement Rate (%)",
                color="white",
                gridcolor="#333",
                tickfont=dict(color="white")
            ),
            yaxis2=dict(
                title="Incremental Revenue (USD)",
                overlaying="y",
                side="right",
                color="white",
                tickfont=dict(color="white")
            ),
            font=dict(color="white"),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            legend=dict(
                font=dict(color="white"),
                bgcolor="#1A1A1A",
                bordercolor="#FFD700",
                borderwidth=1
            ),
            height=520
        )
        
        # Render chart
        st.plotly_chart(fig2, use_container_width=True)
        
        # Inflection Insight Box
        inflection_insight = f"""
        <div style='background-color:#1A1A1A; border-left: 5px solid #FFD700; padding: 12px; margin-top: 10px; color: white;'>
        <b>⚠️ Inflection Insight:</b> After <b>{inflect_gen}%</b> generosity, achievement rate starts to flatten (~<b>{inflect_ach:.1f}%</b>).
        Going beyond this may lead to <span style='color:#FF5733;'>over-investment</span> without proportionate returns.
        </div>
        """
        st.markdown(inflection_insight, unsafe_allow_html=True)


        # Trend Chart: Incremental Redemptions Over Time
        # =========================
        
        # Filter to date and selected scenario
        df_trend_base = df_filtered.copy()
        df_trend_base['Week'] = pd.to_datetime(df_trend_base['Offer_Send_Date']).dt.to_period('W').dt.start_time
        df_trend_summary = df_trend_base.groupby('Week').agg({
            'Customer_ID': 'nunique',
            'Achievement_Rate': 'mean'
        }).reset_index()
        
        # Add base redemptions
        df_trend_summary['Base_Incremental_Redemptions'] = df_trend_summary['Customer_ID'] * df_trend_summary['Achievement_Rate']
        
        # Apply simulation logic for redemptions
        simulated_rate = sim_data['Achievement Rate'] / 100
        sim_redemptions = df_trend_summary['Customer_ID'] * simulated_rate
        df_trend_summary['Sim_Incremental_Redemptions'] = sim_redemptions
        
        # Plot area chart
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=df_trend_summary['Week'],
            y=df_trend_summary['Base_Incremental_Redemptions'],
            mode='lines',
            fill='tozeroy',
            name='Base Scenario',
            line=dict(color='#00BFC4', width=3)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=df_trend_summary['Week'],
            y=df_trend_summary['Sim_Incremental_Redemptions'],
            mode='lines',
            fill='tozeroy',
            name='Simulated Scenario',
            line=dict(color='#FFD700', width=3)
        ))
        
        fig_trend.update_layout(
            title=dict(
                text="📊 Weekly Trend: Incremental Redemptions (Base vs Simulated)",
                font=dict(size=18, color="#FFD700"),
                x=0,
                xanchor='left'
            ),
            xaxis=dict(
                title="Offer Week",
                tickformat="%b %d",
                color="white",
                gridcolor="#333"
            ),
            yaxis=dict(
                title="Incremental Redemptions",
                color="white",
                gridcolor="#333"
            ),
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="white"),
            legend=dict(
                font=dict(color="white"),
                bgcolor="#1A1A1A",
                bordercolor="#FFD700",
                borderwidth=1
            ),
            height=500
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

            
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 20px;
            font-size: 16px;
            border: none;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #ffa500;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# # ========== TAB 4: Offer.AI Insights Engine ==========

with tabs[3]:
    #----------------------------------------------------------
    @st.cache_data
    def load_data(history_path: str, constraints_path: str):
        df = pd.read_csv(
            history_path,
            parse_dates=["Offer_Send_Date", "Offer_Start_Date", "Offer_End_Date",
                         "Offer_Open_Date", "Offer_Activation_Date", "Offer_Redeem_Date"]
        )
        scenario_df = pd.read_csv(constraints_path)
        return df, scenario_df
    
    df, scenario_df = load_data(
        "./Corrected_Offer_Data_With_Variation.csv",
        "./Prepared_Scenario_Constraint_Table.csv",
    )
    
    @st.cache_data
    def compute_metrics(df):
        sim = (
            df.groupby(['Offer_ID', 'Segment', 'SubCategory2', 'Offer_Type'], as_index=False)
              .agg(
                  Base_Achievement=('Redeemed', 'mean'),
                  Base_Incremental_Revenue=('Incremental_Revenue', 'mean')
              )
        )
        sim['Sim_Achievement'] = sim['Base_Achievement']
        sim['Sim_Incremental_Revenue'] = sim['Base_Incremental_Revenue']
        reward_mean = max(df['Reward_Value_USD'].mean(), 1.0)
        sim['Base_ROI'] = sim['Base_Incremental_Revenue'] / (sim['Base_Achievement'] * reward_mean)
        sim['Sim_ROI'] = sim['Base_ROI']
        return sim
    
    sim_summary = compute_metrics(df)
    df = df.merge(sim_summary, on=['Offer_ID', 'Segment', 'SubCategory2', 'Offer_Type'], how='left')
    
    class ChartSpec(BaseModel):
        type: str
        x_col: str
        y_col: List[str]
        group_col: Optional[str]
        data_code: str
        description: str
    
    class AIResult(BaseModel):
        insight: str
        strategic_note: Optional[str] = None
        chart: ChartSpec
    
    _DATA_CONTEXT = """
    📊 DATA CONTEXT:
    You use two primary datasets:
    1. ✅ df: includes Offer_ID, SubCategory2, Segment, Offer_Type, Reward_Value_USD,
             Dates, Activated, Redeemed, Opened, Offer_Period_Visited, Time_to_Respond,
             Incremental_Revenue, Base_ROI, Sim_ROI
    2. 🧠 scenario_df: Contains Generosity, Elasticity, Max_Inc, Penalty_Adjustment, Scenario_Insight
       → For strategy notes, not calculations
       → Use 'Base_ROI' instead of 'ROI' for KPI plots
    """
    
    _PROMPT_TEMPLATE = """
    You are OfferAI, a Senior Data Science and BI expert.
    
    {data_context}
    
    Return JSON in this format:
    {{
      "insight": <text>,
      "strategic_note": <optional>,
      "chart": {{
        "type": "bar|line|area|scatter|pie|radar|waterfall",
        "x_col": <col>,
        "y_col": [<kpi>],
        "group_col": <col or null>,
        "data_code": <python pandas code producing df_chart>,
        "description": <short title for the chart only>
      }}
    }}
    
    Also suggest 1 relevant follow-up question for the user based on the chart data.
    
    Here are 5 rows from df:
    {sample_hist}
    
    Here are 5 rows from scenario_df:
    {sample_scn}
    
    User query: {user_query}
    """
    
    def build_prompt(df, scenario_df, user_query):
        sample_hist = df.sample(5, random_state=42).to_dict('records')
        sample_scn = scenario_df.sample(5, random_state=42).to_dict('records')
        return _PROMPT_TEMPLATE.format(
            data_context=_DATA_CONTEXT,
            sample_hist=json.dumps(sample_hist, default=str),
            sample_scn=json.dumps(sample_scn, default=str),
            user_query=user_query.replace('"','\\"')
        )
    
    def query_offer_ai(prompt: str) -> AIResult:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        content = response.text.strip()
    
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
    
        try:
            result = json.loads(content)
            return AIResult(**result)
        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}\n\n{content}")
    
    def plot_chart(chart: ChartSpec, df_base: pd.DataFrame):
        local_ns = {"df_chart": df_base.copy()}
        try:
            if "'ROI'" in chart.data_code:
                chart.data_code = chart.data_code.replace("'ROI'", "'Base_ROI'")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(chart.data_code, globals(), local_ns)
            dfc = local_ns.get("df_chart", df_base)
        except Exception as e:
            st.error(f"❌ Error executing chart code: {e}")
            return
    
        try:
            fig = None
            if chart.type == "bar":
                fig = px.bar(dfc, x=chart.x_col, y=chart.y_col, color=chart.group_col, barmode="group")
            elif chart.type == "line":
                fig = px.line(dfc, x=chart.x_col, y=chart.y_col, color=chart.group_col, markers=True)
            elif chart.type == "area":
                fig = px.area(dfc, x=chart.x_col, y=chart.y_col, color=chart.group_col)
            elif chart.type == "scatter":
                fig = px.scatter(dfc, x=chart.x_col, y=chart.y_col[0], color=chart.group_col)
            elif chart.type == "pie":
                grouped = dfc.groupby(chart.x_col)[chart.y_col[0]].sum().reset_index()
                fig = px.pie(grouped, names=chart.x_col, values=chart.y_col[0], hole=0.4)
            elif chart.type == "radar":
                melt = dfc.melt(id_vars=chart.x_col, value_vars=chart.y_col)
                fig = px.line_polar(melt, r="value", theta=chart.x_col, color="variable", line_close=True)
            elif chart.type == "waterfall":
                fig = go.Figure(go.Waterfall(x=dfc[chart.x_col], y=dfc[chart.y_col[0]]))
    
            if fig:
                fig.update_layout(
                    paper_bgcolor="#111",
                    plot_bgcolor="#111",
                    font_color="white",
                    title=dict(text=chart.description, font=dict(color="white", size=18)),
                    legend=dict(font=dict(color="white")),
                    xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),
                    yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),
                    margin=dict(t=40, b=40, l=20, r=20), height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Chart rendering failed: {e}")
    
    # UI START
    st.markdown("""
    <div style='text-align:left;font-weight:600;font-size:19px;color:white;'>
      🤖 Offer.AI Insights Engine : Unlock instant offer intelligence with AI-powered queries — analyze trends across customer segments, offer types, KPIs like ROI or redemptions, and uncover hidden patterns from historical offer data with natural language prompts.
    </div>
    <div style="background-color:#222;border:1px solid #FFD700;border-radius:8px;padding:15px;margin:10px 0;">
       <p style="color: white; font-size: 15px; margin-bottom: 5px;"><b style="color:#FFD700;">💡 You can ask things like:</b></p>
       <ul style="color: white; font-size: 15px; padding-left: 20px; margin-top: 0;">
           <li>Which segments had the highest ROI in cashback offers?</li>
           <li>Compare redemptions between different products.</li>
           <li>Plot area chart showing achievement rate progression weekly for hot coffee offers.</li>
           <li>Which Offer Duration performed the best?</li>
           <li>Which offer types had the highest visit activation rate?</li>
           <li>Show a waterfall chart of revenue uplift by subcategory for Loyalist customers.</li>
      </ul>
       <p style="color: white; font-size: 15px; margin-top: 15px;">
            👉 Simply type your question below and click <b style="color:#FFD700;">Analyze</b> to generate insights.
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    query = st.text_input("🔍 Ask a question about your offers:", "Which segments had the best ROI?")
    
    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     analyze_clicked = st.button("Analyze")
    # with col2:
    #     reset_clicked = st.button("Reset History") if st.session_state.history else False
    col1, col2, col3 = st.columns([1, 1, 6])
    
    with col1:
        analyze_clicked = st.button("🧠 Analyze with AI", key="analyze_button")
    
    with col2:
        if st.session_state.history:
            reset_clicked = st.button("Reset History", key="reset_button")
        else:
            reset_clicked = False
    
    if analyze_clicked:
        with st.spinner("🧠 Generating insights from your offer data..."):
            try:
                prompt = build_prompt(df, scenario_df, query)
                result = query_offer_ai(prompt)
                st.session_state.history.append((query, result))
            except Exception as e:
                st.error(str(e))
    
    # if st.session_state.history:
    #     if st.button("🔄 Reset Chat"):
    #         st.session_state.history.clear()
    #         st.experimental_rerun()
    
    
    if reset_clicked:
        st.session_state.history.clear()
        st.experimental_rerun()
    
    if st.session_state.history:
        for q, res in st.session_state.history:
            st.markdown("---")
            st.markdown(f"<div style='font-size:18px;color:#FFD700;'><b>❓ {q}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#222;color:white;border-radius:8px;padding:15px;margin-top:10px;'><b>📊 Insight:</b> {res.insight}</div>", unsafe_allow_html=True)
            if res.strategic_note:
                st.markdown(f"<div style='background-color:#113B24;border-left:4px solid #31D158;padding:12px;margin-top:12px;border-radius:6px;color:white;'><b>📌 Strategic Advice:</b> {res.strategic_note}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            plot_chart(res.chart, df)









# %% [markdown]
# # Part 2: RFM Segmentation & Behavior-Informed Retention Strategy
# This script processes raw transaction data, builds behavioral metrics, merges them with support and web activity logs, and outputs the final segments.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 1. Load Data & Clean Transactions (No Leakage)

# %%
customers = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
tickets = pd.read_csv('support_tickets.csv')
web_events = pd.read_csv('web_events_snapshot.csv')
churn = pd.read_csv('churn_labels.csv')

# Parse dates
orders['order_date'] = pd.to_datetime(orders['order_date'])
SNAPSHOT_DATE = pd.to_datetime('2025-09-30')

# Filter post-snapshot data to avoid target leakage
orders_clean = orders[orders['order_date'] <= SNAPSHOT_DATE].copy()

# Deduplicate orders ending with '_DUP'
orders_clean['order_id'] = orders_clean['order_id'].astype(str).str.replace('_DUP', '')
orders_clean = orders_clean.drop_duplicates(subset=['order_id'], keep='first')

# Impute missing ratings
orders_clean['rating'] = orders_clean['rating'].fillna(orders_clean['rating'].median())

print(f"Cleaned orders shape: {orders_clean.shape}")

# %% [markdown]
# ## 2. Construct RFM Metrics

# %%
# Calculate Recency, Frequency, and Monetary (RFM)
rfm_agg = orders_clean.groupby('customer_id').agg(
    last_order_date=('order_date', 'max'),
    frequency=('order_id', 'count'),
    monetary=('gross_amount', 'sum')
).reset_index()

# Recency calculation relative to snapshot date
rfm_agg['recency'] = (SNAPSHOT_DATE - rfm_agg['last_order_date']).dt.days

# Drop helper column
rfm_agg = rfm_agg.drop(columns=['last_order_date'])

print("RFM Base Metrics:")
print(rfm_agg.head())

# %% [markdown]
# ## 3. Construct Non-RFM Behavioral Signals
# We select two critical signals:
# 1. **Support Ticket Sentiment / Count**
# 2. **Product Return/Refund Rates**

# %%
# 1. Support Tickets Aggregation
tickets['sentiment_score'] = pd.to_numeric(tickets['sentiment_score'], errors='coerce')
tickets_agg = tickets.groupby('customer_id').agg(
    total_tickets=('ticket_id', 'count'),
    avg_sentiment=('sentiment_score', 'mean')
).reset_index()

# 2. Return Rates Aggregation
returns_agg = orders_clean.groupby('customer_id').agg(
    total_items=('quantity', 'sum'),
    returned_items=('returned', 'sum')
).reset_index()
returns_agg['return_rate'] = (returns_agg['returned_items'] / returns_agg['total_items']).fillna(0.0)

# %% [markdown]
# ## 4. Merge All Features & Assign Segments

# %%
# Merge RFM with base profile and non-RFM behavioral metrics
customer_profile = customers.merge(rfm_agg, on='customer_id', how='left')
customer_profile = customer_profile.merge(tickets_agg, on='customer_id', how='left')
customer_profile = customer_profile.merge(returns_agg[['customer_id', 'return_rate']], on='customer_id', how='left')
customer_profile = customer_profile.merge(web_events, on='customer_id', how='left')

# Fill NaNs for non-matching customer records
customer_profile['recency'] = customer_profile['recency'].fillna(365) # Dormant/No order
customer_profile['frequency'] = customer_profile['frequency'].fillna(0)
customer_profile['monetary'] = customer_profile['monetary'].fillna(0.0)
customer_profile['total_tickets'] = customer_profile['total_tickets'].fillna(0)
customer_profile['avg_sentiment'] = customer_profile['avg_sentiment'].fillna(0.0)
customer_profile['return_rate'] = customer_profile['return_rate'].fillna(0.0)

# %%
# Segmentation Logic Execution
def assign_segment(row):
    # Dormant Customers (High Recency, Low Activity)
    if row['recency'] > 150 and row['sessions_30d'] <= 1:
        return 'Dormant Customers'
    
    # High-Value but Unhappy Customers (Cross-referencing RFM + Support + Return Signals)
    if row['monetary'] >= 1200 and (row['avg_sentiment'] < -0.5 or row['return_rate'] > 0.30):
        return 'High-Value but Unhappy'
    
    # Discount-Sensitive Shoppers (Based on web abandoned carts and general profile)
    if row['abandoned_carts_30d'] >= 3:
        return 'Discount-Sensitive Shoppers'
    
    # Champions (Recent, Frequent, and High Spenders)
    if row['recency'] <= 45 and row['frequency'] >= 3:
        return 'Champions'
    
    # Loyal Customers
    if row['recency'] <= 90 and row['frequency'] >= 2:
        return 'Loyal Customers'
    
    # At-Risk Customers
    if row['recency'] > 90 and row['frequency'] >= 2:
        return 'At-Risk Customers'
    
    # Fallback to general low/mid-value category
    return 'Dormant Customers'

customer_profile['segment_name'] = customer_profile.apply(assign_segment, axis=1)

# Inspect distribution
print("\nSegment Distribution:")
print(customer_profile['segment_name'].value_counts())

# %% [markdown]
# ## 5. Visualize Segments & Export Data

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=customer_profile, 
    x='recency', 
    y='monetary', 
    hue='segment_name', 
    palette='Set2', 
    alpha=0.8
)
plt.title("Customer Segments: Recency vs Monetary Value")
plt.xlabel("Recency (Days Since Last Order)")
plt.ylabel("Monetary Value (Total Spend)")
plt.yscale('log') # Scale because of outliers
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('segment_distribution_chart.png')
plt.show()

# %%
# Export segments to CSV
output_cols = [
    'customer_id', 'segment_name', 'recency', 'frequency', 
    'monetary', 'total_tickets', 'avg_sentiment', 'return_rate'
]
customer_profile[output_cols].to_csv('segments.csv', index=False)
print("segments.csv exported successfully!")
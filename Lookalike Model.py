print('** Task 2: Lookalike Model **')
print('=============================')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
customers = pd.read_csv("D:\Python Assessment\Customers.csv") # Customer Information
products = pd.read_csv("D:\Python Assessment\Products.csv") # Product Information
transactions = pd.read_csv("D:\Python Assessment\Transactions.csv") # Transaction History

# Merge transactions with customer and product data
merge_transactions = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Print first few rows to verify
print(merge_transactions.head())

# Feature Engineering
customer_features = merge_transactions.groupby("CustomerID").agg({
    "TotalValue": ["sum", "mean"],  # Total and average spend
    "Quantity": "sum",  # Total quantity purchased
    "Category": lambda x: x.mode()[0] if not x.mode().empty else np.nan,  # Most frequent category
}).reset_index()
customer_features.columns = ["CustomerID", "TotalSpend", "AvgSpend", "TotalQuantity", "TopCategory"]

# Convert categorical variable 'TopCategory' to numeric
customer_features = pd.get_dummies(customer_features, columns=["TopCategory"], drop_first=True)

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.drop("CustomerID", axis=1))

# Compute similarity matrix
similarity_matrix = cosine_similarity(scaled_features)

# Get top 3 similar customers for the first 20 customers
customer_ids = customer_features["CustomerID"].values
lookalike_dict = {}

for i, cust_id in enumerate(customer_ids[:20]):
    similar_customers = np.argsort(similarity_matrix[i])[::-1][1:4]  # Get top 3 excluding itself
    lookalike_dict[cust_id] = [(customer_ids[j], round(similarity_matrix[i][j], 4)) for j in similar_customers]

# Convert to DataFrame and save as CSV
lookalike_df = pd.DataFrame(lookalike_dict.items(), columns=["CustomerID", "Lookalikes"])
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike.csv generated successfully!")
print(' ')
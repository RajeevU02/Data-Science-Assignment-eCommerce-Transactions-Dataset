print('Task 3: Customer Segmentation (Clustering)')

# Step 1: Load the Data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import seaborn as sns
from sklearn.metrics import davies_bouldin_score

#load the datasets
customers = pd.read_csv("D:\Python Assessment\Customers.csv")
transactions = pd.read_csv("D:\Python Assessment\Transactions.csv")

# Step 1: Data Preprocessing for Clustering #
# Merge customer profile with transaction data
customer_profile = customers[['CustomerID', 'Region']]  # Simplified customer profile
transaction_data = transactions.groupby('CustomerID').agg({'TotalValue': 'sum', 'Quantity': 'sum'}).reset_index()

# Merge the two datasets
customer_data = pd.merge(customer_profile, transaction_data, on='CustomerID', how='inner')

# Data Preprocessing: Standardizing the features for clustering
features = ['total_spend', 'num_transactions', 'recency']
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['TotalValue', 'Quantity']])

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42) # You can experiment with the number of clusters
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Inspect cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Step 3: Calculate DB Index for evaluation
db_index = davies_bouldin_score(customer_data_scaled, customer_data['Cluster'])
print(f'Davies-Bouldin Index: {db_index}')

# Step 4: Visualize the Clusters
sns.scatterplot(x=customer_data['TotalValue'], y=customer_data['Quantity'], hue=customer_data['Cluster'], palette='viridis')
plt.title('Customer Segmentation using K-means clusting')
plt.xlabel("Total Spend")
plt.ylabel("Number of Transactions")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
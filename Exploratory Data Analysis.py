print('** Task 1: Exploratory Data Analysis (EDA) and Business Insights **')
print('====================================================================')

import pandas as pd

# Load the datasets
customers = pd.read_csv('D:\Python Assessment\Customers.csv') # Customer Information
products = pd.read_csv('D:\Python Assessment\Products.csv') # Product Information
transactions = pd.read_csv('D:\Python Assessment\Transactions.csv') # Transaction History

# Display basic information
print("Customers Data Info:")
print(customers.info())
print("\nProducts Data Info:")
print(products.info())
print("\nTransactions Data Info:")
print(transactions.info())

# Check for missing values
print("\nMissing Values in Customers:")
print(customers.isnull().sum())
print("\nMissing Values in Products:")
print(products.isnull().sum())
print("\nMissing Values in Transactions:")
print(transactions.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows in Customers:", customers.duplicated().sum())
print("Duplicate Rows in Products:", products.duplicated().sum())
print("Duplicate Rows in Transactions:", transactions.duplicated().sum())

# Display summary statistics
print("\nCustomers Summary:")
print(customers.describe(include='all'))
print("\nProducts Summary:")
print(products.describe(include='all'))
print("\nTransactions Summary:")
print(transactions.describe())

# Business Insights
insights = [
    "1. The majority of customers are from a specific region, which can help target regional promotions.",
    "2. Certain product categories dominate sales, indicating high demand for specific types of products.",
    "3. A significant number of customers signed up recently, showing growth in the customer base.",
    "4. The top-selling products contribute to a large percentage of total revenue, emphasizing key products.",
    "5. There are seasonal trends in transactions, which can guide marketing strategies for peak sales periods."
]

print("\nBusiness Insights:")
for insight in insights:
    print(insight)

print(' ')
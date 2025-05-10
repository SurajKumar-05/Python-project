# Python-project


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Hp\\Downloads\\SmartphonePriceAnalysis\\data\\smartphones_data.csv.csv')

# 1. Analyze the Price Distribution
print(df['Price'].describe())
sns.histplot(df['Price'], kde=True)
plt.title('Price Distribution')
plt.show()

plt.boxplot(df['Price'])
plt.title('Boxplot of Prices')
plt.show()

# 2. Correlation Between Features
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 3. Brand-wise Price Comparison
brand_avg = df.groupby('brand_name')['Price'].mean().sort_values(ascending=False)
brand_avg.plot(kind='bar', title='Average Price by Brand')
plt.ylabel('Average Price')
plt.show()

# 4. Feature Impact on Price (Multivariate Analysis)
sns.pairplot(df, vars=['RAM', 'Battery_cap', 'primery_rear_camera', 'Price'], hue='brand_name')
plt.show()

# 5. Detect Outliers and Anomalies
sns.boxplot(data=df[['RAM', 'Battery_cap', 'storage', 'primery_rear_camera', 'Price']])
plt.title('Outliers in Features')
plt.xticks(rotation=45)
plt.show()

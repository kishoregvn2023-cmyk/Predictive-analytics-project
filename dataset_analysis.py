import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET

df = pd.read_csv("wildfires.csv", low_memory=False)

print("Original dataset size:", df.shape)

# Remove geometry column
df = df.drop(columns=['Shape'], errors='ignore')

# Sample dataset for faster processing
df = df.sample(50000, random_state=42)

print("Sampled dataset size:", df.shape)

# 2. BASIC DATASET INFORMATION

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. FIRE SIZE DISTRIBUTION

plt.figure(figsize=(8,5))
df['FIRE_SIZE'].hist(bins=50)
plt.title("Distribution of Fire Size")
plt.xlabel("Fire Size")
plt.ylabel("Frequency")
plt.show()

# 4. FIRE SIZE CLASS DISTRIBUTION

plt.figure(figsize=(8,5))
df['FIRE_SIZE_CLASS'].value_counts().plot(kind='bar')
plt.title("Wildfire Size Class Distribution")
plt.xlabel("Fire Size Class")
plt.ylabel("Number of Fires")
plt.show()

# 5. WILDFIRES BY YEAR

plt.figure(figsize=(10,5))
df['FIRE_YEAR'].value_counts().sort_index().plot()
plt.title("Number of Wildfires per Year")
plt.xlabel("Year")
plt.ylabel("Number of Fires")
plt.show()

# 6. WILDFIRE CAUSES

plt.figure(figsize=(10,5))
df['STAT_CAUSE_DESCR'].value_counts().head(10).plot(kind='bar')
plt.title("Top Causes of Wildfires")
plt.xlabel("Cause")
plt.ylabel("Count")
plt.show()

# 7. GEOGRAPHIC DISTRIBUTION

plt.figure(figsize=(8,6))
plt.scatter(df['LONGITUDE'], df['LATITUDE'], alpha=0.3)
plt.title("Geographic Distribution of Wildfires")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 8. CORRELATION HEATMAP

numeric_cols = df.select_dtypes(include=['float64','int64'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
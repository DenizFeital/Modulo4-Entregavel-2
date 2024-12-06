import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read the file 
file_path = 'seeds_dataset.csv'
data = pd.read_csv(file_path)

# 2. Show the first lines
print("Step 2 First 5 rows of the dataset:")
print(data.head())

# 3. Calculate descriptive statistics
print("Step 3 Statisticas:")
stats = data.describe()
median = data.median()
print("\nEstatísticas Descritivas:")
print(stats)
print("\nMediana de cada valor:")
print(median)

# 4. Visualize the distribution of features
# Histograms
print("Step 4 - Histogramas:")
data.hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histogramas dos componentes', fontsize=16)
plt.show()

# Boxplots
print("Step 4 - Boxplots:")
plt.figure(figsize=(15, 6))
sns.boxplot(data=data)
plt.title('Boxplots dos componentes', fontsize=16)
plt.xticks(rotation=45)

plt.show()

# 5. Scatter plots to identify possible relationships
print("Step 5 - Possíveis relacionamentos:")
sns.pairplot(data, diag_kind='kde', hue='Target', markers=["o", "s", "D"])
plt.suptitle('Gráfico de dispersão dos componentes', fontsize=16)
plt.show()

# 6. Identify and handle missing values
print("Step 6 - Identificando valores faltantes:")
missing_values = data.isnull().sum()
print("\nValores faltantes em cada componente:")
print(missing_values)

if missing_values.any():
    print("\nPreenchendo os valores faltantes com a média, caso necessário")
    data.fillna(data.mean(), inplace=True)

# 7. Assess the need for scaling and normalization
print("Step 7 - Normalização - se necessário:")
print("\nVerificando o intervalo para dimensionamento:")
print(data.max() - data.min())

# Applying normalization (Min-Max Scaling)
normalized_data = (data - data.min()) / (data.max() - data.min())

# Applying standardization (Z-score Scaling)
standardized_data = (data - data.mean()) / data.std()

# Display normalized and standardized samples
print("\nAmostra do dado normalizado:")
print(normalized_data.head())
print("\nAmostra do dado padronizado:")
print(standardized_data.head())

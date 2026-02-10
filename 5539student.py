
# ==============================
# STUDENT PERFORMANCE DATA ANALYSIS
# 5539 Kaif Mapari
# ==============================

# 0. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Load Dataset
df = pd.read_csv("student_data.csv")
print("\nDataset Loaded Successfully")

# --------------------------------------------------
# 1. Renaming Columns
print("\n--- Op 1: Renaming Columns ---")
df.rename(columns={'G1': 'Grade_1', 'G2': 'Grade_2', 'G3': 'Final_Grade'}, inplace=True)
print(df[['Grade_1', 'Grade_2', 'Final_Grade']].head())

# --------------------------------------------------
# 2. Datatype Conversion
print("\n--- Op 2: Datatype Conversion ---")
df['Final_Grade'] = df['Final_Grade'].astype(float)
print(df['Final_Grade'].dtype)

# --------------------------------------------------
# 3. Creating Total Score
print("\n--- Op 3: Creating Total Score ---")
df['Total_Score'] = df['Grade_1'] + df['Grade_2'] + df['Final_Grade']
print(df[['Total_Score']].head())

# --------------------------------------------------
# 4. Mean
print("\n--- Op 4: Mean ---")
print("Mean Final Grade:", df['Final_Grade'].mean())

# --------------------------------------------------
# 5. Median
print("\n--- Op 5: Median ---")
print("Median Total Score:", df['Total_Score'].median())

# 5539 Kaif Mapari
# --------------------------------------------------
# 6. Mode
print("\n--- Op 6: Mode ---")
print("Mode Final Grade:", df['Final_Grade'].mode()[0])

# --------------------------------------------------
# 7. Detecting Outliers (IQR)
print("\n--- Op 7: Detecting Outliers ---")
Q1 = df['absences'].quantile(0.25)
Q3 = df['absences'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['absences'] < (Q1 - 1.5 * IQR)) | (df['absences'] > (Q3 + 1.5 * IQR))]
print("Outliers Count:", len(outliers))

# --------------------------------------------------
# 8. Binning
print("\n--- Op 8: Binning ---")
bins = [0, 30, 45, 60]
labels = ['Low', 'Medium', 'High']
df['Performance_Level'] = pd.cut(df['Total_Score'], bins=bins, labels=labels)
print(df[['Total_Score', 'Performance_Level']].head())

# --------------------------------------------------
# 9. Aggregation
print("\n--- Op 9: Aggregation ---")
print(df.groupby('sex')['Final_Grade'].mean())

# --------------------------------------------------
# 10. Label Encoding
print("\n--- Op 10: Label Encoding ---")
le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['sex'])
print(df[['sex', 'Sex_Encoded']].head())

# 5539 Kaif Mapari
# --------------------------------------------------
# 11. One-Hot Encoding
print("\n--- Op 11: One-Hot Encoding ---")
df = pd.get_dummies(df, columns=['address'])
print(df.filter(like='address').head())

# --------------------------------------------------
# 12. Normalization
print("\n--- Op 12: Normalization ---")
scaler = MinMaxScaler()
df['Final_Grade_Normalized'] = scaler.fit_transform(df[['Final_Grade']])
print(df[['Final_Grade', 'Final_Grade_Normalized']].head())

# --------------------------------------------------
# 13. Correlation
print("\n--- Op 13: Correlation ---")
print(df[['Grade_1', 'Grade_2', 'Final_Grade']].corr())

# --------------------------------------------------
# 14. Histogram
print("\n--- Op 14: Histogram ---")
plt.hist(df['Final_Grade'], bins=10)
plt.title("Histogram of Final Grade")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.show()

# --------------------------------------------------
# 15. Boxplot
print("\n--- Op 15: Boxplot ---")
plt.boxplot(df['Total_Score'])
plt.title("Boxplot of Total Score")
plt.ylabel("Total Score")
plt.show()

# 5539 Kaif Mapari
# --------------------------------------------------
# 16. Scatter Plot
print("\n--- Op 16: Scatter Plot ---")
plt.scatter(df['Grade_1'], df['Final_Grade'])
plt.xlabel("Grade 1")
plt.ylabel("Final Grade")
plt.show()

# --------------------------------------------------
# 17. Feature Extraction
print("\n--- Op 17: Feature Extraction ---")
df['Average_Score'] = df['Total_Score'] / 3
print(df[['Average_Score']].head())

# --------------------------------------------------
# 18. PCA
print("\n--- Op 18: PCA ---")
pca_cols = ['Grade_1', 'Grade_2', 'Final_Grade']
scaled_data = StandardScaler().fit_transform(df[pca_cols])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA_1'] = pca_result[:, 0]
df['PCA_2'] = pca_result[:, 1]
print(df[['PCA_1', 'PCA_2']].head())

# --------------------------------------------------
# 19. Feature Construction
print("\n--- Op 19: Feature Construction ---")
df['Student_Profile'] = df['school'] + " " + df['sex']
print(df['Student_Profile'].head())

# --------------------------------------------------
# 20. Tokenization
print("\n--- Op 20: Tokenization ---")
df['student_text'] = df['Student_Profile'].str.lower()
df['tokens'] = df['student_text'].apply(lambda x: x.split())
print(df[['student_text', 'tokens']].head())


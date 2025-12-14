#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#-----------------------------
# Create the DataFrame
#-----------------------------
data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Name": ["Ali", "Sara", "Hassan", "Ayesha", "Omar", "Zara", "Usman", "Fatima", "Bilal", "Nadia"],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "Age": [15, 14, 15, 14, 16, 15, 16, 14, 15, 16],
    "Hours_Studied": [5, 6, 4, 7, 3, 8, 2, 6, 5, 7],
    "Marks": [80, 85, 70, 90, 65, 95, 60, 88, 75, 92],
    "Extra_Credit": [5, 7, 3, 10, 2, 8, 1, 6, 4, 9],
    "Sports_Participation": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
}

df = pd.DataFrame(data)
print(df)

#-----------------------------
# Relationship between numeric columns
#-----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x="Hours_Studied", y="Marks", hue="Gender", style="Gender", s=100, data=df)
plt.title("Hours Studied vs Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()

plt.figure(figsize=(8,6))
sns.regplot(x="Marks", y="Extra_Credit", data=df, color="green")
plt.title("Marks vs Extra Credit")
plt.xlabel("Marks")
plt.ylabel("Extra Credit")
plt.show()

sns.jointplot(x="Age", y="Marks", data=df, kind="hist", height=6)
plt.suptitle("Age vs Marks Distribution", y=1.02)
plt.show()

sns.pairplot(df[["Age","Hours_Studied","Marks","Extra_Credit"]], kind="scatter", diag_kind="kde", height=2.5)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

#-----------------------------
# Distribution plots
#-----------------------------
plt.figure(figsize=(8,6))
sns.histplot(df["Marks"], bins=5, kde=True, color="skyblue")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(df["Hours_Studied"], fill=True, color="orange")
plt.title("Distribution of Hours Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Density")
plt.show()

#-----------------------------
# Categorical comparisons
#-----------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x="Marks", y="Gender", data=df, palette="pastel")
plt.title("Marks by Gender")
plt.xlabel("Marks")
plt.ylabel("Gender")
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(x="Marks", y="Sports_Participation", data=df, palette="Set2")
plt.title("Marks by Sports Participation")
plt.xlabel("Marks")
plt.ylabel("Sports Participation")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender", palette="Set1")
plt.title("Number of Students by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

sns.catplot(x="Sports_Participation", y="Marks", hue="Gender", data=df, kind="bar", height=5, aspect=1.2)
plt.title("Average Marks by Sports Participation and Gender")
plt.show()

#-----------------------------
# Correlations and Heatmap
#-----------------------------
numeric_df = df[["Age", "Hours_Studied", "Marks", "Extra_Credit"]]
corr = numeric_df.corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#-----------------------------
# Cluster Map
#-----------------------------
cluster_df = df[["Hours_Studied", "Marks", "Extra_Credit"]]
sns.clustermap(cluster_df, cmap="viridis", standard_scale=1)
plt.title("Cluster Map of Students", y=1.05)
plt.show()

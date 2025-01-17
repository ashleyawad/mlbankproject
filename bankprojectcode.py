import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#synthetically generated data
np.random.seed(42)
n_samples = 1000
data = {
    "Average_Balance": np.random.lognormal(mean=10, sigma=0.5, size=n_samples),
    "Monthly_Income": np.random.lognormal(mean=9, sigma=0.5, size=n_samples),
    "Monthly_Expenses": lambda income: income * np.random.uniform(0.5, 0.9, size=n_samples),
    "Overdrafts": lambda avg_balance: np.random.poisson(lam=np.clip(10 / avg_balance, 0, 3), size=n_samples),
    "Seasonality_Factor": np.random.uniform(0.8, 1.2, size=n_samples),
    "Credit_Score": lambda avg_balance: np.clip(300 + (avg_balance / 1000) * np.random.uniform(1, 50), 300, 850)
}
data["Monthly_Expenses"] = data["Monthly_Expenses"](data["Monthly_Income"])
data["Overdrafts"] = data["Overdrafts"](data["Average_Balance"])
data["Credit_Score"] = data["Credit_Score"](data["Average_Balance"])

df = pd.DataFrame(data)
df["Loan_Approved"] = np.where(
    (df["Average_Balance"] > 2000) & 
    (df["Monthly_Income"] - df["Monthly_Expenses"] > 1000) & 
    (df["Overdrafts"] < 3) & 
    (df["Credit_Score"] > 650), 1, 0
)
df["Average_Balance"] = df["Average_Balance"].round(2)
df["Monthly_Income"] = df["Monthly_Income"].round(2)
df["Monthly_Expenses"] = df["Monthly_Expenses"].round(2)
df["Credit_Score"] = df["Credit_Score"].round()

# Visualizations

# Distribution of Average Balance
plt.figure(figsize=(10, 6))
sns.histplot(df["Average_Balance"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Average Balance", fontsize=16)
plt.xlabel("Average Balance", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y')
plt.show()

# Loan Approval by Credit Score
plt.figure(figsize=(10, 6))
sns.boxplot(x="Loan_Approved", y="Credit_Score", data=df, palette="Set2")
plt.title("Loan Approval by Credit Score", fontsize=16)
plt.xlabel("Loan Approved (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Credit Score", fontsize=12)
plt.grid(axis='y')
plt.show()

# Monthly Income vs. Monthly Expenses (Color by Loan Approval)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Monthly_Income", y="Monthly_Expenses", hue="Loan_Approved", data=df, palette="cool", alpha=0.7
)
plt.title("Monthly Income vs. Monthly Expenses", fontsize=16)
plt.xlabel("Monthly Income", fontsize=12)
plt.ylabel("Monthly Expenses", fontsize=12)
plt.legend(title="Loan Approved")
plt.grid()
plt.show()

# Overdrafts by Loan Approval
plt.figure(figsize=(10, 6))
sns.countplot(x="Overdrafts", hue="Loan_Approved", data=df, palette="viridis")
plt.title("Overdrafts by Loan Approval", fontsize=16)
plt.xlabel("Number of Overdrafts", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title="Loan Approved")
plt.grid(axis='y')
plt.show()

#  Heatmap of Feature Correlations
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

#prepare classifiers
x = df[["Average_Balance", "Monthly_Income", "Monthly_Expenses", "Overdrafts", "Credit_Score"]]
y = df["Loan_Approved"]

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Classifier": SVC(random_state=42, probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}
for model_name, model in models.items():
    model.fit(x_train, y_train)  # Train the model
    y_pred = model.predict(x_test)  # Predict on the test set
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # Plot confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Denied", "Approved"])
    plt.figure(figsize=(8, 6))
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix for {model_name}", fontsize=16)
    plt.show()
    
    # printed classification report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=["Denied", "Approved"]))
    print("-" * 60)

#turning the classification reports into heatmaps for each model
for model_name, model in models.items():
    report = classification_report(y_test, y_pred, target_names=["Denied", "Approved"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # Plot heatmap of classification metrics
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        report_df.iloc[:-1, :-1],  # Exclude support column and overall average row
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar=True
    )
    plt.title(f"Classification Report for {model_name}", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Classes", fontsize=12)
    plt.show()
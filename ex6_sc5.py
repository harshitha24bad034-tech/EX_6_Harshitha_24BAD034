print("Harshitha 24BAD034")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\student\Downloads\fraud_smote.csv")

print(df.head())

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

print("Before SMOTE:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_before = RandomForestClassifier(random_state=42)
model_before.fit(X_train, y_train)
y_pred_before = model_before.predict(X_test)

print("Before SMOTE Report:")
print(classification_report(y_test, y_pred_before))

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train_sm).value_counts())

model_after = RandomForestClassifier(random_state=42)
model_after.fit(X_train_sm, y_train_sm)
y_pred_after = model_after.predict(X_test)

print("After SMOTE Report:")
print(classification_report(y_test, y_pred_after))

y_prob = model_after.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label="PR AUC = %.2f" % pr_auc)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

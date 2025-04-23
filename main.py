import pandas as pd
from sklearn.model_selection import train_test_split

#df = pd.read_csv("processed_titanic1.csv")
df = pd.read_csv("processed_titanic.csv")
print(df.head())
print(df.dtypes, '\n')
df.drop(columns=['Name', 'PassengerId'], inplace=True) #, 'Cabin'


print(df.dtypes)

X = df.drop("Survived", axis=1) #Transported Survived
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)


from sklearn.metrics import (precision_score, recall_score, f1_score)

print("\nForest Precision:", precision_score(y_test, rf_pred))
print("Forest Recall:", recall_score(y_test, rf_pred))
print("Forest F1 Score:", f1_score(y_test, rf_pred))
print("\nBoosting Precision:", precision_score(y_test, gb_pred))
print("Boosting Recall:", recall_score(y_test, gb_pred))
print("Boosting F1 Score:", f1_score(y_test, gb_pred))


from sklearn.model_selection import cross_val_score

rf_cv = cross_val_score(rf, X, y, cv=5, scoring='f1')
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='f1')

print("\nRandom Forest CV F1 mean:", rf_cv.mean())
print("Gradient Boosting CV F1 mean:", gb_cv.mean())


import matplotlib.pyplot as plt

plt.bar(["Random Forest", "Gradient Boosting"],
        [precision_score(y_test, rf_pred), precision_score(y_test, gb_pred)],
        color=["skyblue", "blue"])
plt.title("Сравнение по Precision")
plt.ylabel("Precision")
plt.grid(axis='y')
plt.show()


plt.bar(["Random Forest", "Gradient Boosting"],
        [recall_score(y_test, rf_pred), recall_score(y_test, gb_pred)],
        color=["skyblue", "blue"])
plt.title("Сравнение по Recall")
plt.ylabel("Recall")
plt.grid(axis='y')
plt.show()


plt.bar(["Random Forest", "Gradient Boosting"],
        [f1_score(y_test, rf_pred), f1_score(y_test, gb_pred)],
        color=["skyblue", "blue"])
plt.title("Сравнение по F1 Score")
plt.ylabel("F1 Score")
plt.grid(axis='y')
plt.show()


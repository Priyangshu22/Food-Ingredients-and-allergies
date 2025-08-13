import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df2 = pd.read_csv(r"C:\Users\Priyangshu\OneDrive\Desktop\food_ingredients_and_allergens.csv")
df = pd.read_csv(r"C:\Users\Priyangshu\OneDrive\Desktop\food_ingredients_and_allergens.csv")
df.head()
df.shape
unique_counts = df.apply(lambda x: len(x.unique()))
unique_counts
df.dropna(inplace=True)
df.isnull().sum()


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Prediction'] = LE.fit_transform(df['Prediction'])

category_counts = df['Food Product'].value_counts()
df['Food Product Freq'] = df['Food Product'].map(category_counts)
category_counts = df['Main Ingredient'].value_counts()
df['Main Ingredient Freq'] = df['Main Ingredient'].map(category_counts)
category_counts = df['Sweetener'].value_counts()
df['Sweetener Freq'] = df['Sweetener'].map(category_counts)
category_counts = df['Fat/Oil'].value_counts()
df['Fat/Oil Freq'] = df['Fat/Oil'].map(category_counts)
category_counts = df['Seasoning'].value_counts()
df['Seasoning Freq'] = df['Seasoning'].map(category_counts)
category_counts = df['Allergens'].value_counts()
df['Allergens Freq'] = df['Allergens'].map(category_counts)

df = df.drop(['Food Product', 'Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning','Allergens'],axis=1)

X = df.iloc[:,1:]

y = df.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
para = {
    'criterion' : ["gini", "entropy"],
    'max_depth' : range(1,10)
}
Gd = GridSearchCV(DT,param_grid=para,scoring='accuracy',cv=5)
Gd.fit(X_train,y_train)
Gd.best_params_
Gd.best_score_
DT = DecisionTreeClassifier(criterion = 'gini',max_depth = 1)
DT.fit(X_train,y_train)
y_hat = DT.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(accuracy_score(y_test,y_hat))


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20,random_state=1)
rf.fit(X_train,y_train)
y_hat = rf.predict(X_test)
print(y_hat)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(accuracy_score(y_test,y_hat))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(accuracy_score(y_test,y_hat))


from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(accuracy_score(y_test,y_hat))
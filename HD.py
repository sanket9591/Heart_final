import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('heart1.csv')
df = df.drop_duplicates()
#print(df.shape)
X = np.array(df.iloc[:, 0:11])
y = np.array(df.iloc[:, 11:])

from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y.reshape(-1))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[[0,3,4,7,10]] = scaler.fit_transform(X_train[[0,3,4,7,10]])
X_test[[0,3,4,7,10]] = scaler.transform(X_test[[0,3,4,7,10]])
from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(X_train,y_train)
# y_preden=sv.predict(X_test)
# accuracy_4m=accuracy_score(y_test,y_preden)*100
# print('svm accuracy : ',accuracy_4m)
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
m1 = LogisticRegression(max_iter=1000)
m2 = RandomForestClassifier()
# m3 = GaussianNB() ('GNB', m3),
m4 = SVC(kernel='linear',probability=True)
m5 = DecisionTreeClassifier()
m6 = KNeighborsClassifier(n_neighbors=19)
eclf = VotingClassifier(estimators=[('LR', m1), ('RF', m2), ('SVC', m4),('DT',m5),('KNN',m6)],
                        voting='soft') 

eclf.fit(X_train, y_train)
y_predens=eclf.predict(X_test)
accuracy_4m=accuracy_score(y_test,y_predens)*100
print('ensemble : ',accuracy_4m)

pickle.dump(eclf, open('mypk.pkl', 'wb'))
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Seed for repetiability
np.random.seed(31415)

#Numero variables 15
column_names = ["edad", "sexo", 
                "escolaridad", "tiempoconsintomas", "idg-acr", "fiq-dolor", "fiq-fatiga", "fiq-calidad","autoreporte", "maltrato-fisico", "maltrato-psicologico","abusosexual",
                 "escala-evitacion","escala-depresion", "clase" 
                ]

df = pd.read_csv("datasocio-cuadro-maltratro-escala.csv", names=column_names)
print(df.shape)
df.head()


X = df.iloc[:,:14] #all rows all columns upto 12
print(X.head())

y = df['clase']
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X_test.head()


scaler = StandardScaler() #normalize means we need to give a range in which it is expecting
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train[:5,:]


clf = svm.SVC(kernel='sigmoid') #scikit learn support vector classifier
clf.fit(X_train, y_train)


y_pred = clf.predict(X_train)
print(y_pred)
print(accuracy_score(y_train, y_pred))


for k in ('linear', 'poly','rbf','sigmoid'):
    clf = svm.SVC(kernel=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(k)
    print(accuracy_score(y_train, y_pred)) #this process of trying different parmeters for our svm is known as Hyperparameter optimisation

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

patient = np.array([X_test.iloc[0]])
patient = scaler.transform(patient) #normalizing these features
print(clf.predict(patient))
print(y_test.iloc[0])

X_test = scaler.transform(X_test)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_zero = np.zeros(y_test.shape)
print(accuracy_score(y_test, y_zero))

print(classification_report(y_test, y_pred))
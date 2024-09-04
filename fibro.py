import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Numero variables 80
column_names = ["edad", "sexo", "escolaridad", "tiempo_diagnostico", "tiempo_sintomas", "ena_dolor", "ena_fatiga", "ena_rigidez", "ena_calidad_sueño",
                "dolor_actual", "dolor_max", "dolor_min", "dolor_gral", "parestesia", "fibroniebla", "fatiga", "insomnio", "despertares",
                "sueño_no_reparador", "rigidez", "calambres", "dolor_estomago", "intestino_irr", "dismenorrea", "vulvodinia", "dolor_orinar", "boca_seca",
                "presion_baja_pie", "sensibilidad_tacto", "sensibilidad_luz", "sensibilidad_ruido", "sensibilidad_olores", "sesibilidad_temperatura", "problemas_concentracion",
                "problemas_memoria", "dif_expresarse", "migraña", "tristeza", "ansiedad", "estrés", "enojo", "equilibrio", "ojo_seco",
                "toma_medicamento_dolor", "tiempo_alivio_medicamento", "hipertensión_arterial", "diabetes_mellitusII", "artritis_reumatoide", "espondilitis_anquilosant", "tiroides", 
                "artrosis_osteoartrosis", "radioculopatía", "colon_irritable", "lupus", "esclerosis_multiple", "maltrato_fisico_pasado", "maltrato_psicológico_pasado", "maltrato_sexual_pasado",
                "ningun_maltrato_pasado","prefirio_nodecir_pasado","maltrato_fisico_actual", "maltrato_psicológico_actual", "maltrato_sexual_actual","ningun_maltrato_actual","prefirio_nodecir_actual",
                "alcohol", "tabaco", "marihuana", "ninguna_sustancia", "hace_ejercicio_fisico", "hace_actividades_bajo_impacto", "escala_fiq-r", "escala_discapacidad",
                "pips_evitacion", "pips_fuscog", "pips_total", "escala_ansiedad", "escala_despresion", "ideacion_suicida", "clase"                
                ]

df = pd.read_csv("data.csv", names=column_names)
print(df.shape)
df.head()


X = df.iloc[:,:79] #all rows all columns upto 79
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

clf = svm.SVC(kernel='linear')
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
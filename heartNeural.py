import pandas as pd
import numpy as np
import warnings as wr
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder

# taxa de aprendizado
lr = 0.00001

# carregando dados
df = pd.read_csv("heart.csv")

# dataset de treino
cols_X = df[['age','cp','trestbps','restecg','thalach','exang','oldpeak','slope','ca','thal']]
# rotulos do dataset de treino
cols_y = df[['target']]
df_X = cols_X.copy()
df_y = cols_y.copy()
X_treino, X_teste, y_treino, y_teste = train_test_split(df_X, df_y, test_size=0.5, random_state=42)

mlp = nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=1500, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=lr, activation='identity')

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_treino, y_treino)

# resultado 
print("Resultados")
print("Score de treino: %f" % mlp.score(X_treino, y_treino))
print("Score do teste: %f" % mlp.score(X_teste, y_teste))


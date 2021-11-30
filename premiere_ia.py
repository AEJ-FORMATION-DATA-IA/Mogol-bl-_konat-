#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
import sqlite3
import sqlalchemy


# In[26]:



engine = sqlalchemy.create_engine('sqlite:///projCardiaque')


# # Importer le dataframe coeur

# In[27]:


data = pd.read_excel('Copie de Coeur.xlsx')
df = data.copy()
data


# # Preprocessing
# 

# In[28]:


#normalisé les variables quantitatives

def normalize_quant_var(df):
    #quant = df.select_dtypes('int')
    for i in df.select_dtypes('int'):
        df[i]= df[i] / df[i].max()
    return df
        

normalize_quant_var(data)

data



# In[29]:


#FILTRE

#notre dataframe ne contient pas de doublons 
data.duplicated().value_counts()
#data ne contient pas de données vide
data.isna().value_counts()
#il y a pas de constantes
data.nunique()


# In[30]:


#recoder les variables qualitatives
def recode_quality_var(serie_name):
     
    return serie_name.astype('category').cat.codes

def data_recode_qual(df):
    for col in df.select_dtypes('object').columns:
        df[col]=recode_quality_var(df[col])
    return df
    
data_recode_qual(data)
data.FCMAX.max()


# In[31]:


#deux objet X[age:pente] Y[coeur]

X = data.drop('CŒUR', axis= 1)
print(X)
# on recupere pour Y  df['CŒUR']  qui es les valeurs de coeur non mormalisées
Y = df['CŒUR']
Y


# # Apprentissage
# 

# In[32]:


#Decouper en apprentissage et en texte
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3, random_state = 0)

#initialisation du model de regression
model = LogisticRegression()



#Entrainement du model
model = model.fit(x_train,y_train)

#Verifions le taux d'exatitude de notre model 
model.score(x_test,y_test)

x_test


# In[41]:


conn = sqlite3.connect('projCardiaque')
cur = conn.cursor()
x_test.to_sql('x_test', engine, if_exists='replace',index=False)
req= cur.execute('select * from x_test').fetchall()
pd.read_sql('select * from x_test',conn)


# # Interpretation des perfommance du model

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[ ]:


# La matrice de confusion.

matrice_conf = confusion_matrix(y_test, model.predict(x_test))

#La matrice de confusion est une confrontation entre les valeurs réelles de la variable cible et les
#valeurs prédites par le modèle.

matrice_conf

x_test.shape


# # Remarques
# 
# ### le nombre de vrais négatifs est M(0,0) = matrice_conf[0,0] |  dans notre cas VN = 90
# ### les faux négatifs sont M(1,0) = matrice_conf[1,0]          |  dans notre cas FN = 24
# ### les vrais positifs sont M(1,1) = matrice_conf[1,1]         |  dans notre cas VP = 139
# ### les faux positifs sont (0,1) = matrice_conf[0,1]           |  dans notre cas FP = 23 
# 

# In[ ]:


#la sensibilité


recall_score(y_test,model.predict(x_test))

#Lorsque individu est positif on a 85.27 % de chance que le modèle arrive à bien le prédire (predire qu'il est positif)

#formule mathematique : sensibilité = VP / (VP + FP)


# In[ ]:


#precision 

precision_score(y_test,model.predict(x_test))

#Lorsque le modèle prédit individu est positif on a 85.80 % de chance que cela soit vrai.

#formule mathematique : prediction = VP / (VP + FN)


# In[ ]:


#Calculons le taux de succes de notre model d'apres la fomule suivante : 

accuracy = (matrice_conf[0,0]  + matrice_conf[1,1]) / (matrice_conf[1,1] + matrice_conf[0,0] + matrice_conf[1,0] + matrice_conf[0,1] )
#Multiplions le resultat par 100 pour l'avoire en pourcentage 
accuracy*100

data


# # Conclusion 
# 
# 
# ### D'apres nôtre accuracy (le taux de succes de notre model ) qui est d'environ 0.8297 soit 82.97% 
# ### Nous pouvons dit que nous avons un bon model , Cependant des questions restent posées .
# ### Avons nous eviter tout les biais liés a ce model ?
# ### Le model est t-il le meilleure possible ?
# 

# In[ ]:


#enregistrer le model 
pickle.dump( model ,open('model.pk','wb'))


# 

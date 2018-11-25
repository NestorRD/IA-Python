#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn.datasets import load_boston


# In[4]:


#cargamos la librería
boston = load_boston()

print(boston.DESCR)


# Formula minimizar el error cuadrático medio (MCO): $\beta = (X^{T}X)^{-1}X^{T}Y$

# In[ ]:


#Formula minimizar el error cuadrático medio (MCO): $\beta = (X^{T}X)^{-1}X^{T}Y$ ...PONEMOS EN MARKDOWN Y EJECUTAMOS
#^{T} = a matriz transpuesta.


# In[7]:


X = np.array(boston.data[:,5]) #de los atributos, queremos todos los campos del atributo 5, RM o numero medio de habitaciones por barrio
Y = np.array(boston.target) #Igualamos Y al valor medio.

print(X)
print(Y)

plt.scatter(X,Y, alpha = 0.2) #función para graficar

#Añadimos columna de 1s para termino independiente o w0.
X = np.array([np.ones(506), X]).T
print(X)

B = np.linalg.inv(X.T @ X) @ X.T @Y

plt.plot([4,9], [B[0]+B[1]*4, B[0]+B[1]*9], c="red") #linea que en el eje X empieza en 4 y acaba en 9 y en el eje Y
plt.show()


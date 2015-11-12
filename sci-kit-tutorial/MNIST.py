
# coding: utf-8

# In[1]:

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()


# In[2]:

print(digits.data)


# In[3]:

digits.target


# In[4]:

digits.images[0]


# In[5]:

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)


# In[6]:

clf.fit(digits.data[:-1], digits.target[:-1])


# In[7]:

# Drumroll please...
clf.predict(digits.data[-1:])


# In[10]:

# Refitting model
import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)  
clf.predict(X_test)



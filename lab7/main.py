# низкая обобщающая способность, но быстрее обучаются
import mglearn
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Классификатор BernoulliNB - бинарные
# ненулевые частоты
X = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0]])
Y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(Y):
    counts[label] = X[Y == label].sum(axis=0)
print("Частоты признаков:\n{}".format(counts))
clf = BernoulliNB()
clf.fit(X, Y)
print("clf.predict:\n" + str('BernoulliNB 2:3   ' + str(clf.predict(X[2:3]))))

# Классификатор MultinomialNB - счетные или дискретные
# среднее  значение каждого признака для каждого класса
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
Y = np.array([1, 2, 3, 4, 5, 6])
clf = MultinomialNB()
clf.fit(X, Y)
print('MultinomialNB 2:3    ' + str(clf.predict(X[2:3])))

# Классификатор GaussianNB - непрерывные
# записывает среднее значение и отклонение каждого признака
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print('GaussianNB -0.8,-1  ' + str(clf.predict([[-0.8, -1]])))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print('GaussianNB -0.8,-1  ' + str(clf_pf.predict([[-0.8, -1]])))
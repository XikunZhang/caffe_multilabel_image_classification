# from sklearn import cross_validation
#
# cv = cross_validation.KFold(103, n_folds=10, indices=False)


# from sklearn.model_selection import KFold
# X = ["a", "b", "c", "d"]
# kf = KFold(n_splits=4)
# for train, test in kf.split(X):
#     print("%s %s" % (train, test))
import numpy as np

a = np.array([1,2,3,4])
l = [0,3]
print(type(a[l]))
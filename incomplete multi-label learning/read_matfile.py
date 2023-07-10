import scipy.io
import scipy
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def read_matfile(dataset):

    path = "data/matfile/" + dataset
    mat= scipy.io.loadmat(path)
    # print(mat.keys())
    X = scipy.sparse.lil_matrix(mat['data'])
    Y = scipy.sparse.lil_matrix(mat['target'])
    Y=np.transpose(Y)
    return (X,Y)

if __name__ == '__main__':
    dataset = "ccd.mat"
    X,Y=read_matfile(dataset)
    X = X.todense()
    print(X)
    # a = np.array(Y.todense())
    # print(np.int64(a != 0).sum(0))

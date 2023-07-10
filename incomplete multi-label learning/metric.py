import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import sparse
import math
def precision_at_ks(true_Y, pred_Y, ks=[1,2,3,4,5]):
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[1]) for i in range(true_Y.shape[0])]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / k
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = np.mean(precs)
    return result


import numpy as np
from collections import Counter


def nDCG(y_true, y_pred, k):
    y_true[y_true == -1] = 0
    #    pred = copy.deepcopy(y_pred)
    ndcg = np.zeros(k)
    rank_mat = np.argsort(-y_pred)
    sumY = np.sum(y_true, 1).ravel()
    Ypred = np.zeros([y_true.shape[0], k])
    normFac = np.zeros([y_true.shape[0], k])

    for i in range(k):
        #        Jidx = np.array(pred.argmax(1)).ravel()
        Jidx = rank_mat[:, i]
        Iidx = np.array(range(len(Jidx)))
        lbls = y_true[Iidx, Jidx]
        #        pred[Iidx,Jidx] = 0
        Ypred[:, i] = lbls / np.log2(2 + i)
        sY = np.sum(Ypred[:, :i + 1], 1)

        normFac[:, i] = ((np.float32(sumY >= i + 1) + 1e-12) / np.log2(2 + i))
        sF = np.sum(normFac[:, :i + 1], 1)
        ndcg[i] = np.sum(sY / sF) / y_pred.shape[0]
    #        print np.sum(sY),sF.sum()
    return ndcg


def prec_at(y_true, y_pred, k):
    y_true[y_true == -1] = 0
    #    pred = copy.deepcopy(y_pred)
    p = np.zeros(k)
    rank_mat = np.argsort(-y_pred)
    add = 0
    for i in range(k):
        #        Jidx = np.array(pred.argmax(1)).ravel()
        Jidx = rank_mat[:, i]
        Iidx = np.array(range(len(Jidx)))
        lbls = y_true[Iidx, Jidx]
        add += lbls.sum()
        p[i] = add / (float(i + 1) * len(Jidx))
    #        pred[Iidx,Jidx] = 0

    return p

if __name__ == '__main__':
    y_true = np.array([[0, 0, 1, 1],[1,0,1,0]])
    y_scores = np.array([[0, 0, 0.3, 0.28],[0.9,0.2,0.1,0.4]])

    # y_true = np.array([[0, 0, 1, 1]])
    # y_scores = np.array([[0.1, 0.4, 0.3, 0.7]])
    # print(roc_auc_score(y_true, y_scores))
    print(y_true.shape)
    print(y_scores.shape)
    arr_sparse = sparse.csc_matrix(y_true)
    print(arr_sparse)
    performance = precision_at_ks(arr_sparse, y_scores)

    for k, s in performance.items():
        print('precision@{}: {:.4f}'.format(k, s))

    # print(nDCG(y_true,y_scores))
    print(prec_at(y_true, y_scores,4))
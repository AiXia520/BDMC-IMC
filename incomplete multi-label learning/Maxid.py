import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph
from knn import myKNN
from similarity import calEuclidDistanceMatrix
from laplacian import calLaplacianMatrix
from sklearn.metrics.pairwise import pairwise_kernels,pairwise_distances
from read_arff import *
from metric import *

def readSVMFile():
    train_file = open('yeast_train.svm','r')
    train_file_lines = train_file.readlines(100000000000000)
    train_file.close()
    train_fea = np.zeros((1500,103),dtype=float)
    train_label = np.zeros((1500,14),dtype=int)
    for k in range(1,len(train_file_lines)):
        data_segs = train_file_lines[k].split(' ')
        label_line = data_segs[0]
        labels = label_line.split(',')
        if (len(labels) == 0) or (labels[0] == ''):
            train_label[k-1,0] = 0
        else:
            for i in range(len(labels)):
                train_label[k-1,int(labels[i])-1] = 1

        for i in range(1,len(data_segs)-1):
            fea_pair = data_segs[i].split(':')
            fea_idx = int(fea_pair[0])
            fea_val = float(fea_pair[1])
            train_fea[k-1,fea_idx-1] = fea_val

    #### yeast: classes 14, data: 917, dimensions: 103
    test_file = open('yeast_test.svm','r')
    test_file_lines = test_file.readlines(100000000000000)
    test_file.close()
    test_fea = np.zeros((917,103),dtype=float)
    test_label = np.zeros((917,14),dtype=int)
    for k in range(1,len(test_file_lines)):
        data_segs = test_file_lines[k].split(' ')
        label_line = data_segs[0]
        labels = label_line.split(',')
        if (len(labels) == 0) or (labels[0] == ''):
            test_label[k-1,0] = 0
        else:
            for i in range(len(labels)):
                test_label[k-1,int(labels[i])-1] = 1

        for i in range(1,len(data_segs)-1):
            fea_pair = data_segs[i].split(':')
            fea_idx = int(fea_pair[0])
            fea_val = float(fea_pair[1])
            test_fea[k-1,fea_idx-1] = fea_val

    X = np.concatenate((train_fea,test_fea))
    Y = np.concatenate((train_label,test_label))
    return X,Y

def missing_preprocessing(X,Y,feature_fraction,label_fraction):
    fea_mask = np.random.random(X.shape)
    fea_loc = np.where(fea_mask < feature_fraction)
    random_mat = np.random.random(Y.shape)
    label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix

    featuremask = np.ones(X.shape)
    featuremask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0
    return featuremask,labelmask

# Speedup Matrix Completion with Side Information:Application to Multi-Label Learning
def Maxid(X, Y, U_init, V_init, W_init, H_init,fea_loc, label_loc, alpha, lambda0, lambda1,
                            lambda2, lambda3, delta, kx, lr, max_iter):

    # alpha_weight = tf.constant((1. - alpha), name='alpha', dtype=tf.float64)  ### 1 - alpha
    # beta_weight = tf.constant((2. * alpha - 1), name='beta', dtype=tf.float64)  ### 2* alpha -1

    mask = np.zeros(X.shape)
    mask[fea_loc] = 1.  ## observed entries
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # noisy,the other side information
        U = tf.Variable(initial_value=U_init, name='U', dtype=tf.float64)
        V = tf.Variable(initial_value=V_init.T, name='V', dtype=tf.float64)

        S = tf.matmul(U, V)
        debug_rs = tf.subtract(S, X, name='debug_rs')
        sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs, mask), 2), name='sqr_debug')

        U_frobenius_norm = tf.reduce_mean(tf.pow(U, 2, name='U_frobenius'))
        V_frobenius_norm = tf.reduce_mean(tf.pow(V, 2, name='V_frobenius'))
        norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
        regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

        W = tf.Variable(initial_value=W_init, name='W', dtype=tf.float64)
        H = tf.Variable(initial_value=H_init.T, name='H', dtype=tf.float64)
        B = tf.matmul(W, H)

        R = tf.subtract(tf.matmul(S,B), Y, name='res_label')
        sqr_label = tf.reduce_mean(tf.pow(tf.multiply(R, labelmask), 2), name="sqr_label")
        fused_cost = tf.add(sqr_fea, tf.multiply(sqr_label, delta))

      ##### regularization
        W_frobenius_norm = tf.reduce_mean(tf.pow(W, 2, name='W_frobenius'))
        H_frobenius_norm = tf.reduce_mean(tf.pow(H, 2, name='H_frobenius'))
        norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
        regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')
        cost_reg = tf.add(tf.add(fused_cost, regularizer_fea), regularizer_label)

        training_step = tf.train.RMSPropOptimizer(lr, 0.9, 0.0, 1e-10).minimize(cost_reg)
        # training_step = tf.train.AdagradOptimizer(lr).minimize(cost_reg)
        init = tf.initialize_all_variables()
        # sess = tf.Session()
        sess.run(init)
        print('initial cost function value: ' + str(sess.run(cost_reg)))

        for i in range(max_iter):
            print(str(i) + ' cost function value: ' + str(sess.run(cost_reg)))
            sess.run(training_step)

        return U.eval(sess), V.eval(sess), W.eval(sess), H.eval(sess)


# cost-sensitive
def DirtyIMC2(X, Y, U_init, V_init, W_init, H_init,P_init,Q_init,fea_loc, label_loc, alpha, lambda0, lambda1,
                            lambda2, lambda3, delta, kx, lr, max_iter):

    alpha_weight = tf.constant((1. - alpha), name='alpha', dtype=tf.float64)  ### 1 - alpha
    beta_weight = tf.constant((2. * alpha - 1), name='beta', dtype=tf.float64)  ### 2* alpha -1

    mask = np.zeros(X.shape)
    mask[fea_loc] = 1.  ## observed entries
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    with tf.Session() as sess:

        # noisy,the other side information
        U = tf.Variable(initial_value=U_init, name='U', dtype=tf.float64)
        V = tf.Variable(initial_value=V_init.T, name='V', dtype=tf.float64)

        P = tf.Variable(initial_value=P_init, name='P', dtype=tf.float64)
        Q = tf.Variable(initial_value=Q_init.T, name='Q', dtype=tf.float64)

        S = tf.matmul(U, V)
        debug_rs = tf.subtract(S, X, name='debug_rs')
        sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs, mask), 2), name='sqr_debug')

        U_frobenius_norm = tf.reduce_mean(tf.pow(U, 2, name='U_frobenius'))
        V_frobenius_norm = tf.reduce_mean(tf.pow(V, 2, name='V_frobenius'))
        norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
        regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

        W = tf.Variable(initial_value=W_init, name='W', dtype=tf.float64)
        H = tf.Variable(initial_value=H_init.T, name='H', dtype=tf.float64)
        B = tf.matmul(W, H)
        N=  tf.matmul(P, Q)
        R = tf.subtract(tf.add(tf.matmul(S,B),N), Y, name='res_label')
        sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(tf.multiply(R, labelmask), 2), beta_weight), name="sqr_label1")
        sqr_label_p2 = tf.reduce_mean(tf.multiply(tf.pow(R, 2), alpha_weight), name="sqr_label2")
        sqr_label = tf.add(sqr_label_p1, sqr_label_p2)
        fused_cost = tf.add(sqr_fea, tf.multiply(sqr_label, delta))

      ##### regularization
        W_frobenius_norm = tf.reduce_mean(tf.pow(W, 2, name='W_frobenius'))
        H_frobenius_norm = tf.reduce_mean(tf.pow(H, 2, name='H_frobenius'))
        norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
        regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')
        cost_label = tf.add(tf.add(fused_cost, regularizer_fea), regularizer_label)

        P_frobenius_norm = tf.reduce_mean(tf.pow(P, 2, name='P_frobenius'))
        Q_frobenius_norm = tf.reduce_mean(tf.pow(Q, 2, name='Q_frobenius'))
        norm_sums_noisy = tf.add(P_frobenius_norm, Q_frobenius_norm, name='norm_sums_noisy')
        regularizer_noisy = tf.multiply(norm_sums_noisy, lambda2, 'regularizer_noisy')
        cost_reg = tf.add(cost_label, regularizer_noisy)


        training_step = tf.train.RMSPropOptimizer(lr, 0.9, 0.0, 1e-10).minimize(cost_reg)
        # training_step = tf.train.AdagradOptimizer(lr).minimize(cost_reg)
        init = tf.initialize_all_variables()
        # sess = tf.Session()
        sess.run(init)
        print('initial cost function value: ' + str(sess.run(cost_reg)))

        for i in range(max_iter):
            print(str(i) + ' cost function value: ' + str(sess.run(cost_reg)))
            sess.run(training_step)

        return U.eval(sess), V.eval(sess), W.eval(sess), H.eval(sess),P.eval(sess), Q.eval(sess),

# 使用cost-sensitive 重建Y的同时，加入了特征信息，加入了图信息
def DirtyIMC3(X, Y, U_init, V_init, W_init, H_init,P_init,Q_init,fea_loc, label_loc, alpha, lambda0, lambda1,
                            lambda2, lambda3, delta, kx, lr, max_iter):

    alpha_weight = tf.constant((1. - alpha), name='alpha', dtype=tf.float64)  ### 1 - alpha
    beta_weight = tf.constant((2. * alpha - 1), name='beta', dtype=tf.float64)  ### 2* alpha -1

    mask = np.zeros(X.shape)
    mask[fea_loc] = 1.  ## observed entries
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    with tf.Session() as sess:

        # noisy,the other side information
        U = tf.Variable(initial_value=U_init, name='U', dtype=tf.float64)
        V = tf.Variable(initial_value=V_init.T, name='V', dtype=tf.float64)

        P = tf.Variable(initial_value=P_init, name='P', dtype=tf.float64)
        Q = tf.Variable(initial_value=Q_init.T, name='Q', dtype=tf.float64)

        S = tf.matmul(U, V)
        debug_rs = tf.subtract(S, X, name='debug_rs')
        sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs, mask), 2), name='sqr_debug')

        U_frobenius_norm = tf.reduce_mean(tf.pow(U, 2, name='U_frobenius'))
        V_frobenius_norm = tf.reduce_mean(tf.pow(V, 2, name='V_frobenius'))
        norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
        regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

        W = tf.Variable(initial_value=W_init, name='W', dtype=tf.float64)
        H = tf.Variable(initial_value=H_init.T, name='H', dtype=tf.float64)
        B = tf.matmul(W, H)
        N=  tf.matmul(P, Q)
        R = tf.subtract(tf.add(tf.matmul(S,B),N), Y, name='res_label')
        sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(tf.multiply(R, labelmask), 2), beta_weight), name="sqr_label1")
        sqr_label_p2 = tf.reduce_mean(tf.multiply(tf.pow(R, 2), alpha_weight), name="sqr_label2")
        sqr_label = tf.add(sqr_label_p1, sqr_label_p2)
        fused_cost = tf.add(sqr_fea, tf.multiply(sqr_label, delta))

      ##### regularization
        W_frobenius_norm = tf.reduce_mean(tf.pow(W, 2, name='W_frobenius'))
        H_frobenius_norm = tf.reduce_mean(tf.pow(H, 2, name='H_frobenius'))
        norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
        regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')
        cost_label = tf.add(tf.add(fused_cost, regularizer_fea), regularizer_label)

        P_frobenius_norm = tf.reduce_mean(tf.pow(P, 2, name='P_frobenius'))
        Q_frobenius_norm = tf.reduce_mean(tf.pow(Q, 2, name='Q_frobenius'))
        norm_sums_noisy = tf.add(P_frobenius_norm, Q_frobenius_norm, name='norm_sums_noisy')
        regularizer_noisy = tf.multiply(norm_sums_noisy, lambda2, 'regularizer_noisy')
        cost_1 = tf.add(cost_label, regularizer_noisy)

        init = tf.initialize_all_variables()
        sess.run(init)

        #  graph-regularized
        # L = kneighbors_graph(sess.run(tf.matmul(U,V)), 10, mode='distance', metric='cosine',include_self=True, n_jobs=-1)
        # L=pairwise_kernels(sess.run(tf.matmul(U,V)), sess.run(tf.matmul(U,V)), metric='rbf')
        # D = tf.Variable(initial_value=tf.convert_to_tensor(np.array(L.toarray())), name='D', dtype=tf.float64)

        L = 0
        Similarity = calEuclidDistanceMatrix(sess.run(tf.matmul(U, V)))
        for i in [0.1,1]:
            for k in [5, 10]:
                Adjacent = myKNN(Similarity, k=k, sigma=i)
                result = calLaplacianMatrix(Adjacent)
                L = L + result * (1 / 4)
        # L=L/18

        # Similarity = calEuclidDistanceMatrix(sess.run(tf.matmul(U, V)))
        # Adjacent = myKNN(Similarity, k=40, sigma=1)
        # L = calLaplacianMatrix(Adjacent)

        # D = tf.convert_to_tensor(np.array(L.toarray()), tf.float64, name='D')
        D = tf.convert_to_tensor(L, tf.float64, name='D')
        cost2 = tf.trace(
            tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.transpose(W), tf.transpose(tf.matmul(U, V))), D), tf.matmul(U, V)),
                      W))
        regularizer_graph = tf.multiply(cost2, lambda3, 'regularizer_graph')
        cost_reg = tf.add(cost_1, regularizer_graph)

        training_step = tf.train.RMSPropOptimizer(lr, 0.9, 0.0, 1e-10).minimize(cost_reg)
        # training_step = tf.train.AdagradOptimizer(lr).minimize(cost_reg)
        init = tf.initialize_all_variables()
        # sess = tf.Session()
        sess.run(init)
        print('initial cost function value: ' + str(sess.run(cost_reg)))

        for i in range(max_iter):
            print(str(i) + ' cost function value: ' + str(sess.run(cost_reg)))
            sess.run(training_step)

        return U.eval(sess), V.eval(sess), W.eval(sess), H.eval(sess),P.eval(sess), Q.eval(sess),
if __name__ == '__main__':
    # X,Y=readSVMFile()

    dataset="bibtex"
    label_count=159

    # dataset="mediamill"
    # label_count=101

    # dataset="medical"
    # label_count=45
    
    # dataset="yeast"
    # label_count=14

    path = "data/" + dataset
    # 读入ARFF文件数据
    X, Y = read_arff(path, label_count)
    X=X.todense()
    Y=Y.todense()
    # print(pd.DataFrame(np.array(X)))
    # featuremask,labelmask=missing_preprocessing(X,Y,0.6,0.8)
    alpha = (1.+0.8)/2.
    kx = 10
    fea_fraction = 0.6
    label_fraction = 0.8
    nrank=15
    lrank=15
    lambda0=0.001
    lambda1=0.001
    lambda2=0.001
    lambda3=0.001
    delta=0.01

    # X=X+np.random.normal(loc=0,scale=0.01,size=(X.shape[0], X.shape[1]))

    mean_vector = np.zeros(X.shape[1])
    cov_mat = np.eye(X.shape[1])
    ffproj = np.random.multivariate_normal(mean_vector, cov_mat, 1000)
    no_of_freq = ffproj.shape[0]

    ind_yeast_data = np.array(range(X.shape[0]))
    train_auc_score = []
    test_auc_score = []
    nsample = X.shape[0]
    num_train = int(nsample * 0.8)
    num_test = nsample - num_train
    np.random.shuffle(ind_yeast_data)
    train_data = X[ind_yeast_data[0:num_train],:]
    test_data = X[ind_yeast_data[num_train:],:]
    train_label = Y[ind_yeast_data[0:num_train],:]
    test_label = Y[ind_yeast_data[num_train:],:]

    fea_mask = np.random.random(train_data.shape)
    fea_loc = np.where(fea_mask < (1. - fea_fraction))  ### indexes of the observed entries
    fea_loc_x = fea_loc[0]
    fea_loc_y = fea_loc[1]
    mask = np.zeros(train_data.shape)
    mask[fea_loc] = 1.
    fea_loc_test = np.where(mask < 1)

    pos_entries = np.where(train_label == 1)
    pos_ind = np.array(range(len(pos_entries[0])))
    np.random.shuffle(pos_ind)
    labelled_ind = pos_ind[0:int(float(len(pos_ind)) * (1 - label_fraction))]  # 20% of 1s are preserved
    labelled_mask = np.zeros(train_label.shape)
    for i in labelled_ind:
        labelled_mask[pos_entries[0][i], pos_entries[1][i]] = 1

    label_loc = np.where(labelled_mask == 1)  #### label_loc: observed entries
    label_loc_x = label_loc[0]
    label_loc_y = label_loc[1]
    label_loc_test = np.where(labelled_mask == 0)  #### label_loc_test: missing entries
    train_label_masked = train_label.copy()
    train_label_masked[label_loc_test] = 0.  #### weak label assignments

    U_init = np.random.random((train_data.shape[0], nrank))
    V_init = np.random.random((train_data.shape[1], nrank))

    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
    H_init_DirtyIMC= np.random.random((train_label.shape[1], lrank))

    P_init = np.random.random((train_label.shape[0], lrank))
    Q_init = np.random.random((train_label.shape[1], lrank))

    U,V,W,H,P,Q= DirtyIMC(train_data,train_label_masked,U_init,V_init,W_init_DirtyIMC,H_init_DirtyIMC,P_init,Q_init,
                        fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,nrank,0.01,2500)

    # U,V,W,H,P,Q= DirtyIMC2(train_data,train_label_masked,U_init,V_init,W_init_DirtyIMC,H_init_DirtyIMC,P_init,Q_init,
    #                     fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,nrank,0.01,2500)

    # U,V,W,H,P,Q= DirtyIMC3(train_data,train_label_masked,U_init,V_init,W_init_DirtyIMC,H_init_DirtyIMC,P_init,Q_init,
    #                     fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,nrank,0.01,2500)


    # 对DirtyIMC方法
    X_reconstructed = np.dot(U, V)
    Y_reconstructed = np.add(np.dot(np.dot(U,V),np.dot(W,H)), np.dot(P,Q))


    X_ground_truth = train_data[fea_loc_test]
    X_reconstruction = X_reconstructed[fea_loc_test]
    print('feature error: ' + str(np.linalg.norm(X_ground_truth-X_reconstruction)))

    ground_truth = train_label[label_loc_test].tolist()
    reconstruction = Y_reconstructed[label_loc_test].tolist()
    auc_score = roc_auc_score(np.transpose(np.array(ground_truth)), np.array(reconstruction))
    print('train auc: ' + str(auc_score))

    # precision@ and nDCG@
    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))
    # ground_truth_test = test_label
    # reconstruction_test = Y_test_reconstructed.tolist()
    # auc_score_test = roc_auc_score(np.array(ground_truth_test), np.array(reconstruction_test))  #### train_auc_score
    # print('test auc: ' + str(auc_score_test))

    y_true = np.array(test_label)
    y_pred = 1 / (1 + np.exp((-1) * np.array(Y_test_reconstructed)))
    # y_pred=y_pred.reshape((1,-1))
    print(y_true.shape)
    print(y_pred.shape)

    # precision@
    prec_ats = prec_at(y_true, y_pred, 5)
    k = 0
    for prec_at in prec_ats:
        k = k + 1
        print('precision@{}: {:.4f}'.format(k, prec_at))

    # nDCG@
    nDCGs = nDCG(y_true, y_pred, 5)
    k = 0
    for nDCG in nDCGs:
        k = k + 1
        print('nDCG@{}: {:.4f}'.format(k, nDCG))


    # Y_test_reconstructed = np.dot(test_data, B)
    # # Y_test_reconstructed = test_func(test_data, ffproj, B)
    # ground_truth_test = test_label.tolist()
    # reconstruction_test = Y_test_reconstructed.tolist()
    # auc_score_test = roc_auc_score(np.array(ground_truth_test), np.array(reconstruction_test))  #### train_auc_score
    # print('test auc: ' + str(auc_score_test))

    # y_true = sparse.csc_matrix(ground_truth)
    # print(y_true)
    # performance = precision_at_ks(y_true, np.array(y_pred))
    # print(performance)
    # for k, s in performance.items():
    #     print('precision@{}: {:.4f}'.format(k, s))
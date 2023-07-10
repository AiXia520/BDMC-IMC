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
from sklearn.decomposition import TruncatedSVD

# Transduction with Matrix Completion:Three Birds with One Stone


def MC_1(X, Y, vl_reduced, U_init, V_init, W_init, H_init, fea_loc, label_loc, lambda0,lambda1,lr, max_iter):
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1.  ## observed entries
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        U = tf.Variable(initial_value=U_init, name='U', dtype=tf.float64)
        V = tf.Variable(initial_value=V_init.T, name='V', dtype=tf.float64)

        W = tf.Variable(initial_value=W_init, name='P', dtype=tf.float64)
        H = tf.Variable(initial_value=H_init.T, name='Q', dtype=tf.float64)

        U_frobenius_norm = tf.reduce_mean(tf.pow(U, 2, name='U_frobenius'))
        V_frobenius_norm = tf.reduce_mean(tf.pow(V, 2, name='V_frobenius'))
        norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')

        W_frobenius_norm = tf.reduce_mean(tf.pow(W, 2, name='W_frobenius'))
        H_frobenius_norm = tf.reduce_mean(tf.pow(H, 2, name='H_frobenius'))
        norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')

        UV = tf.matmul(U, V)
        WH= tf.matmul(W, H)
        XUV = tf.matmul(vl_reduced, UV)
        R = tf.subtract(tf.add(XUV,WH), Y, name='res_label')
        sqr_label = tf.reduce_mean(tf.pow(tf.multiply(R, labelmask), 2), name="sqr_label")
        regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')
        regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')
        regularizer_fea_label = tf.add(regularizer_fea, regularizer_label)
        cost_reg=tf.add(sqr_label,regularizer_fea_label)

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


if __name__ == '__main__':
    # X,Y=readSVMFile()

    # 读入arff数据

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

    # 加入noisy
    # X=X+np.random.normal(loc=0,scale=0.01,size=(X.shape[0], X.shape[1]))

    mean_vector = np.zeros(X.shape[1])
    cov_mat = np.eye(X.shape[1])
    ffproj = np.random.multivariate_normal(mean_vector, cov_mat, 1000)
    no_of_freq = ffproj.shape[0]

    # 划分数据集，80%为训练集，20%为测试集
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

    # 处理特征，保留60%特征
    fea_mask = np.random.random(train_data.shape)
    fea_loc = np.where(fea_mask < (1. - fea_fraction))  ### indexes of the observed entries
    fea_loc_x = fea_loc[0]
    fea_loc_y = fea_loc[1]
    mask = np.zeros(train_data.shape)
    mask[fea_loc] = 1.
    fea_loc_test = np.where(mask < 1)

    # 处理标签，20%正标记保留，其余为无标记数据
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


    # 对DirtyIMC 方法进行初始化，N为论文中其他信息;W/H用于side information;P/Q 用于N
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


    # 特征重建误差
    X_ground_truth = train_data[fea_loc_test]
    X_reconstruction = X_reconstructed[fea_loc_test]
    print('feature error: ' + str(np.linalg.norm(X_ground_truth-X_reconstruction)))

    # 标签重建AUC
    ground_truth = train_label[label_loc_test].tolist()
    reconstruction = Y_reconstructed[label_loc_test].tolist()
    auc_score = roc_auc_score(np.transpose(np.array(ground_truth)), np.array(reconstruction))
    print('train auc: ' + str(auc_score))

    # 测试集预测precision@ and nDCG@
    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))
    # ground_truth_test = test_label
    # reconstruction_test = Y_test_reconstructed.tolist()
    # auc_score_test = roc_auc_score(np.array(ground_truth_test), np.array(reconstruction_test))  #### train_auc_score
    # print('test auc: ' + str(auc_score_test))

    # 标签sigmoid
    y_true = np.array(test_label)
    y_pred = 1 / (1 + np.exp((-1) * np.array(Y_test_reconstructed)))
    # y_pred=y_pred.reshape((1,-1))
    print(y_true.shape)
    print(y_pred.shape)

    # 计算precision@
    prec_ats = prec_at(y_true, y_pred, 5)
    k = 0
    for prec_at in prec_ats:
        k = k + 1
        print('precision@{}: {:.4f}'.format(k, prec_at))

    # 计算nDCG@
    nDCGs = nDCG(y_true, y_pred, 5)
    k = 0
    for nDCG in nDCGs:
        k = k + 1
        print('nDCG@{}: {:.4f}'.format(k, nDCG))

    # 测试集预测AUC
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
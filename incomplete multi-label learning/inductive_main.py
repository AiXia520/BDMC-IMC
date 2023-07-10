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
import scipy
import math
from BiasMC import *
from sklearn.decomposition import PCA
from read_matfile import *
from CoEmbed import *
from ColEmbed import *
from DirtyIMC import *
from  our_method import *
from MC_1 import *
from Maxid import *
from COCO import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
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


if __name__ == '__main__':
    # X,Y=readSVMFile()



    # dataset="bibtex"
    # label_count=159

    # dataset="mediamill"
    # label_count=101

    # dataset="medical"
    # label_count=45
    
    # dataset="yeast"
    # label_count=14

    # dataset="CHD_49"
    # label_count=6
    df = pd.DataFrame(columns=["data_set", "method", "X_result", "Y_result", "p@1", "p@2","p@3","p@4","p@5",
                               "nDCG@1","nDCG@2","nDCG@3","nDCG@4","nDCG@5","run_time"])

    # data_lists=["medical","enron", "Langlog", "GnegativeGO","PlantPseAAC","VirusGO","genbase"]
    # label_counts=[45,53,75, 8,12,6,27]


    # data_lists=["CAL500","bibtex","Corel5k","bookmarks", "delicious"]
    # label_counts=[174,159,374,208,983]
    #
    data_lists=["enron"]
    label_counts=[53]
    # methods = ["CoEmbed"]
    # methods=["BiasMC","DirtyIMC","CoEmbed","ColEmbed_L","ColEmbed_NL","Our_method"]
    # methods=["MC_1","Our_method2"]
    methods = ["MC_1", "Maxid","COCO" ,"DirtyIMC", "Bias_DirtyIMC", "CoEmbed", "ColEmbed_L", "ColEmbed_NL", "Our_method", "Our_method2"]
    for i, val in enumerate(data_lists):
        path = "data/" + val
        label_count=label_counts[i]

        X, Y = read_arff(path, label_count)
        # X, Y = read_matfile(val)
        X = X.todense()
        # X = PCA(n_components=500).fit_transform(X)
        Y = Y.todense()
        prec_ats = []
        nDCGs = []
        y_pred = []
        alpha = (1. + 0.8) / 2.
        kx = 10
        fea_fraction = 0.6
        label_fraction = 0.8
        nrank = 30
        lrank = 30
        lambda0 = 0.001
        lambda1 = 0.001
        lambda2 = 0.001
        lambda3 = 0.001
        lambda4 = 0.01
        lambda5 = 0.001
        delta =0.01
        # random k_fold in 5
        for k_fold in range(0,5):

            mean_vector = np.zeros(X.shape[1])
            cov_mat = np.eye(X.shape[1])
            ffproj = np.random.multivariate_normal(mean_vector, cov_mat, 1)
            no_of_freq = ffproj.shape[0]

            # ffproj=np.ones((1,X.shape[1]))
            # no_of_freq = ffproj.shape[0]
            ind_yeast_data = np.array(range(X.shape[0]))
            train_auc_score = []
            test_auc_score = []
            nsample = X.shape[0]
            num_train = int(nsample * 0.8)
            num_test = nsample - num_train
            np.random.shuffle(ind_yeast_data)
            train_data = X[ind_yeast_data[0:num_train], :]
            test_data = X[ind_yeast_data[num_train:], :]
            train_label = Y[ind_yeast_data[0:num_train], :]
            test_label = Y[ind_yeast_data[num_train:], :]

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

            for method in methods:

                # Matrix Completion with Noisy Side Information
                # using feature side information
                if(method=="MC_1"):

                    nsample = X.shape[0]
                    ndim = X.shape[1]
                    nlabel = Y.shape[1]

                    mask = np.zeros(X.shape)
                    mask[fea_loc] = 1.  ## observed entries
                    labelmask = np.zeros(Y.shape)
                    labelmask[label_loc] = 1.

                    vl_reduced = TruncatedSVD(algorithm='randomized', n_components=30).fit_transform(train_data)

                    ndim_reduced = vl_reduced.shape[1]

                    U_init = np.random.random((ndim_reduced, 30))
                    V_init = np.random.random((nlabel, 30))

                    W_init = np.random.random((train_label.shape[0], 30))
                    H_init = np.random.random((nlabel, 30))
                    start_time=time.time()
                    U, V, W, H = MC_1(train_data, train_label_masked, vl_reduced,U_init, V_init, W_init, H_init,
                                                        fea_loc, label_loc,0.01,0.01, 0.01,2500)

                    UV = np.dot(U, V)
                    XUV = np.dot(vl_reduced,UV)
                    WH = np.dot(W,H)
                    Y_reconstructed = XUV + WH
                    end_time = time.time()
                    total_time = end_time - start_time

                elif (method == "COCO"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    Z_init = np.random.random((train_data.shape[0], train_label.shape[1]))
                    M = np.append(train_data, Z_init, axis=1)

                    P_init = np.random.random((M.shape[0], lrank))
                    Q_init = np.random.random((M.shape[1], lrank))
                    start_time = time.time()
                    U, V, P,Q = COCO(train_data, train_label_masked, U_init, V_init, W_init_DirtyIMC,
                                       H_init_DirtyIMC,P_init,Q_init,
                                       fea_loc, label_loc, alpha, lambda0, lambda1, lambda2, lambda3, delta,
                                       nrank,
                                       0.01, 2500)

                    X_reconstructed = np.dot(U, V)
                    M=np.dot(P,Q)
                    Y_reconstructed = M[:,X.shape[1]:]
                    end_time = time.time()
                    total_time = end_time - start_time


                elif(method=="Maxid"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    start_time=time.time()
                    U, V, W, H = Maxid(train_data, train_label_masked, U_init, V_init, W_init_DirtyIMC,
                                                H_init_DirtyIMC,
                                                fea_loc, label_loc, alpha, lambda0, lambda1, lambda2, lambda3, delta,
                                                nrank,
                                                0.01, 2500)
                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.dot(np.dot(U, V), np.dot(W, H))
                    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))

                    end_time = time.time()
                    total_time = end_time - start_time


                elif(method=="DirtyIMC"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    P_init = np.random.random((train_label.shape[0], lrank))
                    Q_init = np.random.random((train_label.shape[1], lrank))

                    start_time=time.time()
                    U, V, W, H, P, Q = DirtyIMC(train_data, train_label_masked, U_init, V_init, W_init_DirtyIMC,
                                                 H_init_DirtyIMC, P_init, Q_init,
                                                 fea_loc, label_loc, alpha, lambda0, lambda1, lambda2, lambda3, delta,
                                                 nrank,
                                                 0.01, 2500)
                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.add(np.dot(np.dot(U, V), np.dot(W, H)), np.dot(P, Q))
                    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))

                    end_time = time.time()
                    total_time = end_time - start_time

                # Matrix Completion with Noisy Side Information
                # using cost-sensitive reconstructed Y and adding feature side information
                elif(method=="Bias_DirtyIMC"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    P_init = np.random.random((train_label.shape[0], lrank))
                    Q_init = np.random.random((train_label.shape[1], lrank))

                    start_time=time.time()
                    U, V, W, H, P, Q = DirtyIMC2(train_data, train_label_masked, U_init, V_init, W_init_DirtyIMC,
                                                H_init_DirtyIMC, P_init, Q_init,
                                                fea_loc, label_loc, alpha, lambda0, lambda1, lambda2, lambda3, delta, nrank,
                                                0.01, 2500)
                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.add(np.dot(np.dot(U, V), np.dot(W, H)), np.dot(P, Q))
                    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))

                    end_time = time.time()
                    total_time = end_time - start_time

                # Convex Co-Embedding for Matrix Completion with predictive side information
                elif(method=="CoEmbed"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))
                    W_init = np.random.random((train_label.shape[0], lrank))
                    H_init = np.random.random((train_label.shape[1], lrank))
                    B_init_CoEmbed = np.random.random((train_data.shape[1], lrank))
                    unit_vector = np.ones((train_data.shape[0], 1), dtype=float)
                    b = np.random.random((train_label.shape[1], 1))

                    start_time=time.time()
                    U, V, W, H, B, b = CoEmbed(train_data, train_label_masked, U_init, V_init, W_init, H_init,
                                               B_init_CoEmbed, unit_vector, b, fea_loc, label_loc,
                                               alpha, lambda0, lambda1, lambda2, lambda3, delta, nrank, 0.01, 2500)

                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.dot(W, H)
                    Y_test_reconstructed = np.dot(np.dot(test_data, B), H) + b
                    end_time = time.time()
                    total_time = end_time - start_time

                # Multi-label Learning with Highly Incomplete Data via Collaborative Embedding

                elif(method=="ColEmbed_L"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))
                    W_init = np.random.random((train_label.shape[0], lrank))
                    H_init = np.random.random((train_label.shape[1], lrank))
                    B_init = np.random.random((train_data.shape[1], train_label.shape[1]))
                    B_init_nonlinear = np.random.random((2 * no_of_freq, train_label_masked.shape[1]))

                    start_time=time.time()
                    U, V, W, H, B = ColEmbed_L(train_data, train_label_masked, U_init, V_init, W_init, H_init, B_init,
                                               fea_loc, label_loc,
                                               alpha, lambda0, lambda1, lambda2, lambda3, delta, nrank, 0.01, 2500)

                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.dot(W, H)
                    Y_test_reconstructed = np.dot(test_data, B)

                    end_time = time.time()
                    total_time = end_time - start_time

                # Multi-label Learning with Highly Incomplete Data via Collaborative Embedding
                # non-linear
                elif(method=="ColEmbed_NL"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))
                    W_init = np.random.random((train_label.shape[0], lrank))
                    H_init = np.random.random((train_label.shape[1], lrank))
                    B_init = np.random.random((train_data.shape[1], train_label.shape[1]))
                    B_init_nonlinear = np.random.random((2 * no_of_freq, train_label_masked.shape[1]))

                    start_time=time.time()
                    U,V,W,H,B = ColEmbed_NL(train_data,train_label_masked,U_init,V_init,W_init,H_init,B_init_nonlinear,ffproj,fea_loc,label_loc,
                                                        alpha,lambda0,lambda1,lambda2,lambda3,delta,nrank,5,3500)

                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.dot(W, H)
                    Y_test_reconstructed = test_func(test_data, ffproj, B)

                    end_time = time.time()
                    total_time = end_time - start_time

                # based on Bias_DirtyIMC,adding ensemble graph information
                elif(method=="Our_method"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    P_init = np.random.random((train_label.shape[0], lrank))
                    Q_init = np.random.random((train_label.shape[1], lrank))

                    start_time=time.time()
                    U, V, W, H, P, Q ,knn_time= our_method(train_data, train_label_masked, U_init, V_init, W_init_DirtyIMC,
                                                  H_init_DirtyIMC, P_init, Q_init,
                                                  fea_loc, label_loc, alpha, lambda0, lambda1, lambda2, lambda3, delta,
                                                  nrank, 0.01, 2500,50)

                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.add(np.dot(np.dot(U, V), np.dot(W, H)), np.dot(P, Q))
                    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))
                    end_time = time.time()
                    total_time = end_time - start_time-knn_time

                # co-embedding framework for transductive multi-label learning
                elif(method=="Our_method2"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    P_init = np.random.random((train_label.shape[0], lrank))
                    Q_init = np.random.random((train_label.shape[1], lrank))
                    G_init = np.random.random((train_label.shape[0], lrank))

                    # N_init = np.random.random((train_label.shape[0], train_label.shape[1]))
                    unit_vector = np.ones((train_data.shape[0], 1), dtype=float)
                    b = np.random.random((train_label.shape[1], 1))

                    start_time=time.time()
                    U,V,W,H,P,Q,G,knn_time= our_method2(train_data,train_label_masked,U_init,V_init,W_init_DirtyIMC,
                    H_init_DirtyIMC,P_init,Q_init,G_init,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,
                    lambda4,lambda5,delta,nrank,0.01,2500,50)

                    # U, V, W, H, N,G, knn_time = our_method4(train_data, train_label_masked, U_init, V_init,
                    #                                             W_init_DirtyIMC,
                    #                                             H_init_DirtyIMC, P_init, Q_init,unit_vector,b, G_init, fea_loc,
                    #                                             label_loc, alpha, lambda0, lambda1, lambda2, lambda3,
                    #                                             lambda4, lambda5, delta, nrank, 0.01, 2500, 30)

                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.add(np.dot(np.dot(U, V), np.dot(W, H)), np.dot(P, Q))
                    # Y_reconstructed = np.dot(G, H)
                    Y_test_reconstructed = np.dot(test_data, np.dot(W, H))

                    end_time = time.time()
                    total_time = end_time - start_time-knn_time

                # co-embedding framework for inductive multi-label learning
                elif(method=="Our_method3"):
                    U_init = np.random.random((train_data.shape[0], nrank))
                    V_init = np.random.random((train_data.shape[1], nrank))

                    W_init_DirtyIMC = np.random.random((train_data.shape[1], lrank))
                    H_init_DirtyIMC = np.random.random((train_label.shape[1], lrank))

                    P_init = np.random.random((train_label.shape[0], lrank))
                    Q_init = np.random.random((train_label.shape[1], lrank))
                    G_init = np.random.random((train_label.shape[0], lrank))
                    unit_vector = np.ones((train_data.shape[0], 1), dtype=float)
                    b = np.random.random((train_label.shape[1], 1))

                    U, V, W, H, P, Q, G, b, knn_time = our_method3(train_data, train_label_masked, U_init, V_init, W_init_DirtyIMC,
                                                         H_init_DirtyIMC, P_init,
                                                         Q_init, G_init, unit_vector, b, fea_loc, label_loc, alpha, lambda0,
                                                         lambda1, lambda2,
                                                         lambda3, lambda4, lambda5, delta, nrank, 0.01, 2500,100)

                    X_reconstructed = np.dot(U, V)
                    Y_reconstructed = np.dot(G, H)
                    Y_test_reconstructed = np.dot(test_data, np.dot(W, H)) + b

                else:
                    print("method error")
                    break

                if(method !="MC_1"):
                    X_ground_truth = train_data[fea_loc_test]
                    X_reconstruction = X_reconstructed[fea_loc_test]
                    X_result = np.linalg.norm(X_ground_truth - X_reconstruction)/np.linalg.norm(X_ground_truth)
                    # print('feature error: ' + str(np.linalg.norm(X_ground_truth - X_reconstruction)))
                else:
                    X_result=0

                ground_truth = train_label[label_loc_test].tolist()
                reconstruction = Y_reconstructed[label_loc_test].tolist()
                Y_result = roc_auc_score(np.transpose(np.array(ground_truth)), np.array(reconstruction))
                # print('train auc: ' + str(Y_result))

                # precision@ and nDCG@
                # Y_test_reconstructed = np.dot(test_data, np.dot(W, H))
                # ground_truth_test = test_label
                # reconstruction_test = Y_test_reconstructed.tolist()
                # auc_score_test = roc_auc_score(np.array(ground_truth_test), np.array(reconstruction_test))  #### train_auc_score
                # print('test auc: ' + str(auc_score_test))

                if (method != "MC_1"  and method !="COCO"):
                    y_true = np.array(test_label)
                    y_pred = 1 / (1 + np.exp((-1) * np.array(Y_test_reconstructed)))
                    # y_pred=y_pred.reshape((1,-1))

                    # precision@
                    prec_ats = prec_at(y_true, y_pred, 5)

                    # nDCG@
                    nDCGs = nDCG(y_true, y_pred, 5)
                else:
                    prec_ats=[0,0,0,0,0]
                    nDCGs=[0,0,0,0,0]

                df = df.append(pd.DataFrame(
                    {'data_set': [val], 'method': [method], 'X_result': [X_result], 'Y_result': [Y_result],
                     'p@1': [prec_ats[0]],'p@2': [prec_ats[1]],'p@3': [prec_ats[2]],'p@4': [prec_ats[3]],
                     'p@5': [prec_ats[4]],'nDCG@1': [nDCGs[0]],'nDCG@2': [nDCGs[1]],'nDCG@3': [nDCGs[2]],
                     'nDCG@4': [nDCGs[3]],'nDCG@5': [nDCGs[4]],'run_time':[total_time]}),
                    ignore_index=True)
                print(df)
            df.to_csv("result/run_time_result_enron.csv")
        df.to_csv("result/"+str(val) + "run_time_result.csv")
from skmultilearn.dataset import load_from_arff
import pandas as pd
import os
import sys
sys.path.append("data")

def read_arff(path,label_count):

    path_to_arff_file=path+".arff"
    arff_file_is_sparse = False
    X, Y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location="end",
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    n_samples, n_features = X.shape
    n_samples, label=Y.shape
    print("样本数："+str(n_samples)+"  特征数："+str(n_features)+"  标签数："+str(label))
    return(X, Y)

if __name__ == '__main__':
    dataset = "medical"
    path = "data"+"/" + dataset
    label_count = 45
    X, Y= read_arff(path, label_count)
    print(type(X))
    print(type(Y))
    # print(pd.DataFrame(Y.todense()))
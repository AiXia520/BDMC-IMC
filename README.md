# Bi-Directional Matrix Completion for Highly Incomplete Multi-label Learning via Co-embedding Predictive Side Information

Motivated by real-world applications such as recommendation systems and social networks where only “likes” or “friendships” are observed, we consider a challenging multi-label learning problem where the observed label consists only of positive and unlabeled entries and the feature matrix contains missing entries. The problem is an enhanced instance of PU (positive-unlabeled) learning. Due to highly incomplete data, traditional multi-label learning algorithms are not directly available in such a scenario. In this paper, we propose a bi-directional matrix completion approach that exploits the matrix low-rank property to recover missing feature entries and label entries. Specifically, we introduce a low-rank co-embedding framework that integrates a low-rank matrix complete model and prediction model to jointly recover missing entries. In our framework, the prediction model can be conducted efficiently by dividing the low-rank matrix into a part capturing feature information and a part capturing information outside the feature space. Furthermore, we incorporate label embedding and graph regularized embedding together to improve matrix completion performance, which not only takes feature graph-structured information into account but also simultaneously exploits label semantic structured information. We provide an efficient alternative minimization scheme to solve the proposed problem. The experiments on transductive and inductive incomplete multi-label learning demonstrate the effectiveness of our proposed approach.

# Incomplete Multi-label Learning

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
pip install tensorflow-gpu
pip install scikit-learn
pip install numpy
pip install scipy
```
### Datasets
it can be downloaded from the websites of KDIS http://www.uco.es/kdis/mllresources/



# Incomplete Multi-view Multi-label Learning


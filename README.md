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
they can be downloaded from the websites of KDIS http://www.uco.es/kdis/mllresources/, you also can use them from /data folder

### Run

For transductive Incomplete Multi-label Learning, you can run the transductive_main.py;

For inductive Incomplete Multi-label Learning, you can run the inductive_main.py;

### Parameter Analysis

We employed the Friedman test to statistically analyze the performance of the inductive incomplete multi-label learning algorithms systematically. 

In our approach, the parameters are mainly divided into two parts, $\lambda $ (i.e. ${\lambda _X}$, ${\lambda _M}$, ${\lambda _N}$, and ${\lambda _Y}$) and  $k$ (i.e. ${k_X}$, ${k_M}$,  ${k_N}$ and ${k_Y}$). The parameter $\lambda $ is the penalty parameter enforcing the low-rank constraint on the matrix approximation. For example, ${\lambda _X}$ is the penalty parameter on the feature matrix approximation of ${\bf{\hat X}}$, ${\lambda _Y}$ for label matrix approximation of ${\bf{\hat Y}}$, ${\lambda _M}$ and ${\lambda _N}$ for controlling the importance between features and residual. The parameter $k$ can be considered the nonconvex relaxation of the proposed problem with a low-dimensional latent space.

<p align="left"> 
<img width="850" src="https://github.com/AiXia520/BDMC-IMC/blob/main/incomplete multi-label learning/fig.png">
</p>


# Incomplete Multi-view Multi-label Learning

In many real-world applications, one instance is not only associated with multiple labels but also often has multiple heterogeneous feature representations from multiple views. For example, an image can be described by color information (RGB), shape cues (SIFT), the Histogram of Oriented Gradients (HOG) and so on, which is an important paradigm referred to as multi-view multi-label learning.

### Datasets
they can be downloaded from the websites http://lear.inrialpes.fr/people/guillaumin/data.php. Core15k, Pascal07, ESPGame, IAPRTC12, Mirflickr contains six views: HUE, SIFT, GIST, HSV, RGB and LAB.

### Run


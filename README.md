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
1. Please replace the path of your own in the 'data_cmp_func.m' file.
2. Add to path - Select folds and Subfolds of 'data' and  'measure' folds. 
3. Run the 'run_main.m' file for the final results.

Our code is largely borrowed from [NAIM3L](https://github.com/EverFAITH/NAIM3L/tree/main/NAIM3L)

### Experimental results

\begin{sidewaystable}
	\sidewaystablefn%
	\small
	\begin{center}
		\begin{minipage}{\textheight}
			\caption{Results on incomplete multi-view multi-label learning, where five datasets with the setting of 50\% incomplete views and 50\% missing labels. ‘$\downarrow$’ indicates that the smaller the value is, the better the performance. ‘$\uparrow$’ indicates that the larger the value is, the better the performance. The best results have been shown in bold.}\label{table:5.5}
			\setlength{\tabcolsep}{0.1mm}{
				\begin{tabular*}{\textheight}{@{\extracolsep{\fill}}cccccccc@{\extracolsep{\fill}}}
					\toprule
					Datasets                     & Metrics           & iMVWL \cite{tan2018incomplete}        & LSML \cite{zhang2018latent}              & ICM2L \cite{tan2019individuality}             & NAIM$^3$L-I \cite{li2021concise}    & NAIM$^3$L-II \cite{li2021concise}    & NAIM$^3$L-II+our method \\
					\midrule
					\multirow{4}{*}{Corel5k}     & Ranking Loss (($\downarrow$))      & 13.5 ± 0.33  & 23.2±0.001         & 20.5±0.003         & 17.27 ± 0.20 & 16.46 ± 0.21 & \textbf{14.16 ± 0.21}      \\
					& Hamming Loss ($\downarrow$)      & 2.16 ± 0.02  & -                  & -                  & \textbf{1.3 ± 0.00}   & \textbf{1.3 ± 0.00}   & 1.31± 0.00        \\
					& Average Precision ($\uparrow$) & 28.31 ± 0.72 & 25.6±0.001         & 27.9±0.004         & 30.20 ± 0.40 & 30.47 ± 0.36 & \textbf{30.82 ± 0.15}      \\
					& Macro AUC ($\uparrow$)         & 86.82 ± 0.32 & 77.4±0.001         & 79.7±0.003         & 82.99 ± 0.20 & 83.80 ± 0.21 & \textbf{84.73 ± 0.20}      \\
					\multirow{4}{*}{Pascal07}    & Ranking Loss (($\downarrow$))      & 26.34 ± 0.93 & 27.4±0.003         & 24.4±0.005         & 22.71 ± 0.18 & 22.65 ± 0.17 & \textbf{21.31 ± 0.32}        \\
					& Hamming Loss ($\downarrow$)      & 11.77 ± 0.38 & -                  & -                  & 7.17 ± 0.00  & 7.17 ± 0.00  & \textbf{7.04±0.02 }        \\
					& Average Precision ($\uparrow$) & 44.08 ± 1.74 & 44.6±0.001         & 45.2±0.001         & 48.64 ± 0.35 & 48.66 ± 0.35 & \textbf{48.97±0.14}        \\
					& Macro AUC ($\uparrow$)         & 76.72 ± 1.20 & 75.8±0.002         & 78.5±0.005         & 79.99 ± 0.17 & 80.55 ± 0.17 & \textbf{82.15±0.23}        \\
					\multirow{4}{*}{ESPGame}     & Ranking Loss (($\downarrow$))      & 19.28 ± 0.14 & 20.4±0.001         & 20.4±0.001         & 20.37 ± 0.20 & 20.2 ± 0.11  & \textbf{19.23 ± 0.13 }     \\
					& Hamming Loss ($\downarrow$)      & 2.81 ± 0.01  &                    &                    & 1.74 ± 0.00  & 1.74 ± 0.00  & \textbf{1.71 ± 0.01}       \\
					& Average Precision ($\uparrow$) & 24.19 ± 0.34 & 20.5±0.001         & 22±0.002           & 24.28 ± 0.20 & 24.34 ± 0.16 & \textbf{24.62 ± 0.14}      \\
					& Macro AUC ($\uparrow$)         & 81.29 ± 0.15 & 78.9±0.000         & 80.3±0.001         & 80.04 ± 0.20 & 80.24 ± 0.13 & \textbf{82.38 ± 0.14}      \\
					\multirow{4}{*}{IAPRTC12}    & Ranking Loss (($\downarrow$))      & 16.7 ± 0.27  & \multirow{4}{*}{-} & \multirow{4}{*}{-} & 17.48 ± 0.00 & 17.3 ± 0.00  & \textbf{16.73 ± 0.06}      \\
					& Hamming Loss ($\downarrow$)      & 3.15 ± 0.02  &                    &                    & \textbf{1.95 ± 0.00} & \textbf{1.95 ± 0.00}  & 1.96 ± 0.00       \\
					& Average Precision ($\uparrow$) & 23.54 ± 0.39 &                    &                    & 25.71 ± 0.10 & 25.76 ± 0.10 & \textbf{26.15 ± 0.15}      \\
					& Macro AUC ($\uparrow$)         & 83.55 ± 0.22 &                    &                    & 82.56 ± 0.10 & 82.76 ± 0.10 & \textbf{84.13 ± 0.06 }     \\
					\multirow{4}{*}{Mirflflickr} & Ranking Loss (($\downarrow$))      & 19.4 ± 1.11  & 23.5±0.003         & 20.4±0.001         & 15.95 ± 0.00 & 15.9 ± 0.00  & \textbf{15.24 ± 0.03}      \\
					& Hamming Loss ($\downarrow$)      & 16.02 ± 0.28 & -                  & -                  & 11.85 ± 0.00 & 11.85 ± 0.00 & \textbf{11.34 ± 0.01}      \\
					& Average Precision ($\uparrow$) & 49.48 ± 1.24 & 48.5±0.001         & 53.6±0.002         & 54.95 ± 0.20 & \textbf{54.98 ± 0.16} & 54.87 ± 0.14      \\
					& Macro AUC ($\uparrow$)         & 79.44 ± 1.46 & 76.9±0.001         & 79±0.001           & 83.33 ± 0.00 & 83.39 ± 0.00 & \textbf{83.47 ± 0.01}      \\
					\bottomrule                    
			\end{tabular*}}
		\end{minipage}
	\end{center}
\end{sidewaystable}


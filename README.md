@article{MFS-MRC_zhou2024,

  title = {Multi-Label feature selection based on Minimizing Redundant Correlation between Label-Specific and Common Features},
  
  author = {Gaozhi Zhou and Runxin Liand Zhenhong Shang and Xiaowu Li and Lianyin Jia },
  
  year = {2024},
  
  abstract = {Multi-label feature selection is an indispensable technology in the preprocessing of multi-label high-dimensional data. Approaches utilizing information theory and sparse models hold promise in 
this domain, demonstrating strong performance. And research indicates that the l1 and l2,1-norms of sparse models can distinguish label-specific and common features in the feature space.However, existing methods rarely integrate the learning of label feature features and common features into the multi-label feature selection model based on mutual information. Moreover, most methods ignore the redundant interference in the learning process of different features. This study introduces a novel multi-label feature selection approach (MFS-MRC, in short). Firstly, we utilize a least-squares problem based on the multi-label features selection of mutual information, so that the sparse models and other learning items can be naturally integrated into the model. Subsequently, to identify label-specific and common features in the feature space, respectively, we introduce two projection weight matrices constrained by the l1 and l2,1-norms. In particular, we define a non-zero correlation constraint that effectively minimizes the redundant correlation between the two matrices. Finally, a manifold regularization term is devised to preserve the local information of the feature space. To solve the optimization model with nonlinear binary regular
terms, we employ a novel solution approach called S-FISTA. Extensive experiments across 15 multi-label benchmark datasets, comparing against 11 top-performing multi-label feature selection methods, demonstrate the superior performance of MFS-MRC.},
  
  keywords = {Multi-label feature selection; Mutual information; Sparse model; Redundant correlation}
}

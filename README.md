@article{MFS-MFR_zhou2024,

  title = {Multi-label feature selection based on minimizing feature redundancy of mutual information},
  
  author = {Gaozhi Zhou and Runxin Liand Zhenhong Shang and Xiaowu Li and Lianyin Jia },
  
  year = {2024},
  
  abstract = {Multi-label feature selection is an indispensable technology in the preprocessing of multi-label high-dimensional data. Approaches utilizing information theory and sparse models hold promise in this domain, demonstrating strong performance. Although there have been extensive literatures using l1 and l{2,1}-norms to identify label-specific features and common features in the feature space, they all ignore the redundant information interference problem when different features are learned simultaneously. In this paper, considering that features and labels in multi-labeled data are usually not linearly related, the MFS-MFR method is proposed by firstly utilizing the nonlinear mutual information between features and labels as well as the sparse model to identify the important features in the feature space. Firstly, a multi-label mutual information loss term represented by least squares is introduced, and the coefficient matrix is used to fit the correlation information between features and labels in the mutual information space. Subsequently, based on the sparse characteristics of l1 and l{2,1}-norms, we identify specific and common features in the feature-label mutual information space by two coefficient matrices constrained by the l1 and l{2,1}-norms, respectively. In particular, we define a non-zero correlation constraint that effectively minimizes the redundant correlation between the two matrices. Finally, a manifold regularization term is devised to preserve the local information of the mutual information space. To solve the optimization model with nonlinear binary regular terms, we employ a novel solution approach called S-FISTA. Extensive experiments across 15 multi-label benchmark datasets, comparing against 11 top-performing multi-label feature selection methods, demonstrate the superior performance of MFS-MFR.},
  
  keywords = {Multi-label feature selection; Mutual information; Sparse model; Redundant correlation}
}

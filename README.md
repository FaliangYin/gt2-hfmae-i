# General Type-2 Hierarchical Fuzzy Model-Agnostic Explanation of Image Classification for XAI

This repository is the implementation of case studies of the paper
"General Type-2 Hierarchical Fuzzy Model-Agnostic Explanation of Image Classification for XAI",
which has been accepted for publication in the IEEE Transactions on Fuzzy Systems.

In this work, general type-2 hierarchical fuzzy model-agnostic explanation of image classification (GT2-HFMAE-I) is proposed to explain decisions of any closed box image classifier. First, the biaxial hierarchical framework is presented to formulate explanations with scalability along the spatial axis with local, domain and universe levels and the semantic axis with different segmentation granularity. Second, the model-agnostic algorithm is developed to identify explanatory features of image with semantic comprehensibility in low-dimensional space by combining superpixels and semantic segmentation, and train a general type-2 fuzzy logic system to approximate the closed box model with robustness to handle uncertainty in explanations. Third, a user interface, with feature salience to highlight image segments as the decision basis and semantic inference to present the decision logic via IF-THEN rules, is designed to deliver explanations to users with transferability.

---

## Citation

If you use this repository in your research, please cite the following paper:

```bibtex
@ARTICLE{gt2hfmaei,
  author={Yin, Faliang and Lam, Hak-Keung and Watson, David},
  journal={IEEE Transactions on Fuzzy Systems}, 
  title={General Type-2 Hierarchical Fuzzy Model-Agnostic Explanation of Image Classification for XAI}, 
  year={2026},
  volume={},
  number={},
  pages={1-10},
  keywords={Explainable AI (XAI);general type-2 fuzzy logic system (GT2FLS);general type-2 hierarchical fuzzy model-agnostic explanation of image classification (GT2-HFMAE-I)},
  doi={10.1109/TFUZZ.2026.3686974}}
```

---

## License

This repository is released under the MIT License. See the `LICENSE` file for details.



## Reference

The data sets and image classifiers used in the experiments are from the following works.

[1] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2818–2826, 2016.

[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” CoRR, vol. abs/1512.03385, 2015.

[3] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 2261–2269, IEEE Computer Society, 2017.

[4] M. Tan and Q. V. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA (K. Chaudhuri and R. Salakhutdinov, eds.), vol. 97 of Proceedings of Machine Learning Research, pp. 6105–6114, PMLR, 2019.

[5] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, “ImageNet large scale visual recognition challenge,” International Journal of Computer Vision, vol. 115, pp. 211–252, Dec. 2015.

[6] F.-F. Li, M. Andreeto, M. Ranzato, and P. Perona, “Caltech 101,” Apr 2022.

[7] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” Tech. Rep. CNS-TR-2011-001, California Institute of Technology, 2011.

[8] M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” in Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.

The image segmentation algorithms are from the following works:

[9] T. Lei, X. Jia, Y. Zhang, S. Liu, H. Meng, and A. K. Nandi, “Superpixel-based fast fuzzy C-means clustering for color image segmentation,” IEEE Transactions on Fuzzy Systems, vol. 27, no. 9, pp. 1753–1766, 2019.


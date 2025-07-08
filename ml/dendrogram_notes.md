# Notes on Dendrogram Features

See `create_dendrogram_v2.py` for image features that lie within Group A, Group B, and Group C. The following rationale was used for the creation of the MLP with these features:

* Four features from each group
* Two first-order features, two higher-order/texture features
* If possible, choose features that are as separate as possible (e.g. choose GLCM and Laws texture features within one group)
* If that is not possible, choose the same features that are as separate as possible (e.g. GLCM correlation with the smallest and largest combinations of distances and angles)

List of features for each group:

From Group A: img_median_intensity, img_mean_intensity, glcm_homogeneity_dist_10_angle_0, laws_L5R5_div_R5L5_mean

From Group B: img_variance_intensity, img_std_intensity, glcm_contrast_dist_5_angle_0, laws_E5E5_mean

From Group C: img_kurtosis_intensity, img_max_intensity, glcm_correlation_dist_1_angle_0, glcm_correlation_dist_50_angle_2_3562
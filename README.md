# BREIL Algonauts challenge
This is the repository that stores code of BREIL members for Algonauts 2023 challenge.
A more detailed report can be found [here](./Algonauts_challenge_report.pdf).

Little review of current existing files:

`algonauts_2023_challenge_tutorial.ipynb` - provided jupyter notebook by challenge hosts. It contains some visualizations of data, and baseline method for prediction using AlexNet.
> Remark: I removed this notebook from the repository due to its huge size. Please access it by [Link](https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link)

`alexnet_submission.py` - .py version of the above notebook. It iterates over 8 subjects and saves prediction scores. A baseline submission.

`coco_image_matching.ipynb` - notebook to match ids of NSD images to the original images from COCO to get corresponding annotations.

`roi_scores.ipynb` - notebook that compares different submission outputs. Eventually, the analysis was used to make ROI based submissions.

### Our works

`clip` - folder with scripts using CLIP model

`sam` - folder with scripts using Segment Anything

> `clip_sam_ensemble.ipynb` - computes average or best-ROI of CLIP and SAM predictions. In the final solution, the average was computed.

> `clip_sam_features_submission.py` - all features of CLIP and SAM were concatanated, and one Linear model mapping these features into the fMRI is constructed.

> `clip_sam_roi_wise_submission.py` - a linear model was constructed for each ROI. For particular ROI, the best features were used. For example, prf-visual ROIs the SAM features performed the best, hence for these ROI I used only the SAM features, whereas for other ROI I used CLIP. `per_roi_sh.sh` - bash script that runs submission file sequentially for all subjects.

`clip_sam_nn` - folder with scripts where Neural Network from CLIP and SAM features are fit to the fMRI voxels

`vith` - end-to-end training of ViT-H model initialized with ImageNet Weights

`ViT` - submission based on pretrained ViT models

`eva` - submission based on EVA model

### Conclusion

The best submission score was `50.6`. It comes from the average of CLIP[ViT-B/32] text+image features submission and SAM features [downsampled with avg pooling of kernel size 8] submission.
Among 107 submissions, this one was placed at number 21.

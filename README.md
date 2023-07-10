# breil-algonauts-challenge
Repository to store code of BREIL members for Algonauts 2023 challenge

Little review of current existing files:

`algonauts_2023_challenge_tutorial.ipynb` - provided jupyter notebook by challenge hosts. It contains some visualizations of data, and baseline method for prediction using AlexNet.
> Remark: I removed this notebook from the repository due to its huge size. Please access it by [Link](https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link)

`alexnet_submission.py` - .py version of the above notebook. It iterates over 8 subjects and saves prediction scores. A baseline submission.

`coco_image_matching.ipynb` - notebook to match ids of NSD images to the original images from COCO to get corresponding annotations.

### Our works

`clip` - folder with scripts using CLIP model

`sam` - folder with scripts using Segment Anything

> `clip_sam_ensemble.ipynb` - computes average of CLIP and SAM predictions

`vith` - end-to-end training of ViT-H model initialized with ImageNet Weights
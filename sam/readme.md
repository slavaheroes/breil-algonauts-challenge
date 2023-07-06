## Segment Anything

`sam_feature_extractor.py` - extracts features from train/test images using image encoder of SAM, and saves as .npy array.

`segment_anything_submission.py` - fits the extracted features to BOLD responses and saves predictions. Since the feature dimension is 256x64x64 (CxHxW), we resphape it to 4096x256, and then apply PCA.

> Update: Pooling works better.
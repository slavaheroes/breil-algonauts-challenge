# CLIP

`feature_extractor.py` - extracts features from train/test images using image encoder of CLIP, and saves as .npy array.

`text_features_extractor.ipynb` - matches the NSD-ids to COCO-ids to get captions for each stimuli image, and extracts the average of all caption embeddings using text encoder of CLIP.

`submission_clip.py` - fits the extracted features to BOLD responses and saves predictions.

`submission_clip_text_image.py` - fits the extracted text and image features to BOLD responses and saves predictions.

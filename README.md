# breil-algonauts-challenge
Repository to store code of BREIL members for Algonauts 2023 challenge

Little review of current existing files:
`algonauts_2023_challenge_tutorial.ipynb` - provided jupyter notebook by challenge hosts. It contains some visualizations of data, and baseline method for prediction using AlexNet. 

`alexnet_submission.py` - .py version of the above notebook. It iterates over 8 subjects and saves prediction scores.

`segment_anything_submission.py` - similar script as alexnet_submission but using Segment Anything model.

`coco_image_matching.ipynb` - notebook to match ids of NSD images to the original images from COCO to get corresponding annotations. 
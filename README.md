# PCA-LGM

This repository is based on [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) (branch pca-gm-archive).
As we all know, PCA-GM input two attribute graphs into the network, where the graphs
 are constructed by Delaunay trigulation or other methods.
 While PCA-LGM can update the graph structures automatically.
 This is the main contribution of this repo.    


# Before Training
 Run this repo. in docker.  
 0. Clone this repository
 1. Install [docker](https://docs.docker.com/engine/install/ubuntu/)
 2. Pull Image of pytorch1.2  
    `docker pull siaimes/pytorch1.2:v1.0.1`
 3. Show docker image id  
    `docker images`
 4. Run docker  
    `docker run -it -v dir_to_PCA_LGM:/home/workdir --gpus all --shm-size 32G IMAGE_ID /bin/bash`


# Training and Evaluation Steps(By ThinkLab)
More information can be found in [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) 

## Preprocessing steps on Pascal VOC Keypoint dataset:
Here we describe our preprocessing steps on Pascal VOC Keypoint dataset for fair comparison and to ease future research.
1. Filter out instances with label 'difficult', 'occluded' and 'truncated', together with 'people' after 2008. 
1. Randomly select two instances from the same category.
1. Crop these two instances from the background images using bounding box annotation.
1. Filter out non-overlapping keypoints (i.e. outliers) in two instances respectively and leave only inliers. **If the resulting inlier number is less than 3, omit it** (because the problem is too trivial).
1. Build graph structures from keypoint positions for two graphs independently (in PCA-GM, it is Delaunay triangulation).

## Get started

1. Install and configure pytorch 1.1+ (with GPU support)
1. Install ninja-build: ``apt-get install ninja-build``
1. Install python packages: ``pip install tensorboardX scipy easydict pyyaml``
1. If you want to run experiment on Pascal VOC Keypoint dataset:
    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``
1. If you want to run experiment on Willow ObjectClass dataset, please refer to [this section](#detailed-instructions-on-willow-object-class-dataset)

## Training

Run training and evaluation

``python train_eval.py --cfg path/to/your/yaml`` 

and replace ``path/to/your/yaml`` by path to your configuration file. Default configuration files are stored in``experiments/``.

## Evaluation

Run evaluation on epoch ``k``

``python eval.py --cfg path/to/your/yaml --epoch k`` 

## Detailed instructions on Willow Object Class dataset

1. Download [Willow ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)
1. Unzip the dataset and make sure it looks like ``data/WILLOW-ObjectClass``
1. If you want to initialize model weights on Pascal VOC Keypoint dataset (as reported in the paper), please:
    1. Remove cached VOC index ``rm data/cache/voc_db_*``
    1. Uncomment L156-159 in ``data/pascal_voc.py`` to filter out overlapping images in Pascal VOC
    1. Train model on Pascal VOC Keypoint dataset, e.g. ``python train_eval.py --cfg experiments/vgg16_pca_voc.yaml``
    1. Copy Pascal VOC's cached weight to the corresponding directory of Willow. E.g. copy Pascal VOC's model weight at epoch 10 for willow
    ```bash
    cp output/vgg16_pca_voc/params/*_0010.pt output/vgg16_pca_willow/params/
    ```
    1. Set the ``START_EPOCH`` parameter to load the pretrained weights, e.g. in ``experiments/vgg16_pca_willow.yaml`` set
    ```yaml
    TRAIN:
       START_EPOCH: 10
    ```


## Benchmark


**Pascal VOC Keypoint** (mean accuracy is on the last column)

| method | aero | bike | bird | boat | bottle | bus  | car  | cat  | chair | cow  | table | dog  | horse | mbike | person | plant | sheep | sofa | train | tv   | mean     |
| ------ | ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ----- | ----- | ------ | ----- | ----- | ---- | ----- | ---- | -------- |
| GMN    | 31.9 | 47.2 | 51.9 | 40.8 | 68.7   | 72.2 | 53.6 | 52.8 | 34.6  | 48.6 | 72.3  | 47.7 | 54.8  | 51.0  | 38.6   | 75.1  | 49.5  | 45.0 | 83.0  | 86.3 | 55.3     |
| PCA-GM | 40.9 | 55.0 | 65.8 | 47.9 | 76.9   | 77.9 | 63.5 | 67.4 | 33.7  | 65.5 | 63.6  | 61.3 | 68.9  | 62.8  | 44.9   | 77.5  | 67.4  | 57.5 | 86.7  | 90.9 | **63.8** |
| PCA-LGM| 51.4 | 62.8 | 61.4 | 61.0 | 78.0   | 71.6 | 72.1 | 71.4 | 38.5  | 63.0 | 62.3  | 65.2 | 62.7 | 61.0  | 47.3   | 77.3  | 65.4   | 56.8 | 79.6 | 88.4  | **64.8**/


**Willow Object Class**

| method        | face      | m-bike   | car      | duck     | w-bottle |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| HARG-SSVM     | 91.2      | 44.4     | 58.4     | 55.2     | 66.6     |
| GMN-VOC       | 98.1      | 65.0     | 72.9     | 74.3     | 70.5     |
| GMN-Willow    | 99.3      | 71.4     | 74.3     | 82.8     | 76.7     |
| PCA-GM-VOC    | **xx** | xx     | xx     | xx     | xx     |
| PCA-GM-Willow | **xx** | **xx** | **xx** | **xx** | **xx** |


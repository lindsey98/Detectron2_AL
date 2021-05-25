# Detectron2 for Active Learning in Object Detection 

## Requirements
torch
torchvision
detectron2 

## Instructions
1. Change configs parameters
2. Run AL training 
```
python train_al_model.py --config-file [cfg_file_path] \
                         --dataset_name [dataset name registered in DatasetCatalog] \
                         --json_annotation_train [training_annot.json] \
                         --image_path_train [training_image_folder] \
                         --json_annotation_val [val_annot.json] \
                         --image_path_val [val_image_folder]
```

## Project structure


```
src
|___ detectron2_al
   |____ dataset
       |____ al_dataset.py
          |_____ build_al_dataset: Supports ImageActiveLearningDataset/ObjectActiveLearningDataset
          |_____ HandyCOCO: Inheritated from COCO, add subsampling ImageId method
          |_____ Budget: Compute remaining budget, allocate equal budget to each round
          |_____ DatasetHistory: Record all datasets used
          |_____ ActiveLearningDataset:
              |___ add_new_dataset_into_catalogs: add new dataset name into catalogs
              |___ get_training_dataloader: get all currently labelled dataset to train
              |___ get_oracle_dataloader: get all unlabelled dataset to label
              |___ create_initial_dataset: create initial labelled set to initialize the model
              |___ create_new_dataset: create new dataloader
              
          |____ ImageActiveLearningDataset: Inheritated from ActiveLearningDataset, select top uncertain images
          |____ ObjectActiveLearningDataset: Inheritated from ActiveLearningDataset, select top uncertain objects, get fused_results (see object_fusion.py) from oracle 
          
       |____ dataset_mapper.py: Define DatasetMapperAL
       |____ object_fusion.py: 
          |___ ObjectFusion: Combine the model predictions with the ground-truth by replacing the objects in the pred with score_al of top replace_ratio. 

       |____ util.py: build_detection_train_loader_drop_ids
       
       
   |____ engine
       |____ al_engine.py
          |_____  ActiveLearningTrainer: Modified based on DefaultTrainer to support active learning functions.   
          |_____ ImageActiveLearningTrainer: Inheritated from ActiveLearningTrainer
          |_____ ObjectActiveLearningTrainer: Inheritated from ActiveLearningTrainer
          |_____ ActiveLearningPredictor
          
          
   |___ modelling
       |___ rcnn.py: 
          |____ ActiveLearningRCNN: define forward_al to generate detection_results which include uncertainty scores
       |___ roi_heads.py
          |____ ROIHeadsAL: supports entropy, margin, least_confidence, perturbation [Localization-aware paper](https://arxiv.org/pdf/1801.05124v1.pdf) and random query strategy. And avg, max, sum, random image aggregation method
       
   |__ configs
       |___ defaults.py: Some important configurations are listed below
            _C.AL.MODE: object or image based AL
            _C.AL.OBJECT_SCORING: query strategy, {'1vs2, 'least_confidence', 'random', 'perturbation'}
            _C.AL.IMAGE_SCORE_AGGREGATION: image-level aggregation method
            _C.AL.PERTURBATION... : for perturbation based query strategy
            _C.AL.DATASET.IMAGE_BUDGET = 20: total budget
            _C.AL.DATASET.OBJECT_BUDGET = 2000: total budget
            _C.AL.DATASET.SAMPLE_METHOD: top/kmeans sampling strategy
            _C.AL.OBJECT_FUSION: Combine pred with gt when you do AL?
            _C.AL.TRAINING.ROUNDS: Number of AL rounds 
            _C.AL.TRAINING.EPOCHS_PER_ROUND_INITIAL: The numbers of epochs for training during each round. 
            _C.AL.TRAINING.EPOCHS_PER_ROUND_DECAY = 'linear': Do you decay number of epochs of trainig for each AL round?
            
```
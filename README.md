# A QA-oriented Pretraining

Here are the links for videos: [3 minutes Video](https://drive.google.com/file/d/1PaaEGvI3VNDh0ouCgQgQwQ2dGqegleFW/view?usp=share_link) amd [Full-length Video](https://drive.google.com/file/d/1mIxxmHmlGJioKSLSjL3rpPAQqZJU4mt1/view?usp=share_link). I really sugget you watch the [Full-length Video](https://drive.google.com/file/d/1mIxxmHmlGJioKSLSjL3rpPAQqZJU4mt1/view?usp=share_link). 

Here is the link for [Slide](https://drive.google.com/file/d/1WyITNV54WLIcrr2AsxAmBB2sZBUsS5m1/view?usp=share_link). Here is the link for [Report](https://drive.google.com/file/d/1DsXdD3_INehkO4mzh1M9m7ufjIfUq8lD/view?usp=share_link).

This is the official code for 'An empirical study of QA-oriented Pretraining'. This repo is based on [12-in-1](https://github.com/facebookresearch/vilbert-multi-task). Thanks a lot for their great work.  

## Repository Setup

1. Create a fresh conda environment, and install all dependencies. 

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop 
```

## Data Setup

## Extracting features

1. Install [`vqa-maskrcnn-benchmark`](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark) repository and download the model and config. 

```text
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```


2. Extract features for images

Run from root directory

```text
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir <path_to_directory_with_images> --output_folder <path_to_output_extracted_features>
```

3. Extract features for images with GT bbox

Generate a `.npy` file with the following format for all the images and their bboxes

```text
{
    {
        'file_name': 'name_of_image_file',
        'file_path': '<path_to_image_file_on_your_disk>',
        'bbox': array([
                        [ x1, y1, width1, height1],
                        [ x2, y2, width2, height2],
                        ...
                    ]),
        'num_box': 2
    },
    ....
}
```

Run from root directory

```text
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --imdb_gt_file <path_to_imdb_npy_file_generated_above> --output_folder <path_to_output_extracted_features>
```

4. Convert the extracted images to an LMDB file

```text
python script/convert_to_lmdb.py --features_dir <path_to_extracted_features> --lmdb_file <path_to_output_lmdb_file>
```

## Datasets

Download the data for different datasets to the `data` directory. Here are the links for downloading all the data for *downstream* tasks used in this project :

1. Run from root directory

```text
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz
tar xf datasets.tar.gz
```

The extracted folder has all the datasets and their cache directories that can be pointed to in the `vilbert_tasks.yaml` file.

2. Download extracted features for COCO, GQA and NLVR2

Some of the features are not present in the extracted folder in Step 1. Those can be downloaded following these commands :

#### COCO features

```text
cd coco

mkdir features_100

cd features_100

mkdir COCO_test_resnext152_faster_rcnn_genome.lmdb

mkdir COCO_trainval_resnext152_faster_rcnn_genome.lmdb

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb/data.mdb && mv data.mdb COCO_trainval_resnext152_faster_rcnn_genome.lmdb/

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/coco/features_100/COCO_test_resnext152_faster_rcnn_genome.lmdb/data.mdb && mv data.mdb COCO_test_resnext152_faster_rcnn_genome.lmdb/
```

#### GQA features

```text
cd gqa

mkdir gqa_resnext152_faster_rcnn_genome.lmdb

cd gqa_resnext152_faster_rcnn_genome.lmdb

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/gqa/gqa_resnext152_faster_rcnn_genome.lmdb/data.mdb
``` 

## Visiolinguistic Pre-training and Multi Task Training

### Training Workflow

Pretraining (First-stage Pretraining) -> Single/Multi-task Second-stage Pretraining -> Multi-task Learning -> Finetuning

### Pretraining on Conceptual Captions

```
python train_concap.py --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --train_batch_size 512 --objective 1 --file_path <path_to_extracted_cc_features>
```
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin) (Pretrained Weights)

### Single-task Second-stage Pretraining

Take VQA as an example.

```
CUDA_VISIBLE_DEVICES=0 python pretrain_second.py --fp16 --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name GQA_SecondPretrain —sep
```

### Multi-task Second-stage Pretraining

Multi-task training includes VQA, GQA, and VG QA.

```
CUDA_VISIBLE_DEVICES=0 python pretrain_second.py --fp16 --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --tasks 1-2-12 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens
```

### Multi-GPU Training

Take VQA as an example.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_tasks3.py --fp16 --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --gpus 4
```

### Multi-task Training

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <pretrained_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1-2-4-7-8-9-10-11-12-13-15-17 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name multi_task_model
```

[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin) (Multi-task Learning Weights)


### Fine-tune from Multi-task trained model

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model
```
 
## License

QA-oriented pretrining is licensed under MIT license available in [LICENSE](LICENSE) file.

# RailGuard
Railway transportation is a reliable and economical
mode of transport in many countries worldwide. Despite its
affordability and widespread use, it suffers from a poor safety
record specifically in developing countries, with numerous acci-
dents occurring, particularly at unmanned level crossings due
to road user negligence. One promising approach to enhancing
railway safety involves leveraging advanced imaging techniques to
improve hazard detection and monitoring. Image fusion integrates
complementary information from multiple modalities such as
infrared and visible images. However, previous approaches have
overlooked the potential of image fusion in railway applications.
Moreover, most state-of-the-art methods perform poor in real-
time edge applications, such as deployment on Jetson devices,
limiting their use in locomotive systems. Furthermore, there is
no standard benchmark for evaluating fusion algorithms in the
context of railway data. This repository introduces RailGuard, a joint
image fusion and semantic segmentation framework optimized
for edge devices. The proposed system cascades image fusion
and semantic segmentation network, enabling both tasks to learn
representations jointly. 
![Alt text](https://github.com/faizan1234567/RailGuard/blob/main/media/railguard_pic.png)

## RRL Visible Infrared Image Fusion and Object Detection Dataset
Existing image fusion datasets are acquired for general-purpose applications across diverse scenarios. However, to the best of our knowledge, there is currently no comprehensive benchmark dataset specifically collected for railway contexts to enable research and development. This domain is particularly susceptible to accidents caused by road user negligence and obstacles on the tracks.

These obstacles pose a significant danger such as derailing locomotives, causing serious injuries, loss of human lives, and significant damage to the train. Timely detection of such obstacles could help avoid catastrophic accidents, especially in low-visibility conditions.

The proposed railFusion dataset contains high-resolution visible and infrared images under various scenarios. The data were captured using the HIKVision thermal and optical bi-spectrum speed dome DS-2TD4167-50/W model, which recorded both visible and infrared scene details.

- Resolution: 1280 × 960 pixels

- Frame rate: 25 FPS

The railFusion dataset consists of:

- 3,100 image pairs (visible + infrared)

- 16,203 annotated objects

The major annotation categories include:

- Vehicle

- Motorcycle

- Person

- Animals

The dataset covers diverse scenarios, including:

- Objects crossing the tracks at level crossings

- People and animals in densely populated areas

The dataset can be downloaded from figshare: https://figshare.com/articles/dataset/rrl_data_zip/28929923

## Installation
Anconda environment recommended.
```
git clone https://github.com/faizan1234567/RailGuard
cd RailGuard
```

create a virtual environment in Anaconda and activate it.
```
conda create -n railguard python=3.9.0 -y 
conda activate railguard
```

Now install all the dependencies.
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
To train and evaluate the RailGuard:
```
python train.py -h
python train.py \
  --model_name RailGuard \
  --dataset_root /drive/data/MSRS/segmentation \
  --batch_size 4 \
  --gpu 0 \
  --num_workers 8 \
  --fusion_epochs 10 \
  --seg_epochs 20000 \
  --pretrained \
  --M 4 \
  --save_path runs/
```

To test the model on the tess set use the following command:
```
python test.py -h
python test.py \
  --model_path ./checkpoints/fusion_model_epoch10.pth \
  --ir_dir ./train_imgs/ir \
  --vi_dir ./train_imgs/vi \
  --save_dir ./runs/SeAFusion_train \
  --batch_size 8 \
  --gpu 0 \
  --num_workers 4 \
   ```
## Cite
If you find this project useful in your research, please consider cite and star the repository:
```
@misc{brats23-tumor-segmentation,
    title={Multi-modal BraTS 2023 brain tumor segmentation},
    author={Muhammad Faizan},
    howpublished = {\url{https://github.com/faizan1234567/Brats-20-Tumors-segmentation}},
    year={2023}
}
```

## Acknowledgements
[1]. https://github.com/Linfeng-Tang/SeAFusion

[2]. https://github.com/JinyuanLiu-CV/TarDAL

[3]. https://github.com/wdzhao123/MetaFusion

[4]. https://github.com/hli1221/imagefusion_densefuse
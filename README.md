# 2025 Fall PKU 智能机器人概论 Final Project: Online Language Splatting
项目基于 2025 ICCV [Online Language Splatting](https://github.com/rpng/online_lang_splatting)

小组成员：于海祥 2300012987  侯玉杰 2300013108

## Environment Setup & Download Dateset
请参考原论文 [github repo](https://github.com/rpng/online_lang_splatting)


## 项目运行：论文复现 (on ReplicaV2 数据集)
1. After following the steps for setting up the language model you should get ```seg_clip_model_l.pth```, Running the Online CLIP model to test lauguage features (demo)
```bash
bash run_lang_features.sh
```

2. Running the 2-stage Pipeline
```bash
bash run_slam.sh
```
根据具体场景替换 ```--config``` 配置文件。根据原论文仓库，需要下载 pretrained checkpoints 并相应更改 ```configs/rgbd/replicav2/base_config.yaml``` 中的```auto_ckpt_path```, ```lang_model_path``` 和 ```hr_ckpt_path```以及场景config(e.g. ```room0.yaml```)中的```dataset_path```。更改```base_config.yaml```中的运行结果保存目录 ```save_dir```，以及场景config 中的训练完成后模型保存路径 ```online_ckpt_path```。

3. Evaluation Step 1: Create Language Label
```bash
bash run_create_labels.sh
```
根据运行 2-stage pipeline 的数据设置相应的 ```--seg_file_config```，根据训练后的结果保存目录，替换 ```--langslam_dir```。

4. Evauation Step 2: Evaluate on the results of 2-stage pipeline.
```bash
bash run_evaluation.sh
```

5. 3D Evaluation Step 1: Prepare colorized GT by running
```bash
cd tsdf_fusion
python3 save_semantic_colors_gt.py
```
Change the ```sem_path``` and ```sem_save``` accordingly. 

6. 3D Evaluation Step 2: To reconstruct TSDF for groundtruth, run
```bash
python3 dim3_recon_gt.py
cd PytorchEMD; python3 setup.py
```
运行结果保存在

copy the compiled .so file to the tsdf-fusion folder (move one level up)

```bash
cd ..
python dim15_recon.py
```
运行结果保存在

7. Run 3D Evaluation on LangSlam
```bash
bash run_recon_mesh.sh
```
运行后会在指定目录得到文件

## 项目运行：GOAT-core
新增配置文件位于 ```configs/rgbd/goatcore```，包含 ```base_config.yaml``` 以及 ```4ok.yaml```

1. Running the Online CLIP model to test lauguage features (demo)
```bash
bash run_lang_features_goatcore.sh
```

2. 运行 2-stage pipeline 进行建图
```bash
bash run_slam_goatcore.sh
```


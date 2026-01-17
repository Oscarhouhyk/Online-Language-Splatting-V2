# 2025 Fall PKU 智能机器人概论 Final Project: Online Language Splatting
项目基于 2025 ICCV [Online Language Splatting](https://github.com/rpng/online_lang_splatting)

小组成员：于海祥 2300012987  侯玉杰 2300013108

## Environment Setup & Download Dateset
请参考原论文 [github repo](https://github.com/rpng/online_lang_splatting)


## 项目运行：论文复现 (on ReplicaV2 数据集)
- After following the steps for setting up the language model you should get ```seg_clip_model_l.pth```, Running the Online CLIP model to test lauguage features (demo)
```bash
bash run_lang_features.sh
```
Please change the configs accordingly to query objects in an image.

- Running the 2-stage Pipeline
```bash
bash run_slam.sh
```
根据具体场景替换 ```--config``` 配置文件。根据原论文仓库，需要下载 pretrained checkpoints 并相应更改 ```configs/rgbd/replicav2/base_config.yaml``` 中的```auto_ckpt_path```, ```lang_model_path``` 和 ```hr_ckpt_path```以及场景config(e.g. ```room0.yaml```)中的```dataset_path```。更改```base_config.yaml```中的运行结果保存目录 ```save_dir```，以及场景config 中的训练完成后模型保存路径 ```online_ckpt_path```。

- Evaluation Step 1: Create Language Label
```bash
bash run_create_labels.sh
```
根据运行 2-stage pipeline 的数据设置相应的 ```--seg_file_config```，根据训练后的结果保存目录，替换 ```--langslam_dir```。

- Evauation Step 2: Evaluate on the results of 2-stage pipeline.
```bash
bash run_evaluation.sh
```
Please change ```--root_dir```, ```--ae_ckpt_dir```, ```--online_ae_ckpt``` accordingly. 运行完成后，2D localization 的检索结果会保存在结果保存目录中。

- 3D Evaluation Step 1: Prepare colorized GT by running
```bash
cd tsdf_fusion
python3 save_semantic_colors_gt.py
```
Change the ```sem_path``` and ```sem_save``` accordingly. 运行完成后会在数据集目录下保存 ```color_code.npy``` 以及 ```semantic_color``` 文件夹。

- 3D Evaluation Step 2: To reconstruct TSDF for groundtruth, run
```bash
python3 dim3_recon_gt.py
cd PytorchEMD; python3 setup.py
```
Change ```color_prefix``` and ```save_path``` in the code. Copy the compiled .so file to the tsdf-fusion folder (move one level up)。运行完成后会在 2-Stage pipeline 运行结果的文件夹内生成带有 language features 的 Ground Truth 点云文件 ```GT_semantic_mesh.ply``` 以及 ```GT_semantic_pc.ply```。

```bash
cd ..
python dim15_recon.py
```
Change ```color_prefix``` in the code. 运行完成后会在 2-Stage pipeline 运行结果的文件夹内生成训练后的带有 language features 的点云文件 ```semantic_mesh.ply``` 以及 ```semantic_mesh_color.ply```。

- Run 3D Evaluation on LangSlam
```bash
bash run_recon_mesh.sh
```
Change the ```query``` in the code corresponding to different objects in the scenes. Change ```path```, ```load_path```, ```ae_ckpt_path```, ```color_mat```, ```gt_pcd``` accordingly. 

运行结果会保存到 2-stage pipeline 训练结果下的 ```3d_mesh``` 文件夹中，包含场景中不同物体的检索结果 3D Localization Mesh。

- 查看 2DGS 渲染效果（利用 pred frames 生成视频）
```bash
bash run_video.sh
```


## 项目运行：GOAT-core
新增配置文件位于 ```configs/rgbd/goatcore```，包含 ```base_config.yaml``` 以及 ```4ok.yaml```

- Running the Online CLIP model to test lauguage features (demo) and generate video.
```bash
bash run_lang_features_goatcore.sh
bash run_video_goatcore.sh
```


- 运行 2-stage pipeline 进行建图
```bash
bash run_slam_goatcore.sh
```


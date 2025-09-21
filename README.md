<div align="center">

# Online Language Splatting

[Saimouli Katragadda<sup>1</sup>](https://saimouli.github.io/), [Cho-Ying Wu<sup>2</sup>](https://choyingw.github.io), [Yuliang Guo<sup>2&dagger;</sup>](https://yuliangguo.github.io/), [Xinyu Huang<sup>2</sup>](https://scholar.google.com/citations?user=cL4bNBwAAAAJ&hl=en), [Guoquan Huang<sup>1</sup>](https://udel.edu/~ghuang/), [Liu Ren<sup>2</sup>](https://sites.google.com/site/liurenshomepage/)  
<sup>1</sup>University of Delaware &emsp;&emsp;&emsp;&emsp;<sup>2</sup>Bosch Research North America &emsp;&emsp;&emsp;&emsp;<sup>&dagger;</sup> Project Lead  
[**Webpage**](https://saimouli.github.io/onlineLang/) | [**Paper**](https://arxiv.org/pdf/2503.09447) | [**Video**](https://www.youtube.com/watch?v=UGdtMb1Y44E) | [**Pretrained Models**](https://huggingface.co/datasets/slamDev/OnlineLanguageSplatting/tree/main)

[**ICCV 2025**](https://iccv.thecvf.com/)

<p align="center">
  <img src="media/langslam_teasor.gif" alt="teaser" style="width: 50%;">
</p>


</div>


**Online Language Splatting** 
- Introduces a **fully online system** that seamlessly integrates dense CLIP features with Gaussian Splatting.
- Provides a potential form of **physical memory** for real-time **human–machine interaction** and **world modeling**.

<div align="center">

<table>
  <tr>
    <td align="center">
      <img src="media/langslam_sofa.gif" width="400px" alt="Sofa Demo"/><br/>
      <b>Sofa</b>
    </td>
    <td align="center">
      <img src="media/langslam_rug.gif" width="400px" alt="Rug Demo"/><br/>
      <b>Rug</b>
    </td>
  </tr>
</table>
</div>

---

## 🔔 Highlights

- Our method realizes Gaussian Splatting and open-vocabulary-preserving language mapping simultenuously in an online SLAM framework.
- This release include both the integrated framework, and the plug-in-and-play pretrained network modules producing dense and sharp CLIP maps (192x192x768) beyond real-time speed, e.g., >40 FPS.

---

### Update:

We add running RGBL-disentanglement in the "lang_disent" branch. See the following section for instructions. Fix some errors in showing visualization.

## 🚀 Getting Started

### 📦 Dataset

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip
unzip vmap.zip
```

## 🛠️ Installation

```bash
git clone https://github.com/rpng/online_lang_splatting.git --recursive
cd online_lang_splatting
```

Setup the environment.

```bash
conda env create -f environment.yaml
conda activate LangGS
```

💬 Language Model Setup

```bash
cd langauge/sed/open_clip
make install
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Download language model weights from

```
https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view?usp=drive_link

```

Edit `language/configs/convnextL_768.yaml` and Set the `WEIGHTS` to the path of the downloaded language model weights

```bash
cd online_lang_splatting
python create_lang_model.py --config language/configs/convnextL_768.yaml
```

# 🧠 Language Features Demo

Downlod the pre-trained weights (HuggingFace link at the top), and place the models under "pretrained_models". We use omni_general indoor trained weights

To test language feature on your own image, run

```bash
python3 language/language_features.py --high-res-model "pretrained_models/omni_general/high_res_71_indoor.ckpt" --lang-model "seg_clip_model_l.pth" --input "sample/replica_room0.jpg" --query-text "vase"
```

This will show up the feature map and heatmap for localization.

# 🧭 Running the Pipeline

Edit paths in config/rgbd/replicav2/base_config.yaml.

- `auto_ckpt_path` to load generalized autoencoder;
- `lang_model_path` to point to the language model weights (e.g., seg_clip_model_l.pth from the last step).
- `hr_ckpt_path` to point to the high resolution module weights.

for room0.yaml edit `dataset_path` to point to the room0 dataset and `online_ckpt_path` to where you want the online AE checkpoint in 2-stage pipeline to be saved.

### To Run ▶️ 2-Stage Pipeline

In base_config.yaml point `auto_ckpt_path` and `hr_ckpt_path` to the respective files and in room0.yaml set `single_stage_ae` to `False`.

### To Run ▶️ 1-Stage Pipeline

To run the 1-stage pipeline, open `room0.yaml` and update the following parameters:

- Set `auto_ckpt_path` to the cross-data generalization checkpoint file.
- Set `single_stage_ae` to `True`.

We use a 4-split strategy for training:

- **Split 1**: `office0`, `room0`
- **Split 2**: `office1`, `room1`
- **Split 3**: `office2`, `room2`
- **Split 4**: `office3`, `office4`
  Training and Testing Example for 4-Split Strategy:
- **Run 1**: Train on Splits 2, 3, 4 → Test on Split 1
- **Run 2**: Train on Splits 1, 3, 4 → Test on Split 2
- **Run 3**: Train on Splits 1, 2, 4 → Test on Split 3
- **Run 4**: Train on Splits 1, 2, 3 → Test on Split 4

The weights are in the pretrained weights folder. Use appropriate weights
**Example**: For evaluating on `room0` and `office0`, use weights from **Run 1**.

```bash
python3 slam.py --config configs/rgbd/replicav2/room0.yaml
```

# 🏹 Run RGBL-disentanglement

0.

```bash
git checkout lang_disent
```

1. Download the processed dataset on Replica-room0 [here](https://www.dropbox.com/scl/fi/3el9t4qk6rb7bipyhxudh/room0_subset.zip?rlkey=111i7c6jev2muya0brgfywfrz&st=gl7l3vrv&dl=0). Download the single-stage pretrained AE [here](https://www.dropbox.com/scl/fi/kibker7f8k39foewj7kse/single_stage_AE_disent.pth?rlkey=6s7ys17o8v290uu44hh1oga3b&st=s0ejtr64&dl=0).

```bash
cd submodules/diff-gaussian-rasterization-disentangle-optim
pip install -e .
cd ../..
```

2. Follow the previous step to donwload the pretrained models. Edit the pretrained model path in configs/rgbd/replicav2/base_config.yaml, room0_disent.yaml and room0_disent_w_labels.yaml (Edit the dataset path and single-stage AE path.)

3. Run the single-stage AE with RGB-L disent

```bash
python3 slam.py --config configs/rgbd/replicav2/room0_disent.yaml
```

4. We also prepare high-resolution language feature labels from LangSplat. Edit the language label path and run

```bash
python3 slam.py --config configs/rgbd/replicav2/room0_disent_w_labels.yaml
```

A GUI window will pop up and show the SLAM results

# Evaluate

🔖 Create Labels

```bash
python3 eval/create_replica_labels.py
```

## ✅ Evaluate 2-Stage Pipeline

To evaluate 2 stage

```bash
python3 eval/evaluate_onlinelangslam.py
```

## ✅ Evaluate 1-Stage Pipeline

To evaluate cross data genenarizable

```bash
python3 eval/evaluate_langslam.py
```

## 🧱 3D Evaluation

⚠️ Note: in each .py file, please read the comment and change path variables that match your local.

Prepare colorized GT by running

```bash
cd eval/tsdf_fusion
python3 save_semantic_colors_gt.py
```

To reconstruct TSDF for groundtruth, run

```bash
python3 dim3_recon_gt.py
```

```bash
cd PytorchEMD; python3 setup.py
```

copy the compiled .so file to the tsdf-fusion folder (move one level up)

▶️ Run 3D Evaluation
LangSlam

```bash
python3 3d_evaluation_and_visualize_langslam_dim15.py
```

LangSplat

```bash
python3 3d_evaluation_and_visualize_langsplat.py
```

🧪 Training

### To train your own AE on your domain for 1-stage

Language feature script can be used to save high or low resolution langauge features labels to train auto encoder on your own domain.

```bash
python3 language/autoencoder/train_encoder_light.py
```

# 🧬 Reprodicibility

There might be minor differences between the released version and the results in the paper. Please bear in mind that multi-process performance has some randomness due to GPU utilisation. We run all our experiments on an RTX A4500 GPU, and the performance may differ when running with a different GPU.

# 🙏 Acknowledgement

This work incorporates many open-source codes. We extend our gratitude to the authors of the software.

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization
  ](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [SED](https://github.com/xb534/SED)
- [MonoGS](https://github.com/muskie82/MonoGS)
- [LangSplat](https://github.com/minghanqin/LangSplat)

# 🎲 Some Error

- If you see error like LangSupervisedNet doesn't have attribute load_from_checkpoint in self.hr_model.load_from_checkpoint(...). This is the issue of Pytorch-lightning version. Edit those lines to LangSupervisedNet.load_from_checkpoint(...)

# 📖 Citation

If you find this work helpful, please consider citing us:

```bibtex
@inproceedings{katragadda2025_onlinelang,
  title     = {{O}nline {L}anguage {S}platting},
  author    = {Saimouli Katragadda and Cho-Ying Wu and Yuliang Guo and Xinyu Huang and Guoquan Huang and Liu Ren},
  booktitle = {ICCV},
  year      = {2025}
}
```

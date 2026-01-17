CUDA_VISIBLE_DEVICES=7 python language/language_features.py \
 --high-res-model "pretrained_models/omni_general/high_res_71_indoor.ckpt" \
 --lang-model "seg_clip_model_l.pth" \
 --input "/data/houyj/robotics/data/Goat-core/dataset/nfv/images" \
 --query-text "couch" \
 --output "results/lang_features_demo/nfv/couch_02" \
 --store-img \
 --sim-threshold 0.6 \

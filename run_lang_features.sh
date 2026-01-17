CUDA_VISIBLE_DEVICES=7 python language/language_features.py \
 --high-res-model "pretrained_models/omni_general/high_res_71_indoor.ckpt" \
 --lang-model "seg_clip_model_l.pth" \
 --input "sample/img0590.png" \
 --query-text "bed" \
 --output "results/lang_features_demo" \

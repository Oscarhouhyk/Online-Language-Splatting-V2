CUDA_VISIBLE_DEVICES=2 python3 eval/evaluate_onlinelangslam.py \
 --dataset_name office4 \
 --root_dir "/data/houyj/robotics/online_lang_splatting/results/2-stage/office_4/2026-01-07-17-06-52/psnr/before_opt" \
 --ae_ckpt_dir "/data/houyj/robotics/online_lang_splatting/pretrained_models/omni_general/ae_149_he.ckpt" \
 --online_ae_ckpt "/data/houyj/robotics/online_lang_splatting/output/omni_data_result/online_15_office4.pth" \
 --label_name label \
 --code_size 15 \
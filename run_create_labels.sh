CUDA_VISIBLE_DEVICES=3 python eval/create_replica_labels.py \
 --langslam_dir "/data/houyj/robotics/online_lang_splatting/results/2-stage/office_4/2026-01-07-17-06-52/psnr/before_opt" \
 --langsplat_dir "/data/houyj/robotics/online_lang_splatting/results/2-stage/office_4/2026-01-07-17-06-52/psnr/before_opt" \
 --seg_file_config /data/houyj/robotics/data/vmap/office_4/imap/00/render_config.yaml \
 --output_name label


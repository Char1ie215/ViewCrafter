python inference.py \
--image_dir test/images/4_last_frame.png \
--out_dir ./output \
--traj_txt None \
--mode 'single_view_target' \
--center_scale 1. \
--elevation=0 \
--seed 123 \
--d_theta 10  \
--d_phi 30 \
--d_r -.2   \
--d_x 50   \
--d_y 25   \
--ckpt_path ./checkpoints/model.ckpt \
--config configs/inference_pvd_1024.yaml \
--ddim_steps 50 \
--video_length 25 \
--device 'cuda:0' \
--height 576 --width 1024 \
--model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
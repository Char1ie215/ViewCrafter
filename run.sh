python inference.py \
--image_dir /apdcephfs_cq10/share_1290939/karmyu/pytorch3d_render_dust/input/figure/family.png \
--out_dir ./output/0723/ \
--center_scale 1. \
--bg_trd .35 \
--mode 'single_view_specify_iterative' \
--d_theta 0 0 -25  \
--d_phi -40 40 0 \
--d_r .8 .8 .8 \
--elevation=5 \
--exp_name 'test_3' \
--seed 123 \
--ckpt_path /apdcephfs_cq10/share_1290939/vg_share/vip3d_share/final_1024_SD-IPA_builton_base512_25frame_DL3DV_R10k_25frame/epoch=0-step=5000.ckpt \
--config configs/inference_1024_vip.yaml \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--text_input \
--video_length 25 \
--frame_stride 10 \
--prompt 'Rotating view of a scene.' \
--device 'cuda:1' \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae \
--bs 1 --height 576 --width 1024


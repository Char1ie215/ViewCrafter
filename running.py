import os
import base64
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from third_party.ViewCrafter.extern.dust3r.dust3r.inference import inference, load_model
from third_party.ViewCrafter.extern.dust3r.dust3r.utils.image import load_images, process_images_directly, process_two_images_directly
from third_party.ViewCrafter.extern.dust3r.dust3r.image_pairs import make_pairs
from third_party.ViewCrafter.extern.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
from third_party.ViewCrafter.utils.pvd_utils import *

opts = OmegaConf.create({
    'device': 'cuda:0',
    'batch_size': 1,
    'height': 576,
    'width': 1024,
    'model_path': '/home/haonan/clean_git/direct_human_demo/third_party/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
    'center_scale': 1.0,
    'elevation': 0,
    'd_theta': [15],
    'd_phi': [10],
    'd_r': [0.2],
})

device = opts.device
dust3r_model = load_model(opts.model_path, device)  #setup duster

#process image
def process_two_images(image, image1, shape):
        
        image = base64.b64decode(image)
        image1 = base64.b64decode(image1)
        shape = tuple(shape)
        restored_tensor = torch.from_numpy(np.frombuffer(image, dtype=np.uint8)).reshape(shape)
        restored_tensor = restored_tensor.numpy().astype(np.uint8)
        restored_tensor1 = torch.from_numpy(np.frombuffer(image1, dtype=np.uint8)).reshape(shape)
        restored_tensor1 = restored_tensor1.numpy().astype(np.uint8)

        img = Image.fromarray(restored_tensor)
        img1 = Image.fromarray(restored_tensor1)
        images = process_two_images_directly(img, img1, size=512,force_1024 = True)
        # images = process_two_images_directly(image, image1, size=512,force_1024 = True)
        # img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        img_gts = []
        img_gts.append((images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)
        img_gts.append((images[1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)

        return images, img_gts

# images, img_ori = process_initial_images(image_data, image_shape)

# 4. 运行 Dust3r 初始化场景
def run_dust3r(input_images,dust3r_model,clean_pc = True):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        device = 'cuda:0'
        output = inference(pairs, dust3r_model, device, batch_size=1)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=10, schedule='cosine', lr=0.01)

        # if clean_pc:
        #     self.scene = scene.clean_pointcloud()
        # else:
        #     self.scene = scene
        return scene.clean_pointcloud()  #scene


def nvs_sparse_view_interp(images,scene, img_ori):

        c2ws = scene.get_im_poses().detach()
        principal_points = scene.get_principal_points().detach()
        focals = scene.get_focals().detach()
        shape = images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in scene.get_pts3d(clip_thred=None)] # a list of points of size whc
        depth = [i.detach() for i in scene.get_depthmaps()]

        if len(images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            scene.min_conf_thr = float(scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = scene.get_masks()
            depth = scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(scene.imgs)
        video_length = 25
        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, video_length, device='cuda:0')
        render_results, viewmask = run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        for i in range(len(img_ori)):
            render_results[i*(video_length - 1)] = img_ori[i]
        # save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        # save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)


#no need
        # diffusion_results = []
        # print(f'Generating {len(self.img_ori)-1} clips\n')
        # for i in range(len(self.img_ori)-1 ):
        #     print(f'Generating clip {i} ...\n')
        #     diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)]))
        # print(f'Finish!\n')
        # diffusion_results = torch.cat(diffusion_results)
        # save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return render_results

def run_render(pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = render_pcd(pcd, imgs, masks, num_views,renderer,device='cuda:0',nbv=False)
        return render_results, viewmask


def render_pcd(pts3d,imgs,masks,views,renderer,device,nbv=False):
        
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None

        return images, view_masks



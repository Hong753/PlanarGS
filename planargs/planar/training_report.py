import torch
import os
import torch
import uuid
from argparse import Namespace
from planargs.scene import Scene
from planargs.common_utils.loss_utils import psnr
from planargs.planar.visualize import visualDepth, visualNorm, visualMask, visualSegmask

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, dn_loss, plane_loss, prior_deploss, prior_normloss, co_planar,
                    l1_loss, elapsed, test_iters, scene : Scene, renderFunc, renderArgs, vis_planar = False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if dn_loss is not None:
            tb_writer.add_scalar('train_loss_patches/normal_loss', dn_loss.item(), iteration) 
        if plane_loss is not None:
            tb_writer.add_scalar('train_loss_patches/plane_loss', plane_loss.item(), iteration)
        if prior_normloss is not None:
            tb_writer.add_scalar('train_loss_patches/prior_normloss', prior_normloss.item(), iteration)
        if prior_deploss is not None:
            tb_writer.add_scalar('train_loss_patches/prior_deploss', prior_deploss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set  
    if iteration in test_iters:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, len(scene.getTrainCameras()), len(scene.getTrainCameras())//6)]})  

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = out["render"].clamp(0.0, 1.0)
                    gt_image = viewpoint.gt_image

                    if tb_writer and (idx < 5):
                        rendered_normal = out["rendered_normal"]
                        depth_normal = out["depth_normal"]
                        depth = out["plane_depth"]
                        min, max, depth_pic = visualDepth(depth)
                        normal_pic = visualNorm(rendered_normal)
                        depthnormal_pic = visualNorm(depth_normal)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rendered_normal".format(viewpoint.image_name), normal_pic[None], global_step=iteration)  
                        tb_writer.add_images(config['name'] + "_view_{}/depth_normal".format(viewpoint.image_name), depthnormal_pic[None], global_step=iteration)  
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_pic[None], global_step=iteration) 

                        if viewpoint.priordepth is not None:
                            _, _, priordepth_pic = visualDepth(viewpoint.priordepth * viewpoint.depth_conf, path=None, filename=None, miner=min, maxer=max) 
                            tb_writer.add_images(config['name'] + "_view_{}/prior_depth".format(viewpoint.image_name), priordepth_pic[None], global_step=iteration)

                        if vis_planar and viewpoint.planarmask is not None:
                            planar_depth = co_planar(depth, viewpoint.planarmask, viewpoint.inv_K)
                            _, _, planardepth_pic = visualDepth(planar_depth, path=None, filename=None, miner=min, maxer=max)  
                            tb_writer.add_images(config['name'] + "_view_{}/planar_depth".format(viewpoint.image_name), planardepth_pic[None], global_step=iteration)

                        if iteration == test_iters[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if viewpoint.priordepth is not None:
                                canny_mask = visualMask(viewpoint.canny_mask)
                                normal_prior_pic = visualNorm(viewpoint.priornormal)
                                tb_writer.add_images(config['name'] + "_view_{}/canny_mask".format(viewpoint.image_name), canny_mask[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/prior_normal".format(viewpoint.image_name), normal_prior_pic[None], global_step=iteration)
                            
                            if viewpoint.planarmask is not None:
                                segmask_pic = visualSegmask(viewpoint.planarmask)
                                tb_writer.add_images(config['name'] + "_view_{}/planar_segmask".format(viewpoint.image_name), segmask_pic[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()

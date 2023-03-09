# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

from torchvision.transforms.functional import rgb_to_grayscale


class Trainer:
    def __init__(self, options):
        self.start_time = None
        self.step = None
        self.epoch = None
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["superpoint"] = networks.SuperPointNet()
        self.models["superpoint"].to(self.device)

        # Superpoint: Train on GPU, deploy on GPU.
        self.parameters_to_train += list(self.models["superpoint"].parameters())  # automatic weights loading

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":  # pose_model_type = "separate_resnet" by default
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        train_filenames = [x.replace('/', '\\') for x in train_filenames]  # WINDOWS PATH ADJUSTMENT
        val_filenames = readlines(fpath.format("val"))
        val_filenames = [x.replace('/', '\\') for x in val_filenames]  # WINDOWS PATH ADJUSTMENT
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:  # opt.no_ssim is False --> not opt.no_ssim is True
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            print('Train Loader iteration', batch_idx)

            before_op_time = time.time()

            inputs = self.pre_process_batch(inputs)

            if ('color_aug_SP_out', 0, 0) not in inputs:
                img: torch.Tensor = rgb_to_grayscale(inputs[('color_aug', 0, 0)])
                inputs[('color_aug_SP_out', 0, 0)] = (torch.zeros((4, 0), device=self.device),
                                                      torch.zeros((256, 0), device=self.device),
                                                      torch.zeros((img.shape[0], img.shape[2], img.shape[3]),
                                                                  device=self.device))
            # elif inputs[('color_aug_SP_out', 0, 0)][-1] is None:
            #     # if heatmap is None -> there's no input color_aug img
            #     img: torch.Tensor = rgb_to_grayscale(inputs[('color_aug', 0, 0)])
            #     inputs[('color_aug_SP_out', 0, 0)][-1] = torch.zeros((img.shape[0], img.shape[2], img.shape[3]))

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            # plot_grad_flow(self.models['superpoint'].named_parameters())  # is a statich method at the end of .py
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    @staticmethod
    def stack_SP_over_imgs(target_img, pts_wrt_batch):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.imshow(torch.permute(target_img, (1, 2, 0)).detach().cpu().numpy())
        pts_coords = pts_wrt_batch[1:-1, :]
        pts_coords_np = torch.t(pts_coords).detach().cpu().numpy()
        plt.scatter(pts_coords_np[:, 0], pts_coords_np[:, 1], marker="o", color="red", s=20)
        fig.tight_layout(pad=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        plt.show()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        print(data.shape)
        final_img = torch.permute(torch.from_numpy(data), (2, 0, 1))
        print(final_img.shape)
        # print(final_img)
        exit(0)
        # TODO: credo che il problema sia sui margini perché non vengono tolti dal canvas
        return final_img

    def pre_process_batch(self, inputs):
        """
        Input: MD2 input which contains 'color_aug' image BxHxWx3 that is initially transformed in grayscale (BxHxW)
        where B is the batch size.

        Output (added in inputs dictionary and returned):
        N: number of unrolled points (multiplied by B) after NMS, threshold, etc.
        pts     --> 4xN     - 'survived' points (1st row: B; 2nd, 3rd row: H, W coords; 4th row: heatmap value)
        desc    --> 256xN   - associated descripors (same order as pts)
        heatmap --> BxHxW   - probability mask

        N.B. there might be a problem with descriptors because the original method was not designed to work with B > 1
        TODO: non-maximum suppression still not implemented! Maybe we have to not implement it and also don't select
        TODO: pixels with logical_OR or similar because we keep all the heatmap values so we must keep all descriptors!
        """
        bord = 4
        dict_to_add = {}
        for key in inputs.keys():
            # we take just the full size images, both the source one, the previous and the following
            # ONLY FORWARD TARGET IMAGE (idx: 0) AND NOT PREVIOUS (idx: -1) AND NEXT (idx: +1)!
            if key == ('color_aug', 0, 0):  # or key == ('color_aug', -1, 0) or key == ('color_aug', 1, 0):
                idx = key[1]
                img = rgb_to_grayscale(inputs[key])
                batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
                # print('input size: {}'.format(img.shape))

                # print('+++++++++++++ pesi SP network:\n')
                # for name, param in self.models["superpoint"].named_parameters():
                #     if name == 'convPb.weight':
                #         print(param[0][:5].view(1, -1))

                sp_enc_out = self.models["superpoint"](img.to(self.device))
                # print("sp_out --> coarse kps: {}\t coarse desc: {}".format(sp_enc_out[0].shape, sp_enc_out[1].shape))
                semi, coarse_desc = sp_enc_out[0], sp_enc_out[1]
                # dense = torch.exp(semi)
                # dense = dense / (torch.sum(dense, dim=1).unsqueeze(1) + 0.00001)  # Should sum to 1
                # dense = self.softmax(semi)  # Use torch Softmax instead of previous 2 lines
                dense = semi.clone()
                # Compute normalization (instead of Softmax) channel-wise
                max_for_each_batch_size = torch.amax(dense, dim=(1, 2, 3))
                min_for_each_batch_size = torch.amin(dense, dim=(1, 2, 3))
                dense_num = (dense - min_for_each_batch_size.view(-1, 1, 1, 1))
                dense_den = (max_for_each_batch_size - min_for_each_batch_size).view(-1, 1, 1, 1)
                dense = dense_num / dense_den

                nodust = dense[:, :-1, :, :]
                Hc = int(H / 8)
                Wc = int(W / 8)
                nodust = torch.permute(nodust, (0, 2, 3, 1))
                heatmap = torch.reshape(nodust, [batch_size, Hc, Wc, 8, 8])
                heatmap = torch.permute(heatmap, [0, 1, 3, 2, 4])
                heatmap = torch.reshape(heatmap, [batch_size, Hc * 8, Wc * 8])
                # print('heatmap final shape {}'.format(heatmap.shape))
                # print("heatmap - max {}\tmin {}".format(heatmap.max(), heatmap.min()))
                bs, xs, ys = torch.where(heatmap >= self.opt.conf_thresh * torch.ones([batch_size, heatmap.shape[1],
                                                                                       heatmap.shape[2]]).to(
                    self.device))  # OLD indexing considering threshold of SuperPoint
                # bs, xs, ys = torch.where(heatmap >=
                #                          torch.zeros([batch_size, heatmap.shape[1], heatmap.shape[2]]).to(self.device))
                if len(xs) == 0:
                    dict_to_add[('color_aug_SP_out', idx, 0)] = (torch.zeros((4, 0), device=self.device),
                                                                 torch.zeros((256, 0), device=self.device),
                                                                 torch.zeros((img.shape[0], img.shape[2], img.shape[3]),
                                                                             device=self.device))
                    return {**inputs, **dict_to_add}
                pts = torch.zeros((4, len(xs)))  # Populate point data sized 3(or 4 if batch is considered)xN.
                pts[0, :] = bs
                pts[1, :] = ys
                pts[2, :] = xs
                pts[3, :] = heatmap[bs, xs, ys]
                # pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # See starting method comments!
                inds = torch.argsort(pts[3, :], descending=True)
                pts = pts[:, inds]  # Sort by confidence.

                # Remove points along border.
                toremoveB = torch.zeros((pts[0, :].shape[0],), dtype=torch.bool)
                toremoveW = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (W - bord))
                toremoveH = torch.logical_or(pts[2, :] < bord, pts[2, :] >= (H - bord))
                toremove = torch.logical_or(torch.logical_or(toremoveB, toremoveW), toremoveH).to(self.device)
                pts = pts[:, ~toremove]

                # print('pts final shape {}'.format(pts.shape))
                # --- Process descriptor.
                D = coarse_desc.shape[1]
                if pts.shape[1] == 0:
                    desc = torch.zeros((batch_size, D, 0))
                else:
                    # Interpolate into descriptor map using 2D point locations.
                    samp_pts = pts[1:3, :].clone()
                    samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
                    samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
                    samp_pts = samp_pts.transpose(0, 1).contiguous()
                    samp_pts = samp_pts.view(1, 1, -1, 2)
                    samp_pts = samp_pts.float().to(self.device)
                    # print('grid_sample --> input:', coarse_desc.view(1, 256, 24, -1).shape)
                    # print('grid_sample --> grid:', samp_pts.shape)
                    desc = torch.nn.functional.grid_sample(
                        coarse_desc.view(1, 256, 24, -1).cpu(), samp_pts.cpu(), align_corners=True)
                    desc = desc.reshape(D, -1)
                    desc = desc / torch.norm(desc, dim=0)
                    # print('desc final shape:', desc.shape)
                dict_to_add[('color_aug_SP_out', idx, 0)] = (pts, desc, heatmap)
        return {**inputs, **dict_to_add}

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if isinstance(ipt, tuple): continue  # SuperPoint outs are in a tuple and already in cuda
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":  # pose_model_type != "shared" by default
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:  # opt.predictive_mask is False by default
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:  # use_pose_net is True by default
            outputs.update(self.predict_poses(inputs, features))  # cam_T_cam is added and poses too to output dict

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)  # 3 losses (img scale dependent) + total loss

        print(losses)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:  # num_pose_frames = 2 by default
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":  # pose_model_type = 'separate_resnet' by default
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}  # keys: [0, -1, +1]

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]  # is the order of T (from -1 to 0)
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]  # viceversa (0 to +1)

                    if self.opt.pose_model_type == "separate_resnet":  # pose_encoder returns the pose encoded
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]  # cat pose_feats
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    # print('axisangle: {}\ntranslation: {}'.format(axisangle.shape, translation.shape))
                    outputs[("axisangle", 0, f_i)] = axisangle  # B, 2, 1, 3 --> there are 2 poses, but we need 1
                    outputs[("translation", 0, f_i)] = translation  # B, 2, 1, 3 --> there are 2 poses, but we need 1

                    # Invert the matrix if the frame id is negative (taking [:, 0] means the first pose)
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            inputs = self.pre_process_batch(inputs)

            if ('color_aug_SP_out', 0, 0) not in inputs:
                img: torch.Tensor = rgb_to_grayscale(inputs[('color_aug', 0, 0)])
                inputs[('color_aug_SP_out', 0, 0)] = (torch.zeros((4, 0)),
                                                      torch.zeros((256, 0)),
                                                      torch.zeros((img.shape[0], img.shape[2], img.shape[3])))
            elif inputs[('color_aug_SP_out', 0, 0)][-1] is None:  # if heatmap is None -> there's no input color_aug img
                img: torch.Tensor = rgb_to_grayscale(inputs['color_aug'])
                inputs[('color_aug_SP_out', 0, 0)][-1] = torch.zeros((img.shape[0], img.shape[2], img.shape[3]))

            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:  # v1_multiscale is False by default
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":  # frame_id != "s" by default --> else branch
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":  # pose_model_type == "separete_cnn" by default
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")  # this is the reprojected sampled image color

                # plt.figure(0)
                # plt.title(("reprojected", frame_id, scale))
                # plt.imshow(outputs[("color", frame_id, scale)][0].permute(1, 2, 0).detach().cpu())
                # plt.figure(1)
                # plt.title(("target (0)", 0, scale))
                # plt.imshow(inputs[("color_aug", 0, 0)][0].permute(1, 2, 0).detach().cpu())
                # plt.figure(2)
                # plt.title(("source (-1)", 0, scale))
                # plt.imshow(inputs[("color_aug", -1, 0)][0].permute(1, 2, 0).detach().cpu())
                # plt.show()
                # exit(0)

                if not self.opt.disable_automasking:  # opt.disable_automasking is False by default -> not (...) is True
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:  # opt.no_ssim is False --> else branch
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_reprojection_loss_w_heatmap(self, pred, target, heatmap):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        # l1_loss_times_heat = l1_loss * heatmap.unsqueeze(1)

        if self.opt.no_ssim:  # opt.no_ssim is False --> else branch
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            # ssim_loss_times_heat = ssim_loss * heatmap.unsqueeze(1)
            # why is not (1-SSIM())?? why is not 0.85/2 like in formule above equation 3 in the paper?
            # --> maybe the answer is in the SSIM() method!
            reprojection_loss = heatmap.unsqueeze(1) * (0.85 * ssim_loss + 0.15 * l1_loss)

        return reprojection_loss

    @staticmethod
    def compute_regularization_SP_loss(heatmap):
        diff = torch.ones_like(heatmap, requires_grad=True) - heatmap
        sum_reg = torch.sum(diff)
        return sum_reg

    def favour_SP_sparsity_loss(self, heatmap):
        tot_pixels = torch.tensor(heatmap.shape[0] * heatmap.shape[1] * heatmap.shape[2], device=self.device)
        tot_wanted_pixels = tot_pixels / 60
        diff = torch.abs(torch.sum(heatmap) - tot_wanted_pixels)
        return diff

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        global idxs, idxs_SP, identity_reprojection_loss, identity_reprojection_loss_SP
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss, loss_SP = 0, 0
            reprojection_losses = []  # indicated by L_p in the MD2 paper
            reprojection_losses_SP = []  # SP loss term

            if self.opt.v1_multiscale:  # opt.v1_multiscale False by default --> else branch
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:  # -1 and 1 --> prev and next image
                pred = outputs[("color", frame_id, scale)]

                # plt.figure(0)
                # plt.title(("reprojected", frame_id, scale))
                # plt.imshow(outputs[("color", frame_id, scale)][0].permute(1, 2, 0).detach().cpu())
                # plt.figure(1)
                # plt.title(("target (0)", 0, scale))
                # plt.imshow(inputs[("color_aug", 0, 0)][0].permute(1, 2, 0).detach().cpu())
                # plt.figure(2)
                # plt.title(("source (-1)", 0, scale))
                # plt.imshow(inputs[("color_aug", -1, 0)][0].permute(1, 2, 0).detach().cpu())
                # plt.show()
                # exit(0)

                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:  # False by default -> else branch
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:  # False by default
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:  # False by default -> else branch
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:  # combined.shape[1] != 1
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)  # min among identity_repr_loss and reprojected_loss

            if not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)  # indicated by L_s in the MD2 paper

            loss = loss + self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

            ########################################################################################
            ############## LOSS SUPERPOINT SECTION - REPROJECTION AND REGULARIZAZION ###############
            ########################################################################################
            if scale != 0: continue  # FOR SP LOSS TERM WE IGNORE SCALE != 0
            # TODO: different scales for heatmap (?)
            heatmap = inputs[('color_aug_SP_out', 0, scale)][2]  # heatmap of target image (indexed by 0)

            loss_SP_reg = self.compute_regularization_SP_loss(heatmap)
            loss_SP_sparsity = self.favour_SP_sparsity_loss(heatmap)  # about 2000 pixels per image (16k for batch=8)
            losses["loss_SP_reg/{}".format(scale)] = loss_SP_reg
            losses["loss_SP_spars/{}".format(scale)] = loss_SP_sparsity

            for frame_id in self.opt.frame_ids[1:]:  # -1 and 1 --> prev and next image
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses_SP.append(self.compute_reprojection_loss_w_heatmap(pred, target, heatmap))

            reprojection_losses_SP = torch.cat(reprojection_losses_SP, 1)

            if not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                identity_reprojection_losses_SP = []
                # N.B. identity is the reprojection with target image -> we compare taget, +1, -1, and we choose the one
                # with reprojection error which is the minimum
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses_SP.append(
                        self.compute_reprojection_loss_w_heatmap(pred, target, heatmap))

                identity_reprojection_losses_SP = torch.cat(identity_reprojection_losses_SP, 1)

                if self.opt.avg_reprojection:  # False by default -> else branch
                    identity_reprojection_loss_SP = identity_reprojection_losses_SP.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss_SP = identity_reprojection_losses_SP

            elif self.opt.predictive_mask:  # False by default
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:  # False by default -> else branch
                reprojection_loss_SP = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss_SP = reprojection_losses_SP

            if not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                # add random numbers to break ties (NB: it was randn, now is rand to avoid loss < 0)
                identity_reprojection_loss_SP += torch.rand(
                    identity_reprojection_loss_SP.shape, device=self.device) * 0.00001

                combined_SP = torch.cat((identity_reprojection_loss_SP, reprojection_loss_SP), dim=1)
            else:
                combined_SP = reprojection_loss_SP

            if combined_SP.shape[1] == 1:  # combined_SP.shape[1] != 1
                to_optimise_SP = combined_SP
            else:
                # min among identity_repr_loss_SP and reprojected_loss_SP
                to_optimise_SP, idxs_SP = torch.min(combined_SP, dim=1)

            if not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                outputs["identity_selection/{}".format(scale)] = (
                        idxs_SP > identity_reprojection_loss_SP.shape[1] - 1).float()

            loss_SP += to_optimise_SP.mean()
            total_loss += (self.opt.SP_loss_gamma * loss_SP)
            total_loss += (self.opt.SP_regulariz_loss_decay * loss_SP_reg)
            total_loss += (self.opt.SP_loss_sparsity_weight * loss_SP_sparsity)
            print('SUM HEATMAP VALS:', torch.sum(heatmap), '\n')
            losses["loss_SP/{}".format(scale)] = loss_SP

            '''
            # DELETE THIS PART WITH PARSIMONIA!!!!

            # Grab SP probs to mask input image and disparity too. L_s computed just for img_0 as MD2 standard pipeline
            heatmap = inputs[('color_aug_SP_out', 0, scale)][-1]
            # heatmap.retain_grad()  # remove
            # l = torch.norm(heatmap)
            # l.backward()
            # print('706 - ORA DOVREBBE AVERE IL GRADIENTE:', heatmap.grad.shape)
            # exit(0)

            # temp = torch.ones(heatmap.shape).to(self.device)  # remove
            # loss_SP = torch.norm(temp-heatmap)  # remove

            # print('loss_SP:', loss_SP)
            # print('loss_SP shape:', loss_SP.shape)
            # print('1:', heatmap.grad)
            # self.model_optimizer.zero_grad()
            # print(losses)
            # loss_SP.backward()
            # # plot_grad_flow(self.models['superpoint'].named_parameters())
            # self.model_optimizer.step()
            # print('2:', heatmap.grad)
            # exit(0)
            B, H, W = heatmap.shape
            # heat_bool = torch.where(heatmap.unsqueeze(1) > self.opt.conf_thresh * torch.ones([B, 1, H, W]).to(self.device))
            # heat_bool.retain_grad()  # remove
            # print('heat bool shape:', heat_bool.shape)
            # mask_sp = torch.zeros(heatmap.unsqueeze(1).shape, requires_grad=True).to(self.device)
            # mask_sp[heat_bool] = 1.0
            # mask_sp.retain_grad()  # remove
            # masked_img = mask_sp * color
            # masked_disp = mask_sp * disp

            # masked_img = heatmap.unsqueeze(1).repeat(1, 3, 1, 1) * color
            # masked_disp = heatmap.unsqueeze(1) * disp

            masked_img = heatmap.unsqueeze(1).repeat(1, 3, 1, 1) * color
            masked_disp = heatmap.unsqueeze(1) * disp

            # masked_img.reatain_grad()  # remove (SURE??????)
            # masked_disp.retain_grad()  # remove
            # print(masked_img)
            # exit(0)
            # masked_disp.retain_grad()
            smooth_loss_SP = get_smooth_loss(masked_disp, masked_img)
            # smooth_loss_SP.retain_grad()  # remove
            # loss += (self.opt.SP_disparity_smoothness * smooth_loss_SP / (2 ** scale))  # old formula
            loss_SP = loss_SP + (self.opt.SP_disparity_smoothness * smooth_loss_SP)

            # loss_SP.retain_grad()
            # print('requires_grad SP:', loss_SP.requires_grad, '\tgrad:', loss_SP.grad)
            # print('leaf SP:', loss_SP.is_leaf)
            total_loss = total_loss + loss_SP
            # print("autograd grad SP:", torch.autograd.grad(total_loss, masked_img, retain_graph=True)[0][0][0][0][0])
            losses["loss_SP/{}".format(scale)] = loss_SP
            '''

        # total_loss is not properly correct because it's scaled by 4 (num_scales) while SP_loss is computed just with
        # one single scale level. This global scale could be interpretated as an additional scaling of the SP_loss
        total_loss = total_loss / self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        pts = inputs[('color_aug_SP_out', 0, 0)][0]  # pts coords of heatmap values

        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:  # False by default
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:  # False by default -> not disable_automasking is True
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

            writer.add_image("SP_heatmap_on_target_0/{}".format(j),
                             self.stack_SP_over_imgs(inputs[("color", 0, 0)][j],
                                                     pts[:, pts[0] == j]))

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        print('Save model in:', save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    @staticmethod
    def plot_grad_flow(named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            print("name:", n)
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                print('param grad:', p.grad)
                ave_grads.append(p.grad.abs().mean().cpu())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.pause(1)

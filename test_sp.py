import os
import torch
import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(192 * 640, 50)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = self.relu1(a1)
        return h1


class Options:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.learning_rate = 1e-4
        self.scheduler_step_size = 15
        self.load_weights_folder = os.path.expanduser('.\\tmp\\mono_model\\models\\weights_19_copia_originale\\')
        self.epochs = 3


opt = Options()


def load_model(model: SuperPointNet):
    print("Loading Superpoint weights...")
    path = os.path.join(opt.load_weights_folder, "superpoint.pth")
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for param_name, parameter in model.named_parameters():
        print(param_name)
        parameter.requires_grad_()


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def main():
    model = SuperPointNet()
    # model = Net()
    model.to(opt.device)
    parameters_to_train = list(model.parameters())
    model_optimizer = optim.Adam(parameters_to_train, opt.learning_rate)
    model_lr_scheduler = optim.lr_scheduler.StepLR(model_optimizer, opt.scheduler_step_size, 0.1)
    load_model(model=model)

    for name, param in model.named_parameters():
        # print(name)
        if name == 'convPb.weight':  # 'fc1.weight':
            print(name)
            print(param[0][:5].view(1, -1))

    old_weights = list(model.parameters())[-6].clone()  # convPb.weight
    print('Same weigths?', torch.equal(list(model.parameters())[-6], old_weights))

    model.train()
    for t in range(200):
        loss_SP, total_loss = 0, 0

        model_lr_scheduler.step()
        img = rgb_to_grayscale(torch.randn(8, 3, 192, 640).to(opt.device))
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        # img = torch.randn(8, 1, 192, 640).to(opt.device)  # for Net()
        semi, coarse_desc = model(img)
        # semi = model(img.squeeze(1).flatten())  # for Net()

        dense = torch.exp(semi)
        dense = dense / (torch.sum(dense, dim=1).unsqueeze(1) + 0.00001)  # Should sum to 1
        # dense.retain_grad()  # remove
        nodust = dense[:, :-1, :, :]
        Hc = int(H / 8)
        Wc = int(W / 8)
        nodust = torch.permute(nodust, (0, 2, 3, 1))
        # nodust.retain_grad()  # remove
        heatmap = torch.reshape(nodust, [batch_size, Hc, Wc, 8, 8])
        heatmap = torch.permute(heatmap, [0, 1, 3, 2, 4])
        heatmap = torch.reshape(heatmap, [batch_size, Hc * 8, Wc * 8])
        bs, xs, ys = torch.where(heatmap >= 0.01 * torch.ones([batch_size, heatmap.shape[1],
                                                               heatmap.shape[2]]).to(opt.device))
        pts = torch.zeros((4, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = bs
        pts[1, :] = ys
        pts[2, :] = xs
        pts[3, :] = heatmap[bs, xs, ys]
        # pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # NMS yes or not? if yes, implement it
        inds = torch.argsort(pts[3, :], descending=True)
        pts = pts[:, inds]  # Sort by confidence.

        # Remove points along border.
        # toremoveB = torch.zeros((pts[0, :].shape[0],), dtype=torch.bool)
        # toremoveW = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (W - bord))
        # toremoveH = torch.logical_or(pts[2, :] < bord, pts[2, :] >= (H - bord))
        # toremove = torch.logical_or(torch.logical_or(toremoveB, toremoveW), toremoveH).to(self.device)
        # pts = pts[:, ~toremove]

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
            samp_pts = samp_pts.float().to(opt.device)
            # print('grid_sample --> input:', coarse_desc.view(1, 256, 24, -1).shape)
            # print('grid_sample --> grid:', samp_pts.shape)
            desc = torch.nn.functional.grid_sample(
                coarse_desc.view(1, 256, 24, -1).cpu(), samp_pts.cpu(), align_corners=True)
            desc = desc.reshape(D, -1)
            desc = desc / torch.norm(desc, dim=0)

        #####################################
        # loss computation part #############
        #####################################

        # Grab SP probs to mask input image and disparity too. L_s computed just for img_0 as MD2 standard pipeline

        # heat_bool = torch.where(
        #     heatmap.unsqueeze(1) > 0.01 * torch.ones([batch_size, 1, H, W]).to(opt.device))
        # mask_sp = torch.zeros(heatmap.unsqueeze(1).shape, requires_grad=True).to(opt.device)
        # mask_sp[heat_bool] = 1.0
        # masked_img = mask_sp * torch.randn(8, 3, 192, 640).to(opt.device)
        # masked_disp = mask_sp * torch.randn(8, 1, 192, 640).to(opt.device)
        masked_img = heatmap.unsqueeze(1).repeat(1, 3, 1, 1) * torch.randn(8, 3, 192, 640).to(opt.device)
        masked_disp = heatmap.unsqueeze(1) * torch.randn(8, 1, 192, 640).to(opt.device)
        smooth_loss_SP = get_smooth_loss(masked_disp, masked_img)
        # loss += (self.opt.SP_disparity_smoothness * smooth_loss_SP / (2 ** scale))  # old formula
        loss_SP = loss_SP + (0.15 * smooth_loss_SP)

        total_loss = total_loss + loss_SP

        # loss = torch.norm(heatmap)
        model_optimizer.zero_grad()
        total_loss.backward()
        model_optimizer.step()
        for name, param in model.named_parameters():
            if name == 'convPb.weight':  # 'fc1.weight':
                # print(name, 'loss:', loss)
                print(param[0][:5].view(1, -1))
        print('Same weigths?', torch.equal(list(model.parameters())[-6], old_weights))


if __name__ == "__main__":
    main()

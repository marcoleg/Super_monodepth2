import os
import torch
import numpy as np
import torchvision
from torchvision.io import ImageReadMode
from networks.superpoint import SuperPointNet
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#matplotlib.use('Agg')

class Configuration:

    def __init__(self):
        # self.osys = os.path.join('C:\\', 'Users', os.getlogin()) if os.sep == '\\' \
        #     else os.path.join('/root', os.getlogin())
        self.osys = os.path.join('/', 'root', 'Documents')
        self.proj_dir = os.path.join(self.osys, 'Super_monodepth2')
        self.weights_path = os.path.join(self.proj_dir, 'superpoint.pth')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cfg = Configuration()

def main():
    print(cfg.weights_path, cfg.device)
    sp = SuperPointNet()
    sp.load_state_dict(torch.load(cfg.weights_path))
    sp.to(cfg.device)


def stack_SP_over_imgs(target_img, pts_wrt_batch):
    # imshow target image
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(target_img.permute(1, 2, 0).detach().cpu())  # permute to match the order of dims expected by imshow
    ax.scatter(pts_wrt_batch[1, :].detach().cpu(), pts_wrt_batch[2, :].detach().cpu(), marker="o", s=5, c='red')

    # Convert the output to a tensor
    canvas = FigureCanvas(fig)
    canvas.draw()
    output_img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    # Remove alpha channel and convert to PyTorch tensor
    output_img = torch.from_numpy(output_img[:, :, :3].copy()).permute(2, 0, 1)
    b, h, w = target_img.shape
    canvas_w, canvas_h = canvas.get_width_height()
    output_img = output_img[:, int((canvas_h - h)/2):int(canvas_h - ((canvas_h - h)/2)), :]
    plt.show()
    return output_img

# if __name__=='__main__':
#     main()
softmax = nn.Softmax()
print(cfg.weights_path, cfg.device)
sp = SuperPointNet()
sp.load_state_dict(torch.load(cfg.weights_path))
sp.to(cfg.device)
img = torchvision.io.read_image('/root/Archive/Dataset/01/image_0/000000.png', mode=ImageReadMode.RGB).float()
img = img / 255.0
print(img.shape)
img = img.to(cfg.device)
# print(img)

img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
# output transformation
batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]

with torch.no_grad():
    semi, coarse_desc = sp(torchvision.transforms.functional.rgb_to_grayscale(img))
    dense = softmax(semi)  # Use torch Softmax instead of previous 2 lines (sum = 1 in channel dim)

nodust = dense[:, :-1, :, :]
print('dustbin sum batch 0: {}'.format(torch.sum(dense[0, -1, :, :])))  # 24x80
Hc = int(H / 8)
Wc = int(W / 8)
nodust = torch.permute(nodust, (0, 2, 3, 1))
heatmap = torch.reshape(nodust, [batch_size, Hc, Wc, 8, 8])
heatmap = torch.permute(heatmap, [0, 1, 3, 2, 4])
heatmap = torch.reshape(heatmap, [batch_size, Hc * 8, Wc * 8])
# print('heatmap final shape {}'.format(heatmap.shape))
# print("heatmap - max {}\tmin {}".format(heatmap.max(), heatmap.min()))

# OLD indexing considering threshold of SuperPoint
bs, xs, ys = torch.where(heatmap >= 1.0 * torch.ones_like(heatmap))

pts = torch.zeros((4, len(xs)))  # Populate point data sized 3(or 4 if batch is considered)xN.
pts[0, :] = bs
pts[1, :] = ys
pts[2, :] = xs
pts[3, :] = heatmap[bs, xs, ys]

img_out = stack_SP_over_imgs(img[0], pts)




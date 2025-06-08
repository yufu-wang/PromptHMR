import numpy as np
import cv2
import os
import matplotlib.pylab as plt
from matplotlib.offsetbox import AnchoredText
import cv2
import torch


def rescale_intr(_intr, size_factor=[1, 1], crop=[0, 0]):
    
    if torch.is_tensor(_intr):
        intr = _intr.clone()
    else:
        intr = _intr.copy()

    intr[2] += crop[0]
    intr[3] += crop[1]
    intr[0] /= size_factor[0]
    intr[1] /= size_factor[1]
    intr[2] /= size_factor[0]
    intr[3] /= size_factor[1]
    return intr

def save_image(i, image, intr, time, intr_gt=None):

    image = image[0].permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if intr_gt is not None:
        if len(intr) == 4:
            at = r"Ground-truth: $f_x$={:.1f}, $f_y$={:.1f}, $c_x$={:.1f}, $c_y$={:.1f} \
                Estimate: $f_x$={:.1f}, $f_y$={:.1f}, $c_x$={:.1f}, $c_y$={:.1f}".format(*intr_gt, *intr)
        else:
            at = r"Ground-truth: $f_x$={:.1f}, $f_y$={:.1f}, $c_x$={:.1f}, $c_y$={:.1f}, $\xi$={:.2f} \
                Estimate: $f_x$={:.1f}, $f_y$={:.1f}, $c_x$={:.1f}, $c_y$={:.1f}, $\xi$={:.2f}".format(*intr_gt, *intr)
    else:
        if len(intr) == 4:
            at = r"$f_x$={:.1f}, $f_y$={:.1f}, $c_x$={:.1f}, $c_y$={:.1f}".format(*intr)
        else:
            at = r"$f_x$={:.1f}, $f_y$={:.1f}, $c_x$={:.1f}, $c_y$={:.1f}, $\xi$={:.2f}".format(*intr)

    plt.text(0.025, 0.95, at, fontdict=dict(fontsize=12), bbox=dict(facecolor='white', alpha=0.6, 
             edgecolor='gray'), transform=ax.transAxes)

    plt.text(0.025, 0.05, "Image {} / Time {:.1f} s".format(i, time), fontdict=dict(fontsize=12), 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'), transform=ax.transAxes)
    
    plt.axis('off')

    if not os.path.isdir("figures/images"):
        os.makedirs("figures/images")

    plt.savefig("figures/images/{:04d}.png".format(i), bbox_inches="tight", dpi=300)
    plt.close()


def create_video():

    image_folder = 'figures/images'
    video_name = 'figures/images/video.mp4'

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 4, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # to see last result from global BA longer
    for _ in range(10):
        video.write(cv2.imread(os.path.join(image_folder, images[-1]))) 

    cv2.destroyAllWindows()
    video.release()

    images = os.listdir(image_folder)
    for image in images:
        if image.endswith(".png"):
            os.remove(os.path.join(image_folder, image))
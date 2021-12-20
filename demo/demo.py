import glob
import tqdm
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pseudo_labeling.mask_processing import dilate_pseudomasks
from demo.dilation_demo import plot_bboxes_yolo

input_masks = ['./pseudo_masks/new_dataset/*.npz']
# input_masks = ['./pseudo_masks/new_dataset/IMG_20210924_160155838.npz']
# input_masks = ['./pseudo_masks/new_dataset/IMG_20210924_161902460.npz']

data_path = './new_dataset/train'

if len(input_masks) == 1:
    input_masks = glob.glob(os.path.expanduser(input_masks[0]))
    assert input_masks, "The input path(s) was not found"
for path in tqdm.tqdm(input_masks):
    masks = np.load(path)['arr_0'].astype(np.uint8)  # H x W x n
    masks_id = os.path.basename(path)
    masks_id = os.path.splitext(masks_id)[0]

    bboxes = np.loadtxt(os.path.join(data_path, f'{masks_id}.txt'), delimiter=" ", dtype=np.float32)
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, axis=0)
    bboxes = bboxes[:, 1:]  # remove classes

    image = cv2.imread(os.path.join(data_path, f'{masks_id}.jpg'))

    for obj in range(masks.shape[2]):
        fig1, ax1 = plt.subplots(1, 1)

        mask = masks[:, :, obj]
        bbox = bboxes[obj]

        # # Before processing
        # ax1.imshow(image)
        # ax1.imshow(mask, alpha=0.5)
        # ax1.set_axis_off()
        # plot_bboxes_yolo(image, [bbox], ax1)

        dilated_mask = dilate_pseudomasks(mask[:, :, None].copy(), [bbox])
        dilated_mask = dilated_mask.squeeze()

        ax1.imshow(image)
        ax1.imshow(dilated_mask + mask, alpha=0.5)
        ax1.set_axis_off()
        plot_bboxes_yolo(image, [bbox], ax1)

        manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        manager.set_window_title(f'{masks_id}')
        plt.savefig(f'{masks_id}_dilation.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()

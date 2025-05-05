# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
from pathlib import Path


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames



from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class Fusion_dataset(Dataset):
    def __init__(self, 
                 dataset_root="/drive/faizanai.rrl/faizan/APWNet/datasets", 
                 type="MSRS", 
                 split='train', 
                 ir_path=None, 
                 vi_path=None,
                 resize=None):
        super(Fusion_dataset, self).__init__()
        self.type = type
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        assert type in ["MSRS", 'TNO', 'rrl', 'RoadScene'], 'Type must be "MSRS", "TNO", "rrl", or "RoadScene"'

        self.root = Path(dataset_root) / type
        self.resize = resize  # Resize option as (width, height)

        if split == 'train':
            data_dir_vis = self.root / type.lower() / "train" / "vi"
            data_dir_ir = self.root / type.lower() / "train" / "ir"
            data_dir_label = self.root / type.lower() / "train" / "labels"
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def preprocess_image(self, image, is_gray):
        """
        Preprocess the image:
        - Resize if resize is set.
        - Add batch and channel dimensions for grayscale images.
        """
        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_LINEAR)
        
        if is_gray:
            if len(image.shape) == 2:  # If image is grayscale without channel dimension
                image = np.expand_dims(image, axis=0)  # Add channel dimension
        else:
            if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[-1] == 1:
                raise ValueError("Expected a color image with 3 channels.")
        
        return image

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]

            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)  # Load grayscale image
            label = np.array(Image.open(label_path))

            # Preprocess images
            image_vis = self.preprocess_image(image_vis, is_gray=False)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            )

            image_ir = self.preprocess_image(image_inf, is_gray=True)
            image_ir = np.asarray(image_ir, dtype=np.float32) / 255.0

            label = np.asarray(Image.fromarray(label), dtype=np.int64)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )

        elif self.split == 'val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)  # Load grayscale image

            # Preprocess images
            image_vis = self.preprocess_image(image_vis, is_gray= True if self.type == "TNO" else False)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            )

            image_ir = self.preprocess_image(image_inf, is_gray=True)
            image_ir = np.asarray(image_ir, dtype=np.float32) / 255.0

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length



# if __name__ == '__main__':
    # data_dir = '/data1/yjt/MFFusion/dataset/'
    # train_dataset = MF_dataset(data_dir, 'train', have_label=True)
    # print("the training dataset is length:{}".format(train_dataset.length))
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # train_loader.n_iter = len(train_loader)
    # for it, (image_vis, image_ir, label) in enumerate(train_loader):
    #     if it == 5:
    #         image_vis.numpy()
    #         print(image_vis.shape)
    #         image_ir.numpy()
    #         print(image_ir.shape)
    #         break

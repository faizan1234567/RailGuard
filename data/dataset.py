import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from natsort import natsorted
from pathlib import Path

class Fusion_dataset(Dataset):
    def __init__(self, split, data_name= "MSRS", shape = (480, 640)):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.data_root = Path("/drive/faizanai.rrl/faizan/APWNet/datasets")
        self.data_name = data_name
        self.tno = self.data_root / "TNO" / "tno"
        self.road_scene = self.data_root / "RoadScene" / "roadscene"
        self.msrs = self.data_root / "MSRS" / "msrs"
        self.rrl = self.data_root / "rrl"

        self.imgh, self.imgw= shape[0], shape[1]
        if split == 'train' and self.data_name == "MSRS":
            self.vis_dir = self.msrs / "train" / "vi"
            self.ir_dir = self.msrs / "train" / "ir"
            self.label_dir = self.msrs / "train" / "labels"
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif (split == 'val' or split == 'test') and self.data_name == "MSRS":
            self.vis_dir = self.msrs / "test" / "vi"
            self.ir_dir = self.msrs / "test" / "ir"
            self.label_dir = self.msrs / "test" / "labels"
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif (split == 'val' or split == 'test') and self.data_name == "TNO":
            self.vis_dir = self.tno / "vi"
            self.ir_dir = self.tno / "ir"
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))
        
        elif (split == 'val' or split == 'test') and self.data_name == "RoadScene":
            self.vis_dir = self.road_scene / "vi"
            self.ir_dir = self.road_scene / "ir"
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))
            
        elif (split == 'val' or split == 'test') and self.data_name == "rrl":
            self.vis_dir = self.rrl / "vi"
            self.ir_dir = self.rrl / "ir"
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

    def resize_tensor(self, tensor, new_height, new_width):
        """
        Reshapes the spatial dimensions (height, width) of a 4D tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (batch, channel, height, width).
            new_height (int): Desired height.
            new_width (int): Desired width.

        Returns:
            torch.Tensor: Reshaped tensor with spatial dimensions (new_height, new_width).
        """
        
        # Use torch.nn.functional.interpolate for resizing
        reshaped_tensor = torch.nn.functional.interpolate(
            tensor, size=(new_height, new_width), mode='bilinear', align_corners=False
        )
        
        return reshaped_tensor


    def __getitem__(self, index):
        img_name = self.filelist[index]
        vis_path = os.path.join(self.vis_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)          
        img_vis = self.imread(path=vis_path, vis_flage=True)
        img_ir = self.imread(path=ir_path, vis_flage=False)            
        if (self.split=='train' or self.split == "val") and self.data_name == "MSRS":            
            label_path = os.path.join(self.label_dir, img_name.split(".")[0] + ".txt")  
            label = self.imread(path=label_path, label=True, vis_flage=False)
            # label = label.type(torch.LongTensor)   
                  
        if (self.split=='train' or self.split == "val") and self.data_name == "MSRS" : 
            return img_vis, img_ir, label, img_name
        else:
            return img_vis, img_ir, img_name

    def __len__(self):
        return self.length
    
    
    def imread(self, path, label=False, vis_flage=True):
        if label:
            labels = []
            try:
                with open(path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])  
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([class_id, x_center, y_center, width, height])

                class_ids = [label[0] for label in labels]
                other_values = [label[1:] for label in labels]
                class_ids_tensor = torch.tensor(class_ids, dtype=torch.int32)
                other_values_tensor = torch.tensor(other_values, dtype=torch.float32)
                labels_tensor = torch.cat([class_ids_tensor.unsqueeze(1), other_values_tensor], dim=1)
                nl = len(labels_tensor)
                labels_out = torch.zeros((nl, 6))
                if nl:
                    labels_out[:, 1:] = labels_tensor
                return labels_out

            except FileNotFoundError:
                print(f"File not found: {path}")
                return torch.empty(0)  # Return an empty tensor if the file doesn't exist
            except Exception as e:
                print(f"An error occurred: {e}")
                return torch.empty(0)
          
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img)
                im_ts = TF.resize(im_ts, [self.imgh, self.imgw])
            else: ## infrared images single channel 
                img = Image.open(path).convert('L') 
                im_ts = TF.to_tensor(img)
                im_ts = TF.resize(im_ts, [self.imgh, self.imgw])
        return im_ts
    
    @staticmethod
    def custom_collate_fn(batch):
        vis_images, ir_images, labels, names = [], [], [], []
        for item in batch:
            vis_images.append(item[0])
            ir_images.append(item[1])
            labels.append(item[2])  # Keep labels as a list of tensors
            names.append(item[3])

        for i, lb in enumerate(labels):
            lb[:, 0] = i  # add target image index for build_targets()
        vis_images = torch.stack(vis_images)
        ir_images = torch.stack(ir_images)
        return vis_images, ir_images, torch.cat(labels, 0), names

    

if __name__ == "__main__":
    # root = '/home/radar/faizan/lab_codes/fusion_paper/baseline_methods/APWNet/datasets/TNO/tno'
    # dataset = fusionDataset(root= root, img_size=512)
    dataset = Fusion_dataset(split= "val", data_name= "TNO", shape= (320, 320))
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=12)
    # data = 
    data = next(iter(loader))
    img_ir, img_vi, names = data
    print(img_ir.shape, img_vi.shape, names)




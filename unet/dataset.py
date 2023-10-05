import os
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T

def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively
    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        if file.endswith('.jpg'):
            orig_img.append(file)
    for file in mask_dataset:
        if file.endswith('.png'):
            mask_img.append(file)

    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    orig_img.sort()
    mask_img.sort()
    
    return orig_img, mask_img

def preprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    # Define X and Y as number of images along with shape of one image
    # X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
    # y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
    
    data = []
    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h,i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = single_img/256.
        
        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
        single_mask = single_mask - 1 # to ensure classes #s start from 0
        data.append((single_img, single_mask))
    return data



class PetDataset(Dataset):
    #Init dataset
    def __init__(self, base_path):
        path1 = base_path + "originals/"
        path2 = base_path + "masks/"
        self.img, self.mask = LoadData(path1, path2)
        # X_train, X_valid = train_test_split(data, test_size=0.2, random_state=123)

    def __getitem__(self, index):
        i_h,i_w,i_c = [128, 128, 3]
        m_h,m_w,m_c = [128, 128, 1]
        
        # Define X and Y as number of images along with shape of one image
        # X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
        # y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
        
        # convert image into an array of desired shape (3 channels)
        file = img[index]
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h,i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = (single_img/256.).astype(np.float32)
        
        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
        single_mask = single_mask - 1 # to ensure classes #s start from 0
        return (single_img, single_mask)

    def __len__(self):
        return len(self.img)



# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()


# Oxford IIIT Pets Segmentation dataset loaded via torchvision.

def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

def trimap2f(trimap):
    return (img2t(trimap) * 255.0 - 1) / 2

class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=True,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)
    
    
def get_pet_dataloader(working_dir: str="/kaggle/working/", 
                       batch_size: int=4,
                       resize: int= 128
                       ):
    pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
    pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')
    transform_dict = args_to_dict(
        pre_transform=T.ToTensor(),
        pre_target_transform=T.ToTensor(),
        common_transform=T.Compose([
            T.Resize((resize, resize), interpolation=T.InterpolationMode.NEAREST),
        ]),
        post_target_transform=T.Compose([
            T.Lambda(tensor_trimap),
        ]),
    )

    pets_train = OxfordIIITPetsAugmented(
        root=pets_path_train,
        split="trainval",
        target_types="segmentation",
        download=True,
        **transform_dict,
    )
    pets_test = OxfordIIITPetsAugmented(
        root=pets_path_test,
        split="test",
        target_types="segmentation",
        download=True,
        **transform_dict,
    )
    pets_train_loader = torch.utils.data.DataLoader(
        pets_train,
        batch_size=batch_size,
        shuffle=True,
    )
    pets_test_loader = torch.utils.data.DataLoader(
        pets_test,
        batch_size=batch_size,
        shuffle=True,
    )
    return pets_train_loader, pets_test_loader


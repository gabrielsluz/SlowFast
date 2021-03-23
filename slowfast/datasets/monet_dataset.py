"""
Used in MONet-pytorch code, not here

python3 train.py --dataroot /datasets/dcl_clevrer --dataset_mode clevrer --name clevrer_monet --model monet --gpu_ids 0 --continue_train --num_slots 8 --save_epoch_freq 2 --batch_size 50 --print_freq 1000
"""
import os
import random

import torchvision.transforms.functional as TF

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

def get_filepath(video_id):
    """
    Returns the filepath of the video in the clevrer directory
    Args:
        video_id(int): The id of the video obtained from the json
    Returns:
        filepath(string): the path from the dataset directory to the video
            Ex: image_00000-01000/video_00428
    """
    filepath = ''
    #Find the interval
    min_id = (video_id // 1000)*1000
    max_id = min_id + 1000
    #Convert to 5 character strings
    video_id_s = str(video_id).zfill(5)
    min_id = str(min_id).zfill(5)
    max_id = str(max_id).zfill(5)

    filepath += 'image_' + min_id + '-' + max_id + '/'
    filepath += 'video_' + video_id_s
    return filepath

class CLEVRERDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3,
                            crop_size=192, # crop is done first
                            load_size=64,  # before resize
                            num_slots=11, display_ncols=11)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if opt.isTrain:
            max_i = 9999
            min_i = 0
        else:
            max_i = 14999
            min_i = 10000
        self.A_paths = [] #List of video frames paths
        #Sample 30 frames per video
        for index in range(min_i, max_i+1):
            sampled_frames = random.sample(range(0,127), 15)
            for frame_id in sampled_frames:
                self.A_paths.append(os.path.join(opt.dataroot, get_filepath(index), str(frame_id)+'.png'))

    def _transform(self, img):
        crop_size = 192
        load_size = 64
        input_nc = 3
        #img = TF.to_pil_image(img)
        img = TF.resize(img, (64,64))
        #img = TF.resized_crop(img, 64, 29, crop_size, crop_size, load_size)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * input_nc, [0.5] * input_nc)
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self._transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
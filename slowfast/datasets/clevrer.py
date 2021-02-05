#!/usr/bin/env python3

import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
import json

import torchvision.transforms as transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class ClevrerFrame(torch.utils.data.Dataset):
    """
    Dataset of frames of the CLEVRER dataset.
    __getitem__ Returns a random frame from a video
    """
    def __init__(self, cfg, mode):
        """
        Constructs the Clevrer frame loader with a given json file. The format of
        the csv file is the one used in Clevrer
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Clevrer Frame".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}

        logger.info("Constructing Clevrer Frame {}...".format(mode))
        self._construct_loader()
    
    def _get_filepath(self, mode, video_id):
        """
        Return the filepath of the video in the clevrer directory
        Args:
            mode(string): Options includes `train`, `val`, or `test` mode.
            video_id(int): The id of the video obtained from the json
        Returns:
            filepath(string): the path from the dataset directory to the video
                Ex: video_train/video_00000-01000/video_00428.mp4
        """
        filepath = 'video_' + mode + '/'
        #Find the interval
        min_id = (video_id // 1000)*1000
        max_id = min_id + 1000
        #Convert to 5 character strings
        video_id_s = str(video_id).zfill(5)
        min_id = str(min_id).zfill(5)
        max_id = str(max_id).zfill(5)

        filepath += 'video_' + min_id + '-' + max_id + '/'
        filepath += 'video_' + video_id_s + '.mp4'
        return filepath

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.json".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []

        with g_pathmgr.open(path_to_file, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                path = self._get_filepath(self.mode, int(data[i]['scene_index']))
                self._path_to_videos.append(
                    os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                )
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Clevrer Frame from {}".format(
            path_to_file
        )
        logger.info(
            "Constructing Clevrer Frame dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )


    def __getitem__(self, index):
        """
        Given the video index, return a random frame
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frame sampled from the video. The dimension
                is `channel` x `height` x `width`.
        """
        try:
            video_container = container.get_video_container(
                self._path_to_videos[index],
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.DATA.DECODING_BACKEND,
            )
        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(
                    self._path_to_videos[index], e
                )
            )
            exit()
        # Decode video. Meta info is used to perform selective decoding.
        frames = decoder.decode(
            container=video_container,
            sampling_rate=1,
            num_frames=1,
            clip_idx=-1,
            num_clips=1,
            video_meta=None,
            target_fps=self.cfg.DATA.TARGET_FPS,
            backend=self.cfg.DATA.DECODING_BACKEND,
            max_spatial_scale=0
        )

        if frames is None:
            logger.info(
                "Failed to decode video from {}".format(
                    self._path_to_videos[index]
                )
            )
        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        #C T H W -> C H W
        frames = torch.squeeze(frames, 1)
        # Perform resize
        frames = transforms.Resize([64, 64])(frames)
        return frames

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
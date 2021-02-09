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



def get_filepath(mode, video_id):
    """
    Returns the filepath of the video in the clevrer directory
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

#Transforms a string into a list of lowercase tokens, separating punctuation.
def string_to_token_list(text):
    text = text.lower()
    new_text = ''
    for c in text:
        if c.isalnum():
            new_text += c
        else:
            new_text += ' '
            new_text += c
    text_split = new_text.split()
    return text_split
        
#Updates vocab dict in place from a token list    
def update_vocab(vocab, token_list):
    for token in token_list:
        if token in vocab:
            continue
        else:
            vocab[token] = vocab[' counter ']
            vocab[' counter '] += 1

#Returns the token for a given index
def get_token_for_index(vocab, tgt_index):
    for token, index in vocab.items():
        if index == tgt_index:
            return token
    logger.info("Token for index {} not found".format(tgt_index))
    return "TOKEN NOT FOUND"

@DATASET_REGISTRY.register()
class Clevrer(torch.utils.data.Dataset):
    """
    CLEVRER Dataset.
    __getitem__ Returns a list o frames from a video and two questions with the anwers
    One descriptive question and one multiple choice
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

        self._num_retries = 10  

        logger.info("Constructing Clevrer Frame {}...".format(mode))
        self._construct_loader()


    def _create_vocabs(self, path_to_file):
        """
        Creates the vocabularies used to tokenize questions and answers
        It uses only the train.json file
        """
        ans_vocab = {} #Used for descriptive questions
        ans_vocab[' counter '] = 0
        possible_ans = ["0", "1", "2", "3", "4", "5", 
                        "yes", "no", "rubber", "metal",
                        "sphere", "cube", "cylinder",
                        "gray", "brown", "green", "red",
                        "blue", "purple", "yellow", "cyan"
        ]
        update_vocab(ans_vocab, possible_ans)

        vocab = {}
        vocab['CLS'] = 0
        vocab['|'] = 1
        vocab[' counter '] = 2 #Has spaces in key => not a valid token

        with open(path_to_file, "r") as f:
            data = json.load(f)

            for i in range(len(data)):
                questions = data[i]['questions']
                for q in questions:
                    split_q = string_to_token_list(q['question'])
                    update_vocab(vocab, split_q)
                    if q['question_type'] != 'descriptive':
                        choices = q['choices']
                        for c in choices:
                            split_c = string_to_token_list(c['choice'])
                            update_vocab(vocab, split_c)
        self.vocab = vocab
        self.ans_vocab = ans_vocab

    
    def _constructs_questions_ans(self, data, video_path):
        """
        Fills new_dict with the information contained in data extracted from the json
        Uses the mode to determine if the answers will be returned
        """
        new_dict = {}
        new_dict['video_path'] = video_path

        new_dict['des_q'] = []
        new_dict['mc_q'] = []

        if self.mode != "test":
            new_dict['des_ans'] = []
            new_dict['mc_ans'] = []

        questions = data['questions']
        for q in questions:
            split_q = string_to_token_list(q['question'])
            if q['question_type'] == 'descriptive':
                new_dict['des_q'].append(torch.tensor([self.vocab[token] for token in split_q],dtype=torch.long))

                if self.mode != 'test':
                    new_dict['des_ans'].append(self.ans_vocab[q['answer']])
            else:
                choices = q['choices']
                choice_ans = []
                for c in choices:
                    split_c = string_to_token_list(c['choice'])
                    split_q += ['|'] + split_c

                    if self.mode != 'test':
                        answer = 1 if c['answer'] == 'correct' else 0
                        choice_ans.append(answer)

                for _ in range(4 - len(choice_ans)): #Assume that the maximum of choices is 4
                    choice_ans.append(0)
                new_dict['mc_q'].append(torch.tensor([self.vocab[token] for token in split_q],dtype=torch.long))
                new_dict['mc_ans'].append(choice_ans)
        return new_dict

    def _construct_loader(self):
        """
        Construct the video loader.
        self._dataset:
            The main data structure: list of dicts: video_path 
                                    + list of descriptive questions (des_q) 
                                    + list of multiple choice questions (mc_q)
                                    + list of descriptive answers (des_ans)
                                    + list of multiple choice answers (mc_ans)
        The questions are already in LongTensor for the Embedding layer
        The answers for descriptive answers are a single integer indicating the
        index of the ans_vocab.
        The answers for multiple choice answers are a list of 4 binary numbers. 
        1 indicates correct and 0 indicates wrong.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.json".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._create_vocabs(path_to_file)
        #Main data structure
        self._dataset = []

        with g_pathmgr.open(path_to_file, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                path = get_filepath(self.mode, int(data[i]['scene_index']))
                full_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                self._dataset.append(self._constructs_questions_ans(data[i], full_path))

        assert (
            len(self._dataset) > 0
        ), "Failed to load Clevrer Frame from {}".format(
            path_to_file
        )
        logger.info(
            "Constructing Clevrer Frame dataloader (size: {}) from {}".format(
                len(self._dataset), path_to_file
            )
        )
    
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, question_dict, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
        A dict containing:
                frames (tensor): the frames of sampled from the video. The dimension
                    is `num frames` x `channel` x `height` x `width`.
                question_dict (dict): A dictionary with the questions and answers (if not test)
                index (int): if the video provided by pytorch sampler can be
                    decoded, then return the index of the video. If not, return the
                    index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        # -1 indicates random sampling.
        temporal_sample_index = -1
        spatial_sample_index = -1
        min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
        max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
        crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        if short_cycle_idx in [0, 1]:
            crop_size = int(
                round(
                    self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                    * self.cfg.MULTIGRID.DEFAULT_S
                )
            )
        if self.cfg.MULTIGRID.DEFAULT_S > 0:
            # Decreasing the scale is equivalent to using a larger "span"
            # in a sampling grid.
            min_scale = int(
                round(
                    float(min_scale)
                    * crop_size
                    / self.cfg.MULTIGRID.DEFAULT_S
                )
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._dataset[index]['video_path'],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._dataset[index]['video_path'], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._dataset[index]['video_path'], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                num_clips=1,
                video_meta=None,
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._dataset[index]['video_path'], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            # frames = utils.tensor_normalize(
            #     frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            # )
            # T H W C -> T C H W.
            frames = frames.permute(0, 3, 1, 2)
            # Perform resize
            transform_rs = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W]),
            transforms.ToTensor()
            ])
            frames_size = frames.size()
            resized_frames = torch.zeros(frames_size[0], frames_size[1], self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W)
            for i in range(frames_size[0]):
                resized_frames[i] = transform_rs(frames[i])
            #resized_frames = utils.pack_pathway_output(self.cfg, resized_frames)

            #Get the questions => a random descriptive and a random multiple choice
            question_dict = {}
            pick_des = random.randint(0, len(self._dataset[index]['des_q']) - 1)
            pick_mc = random.randint(0, len(self._dataset[index]['mc_q']) - 1)
            question_dict['des_q'] = self._dataset[index]['des_q'][pick_des]
            question_dict['mc_q'] = self._dataset[index]['mc_q'][pick_mc]
            if self.mode != "test":
                question_dict['des_ans'] = self._dataset[index]['des_ans'][pick_des]
                question_dict['mc_ans'] = self._dataset[index]['mc_ans'][pick_mc]
            
            output_dict = {}
            output_dict['frames'] = resized_frames
            output_dict['question_dict'] = question_dict
            output_dict['index'] = index
            return output_dict
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )
    
    def decode_question(self, q_tensor):
        """
        Decodes a vector of indexes into a string of words
        Args:
            q_tensor (LongTensor): Encoded question
        Returns:
            decoded_q (string): Decoded question
        """
        decoded_q = ""
        for index in q_tensor:
            decoded_q += get_token_for_index(self.vocab, index)
            decoded_q += " "
        return decoded_q
    
    def decode_answer(self, ans_index):
        """
        Decodes an integer into the corresponding answer
        Args:
            ans_index (int): Encoded answer
        Returns:
            (string): Decoded answer
        """
        return get_token_for_index(self.ans_vocab, ans_index)
    
    def get_video_info(self, index):
        """
        Args:
            index (int): Returned from __get_item__
        Returns:
            video_info (dict): video_path 
                            + list of descriptive questions (des_q) 
                            + list of multiple choice questions (mc_q)
                            + list of descriptive answers (des_ans)
                            + list of multiple choice answers (mc_ans)
        """
        if index < 0 or index >= len(self._dataset):
            logger.info("Video for index {} not found".format(index))
            return None
        video_dict = self._dataset[index]
        video_info = {}
        video_info['video_path'] =  video_dict['video_path']
        #Questions
        video_info['des_q'] = []
        for q in video_dict['des_q']:
            video_info['des_q'].append(self.decode_question(q))

        video_info['mc_q'] = []
        for q in video_dict['mc_q']:
            video_info['mc_q'].append(self.decode_question(q))
        
        if self.mode == 'test':
            return video_info

        #Answers
        video_info['des_ans'] = []
        for ans in video_dict['des_ans']:
            video_info['des_ans'].append(self.decode_answer(ans))

        video_info['mc_ans'] = []
        for ans in video_dict['mc_ans']:
            video_info['mc_ans'].append(ans)
        
        return video_info

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._dataset)


# ////////////////////////////////////////////////////////////////////////////


@DATASET_REGISTRY.register()
class Clevrerframe(torch.utils.data.Dataset):
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

        self._num_retries = 10

        logger.info("Constructing Clevrer Frame {}...".format(mode))
        self._construct_loader()

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
                path = get_filepath(self.mode, int(data[i]['scene_index']))
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
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
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
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

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

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            # frames = utils.tensor_normalize(
            #     frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            # )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            #C T H W -> C H W
            frames = torch.squeeze(frames, 1)
            # Perform resize
            transform_rs = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W]),
            transforms.ToTensor()
            ])
            frames = transform_rs(frames)
            return frames
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
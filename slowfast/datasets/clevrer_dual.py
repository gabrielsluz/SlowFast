#Clevrer_des and Clevrer_mc
import torch 
import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
import json
import copy
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

#_---____-___---_-____-_-__DES_---____-___---_-____-_-__
@DATASET_REGISTRY.register()
class Clevrer_des(torch.utils.data.Dataset):
    """
    CLEVRER Dataset.
    __getitem__ Returns a list o frames from a video and a question
    """
    def __init__(self, cfg, mode):
        """
        Constructs the Clevrer  loader with a given json file. The format of
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
        ], "Split '{}' not supported for Clevrer_des".format(mode)
        self.mode = mode
        self.cfg = cfg  

        self._num_retries = 10  

        logger.info("Constructing Clevrer_des {}...".format(mode))
        self._construct_loader()


    def _create_vocabs(self, path_to_file):
        """
        Creates the vocabularies used to tokenize questions and answers
        It uses only the train.json file
        """
        # ans_vocab = {} #Used for descriptive questions
        # ans_vocab[' counter '] = 0
        # possible_ans = ["0", "1", "2", "3", "4", "5", 
        #                 "yes", "no", "rubber", "metal",
        #                 "sphere", "cube", "cylinder",
        #                 "gray", "brown", "green", "red",
        #                 "blue", "purple", "yellow", "cyan"
        # ]
        # update_vocab(ans_vocab, possible_ans)

        # vocab = {}
        # vocab[' CLS '] = 0
        # vocab[' PAD '] = 1
        # vocab['|'] = 2
        # vocab[' counter '] = 3 #Has spaces in key => not a valid token

        des_q_lens = [] #Description question lens

        with open(path_to_file, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                questions = data[i]['questions']
                for q in questions:
                    split_q = string_to_token_list(q['question'])
                    # update_vocab(vocab, split_q)
                    if q['question_type'] == 'descriptive':
                        des_q_lens.append(len(split_q))
        # self.vocab = vocab
        # self.ans_vocab = ans_vocab
        self.vocab = {' CLS ': 0, ' PAD ': 1, '|': 2, ' counter ': 78, 'what': 3, 'is': 4, 'the': 5, 'shape': 6, 'of': 7, 'object': 8, 'to': 9, 'collide': 10, 'with': 11, 'purple': 12, '?': 13, 'color': 14, 'first': 15, 'gray': 16, 'sphere': 17, 'material': 18, 'how': 19, 'many': 20, 'collisions': 21, 'happen': 22, 'after': 23, 'cube': 24, 'enters': 25, 'scene': 26, 'objects': 27, 'enter': 28, 'are': 29, 'there': 30, 'before': 31, 'stationary': 32, 'rubber': 33, 'metal': 34, 'that': 35, 'moving': 36, 'cubes': 37, 'when': 38, 'blue': 39, 'exits': 40, 'any': 41, 'brown': 42, 'which': 43, 'following': 44, 'responsible': 45, 'for': 46, 'collision': 47, 'between': 48, 'and': 49, 'presence': 50, "'s": 51, 'entering': 52, 'colliding': 53, 'event': 54, 'will': 55, 'if': 56, 'cylinder': 57, 'removed': 58, 'collides': 59, 'without': 60, ',': 61, 'not': 62, 'red': 63, 'spheres': 64, 'exit': 65, 'cylinders': 66, 'video': 67, 'begins': 68, 'ends': 69, 'next': 70, 'last': 71, 'yellow': 72, 'cyan': 73, 'entrance': 74, 'green': 75, 'second': 76, 'exiting': 77}
        self.ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
        self.max_seq_len = max(des_q_lens)
    
    def _token_list_to_tensor(self, token_list, question_type):
        """
        Transforms a token list into a tensor with padding.
        """
        tensor = torch.ones(self.max_seq_len, dtype=torch.long) * self.vocab[' PAD ']
        for i in range(len(token_list)):
            tensor[i] = self.vocab[token_list[i]]
        return tensor

    
    def _constructs_questions_ans(self, data, video_path):
        """
        Creates a list of dicts with the information contained in data extracted from the json
        Uses the mode to determine if the answers will be returned
        """
        num_choices = 4 #Numer of multiple choice answers per question
        data_list = []
        l_index = -1

        questions = data['questions']
        for q in questions:
            if q['question_type'] != 'descriptive':
                continue
            l_index += 1
            data_list.append({})
            data_list[l_index]['video_path'] = video_path

            split_q = string_to_token_list(q['question'])
            
            data_list[l_index]['is_des'] = True
            data_list[l_index]['len'] = len(split_q)
            data_list[l_index]['question'] = self._token_list_to_tensor(split_q, 'descriptive')
            if self.mode != 'test':
                data_list[l_index]['ans'] = self.ans_vocab[q['answer']]
        return data_list

    def _construct_loader(self):
        """
        Construct the video loader.
        self._dataset:
            The main data structure: list of dicts: video_path 
                                    + is_des: bool flag indicating questiontype
                                    + question
                                    + ans
                                    + len
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
        train_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "train.json"
        )

        self._create_vocabs(train_file)

        #Main data structure
        self._dataset = []
        max_len = self.cfg.DATA.MAX_TRAIN_LEN if self.mode == 'train' else self.cfg.DATA.MAX_VAL_LEN

        with g_pathmgr.open(path_to_file, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                path = get_filepath(self.mode, int(data[i]['scene_index']))
                full_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                self._dataset += self._constructs_questions_ans(data[i], full_path)
                if not max_len is None:
                    if len(self._dataset) >= max_len:
                        break

        assert (
            len(self._dataset) > 0
        ), "Failed to load Clevrer from {}".format(
            path_to_file
        )
        logger.info(
            "Constructing Clevrer dataloader (size: {}) from {}".format(
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
                question_dict (dict): A dictionary with the question and answer (if not test)
                index (int): if the video provided by pytorch sampler can be
                    decoded, then return the index of the video. If not, return the
                    index of the video replacement that can be decoded.
        """
        # -1 indicates random sampling.
        temporal_sample_index = -1
        spatial_sample_index = -1
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        # Try to decode and sample a clip from a video. If the video cannot be
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
                max_spatial_scale=0
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
                
            if cfg.MODEL.ARCH == "slowfast":
                # T H W C -> T C H W. 
                frames = frames.permute(0, 3, 1, 2)
                # Perform resize
                transform_rs = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                frames_size = frames.size()
                resized_frames = torch.zeros(frames_size[0], frames_size[1], self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W)
                for i in range(frames_size[0]):
                    resized_frames[i] = transform_rs(frames[i])
                # T C H W -> C T H W. 
                resized_frames = resized_frames.permute(1, 0, 2, 3)
                resized_frames = utils.pack_pathway_output(self.cfg, resized_frames)

            else:
                # T H W C -> T C H W.
                frames = frames.permute(0, 3, 1, 2)
                # Perform resize
                transform_rs = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                frames_size = frames.size()
                resized_frames = torch.zeros(frames_size[0], frames_size[1], self.cfg.DATA.RESIZE_H, self.cfg.DATA.RESIZE_W)
                for i in range(frames_size[0]):
                    resized_frames[i] = transform_rs(frames[i])

            question_dict = copy.deepcopy(self._dataset[index])
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
                            + is_des
                            + Decoded question
                            + Decoded answer
        """
        if index < 0 or index >= len(self._dataset):
            logger.info("Video for index {} not found".format(index))
            return None
        video_dict = self._dataset[index]
        video_info = {}
        video_info['video_path'] =  video_dict['video_path']
        #Questions
        video_info['is_des'] = video_dict['is_des']
        video_info['len'] = video_dict['len']
        video_info['question'] = self.decode_question(video_dict['question'])    
        if self.mode == 'test':
            return video_info
        video_info['ans'] = self.decode_answer(video_dict['ans'])
        return video_info

    def get_vocab_len(self):
        return len(self.vocab.keys())
    
    def get_ans_vocab_len(self):
        return len(self.ans_vocab.keys())
    
    def get_video_path(self, index):
        return self._dataset[index]['video_path']

    def get_vocab(self):
        vocab_copy = copy.deepcopy(self.vocab)
        return vocab_copy

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


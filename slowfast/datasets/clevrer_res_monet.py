import torch 
import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
import json
import copy
import h5py

import slowfast.utils.logging as logging

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

#-_-__---_------------___-___--_____MC as DES questions-_-__---_------------___-___--_____ 

@DATASET_REGISTRY.register()
class Clevrer_res_monet(torch.utils.data.Dataset):
    """
    CLEVRERmac_des Dataset.
    __getitem__  Returns one descriptive question and ResNet or MONet features
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
        ], "Split '{}' not supported for Clevrerresnet".format(mode)
        self.mode = mode
        self.cfg = cfg  

        logger.info("Constructing Clevrerresnet {}...".format(mode))
        self._construct_loader()

        res_h5_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, '{}_res50_features.hdf5'.format(mode))
        self.res_f_h5 = h5py.File(res_h5_path, 'r')
        self.res_fts = self.res_f_h5['data']
        
        mon_h5_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, '{}_monet_features.hdf5'.format(mode))
        self.mon_f_h5 = h5py.File(mon_h5_path, 'r')
        self.monet_fts = self.mon_f_h5['data']

    def close(self):
        self.res_f_h5.close()
        self.mon_f_h5.close()

    def _create_vocabs(self, path_to_file):
        """
        Creates the vocabularies used to tokenize questions and answers
        It uses only the train.json file
        """
        self.vocab = {'[CLS]': 0, '[PAD]': 1, '[SEP]': 2, ' counter ': 78, 'what': 3, 'is': 4, 'the': 5, 'shape': 6, 'of': 7, 'object': 8, 'to': 9, 'collide': 10, 'with': 11, 'purple': 12, '?': 13, 'color': 14, 'first': 15, 'gray': 16, 'sphere': 17, 'material': 18, 'how': 19, 'many': 20, 'collisions': 21, 'happen': 22, 'after': 23, 'cube': 24, 'enters': 25, 'scene': 26, 'objects': 27, 'enter': 28, 'are': 29, 'there': 30, 'before': 31, 'stationary': 32, 'rubber': 33, 'metal': 34, 'that': 35, 'moving': 36, 'cubes': 37, 'when': 38, 'blue': 39, 'exits': 40, 'any': 41, 'brown': 42, 'which': 43, 'following': 44, 'responsible': 45, 'for': 46, 'collision': 47, 'between': 48, 'and': 49, 'presence': 50, "'s": 51, 'entering': 52, 'colliding': 53, 'event': 54, 'will': 55, 'if': 56, 'cylinder': 57, 'removed': 58, 'collides': 59, 'without': 60, ',': 61, 'not': 62, 'red': 63, 'spheres': 64, 'exit': 65, 'cylinders': 66, 'video': 67, 'begins': 68, 'ends': 69, 'next': 70, 'last': 71, 'yellow': 72, 'cyan': 73, 'entrance': 74, 'green': 75, 'second': 76, 'exiting': 77}
        self.ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
        self.max_seq_len = 45
    
    def _token_list_to_tensor(self, token_list):
        """
        Transforms a token list into a tensor with padding.
        Adds 
        """
        tensor = torch.ones(self.max_seq_len, dtype=torch.long) * self.vocab['[PAD]']
        for i in range(len(token_list)):
            tensor[i] = self.vocab[token_list[i]]
        return tensor

    
    def _constructs_questions_ans(self, data, video_path):
        """
        Creates a list of dicts with the information contained in data extracted from the json
        Uses the mode to determine if the answers will be returned
        """
        data_list = []
        l_index = -1

        questions = data['questions']
        for q in questions:
            if q['question_type'] == 'descriptive':
                l_index += 1
                data_list.append({})
                data_list[l_index]['video_path'] = video_path
                data_list[l_index]['question_type'] = q['question_type']

                split_q = string_to_token_list(q['question'])
                split_q = ['[CLS]'] + split_q + ['[SEP]']
                data_list[l_index]['len'] = len(split_q)
                attention_mask = torch.zeros(self.max_seq_len)
                for i in range(len(split_q)):
                    attention_mask[i] = 1.0
                data_list[l_index]['attention_mask'] = attention_mask
                data_list[l_index]['question'] = self._token_list_to_tensor(split_q)
                if self.mode != 'test':
                    data_list[l_index]['ans'] = self.ans_vocab[q['answer']]
            else: #Multiple choice questions
                choices = q['choices']
                for c in choices: #A question for each choice
                    l_index += 1
                    data_list.append({})
                    data_list[l_index]['video_path'] = video_path
                    data_list[l_index]['question_type'] = q['question_type']

                    split_q = string_to_token_list(q['question']) + ['[SEP]'] + string_to_token_list(c['choice'])
                    split_q = ['[CLS]'] + split_q + ['[SEP]']
                    data_list[l_index]['len'] = len(split_q)
                    attention_mask = torch.zeros(self.max_seq_len)
                    for i in range(len(split_q)):
                        attention_mask[i] = 1.0
                    data_list[l_index]['attention_mask'] = attention_mask
                    data_list[l_index]['question'] = self._token_list_to_tensor(split_q)
                    if self.mode != 'test':
                        trans_mc_ans = 'yes' if c['answer'] == 'correct' else 'no'
                        data_list[l_index]['ans'] = self.ans_vocab[trans_mc_ans]

        return data_list

    def _construct_loader(self):
        """
        Construct the video loader.
        self._dataset:
            The main data structure: list of dicts: video_path 
                                    + question_type
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

        with g_pathmgr.open(path_to_file, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                path = get_filepath(self.mode, int(data[i]['scene_index']))
                full_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                self._dataset += self._constructs_questions_ans(data[i], full_path)

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
    
    def get_ft_index(self, video_path):
        """
        Gets the index for the feature in the h5py feature file
        """
        if self.mode == "train":
            base_i = 0
        elif self.mode == "val":
            base_i = 10000
        elif self.mode == "test":
            base_i = 15000
        video_i = int(video_path[-9:-4])
        return video_i - base_i


    def __getitem__(self, index):
        """
        Args:
            index (int): the dataset index provided by the pytorch sampler.
        Returns:
        A dict containing:
                question_dict (dict): A dictionary with the questions and answers (if not test)
                index (int): if the video provided by pytorch sampler can be
                    decoded, then return the index of the video. If not, return the
                    index of the video replacement that can be decoded.
        """
        question_dict = copy.deepcopy(self._dataset[index])
        output_dict = {}
        ft_i = self.get_ft_index(self._dataset[index]['video_path'])
        output_dict['res_ft'] = {'res_ft':torch.from_numpy(self.res_fts[ft_i]), 'monet_ft':torch.from_numpy(self.monet_fts[ft_i])}
        output_dict['question_dict'] = question_dict
        output_dict['index'] = index
        # print("DATASET PRINT BEGIN ---")
        # print("Dataset entry = {}".format(self._dataset[index]))
        # print("Dataset index = {}".format(index))
        # print("Feature index index = {}".format(ft_i))
        # print("Dataset res ft sum = {}".format(torch.from_numpy(self.res_fts[ft_i]).sum()))
        # print("DATASET PRINT END---")
        return output_dict

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
                            + question_type
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
        video_info['question_type'] = video_dict['question_type']
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
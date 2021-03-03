import torch 
import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
import json
import copy

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

@DATASET_REGISTRY.register()
class Clevrertext(torch.utils.data.Dataset):
    """
    CLEVRERtext Dataset.
    __getitem__  Two questions with the anwers
    One descriptive question and one multiple choice
    Uses padding.
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
        ], "Split '{}' not supported for ClevrerText".format(mode)
        self.mode = mode
        self.cfg = cfg  

        logger.info("Constructing Clevrertext {}...".format(mode))
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
        vocab[' CLS '] = 0
        vocab[' PAD '] = 1
        vocab['|'] = 2
        vocab[' counter '] = 3 #Has spaces in key => not a valid token

        des_q_lens = [] #Description question lens
        mc_q_lens = [] #Multiple choice question lens

        with open(path_to_file, "r") as f:
            data = json.load(f)
            for i in range(len(data)):
                questions = data[i]['questions']
                for q in questions:
                    split_q = string_to_token_list(q['question'])
                    update_vocab(vocab, split_q)
                    if q['question_type'] == 'descriptive':
                        des_q_lens.append(len(split_q))
                    else:
                        choices = q['choices']
                        q_len = len(split_q)
                        for c in choices:
                            split_c = string_to_token_list(c['choice'])
                            update_vocab(vocab, split_c)
                            q_len += len(split_c) + 1 #Plus 1 because adds |
                        mc_q_lens.append(q_len)
                    
        self.vocab = vocab
        self.ans_vocab = ans_vocab

        self.max_des_len = max(des_q_lens)
        self.max_mc_len = max(mc_q_lens)
    
    def _token_list_to_tensor(self, token_list, question_type):
        """
        Transforms a token list into a tensor with padding 
        according to the question type
        """
        if question_type == 'descriptive':
            tensor = torch.ones(self.max_des_len, dtype=torch.long) * self.vocab[' PAD ']
        else:
            tensor = torch.ones(self.max_mc_len, dtype=torch.long) * self.vocab[' PAD ']
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
            l_index += 1
            data_list.append({})
            data_list[l_index]['video_path'] = video_path

            split_q = string_to_token_list(q['question'])
            if q['question_type'] == 'descriptive':
                data_list[l_index]['is_des'] = True
                data_list[l_index]['question'] = self._token_list_to_tensor(split_q, 'descriptive')
                if self.mode != 'test':
                    data_list[l_index]['ans'] = self.ans_vocab[q['answer']]
            else:
                data_list[l_index]['is_des'] = False
                choices = q['choices']
                choice_ans = torch.zeros(num_choices)
                c_index = 0
                for c in choices:
                    split_c = string_to_token_list(c['choice'])
                    split_q += ['|'] + split_c

                    if self.mode != 'test':
                        choice_ans[c_index] = 1 if c['answer'] == 'correct' else 0
                        c_index += 1

                data_list[l_index]['question'].append(self._token_list_to_tensor(split_q, 'mc'))
                data_list[l_index]['ans'].append(choice_ans)
        return data_list

    def _construct_loader(self):
        """
        Construct the video loader.
        self._dataset:
            The main data structure: list of dicts: video_path 
                                    + is_des: bool flag indicating questiontype
                                    + question
                                    + ans
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
    
    def __getitem__(self, index):
        """
        Given the dataset index, return question_dict, and video
        index
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
        output_dict['question_dict'] = question_dict
        output_dict['index'] = index
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
        video_info['question'] = self.decode_question(video_dict['question'])    
        if self.mode == 'test':
            return video_info
        if video_info['is_des']:
            video_info['ans'] = self.decode_answer(video_dict['ans'])
        else:
            video_info['ans'] = video_dict['ans']
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
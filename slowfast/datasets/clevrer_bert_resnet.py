import torch 
import os
import random
import torch.utils.data
from iopath.common.file_io import g_pathmgr
import json
import copy
from transformers import BertTokenizer
import torchvision.transforms as transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container

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

#Returns the token for a given index
def get_token_for_index(vocab, tgt_index):
    for token, index in vocab.items():
        if index == tgt_index:
            return token
    logger.info("Token for index {} not found".format(tgt_index))
    return "TOKEN NOT FOUND"

class Clevrerbert_resnet(torch.utils.data.Dataset):
    """
    Interprets multiple choice questions as a descriptive yes/no question
    Uses BERT and resnet features
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
        ], "Split '{}' not supported for Clevrerbert_resnet".format(mode)
        self.mode = mode
        self.cfg = cfg  
        self._num_retries = 10

        logger.info("Constructing Clevrerbert_resnet {}...".format(mode))
        self._construct_loader()

        h5_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, '{}_{}_features.hdf5'.format(mode, cfg.RESNET_SZ))
        self.f_h5 = h5py.File(h5_path, 'r')
        self.res_fts = self.f_h5['data']
        self.res_fts_index = self.f_h5['indexes']
    
    def close(self):
        self.f_h5.close()

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
                data_list[l_index]['question'] = self.bert_tokenizer(q['question'], add_special_tokens = True, 
                                                                     padding='max_length', max_length=self.q_max_len)
                #Convert to tensor
                data_list[l_index]['question']['input_ids'] = torch.tensor(data_list[l_index]['question']['input_ids'])
                data_list[l_index]['question']['token_type_ids'] = torch.tensor(data_list[l_index]['question']['token_type_ids'])
                data_list[l_index]['question']['attention_mask'] = torch.tensor(data_list[l_index]['question']['attention_mask'])

                if self.mode != 'test':
                    data_list[l_index]['ans'] = self.ans_vocab[q['answer']]
            else: #Multiple choice questions
                choices = q['choices']
                for c in choices: #A question for each choice
                    l_index += 1
                    data_list.append({})
                    data_list[l_index]['video_path'] = video_path
                    data_list[l_index]['question_type'] = q['question_type']
                    trans_mc_q = q['question'] + ' [SEP] ' + c['choice']
                    data_list[l_index]['question'] = self.bert_tokenizer(trans_mc_q , add_special_tokens = True, 
                                                                         padding='max_length', max_length=self.q_max_len)
                    if self.mode != 'test':
                        trans_mc_ans = 'yes' if c['answer'] == 'correct' else 'no'
                        data_list[l_index]['ans'] = self.ans_vocab[trans_mc_ans]
                    #Convert to tensor
                    data_list[l_index]['question']['input_ids'] = torch.tensor(data_list[l_index]['question']['input_ids'])
                    data_list[l_index]['question']['token_type_ids'] = torch.tensor(data_list[l_index]['question']['token_type_ids'])
                    data_list[l_index]['question']['attention_mask'] = torch.tensor(data_list[l_index]['question']['attention_mask'])

        return data_list

    def _construct_loader(self):
        """
        Construct the video loader.
        self._dataset:
            The main data structure: list of dicts: video_path 
                                    + question_type: indicates question type
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

        self.ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.q_max_len = 64
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
        Given the video index, return the list of frames, question_dict, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
        A dict containing:
                res_ft (tensor): features extracted by a pretrained resnet
                question_dict (dict): A dictionary with the question and answer (if not test)
                index (int): if the video provided by pytorch sampler can be
                    decoded, then return the index of the video. If not, return the
                    index of the video replacement that can be decoded.
        """
        

        question_dict = copy.deepcopy(self._dataset[index])
        output_dict = {}
        ft_i = self.get_ft_index(self._dataset[index]['video_path'])
        output_dict['res_ft'] = torch.from_numpy(self.res_fts[ft_i])
        output_dict['res_ft_index'] = torch.from_numpy(self.res_fts_index[ft_i])
        output_dict['question_dict'] = question_dict
        output_dict['index'] = index
        return output_dict


    def decode_question(self, ids):
        """
        Decodes a vector of indexes into a string of words
        Args:
            q_tensor (LongTensor): Encoded question
        Returns:
            decoded_q (string): Decoded question
        """
        return self.bert_tokenizer.decode(ids.input_ids)
    
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
        video_info['question'] = self.decode_question(video_dict['question'])    
        if self.mode == 'test':
            return video_info
        video_info['ans'] = self.decode_answer(video_dict['ans'])
        return video_info
    
    def get_ans_vocab_len(self):
        return len(self.ans_vocab.keys())
    
    def get_video_path(self, index):
        return self._dataset[index]['video_path']

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
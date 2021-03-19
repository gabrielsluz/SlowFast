import torch 
import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
import json
import copy
import slowfast.utils.logging as logging
from transformers import BertTokenizer

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

class Clevrerbert_des(torch.utils.data.Dataset):
    """
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
        ], "Split '{}' not supported for Clevrerbert_des".format(mode)
        self.mode = mode
        self.cfg = cfg  

        logger.info("Constructing Clevrerbert_des {}...".format(mode))
        self._construct_loader()

    
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
            data_list[l_index]['is_des'] = True
            data_list[l_index]['question'] = self.bert_tokenizer(q['question'], add_special_tokens = True, 
                                                                 padding='max_length', max_length=self.q_max_len)
            #Convert to tensor
            data_list[l_index]['question']['input_ids'] = torch.tensor(data_list[l_index]['question']['input_ids'])
            data_list[l_index]['question']['token_type_ids'] = torch.tensor(data_list[l_index]['question']['token_type_ids'])
            data_list[l_index]['question']['attention_mask'] = torch.tensor(data_list[l_index]['question']['attention_mask'])

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

        self.ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.q_max_len = 50
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
    
    def _get_video(self, video_path):
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
                    video_path,
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        video_path, e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, video_path, i_try
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
                        index, video_path, i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.cfg.MODEL.ARCH == "slowfast":
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
        return resized_frames

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
        output_dict['frames'] = self._get_video(self._dataset[index]['video_path'])
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
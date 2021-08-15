import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import csv

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def get_default_image_loader():
    return pil_loader

def get_video(video_path, frame_indices):
    """
    generate a video clip which is a list of selected frames
    :param video_path: path of video folder which contains video frames
    :param frame_indices: list of selected indices of frames. e.g. if index is 1, then selected frame's name is "img_1.png"
    :return: a list of selected frames which are PIL.Image or accimage form
    """
    image_reader = get_default_image_loader()
    video = []
    for image_index in frame_indices:
        image_name = 'img_' + str(image_index) + '.png'
        image_path = os.path.join(video_path, image_name)
        img = image_reader(image_path)
        img = img.resize((192, 192), Image.ANTIALIAS)
        video.append(img)
    return video

def get_clips(video_path, video_begin, video_end, label, view, sample_duration):
    """
    be used when validation set is generated. be used to divide a video interval into video clips
    :param video_path: validation data path
    :param video_begin: begin index of frames
    :param video_end: end index of frames
    :param label: 1(normal) / 0(anormal)
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :return: a list which contains  validation video clips
    """
    clips = []
    sample = {
        'video': video_path,
        'label': label,
        'subset': 'validation',
        'view': view,
    }
    interval_len = (video_end - video_begin + 1)
    num = int(interval_len / sample_duration)
    for i in range(num):
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_begin + sample_duration))
        clips.append(sample_)
        video_begin += sample_duration
    if interval_len % sample_duration != 0:
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_end+1)) + [video_end] * (sample_duration - (video_end - video_begin + 1))
        clips.append(sample_)
    return clips


def listdir(path):
    """
    show every files or folders under the path folder
    """
    for f in os.listdir(path):
            yield f


def make_dataset(root_path, subset, view, sample_duration, type=None):
    """
    :param root_path: root path of the dataset"
    :param subset: train / validation
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = normal / anormal ; during validation or test process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 'label': 0/1, 'subset': 'train'/'validation', 'view': 'front_depth' / 'front_IR' / 'top_depth' / 'top_IR', 'action': 'normal' / other anormal actions}
    """
    dataset = []
    if subset == 'train' and type == 'normal':
        # load normal training data
        train_folder_list = list(filter(lambda string: string.find('Tester') != -1, list(listdir(root_path))))

        for train_folder in train_folder_list:
            normal_video_list = list(filter(lambda string: string.split('_')[0] == 'normal', list(listdir(os.path.join(root_path, train_folder)))))

            for normal_video in normal_video_list:
                video_path = os.path.join(root_path, train_folder, normal_video, view)
                if not os.path.exists(video_path):
                    print(f"Video path doesn't exit: {video_path}")
                    continue

                n_frames = len(os.listdir(video_path))
                if n_frames <= 0:
                    print(f"Path {video_path} does't contain any data")
                    continue

                sample = {
                    'video': video_path,
                    'label': 1,
                    'subset': 'train',
                    'view': view,
                    'action': 'normal'
                }
                for i in range(0, n_frames, sample_duration):
                    sample_ = sample.copy()
                    sample_['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])
                    dataset.append(sample_)


    elif subset == 'train' and type == 'anormal':
        #load anormal training data
        train_folder_list = list(filter(lambda string: string.find('Tester') != -1, list(listdir(root_path))))

        for train_folder in train_folder_list:
            anormal_video_list = list(filter(lambda string: string.split('_')[0] != 'normal', list(listdir(os.path.join(root_path, train_folder)))))

            for anormal_video in anormal_video_list:
                video_path = os.path.join(root_path, train_folder, anormal_video, view)
                if not os.path.exists(video_path):
                    print(f"Video path doesn't exit: {video_path}")
                    continue
                n_frames = len(os.listdir(video_path))
                if n_frames <= 0:
                    print(f"Path {video_path} does't contain any data")
                    continue
                sample = {
                    'video': video_path,
                    'label': 0,
                    'subset': 'train',
                    'view': view,
                    'action': anormal_video,
                }

                for i in range(0, n_frames, sample_duration):
                    sample_ = sample.copy()
                    sample_['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])

                    dataset.append(sample_)

    elif subset == 'validation' and type == None:
        #load valiation data as well as thier labels
        csv_path = root_path + 'LABEL.csv'
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[-1] == '':
                    continue
                if row[0] != '':
                    which_val_path = os.path.join(root_path, row[0].strip())
                if row[1] != '':
                    video_path = os.path.join(which_val_path, row[1], view)
                video_begin = int(row[2])
                video_end = int(row[3])
                if row[4] == 'N':
                    label = 1
                elif row[4] == 'A':
                    label = 0
                clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
                dataset = dataset + clips
    else:
        print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    return dataset


class DAD(data.Dataset):
    """
    generate normal training/ anormal training/ validation dataset according to requirement
    """
    def __init__(self,
                 root_path,
                 subset,
                 view,
                 sample_duration=1,
                 type=None,
                 get_loader=get_video,
                 ):
        self.data = make_dataset(root_path, subset, view, sample_duration, type)
        self.sample_duration = sample_duration
        self.subset = subset
        self.loader = get_loader

    def __getitem__(self, index):
        if self.subset == 'train':
            video_path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']
            #print(frame_indices)
           
            clip = self.loader(video_path, frame_indices)
            trans1 = transforms.ToTensor()
            clip = [trans1(img).float() for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clip = torch.squeeze(clip,1)
                          #data with shape (channels, timesteps, height, width)
            return clip, index
        elif self.subset == 'validation':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']

            clip = self.loader(video_path, frame_indices)
            trans1 = transforms.ToTensor()
            clip = [trans1(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clip = torch.squeeze(clip,2)

            return clip, ground_truth

        else:
            print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    def __len__(self):
        return len(self.data)


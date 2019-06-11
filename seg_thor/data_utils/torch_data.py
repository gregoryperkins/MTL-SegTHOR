import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pdb


def get_cross_validation_paths(test_flag):
    """
    we use 4-fold cross-validation for testing
    we split data in order 0..9, 10..19, 20..29, 30..39
    Args:
        test_flag: range in [0, 3]
    """
    assert test_flag > -1 and test_flag < 4, 'the test flag is not in range !'
    train_files = []
    test_files = []
    test_nums = [i for i in range(test_flag * 10 + 1, test_flag * 10 + 11)]
    for i in range(1, 41):
        flag = False
        if i in test_nums:
            flag = True
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)
        if flag:
            test_files.append("Patient_" + i)
        else:
            train_files.append("Patient_" + i)
    return train_files, test_files


def get_global_alpha(patient_ids, data_path):
    """
    given the patient ids, we count the data number of each class and then
    we calculate the global condition probability of each class
    the condition probability: P(B|A) --> P[A, B] in the code
    the return value is the alpha
    """
    total_patients = []
    for patient_id in os.listdir(data_path):
        total_patients.append(patient_id)
    n_class = 4  # four organs including Esophagus, heart, trachea, aorta
    events = np.zeros((n_class, n_class))
    P = np.zeros((n_class, n_class))
    for patient_id in patient_ids:
        if not patient_id in total_patients:
            assert 'there is a not found patient, please check the data !'
        cur_label_path = os.path.join(data_path, patient_id)
        for img_file in os.listdir(cur_label_path):
            if 'label.npy' in img_file:
                img_path = os.path.join(cur_label_path, img_file)
                cur_label = np.load(img_path)
                event = np.zeros((n_class, 1))
                for i in range(n_class):
                    if np.sum(cur_label == i + 1) > 0:
                        event[i] = 1
                events += np.dot(event, event.transpose(1, 0))
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            P[i, j] = events[i, j] / float(events[i, i])
    alpha = np.copy(P)
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            alpha[i, i] -= alpha[j, i]
        alpha[i, i] += n_class
    return alpha


class THOR_Data(Dataset):
    '''
        parameters: 
            transform: the data augmentation methods
            path: the processed data path (training or testing)
        functions:
    '''
    def __init__(self, transform=None, path=None, file_list=None):
        data_listdirs = os.listdir(path)
        data_files = []
        label_files = []
        for cur_listdir in data_listdirs:
            if not cur_listdir in file_list:
                continue
            cur_file_dir = os.path.join(path, cur_listdir)
            for cur_file_image in os.listdir(cur_file_dir):
                if 'image.npy' in cur_file_image:
                    data_files.append(os.path.join(cur_file_dir, cur_file_image))
                    label_files.append(
                        os.path.join(
                            cur_file_dir,
                            cur_file_image.split('image.npy')[0] + 'label.npy'))
        self.data_files = []
        self.label_files = []
        shuffle_idx = [i for i in range(len(data_files))]
        np.random.shuffle(shuffle_idx)
        for i in shuffle_idx:
            self.data_files.append(data_files[i])
            self.label_files.append(label_files[i])
        self.transform = transform
        assert (len(self.data_files) == len(self.label_files))
        print('the data length is %d' % len(self.data_files))

    def __len__(self):
        L = len(self.data_files)
        return L

    def __getitem__(self, index):
        _img = np.load(self.data_files[index])
        _img = Image.fromarray(_img)
        _target = np.load(self.label_files[index])
        _target = Image.fromarray(np.uint8(_target))
        sample = {'image': _img, 'label': _target}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __str__(self):
        pass

    
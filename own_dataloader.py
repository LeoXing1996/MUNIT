from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import random
from PIL import Image
import os


def get_data_loader_monitor(root, file_list):
    # todo this function return a dataset which handle images separately
    pass


############################################
#            load images in pair           #
############################################

def get_all_data_loader_pair(conf):
    # this function is use to get pair data_loader
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    root = conf['data_root']
    rain_dir = conf['rain_dir']
    sun_dir = conf['sun_dir']
    train_loader = get_data_loader_monitor_pair(root, rain_dir, sun_dir, batch_size, True,
                                                new_size, height, width, num_workers)

    # TODO: add test loader code get the rest of the dataset
    test_loader = get_data_loader_monitor_pair(root, rain_dir, sun_dir, batch_size, False,
                                               new_size, height, width, num_workers)
    return train_loader, test_loader


def get_data_loader_monitor_pair(root, rain_dir, sun_dir, batch_size, train, new_size=None,
                                 height=256, width=256, num_worker=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImagePair(root, rain_dir, sun_dir, transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_worker)
    return loader


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImagePair(Dataset):
    def __init__(self, root, rain_dir, sun_dir, transform, loader=default_loader, num_ceiling=100):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.ceiling = num_ceiling

        self.r_dir = os.path.join(self.root, rain_dir)
        self.s_dir = os.path.join(self.root, sun_dir)
        self.r_pos = os.listdir(self.r_dir)
        self.s_pos = os.listdir(self.s_dir)
        assert self.r_pos == self.s_pos
        self.r_img_dict = {dir_: os.listdir(os.path.join(self.r_dir, dir_)) for dir_ in self.r_pos}
        self.s_img_dict = {dir_: os.listdir(os.path.join(self.s_dir, dir_)) for dir_ in self.s_pos}
        self.aligned_dataset = self.align_dataset(self.r_img_dict, self.s_img_dict)

    def align_dataset(self, r_img_dict, s_img_dict):
        assert r_img_dict.keys() == s_img_dict.keys()
        aligned_dataset = []
        for k in r_img_dict.keys():
            r_imgs = r_img_dict[k]
            s_imgs = s_img_dict[k]
            r_img_num = len(r_imgs)
            s_img_num = len(s_imgs)
            sample_num = min(min(r_img_num, s_img_num), self.ceiling)

            r_img_sample = random.sample(r_imgs, sample_num)
            s_img_sample = random.sample(s_imgs, sample_num)
            for i in range(sample_num):
                aligned_dataset.append((os.path.join(self.r_dir, k, r_img_sample[i]),
                                        os.path.join(self.s_dir, k, s_img_sample[i])))

        return aligned_dataset

    def __len__(self):
        return len(self.aligned_dataset)

    def __getitem__(self, index):
        img_pair = {'r_img': self.loader(self.aligned_dataset[index][0]),
                    's_img': self.loader(self.aligned_dataset[index][1])}
        if self.transform:
            img_pair['r_img'] = self.transform(img_pair['r_img'])
            img_pair['s_img'] = self.transform(img_pair['s_img'])

        return img_pair


if __name__ == '__main__':
    pass

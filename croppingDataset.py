import os
import torch.utils.data as data
import cv2
import math
import numpy as np
from augmentations import CropAugmentation

MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class TransformFunction(object):

    def __call__(self, sample,image_size):
        image, annotations = sample['image'], sample['annotations']

        scale = image_size / min(image.shape[:2])
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image,(int(w),int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = float(resized_image.shape[0]) / image.shape[0]
        scale_width = float(resized_image.shape[1]) / image.shape[1]

        transformed_bbox = {}
        transformed_bbox['xmin'] = []
        transformed_bbox['ymin'] = []
        transformed_bbox['xmax'] = []
        transformed_bbox['ymax'] = []
        MOS = []
        for annotation in annotations:
            transformed_bbox['xmin'].append(math.floor(float(annotation[1]) * scale_width))
            transformed_bbox['ymin'].append(math.floor(float(annotation[0]) * scale_height))
            transformed_bbox['xmax'].append(math.ceil(float(annotation[3]) * scale_width))
            transformed_bbox['ymax'].append(math.ceil(float(annotation[2]) * scale_height))

            MOS.append((float(annotation[-1]) - MOS_MEAN) / MOS_STD)

        resized_image = resized_image.transpose((2, 0, 1))
        return {'image': resized_image, 'bbox': transformed_bbox, 'MOS': MOS}

class GAICD(data.Dataset):

    def __init__(self, image_size=256, dataset_dir='dataset/GAIC/', set = 'train',
                 transform=TransformFunction(), augmentation=False):
        self.image_size = float(image_size)
        self.dataset_dir = dataset_dir
        self.set = set
        image_lists = os.listdir(self.dataset_dir + 'images/' + set)
        self._imgpath = list()
        self._annopath = list()
        for image in image_lists:
          self._imgpath.append(os.path.join(self.dataset_dir, 'images', set, image))
          self._annopath.append(os.path.join(self.dataset_dir, 'annotations', set, image[:-3]+"txt"))
        self.transform = transform
        if augmentation:
            self.augmentation = CropAugmentation()
        else:
            self.augmentation = None


    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath[idx])

        with open(self._annopath[idx],'r') as fid:
            annotations_txt = fid.readlines()

        annotations = list()
        for annotation in annotations_txt:
            annotation_split = annotation.split()
            if float(annotation_split[4]) != -2:
                annotations.append([float(annotation_split[0]),float(annotation_split[1]),float(annotation_split[2]),float(annotation_split[3]),float(annotation_split[4])])

        if self.augmentation:
            image, annotations = self.augmentation(image, annotations)

        # to rgb
        image = image[:, :, (2, 1, 0)]

        sample = {'image': image, 'annotations': annotations}

        if self.transform:
            sample = self.transform(sample,self.image_size)

        return sample

    def __len__(self):
        return len(self._imgpath)


class TransformFunctionTest(object):
    def __init__(self, aspect_ratio=3.0/4.0):
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, image_size, bboxes=None):

        scale = image_size / min(image.shape[:2])
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image,(int(w),int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = image.shape[0] / float(resized_image.shape[0])
        scale_width = image.shape[1] / float(resized_image.shape[1])

        if bboxes is None:
            bboxes = generate_bboxes_custom(resized_image, self.aspect_ratio)

        transformed_bboxes = []
        source_bboxes = []

        for bbox in bboxes:
            source_bboxes.append([round(bbox[0] * scale_width), round(bbox[1] * scale_height), \
                    round(bbox[2] * scale_width), round(bbox[3] * scale_height)])
            transformed_bboxes.append((0, bbox[0], bbox[1], bbox[2], bbox[3]))

        resized_image = resized_image.transpose((2, 0, 1))
        return resized_image, transformed_bboxes, source_bboxes


def tiling_range(start, end, step, width):
    if step == 0:
        if start + width <= end:
            yield start
        return
    while start + width <= end:
        yield start
        start += step
    if start + width != end + step:
        yield end - width


def generate_bboxes_custom(image, wh_ratio, n_samples=16):
    fragments = []

    h = float(image.shape[0])
    w = float(image.shape[1])

    for size in range(n_samples, n_samples // 2 - 1, -1):
        steps = n_samples - size + 1
        size = float(size)

        if wh_ratio > w / h:
            w_sample = size / float(n_samples) * w
            h_sample = w_sample / wh_ratio
            w_step = w * (1.0 / (size + 1.0))
            h_step = w_step * (h_sample / w_sample)
        else:
            h_sample = size / float(n_samples) * h
            w_sample = h_sample * wh_ratio
            h_step = h * (1.0 / (size + 1.0))
            w_step = h_step * (w_sample / h_sample)

        x_range = list(tiling_range(0.0, w, w_step, w_sample))
        y_range = list(tiling_range(0.0, h, h_step, h_sample))

        for x_start in x_range:
            for y_start in y_range:
                x_end = x_start + w_sample
                y_end = y_start + h_sample
                fragments.append([x_start, y_start, x_end, y_end])
                assert x_end <= w and y_end <= h
    
    return fragments


class setup_test_dataset(data.Dataset):

    def __init__(self, image_size=256.0, dataset_dir='testsetDir', transform=TransformFunctionTest()):
        self.image_size = float(image_size)
        self.dataset_dir = dataset_dir
        image_lists = os.listdir(self.dataset_dir)
        self._imgpath = list()
        self._annopath = list()
        for image in image_lists:
            self._imgpath.append(os.path.join(self.dataset_dir, image))
        self.transform = transform


    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath[idx])

        # to rgb
        image = image[:, :, (2, 1, 0)]

        if self.transform:
            resized_image,transformed_bbox,source_bboxes = self.transform(image,self.image_size)

        sample = {'imgpath': self._imgpath[idx], 'image': image, 'resized_image': resized_image, 'tbboxes': transformed_bbox, 'sourceboxes': source_bboxes}

        return sample

    def __len__(self):
        return len(self._imgpath)


class SingleImageDataset(data.Dataset):
    def __init__(self, image_path, transform=TransformFunctionTest(), image_size=256.0):
        self.image_size = float(image_size)
        self._imgpath = image_path
        self.transform = transform


    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath)

        # to rgb
        image = image[:, :, (2, 1, 0)]

        if self.transform:
            resized_image, transformed_bbox, source_bboxes = self.transform(image, self.image_size)

        sample = {
            'image': image,
            'resized_image': resized_image,
            'tbboxes': transformed_bbox,
            'sourceboxes': source_bboxes
        }

        return sample

    def __len__(self):
        return 1

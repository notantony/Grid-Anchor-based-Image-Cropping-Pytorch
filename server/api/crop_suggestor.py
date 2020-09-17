from croppingModel import build_crop_model
from croppingDataset import setup_test_dataset, TransformFunctionTest
import os
import torch
import cv2
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import time
import math
import heapq
import numpy as np
import itertools
from server.api.clustering import clusterize


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Grid anchor based image cropping With Pytorch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--net_path', default='pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth',
                    help='Directory for saving checkpoint models')
parser.add_argument('--aspect_ratio', default=4.0/3.0, type=float, help='Resulting image aspect ratio')

args = parser.parse_args()

# if not os.path.exists(args.output_dir):
    # os.makedirs(args.output_dir)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')

else:
    torch.set_default_tensor_type('torch.FloatTensor')

args.input_dir = "./output/test/"
dataset = setup_test_dataset(dataset_dir=args.input_dir, transform=TransformFunctionTest(aspect_ratio=args.aspect_ratio))


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def main():

    net = build_crop_model(scale='multi', alignsize=9, reddim=8, loadweight=True, model='mobilenetv2',downsample=4)
    net.load_state_dict(torch.load(args.net_path, map_location='cpu'))
    net.eval()

    if args.cuda:
        net = torch.nn.DataParallel(net,device_ids=[0])
        cudnn.benchmark = True
        net = net.cuda()


    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)


    best_results = []
    for id, sample in enumerate(data_loader):
        imgpath = sample['imgpath']
        image = sample['image']
        bboxes = sample['sourceboxes']
        resized_image = sample['resized_image']
        print(type(resized_image))
        print(resized_image.shape)
        # (3, 256, 448)
        tbboxes = sample['tbboxes']

        # if len(tbboxes['xmin']) == 0:
        #     continue

        roi = []

        # for idx in range(0,len(tbboxes['xmin'])):
            # roi.append((0, tbboxes['xmin'][idx],tbboxes['ymin'][idx],tbboxes['xmax'][idx],tbboxes['ymax'][idx]))
        roi = tbboxes
        print(roi)

        if args.cuda:
            resized_image = Variable(resized_image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            resized_image = Variable(resized_image)
            roi = Variable(torch.Tensor(roi))

        out = net(resized_image, roi)
    
        print(out)
        #HERE:::

    
        # print(len(out))

        id_out = range(len(out))
        image = image.cpu().numpy().squeeze(0)

        # width = 800
        # height = 600 # TODO: ??

        # Amount of best crops for clustering
        n_top = 50
        top = list(sorted(range(len(out)), key=lambda k: out[k]))[-n_top:]

        # Amount of crops after clustering
        n_results = 4
        k_best = clusterize([bboxes[i] for i in top], [out[i] for i in top], n_results)
        
        # score_boxes_all = sorted([(out[i], bboxes[i]) for i in top], key=lambda (k, _): k, reverse=True)
        score_boxes_final = sorted([(out[top[i]], bboxes[top[i]]) for i in k_best], key=lambda (k, _): k, reverse=True)

        best_results = []
        # for i, (score, box) in enumerate(score_boxes_all):
        for i, (score, box) in enumerate(score_boxes_final):
            box = [box[0].numpy()[0],box[1].numpy()[0],box[2].numpy()[0],box[3].numpy()[0]]
            
            # print('crop: {}'.format(box), 'score: {}'.format(float(score)))
            # crop = image[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
            
            crop = image[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
            resized_image = crop # cv2.resize(crop,(int(width),int(height))) # TODO: watch above
            imgname = imgpath[0].split('/')[-1]
            #USED
            # cv2.imwrite(args.output_dir + '/' + imgname[:-4] + '_' +str(i) + imgname[-4:],resized_image[:,:,(2, 1, 0)])
            # if len(best_results) < 50:
            #     heapq.heappush(best_results, (float(score), imgname[:-4], resized_image[:,:,(2, 1, 0)]))
            # else:
            #     heapq.heappushpop(best_results, (float(score), imgname[:-4], resized_image[:,:,(2, 1, 0)]))

        print([int(box[0]),int(box[2]),int(box[1]),int(box[3])])
        # print([[91, 0], [502, 0], [502, 547], [91, 547]])

        # crop = image[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
        # [[91, 0], [502, 0], [502, 547], [91, 547]]
        # imgname = imgpath[0].split('/')[-1]
        # cv2.imwrite(args.output_dir + '/' + imgname[:-4] + '_' +str(i) + imgname[-4:],resized_image[:,:,(2, 1, 0)])

        # for i in list(reversed(range(len(out))))[-4:]:

        #     top1_box = bboxes[id_out[i]]
        #     top1_box = [top1_box[0].numpy()[0],top1_box[1].numpy()[0],top1_box[2].numpy()[0],top1_box[3].numpy()[0]]
        #     top1_crop = image[int(top1_box[0]):int(top1_box[2]),int(top1_box[1]):int(top1_box[3])]
            
        #     resized_image = cv2.resize(top1_crop,(int(600),int(800)))
        #     imgname = imgpath[0].split('/')[-1]
        #     cv2.imwrite(args.output_dir + '/' + imgname[:-4] + '_' +str(i) + imgname[-4:],resized_image[:,:,(2, 1, 0)])

    # os.makedirs('./crops')
    for i, (score, _, img) in enumerate(sorted(best_results, key=lambda x: x[0], reverse=True)):
        cv2.imwrite('./crops/{}.jpg'.format(i), img)
        print("{} {}".format(i, score))


class CropSuggestorModel():
    def __init__(self, cuda=False, image_size=256.0):
        self.cuda = cuda
        self.image_size = image_size

        net = build_crop_model(scale='multi', alignsize=9, reddim=8, loadweight=True, model='mobilenetv2', downsample=4)
        net.load_state_dict(torch.load(args.net_path, map_location='cpu')) 
        net.eval()

        if self.cuda:
            net = torch.nn.DataParallel(net, device_ids=[0])
            net = net.cuda()
        
        self.net = net

    # core_bbox format: [x_min, y_min, x_max, y_max]
    def suggest(self, image, aspect_ratio=3.0/4.0, n_results=2, core_bbox=None):
 
        transform = TransformFunctionTest(aspect_ratio=aspect_ratio)
        resized_image, transformed_bbox, bboxes = transform(image, self.image_size)

        if core_bbox is not None:
            good = []
            for bbox in bboxes:
                if bbox[0] > core_bbox[0] or bbox[1] > core_bbox[1] \
                        or bbox[2] < core_bbox[2] or bbox[3] < core_bbox[3]:
                    good.append(False)
                else:
                    good.append(True)

            bboxes = list(itertools.compress(bboxes, good))
            transformed_bbox = list(itertools.compress(transformed_bbox, good))
            if len(bboxes) == 0:
                return []

        
        resized_image = torch.FloatTensor([resized_image])

        if self.cuda:
            resized_image = Variable(resized_image.cuda())
        else:
            resized_image = Variable(resized_image)

        roi = Variable(torch.Tensor(transformed_bbox))

        out = self.net(resized_image, roi)

        # Amount of best crops for clustering
        n_top = min(50, len(out))
        top = list(sorted(range(len(out)), key=lambda k: out[k]))[-n_top:]

        k_best = clusterize([bboxes[i] for i in top], [out[i] for i in top], n_results, dist=dist)

        score_boxes_final = sorted([(out[top[i]], bboxes[top[i]]) for i in k_best], key=lambda (k, _): k, reverse=True)

        return list(score_boxes_final)


# suggestor = CropSuggestorModel()

# best_results = []

# for i in range(30):
#     image = cv2.imread('/home/notantony/tmp/Grid-Anchor-based-Image-Cropping-Pytorch/output/tmp{}.png'.format(i))
#     image = image[:, :, (2, 1, 0)]
#     q = suggestor.suggest(image, core_bbox=[0, 0, 100, 100])
#     # print(q)

#     for score, box in q:    
#         cropped = image[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

#         if len(best_results) < 10:
#             heapq.heappush(best_results, (float(score), cropped[:, :, (2, 1, 0)]))
#         else:
#             heapq.heappushpop(best_results, (float(score), cropped[:, :, (2, 1, 0)]))

# for i, (score, img) in enumerate(sorted(best_results, key=lambda x: x[0], reverse=True)):
#     cv2.imwrite('./crops/{}.jpg'.format(i), img)
#     print("{} {}".format(i, score))



# main()
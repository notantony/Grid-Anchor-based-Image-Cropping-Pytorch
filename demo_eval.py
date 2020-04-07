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


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Grid anchor based image cropping With Pytorch')
parser.add_argument('--input_dir', required=True,
                    help='root directory path of testing images')
parser.add_argument('--output_dir', required=True,
                    help='root directory path of testing images')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--net_path', default='pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth',
                    help='Directory for saving checkpoint models')
parser.add_argument('--aspect_ratio', default=4.0/3.0, type=float, help='Resulting image aspect ratio')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')

else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset = setup_test_dataset(dataset_dir = args.input_dir, transform=TransformFunctionTest(aspect_ratio=args.aspect_ratio))


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def clusterize(crops, scores, n_results):
    n_crops = len(crops)
    centers = [((crop[3] + crop[1]) / 2, (crop[2] + crop[0]) / 2) for crop in crops]
    
    dists = []
    for i_center in range(n_crops):
        for j_center in range(i_center, n_crops):
            dists.append((dist(centers[i_center], centers[j_center]), (i_center, j_center)))

    parents = list(range(n_crops))
    cur_classes = n_crops
    
    def get_head(c):
        if parents[c] != c:
            parents[c] = get_head(parents[c])
        return parents[c]

    for _, (i, j) in sorted(dists, key=lambda (x, _): x):
        if cur_classes == n_results:
            break
        i_head = get_head(i)
        j_head = get_head(j)
        if i_head != j_head:
            cur_classes -= 1
            parents[j_head] = i_head
    
    classes = {}
    for i in range(n_crops):
        head = get_head(i)
        if head not in classes:
            classes[head] = []
        
        classes[head].append(i)

    return [max(members, key=lambda k: scores[k]) for members in classes.values()]


def main():

    net = build_crop_model(scale='multi', alignsize=9, reddim=8, loadweight=True, model='mobilenetv2',downsample=4)
    net.load_state_dict(torch.load(args.net_path, map_location='cpu'))
    net.eval()

    if args.cuda:
        net = torch.nn.DataParallel(net,device_ids=[0])
        cudnn.benchmark = True
        net = net.cuda()


    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    for id, sample in enumerate(data_loader):
        imgpath = sample['imgpath']
        image = sample['image']
        bboxes = sample['sourceboxes']
        resized_image = sample['resized_image']
        tbboxes = sample['tbboxes']

        if len(tbboxes['xmin'])==0:
            continue

        roi = []

        for idx in range(0,len(tbboxes['xmin'])):
            roi.append((0, tbboxes['xmin'][idx],tbboxes['ymin'][idx],tbboxes['xmax'][idx],tbboxes['ymax'][idx]))

        if args.cuda:
            resized_image = Variable(resized_image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            resized_image = Variable(resized_image)
            roi = Variable(torch.Tensor(roi))


        # t0 = time.time()
        # for _ in range(0,100):
        #     out = net(resized_image,roi)
        # t1 = time.time()
        # print('timer: %.4f sec.' % (t1 - t0))

        out = net(resized_image,roi)


        print(len(out))

        id_out = range(len(out))
        image = image.cpu().numpy().squeeze(0)

        width = 800
        height = 600

        # Amount of best crops for clustering
        n_top = 50
        top = list(sorted(range(len(out)), key=lambda k: out[k]))[-n_top:]

        # Amount of crops after clustering
        n_results = 4
        k_best = clusterize([bboxes[i] for i in top], [out[i] for i in top], n_results)
        
        score_boxes_all = sorted([(out[i], bboxes[i]) for i in top], key=lambda (k, _): k, reverse=True)
        score_boxes_final = sorted([(out[top[i]], bboxes[top[i]]) for i in k_best], key=lambda (k, _): k, reverse=True)

        # for i, (score, box) in enumerate(score_boxes_final):
        for i, (score, box) in enumerate(score_boxes_all):
            box = [box[0].numpy()[0],box[1].numpy()[0],box[2].numpy()[0],box[3].numpy()[0]]
            
            print('crop: {}'.format(box), 'score: {}'.format(float(score)))
            crop = image[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
            
            crop = image[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
            resized_image = cv2.resize(crop,(int(width),int(height)))
            imgname = imgpath[0].split('/')[-1]
            cv2.imwrite(args.output_dir + '/' + imgname[:-4] + '_' +str(i) + imgname[-4:],resized_image[:,:,(2, 1, 0)])

        print([int(box[0]),int(box[2]),int(box[1]),int(box[3])])
        print([int(box[0]),int(box[2]),int(box[1]),int(box[3])])
        print([[91, 0], [502, 0], [502, 547], [91, 547]])

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


if __name__ == '__main__':
    main()

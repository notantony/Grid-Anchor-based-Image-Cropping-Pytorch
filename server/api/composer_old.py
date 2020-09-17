import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import data

from server.api.utils import tiling_range, n_dim_iter


class ImageComposer():
    def __init__(self):
        pass

    @staticmethod
    def paste_all(bg, obj_positons):
        bg = bg.copy()
        for img, position in obj_positons:
            x1, y1 = position
            y2 = y1 + img.shape[0]
            x2 = x1 + img.shape[1]

            alpha_s = img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                bg[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])
        
        return bg



    @staticmethod
    def generate_compositions(bg_dims, dims, complexity):
        w, h = bg_dims

        compositions = []
        for dim in dims:
            (w_sample, h_sample) = dim
            w_step = (w - w_sample) / float(complexity) # TODO: sample size
            h_step = (h - h_sample) / float(complexity)
            x_range = list(tiling_range(0.0, w, w_step, w_sample, dtype=int))
            y_range = list(tiling_range(0.0, h, h_step, h_sample, dtype=int))
            compositions.append(n_dim_iter([x_range, y_range]))

        return compositions

    def evaluate(self, image):
        pass


    def compose(self, bg_image_path, image_paths, complexity=2): # TODO: image doesn't exist: imgread -> None
        bg_image = cv2.imread(bg_image_path)
        # x-y-like
        bg_dims = bg_image.shape[1], bg_image.shape[0]
        dims = []
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            dims.append((image.shape[1], image.shape[0]))
            images.append(image)
            
        compostitions = self.generate_compositions(bg_dims, dims, complexity)
        
        result = 0
        for coord_list in n_dim_iter(compostitions):
            image = self.paste_all(bg_image, zip(images, coord_list))
            result += 1
            cv2.imwrite("./output/{}.jpg".format(result), image)

composer = ImageComposer()
# composer.compose("./input/bg_beach.jpg", ["./input/dog_obj.png", "./input/man_obj.png"])
# composer.compose("./input/bg_savannah.jpg", ["./input/lion_obj.png", "./input/rhino_obj.png"])
# composer.compose("./input/bg_beach.jpg", ["./input/boat_obj.png", "./input/parrot_obj.png"])
composer.compose("./input/bg_castle2.jpg", ["./input/knight_obj.png", "./input/knight2_obj.png"])



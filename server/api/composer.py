# # -*- coding: utf-8 -*-
import math
import random
import numpy as np
import heapq

from io import BytesIO
from PIL import Image, ImageOps
from server.api.crop_suggestor import CropSuggestorModel
from server.api.utils import compress_bg, read_image, nd_deserialize, np_isground, tiling_range, n_dim_iter, \
        np_bg_goodness, np_iswater
from server.api.graphics import paste_obj


class ObjImg():
    def __init__(self, image_data, obj_class=None, obj_tags=None, size_ratio=1.0, pad=5, well_cropped=False):
        image = Image.open(BytesIO(image_data)).convert('RGBA')
        bbox = image.getbbox()
        image = image.crop(bbox)

        if pad != 0:
            image = ImageOps.expand(image, pad)

        self.image = image
        self.size_ratio = float(size_ratio)
        self.tags = set(obj_tags.split(";"))
        self.well_cropped = well_cropped

        self.expected_size = None
        self.tiles_size = None

    def is_grounded(self):
        return 'grounded' in self.tags

    def is_air(self):
        return 'air' in self.tags

    def is_bg_obj(self):
        return 'bg_obj' in self.tags

    def get_xy_ratio(self):
        return float(self.image.size[0]) / self.image.size[1]


    def normalize(self, tile_size, standart_size):
        if standart_size is None:
            self.expected_size = (int(self.size_ratio * self.image.size[0]), int(self.size_ratio * self.image.size[1]))
        else:
            self.expected_size = (int(self.size_ratio * standart_size * self.get_xy_ratio()), \
                int(self.size_ratio * standart_size))

        self.tiles_size = (int(math.ceil(float(self.expected_size[0]) / tile_size[0])), \
                int(math.ceil(float(self.expected_size[1]) / tile_size[1])))


class ImageComposer():
    def __init__(self, complexity=6):
        self.complexity = complexity

    def compose(self, bg_image, objs, bg_cm, bg_dm, search='complete', standart_size=None, tries=None, allow_not_all=True):
        if search not in ['complete', 'random']:
            raise ValueError("Unexpected `search` parameter value: {}".format(search))
        # if search == 'complete' and tries is not None:
        #     raise ValueError("Parameter `tries` cannot be used with parameter `search` == 'complete'")

        if not isinstance(standart_size, (int, float)) and standart_size not in [None, 'depth', 'compostition']:
            raise ValueError("Unexpected `standart_size` parameter value: {}".format(search))

        if bg_image.size[0] > bg_image.size[1]:
            steps_x = self.complexity
            steps_y = int(float(steps_x) * bg_image.size[1] / bg_image.size[0])
        else:
            steps_y = self.complexity
            steps_x = int(float(steps_y) * bg_image.size[0] / bg_image.size[1])

        steps = (steps_x, steps_y)

        # [x][y]-like PIL upper-left corner format
        bg_cm = np.flip(np.rot90(bg_cm, k=-1), axis=1)
        bg_dm = np.flip(np.rot90(bg_dm, k=-1), axis=1)
        cm_tiles = compress_bg(bg_cm, steps)
        dm_tiles = compress_bg(bg_dm, steps)

        if len(objs) == 0: # TODO
            return []

        step_pxs = (float(bg_image.size[0]) / steps[0], float(bg_image.size[1]) / steps[1])

        isground_tiles = np_isground(cm_tiles)
        bg_goodness_tiles = np_bg_goodness(cm_tiles)
        dists_mean = dm_tiles[isground_tiles != 0].mean()

        dist_std = 0.25 * bg_image.size[1] #* max(objs, key=lambda x: x.size_ratio).size_ratio

        for obj in objs:
            obj.normalize(step_pxs, dist_std)

        compositions = []
        for obj in objs:
            x_range = list(tiling_range(0, steps_x, obj.tiles_size[0], 1))
            y_range = list(tiling_range(0, steps_y, obj.tiles_size[1], 1))
            compositions.append(n_dim_iter([x_range, y_range]))

        good = []
        for coord_list in n_dim_iter(compositions):
            fail = False
            bg_cur = bg_goodness_tiles.copy()
            for obj, (x_pos, y_pos) in zip(objs, coord_list):
                bottom_xy = (int(x_pos + math.ceil(float(obj.tiles_size[0]) / 2) - 1), y_pos + int(obj.tiles_size[1]) - 1)
                if obj.is_grounded() and not isground_tiles[bottom_xy]:
                    fail = True
                    break
                if obj.is_air() and isground_tiles[bottom_xy]:
                    fail = True
                    break
                if not obj.is_bg_obj():
                    bg_cur[x_pos:x_pos + obj.tiles_size[0], y_pos:y_pos + obj.tiles_size[1]] -= 1
                
            if np.min(bg_cur) < -1:
                fail = True
            
            if not fail:
                core_bbox = None
                if bg_cur[bg_cur == -1].any():
                    core_bbox = [float("inf"), float("inf"), float("-inf"), float("-inf")]
                for i in range(steps_x):
                    for j in range(steps_y):
                        if bg_cur[i][j] == -1:
                            core_bbox[0] = min(core_bbox[0], int(i * step_pxs[0]))
                            core_bbox[1] = min(core_bbox[1], int(j * step_pxs[1]))
                            core_bbox[2] = max(core_bbox[2], int((i + 1) * step_pxs[0]))
                            core_bbox[3] = max(core_bbox[3], int((j + 1) * step_pxs[1]))

                good.append((coord_list, core_bbox))


        if tries is not None:
            random.shuffle(good)
            good = good[:tries]

        path_bboxes = []
        for i, (coord_list, core_bbox) in enumerate(good):
            bg_cur = bg_image
            for obj, (x_pos, y_pos) in zip(objs, coord_list):
                bottom_xy = (int(x_pos + math.ceil(obj.tiles_size[0] / 2) - 1), y_pos + int(obj.tiles_size[1]) - 1)
                
                if obj.is_air():
                    dist_coef = 1
                else:
                    dist_coef = dists_mean / dm_tiles[bottom_xy]
                new_size = (int(obj.expected_size[0] * dist_coef), int(obj.expected_size[1] * dist_coef))

                x_coord = int(step_pxs[0] * (x_pos + float(obj.tiles_size[0]) / 2))
                y_coord = int(step_pxs[1] * (y_pos + obj.tiles_size[1])) - 10
                try:
                    # (not obj.well_cropped)
                    bg_cur = paste_obj(bg_cur, obj.image, new_size, lower_middle=(x_coord, y_coord), smooth=True)
                except ValueError:
                    pass
            path = "./output/tmp{}.png".format(i)
            # bg_cur.save(path)
            cv2.imwrite(path, bg_cur)
            path_bboxes.append((path, core_bbox))
        suggest(path_bboxes)



import json
import base64

def loadnp(filepath, name='data'):
    with open(filepath, 'r') as file:
        json_obj = json.load(file)
    
    return nd_deserialize(json_obj, name)

cat_data = read_image('./input/obj/cat.png')
cat_obj = ObjImg(cat_data, "cat", "object;grounded", size_ratio=0.4, well_cropped=False)

dog_data = read_image('./input/obj/dog.png')
dog_obj = ObjImg(dog_data, "dog", "object;grounded", size_ratio=0.75, well_cropped=True)

person_data = read_image('./input/obj/man.png')
person_obj = ObjImg(person_data, "person", "object;grounded", size_ratio=1.8)

rhino_data = read_image('./input/obj/rhino.png')
rhino_obj = ObjImg(rhino_data, "rhino", "object;grounded", size_ratio=1.5)

zebra_data = read_image('./input/obj/zebra.png')
zebra_obj = ObjImg(zebra_data, "zebra", "object;grounded", size_ratio=1.5, well_cropped=True)

lion_data = read_image('./input/obj/lion.png')
lion_obj = ObjImg(lion_data, "zebra", "object;grounded", size_ratio=1.5)

giraffe_data = read_image('./input/obj/giraffe.png')
giraffe_obj = ObjImg(giraffe_data, "giraffe", "object;grounded", size_ratio=3.0)

person_data = read_image('./input/obj/man.png')
person_obj = ObjImg(person_data, "person", "object;grounded", size_ratio=1.8)

pirate_data = read_image('./input/obj/pirate.png')
pirate_obj = ObjImg(pirate_data, "pirate", "object;grounded", size_ratio=1.8)

boat_data = read_image('./input/obj/boat.png')
boat_obj = ObjImg(boat_data, "boat", "object;water", size_ratio=1.0, well_cropped=True)

parrot_data = read_image('./input/obj/parrot.png')
parrot_obj = ObjImg(parrot_data, "parrot", "object;air", size_ratio=1.0, well_cropped=True)

field_cm = loadnp('./input/bg/field_cm.json', 'colormap')
field_dm = loadnp('./input/bg/field_dm.json', 'depthmap')
field_bg = Image.open("./input/bg/field.jpg")

africa_cm = loadnp('./input/bg/africa_cm.json', 'colormap')
africa_dm = loadnp('./input/bg/africa_dm.json', 'depthmap')
africa_bg = Image.open("./input/bg/africa.jpg")

beach_cm = loadnp('./input/bg/beach_cm.json', 'colormap')
beach_dm = loadnp('./input/bg/beach_dm.json', 'depthmap')
beach_bg = Image.open("./input/bg/beach.jpg")

autumn_cm = loadnp('./input/bg/autumn_cm.json', 'colormap')
autumn_dm = loadnp('./input/bg/autumn_dm.json', 'depthmap')
autumn_bg = Image.open("./input/bg/autumn.jpg")


composer = ImageComposer()
# composer.compose(field_bg, [person_obj], field_cm, field_dm)
# composer.compose(field_bg, [cat_obj, person_obj], field_cm, field_dm)



import cv2

def suggest(path_bboxes, amount=10):
    suggestor = CropSuggestorModel()

    best_results = []

    for path, core_bbox in path_bboxes:
        image = cv2.imread(path)
        # image = cv2.imread('/home/notantony/tmp/Grid-Anchor-based-Image-Cropping-Pytorch/output/tmp{}.png'.format(i))
        image = image[:, :, (2, 1, 0)]
        q = suggestor.suggest(image, core_bbox=core_bbox, n_results=1)
        # print(q)

        for score, box in q:    
            cropped = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            if len(best_results) < amount:
                heapq.heappush(best_results, (float(score), path, cropped[:, :, (2, 1, 0)]))
            else:
                heapq.heappushpop(best_results, (float(score), path, cropped[:, :, (2, 1, 0)]))

    for i, (score, orig_path, img) in enumerate(sorted(best_results, key=lambda x: x[0], reverse=True)):
        cv2.imwrite('./crops/{}.jpg'.format(i), img)
        print("{} {} {}".format(i, score, orig_path))



# composer.compose(africa_bg, [rhino_obj], africa_cm, africa_dm)
# composer.compose(africa_bg, [zebra_obj], africa_cm, africa_dm)
# composer.compose(africa_bg, [lion_obj, giraffe_obj], africa_cm, africa_dm)
# composer.compose(africa_bg, [lion_obj], africa_cm, africa_dm)
composer.compose(field_bg, [cat_obj, dog_obj], field_cm, field_dm)

# composer.compose(beach_bg, [parrot_obj, pirate_obj], beach_cm, beach_dm)
# composer.compose(autumn_bg, [person_obj], autumn_cm, autumn_dm)
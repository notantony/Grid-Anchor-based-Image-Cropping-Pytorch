import base64
import numpy as np
import csv

from ast import literal_eval 


def tiling_range(start, end, width, step):
    if step == 0:
        if start + width <= end:
            yield start
        return
    while start + width <= end:
        yield start
        start += step


def n_dim_iter(iterators):
    iterators = [list(iterator) for iterator in iterators]
    for iterator in iterators:
        if len(iterator) == 0:
            return

    positions = [0] * len(iterators)
    cur = len(positions) - 1
    while cur >= 0:
        yield [iterator[position] for iterator, position in zip(iterators, positions)]

        positions[cur] += 1
        if positions[cur] == len(iterators[cur]):
            while positions[cur] == len(iterators[cur]):
                positions[cur] = 0
                cur -= 1
                if cur < 0:
                    return
                else:
                    positions[cur] += 1

            cur = len(positions) - 1


def read_classes(filepath):
    with open(filepath, mode='r') as infile:
        reader = csv.reader(infile)
        header = (next(reader))
        return dict((int(rows[0]), dict((header[i], rows[i]) for i in range(len(header)))) for rows in reader)


classes_table = read_classes("./covers_ade20k.csv")
# from PIL import Image

# def merge(bg_image, objects):
#     for obj_image, obj_position in objects:
#         bg_image.paste(obj_image, obj_position, obj_image)
#     return bg_image

# bg = Image.open("./input/containers.jpg")
# obj1 = Image.open("./sample_data/cat_obj.png")
# obj2 = Image.open("./sample_data/dog_obj.png")

# result = merge(bg, [(obj1, (0, 0)), (obj2, (100, 100))])
# result.save("./output/result.jpeg")

def compress_bg(orig, steps):
    step_x = float(orig.shape[0]) / steps[0]
    step_y = float(orig.shape[1]) / steps[1]
    tiles = np.zeros(steps)
    for i in range(steps[0]):
        for j in range(steps[1]):
            values, counts = np.unique(orig[int(i * step_x):int((i + 1) * step_x), int(j * step_y):int((j + 1) * step_y)], \
                    return_counts=True)
            ind = np.argmax(counts)
            tiles[i][j] = values[ind]
    return tiles


def read_image(image_path):
    with open(image_path, "rb") as image_fp:
        image_data = image_fp.read()
    return image_data


def nd_deserialize(json_obj, data_field="data"):
    data = np.frombuffer(base64.b64decode(json_obj[data_field]), dtype=json_obj["dtype"])
    data = data.reshape(literal_eval(json_obj["shape"]))
    return data

np_isground = np.vectorize(lambda x: classes_table[x + 1]['Position'] == 'ground')

np_iswater = np.vectorize(lambda x: classes_table[x + 1]['Position'] == 'water')

def bg_goodness(x):
    table_goodness = classes_table[x + 1]['Bg_goodness']
    return float(table_goodness) if table_goodness != "" else 0.0

np_bg_goodness = np.vectorize(bg_goodness)


x = 5
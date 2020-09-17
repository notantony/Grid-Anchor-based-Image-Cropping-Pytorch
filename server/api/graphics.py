import io
import base64
from PIL import Image
import numpy as np
import cv2
import math


def str64_to_cv2(str64):
    imgdata = base64.b64decode(str64)
    img = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)


def to_cv2(image):
    if isinstance(image, np.ndarray):
        return image
    if image.mode == 'RGB':
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if image.mode == 'RGBA':
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)


def rgba_split(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR), \
            np.array(pil_image)[:, :, 3]


def cv2_paste(obj, bg, mask, upper_left, inplace=False):
    (x1, y1) = upper_left
    y2 = y1 + obj.shape[0]
    x2 = x1 + obj.shape[1]

    alpha_s = mask / 255.0
    alpha_l = 1.0 - alpha_s

    if not inplace:
        bg = bg.copy()

    if len(bg.shape) == 3: # RGB-like
        for c in range(0, 3):
            bg[y1:y2, x1:x2, c] = (alpha_s * obj[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])
    elif len(bg.shape) == 2: # Single channel
        bg[y1:y2, x1:x2] = (alpha_s * obj[:, :] + alpha_l * bg[y1:y2, x1:x2])

    return bg


def blend(src_image, dst_image, upper_left, blending_type="normal", smooth=True):
    src, src_mask = rgba_split(src_image)
    dst = to_cv2(dst_image)

    center = (upper_left[0] + src.shape[1] / 2, upper_left[1] + src.shape[0] / 2)

    # Clone seamlessly.
    if blending_type == "mixed":
        option = cv2.MIXED_CLONE 
    elif blending_type == "normal":
        option = cv2.NORMAL_CLONE
    elif blending_type == "monochrome":
        option = cv2.MONOCHROME_TRANSFER
    else:   
        raise ValueError("Unexpected `blending_type` parameter value: {}".format(blending_type))
    # cv2.NORMAL_CLONE -- without bg
    # cv2.MIXED_CLONE -- with bg

    # clone_mask = 255 * np.ones(src_mask.shape[:2], src_mask.dtype)
    if smooth:
        dst = cv2.seamlessClone(src, dst, src_mask.copy(), center, option)
        
        kernel = np.ones((3, 3), np.uint8) 
        src_mask = cv2.erode(src_mask, kernel)
        src_mask = cv2.GaussianBlur(src_mask, (3, 3), 1.0)

    dst = cv2_paste(src, dst, src_mask, upper_left, inplace=True)

    return dst


def paste_obj(bg_image, obj_image, new_size=None, upper_left=None, lower_middle=None, smooth=False):
    if (upper_left is None) == (lower_middle is None):
        raise ValueError("One of arguments `upper_left` and `lower_middle` shouldn't be None")

    bg_image = bg_image.copy()
    if new_size is not None:
        obj_image = obj_image.resize(new_size)

    if upper_left is None:
        upper_left = (int(lower_middle[0] - float(obj_image.size[0]) / 2), \
                int(lower_middle[1] - obj_image.size[1]))

    return blend(obj_image, bg_image, upper_left, smooth=smooth)


# Source: https://github.com/PPPW/poisson-image-editing
import scipy.sparse
from scipy.sparse.linalg import spsolve

def laplacian_matrix(n, m):   
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def blend_alt(src_image, dst_image, upper_left):
    src, src_mask = rgba_split(src_image)
    dst = to_cv2(dst_image)
    src = cv2_paste(src, np.zeros(dst.shape, dst.dtype), src_mask, upper_left=upper_left)
    mask = cv2_paste(src_mask, np.zeros(dst.shape[:2], dst.dtype), src_mask, upper_left=upper_left)

    print(src.shape)

    y_max, x_max = dst.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min

    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()

    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(src.shape[2]):
        source_flat = src[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = dst[y_min:y_max, x_min:x_max, channel].flatten()        

        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
        
        x = spsolve(mat_A, mat_b)    
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        
        dst[y_min:y_max, x_min:x_max, channel] = x

    return dst


# obj_img = Image.open("./input/obj/cat.png")
# bg_img = Image.open("./input/bg/field.jpg")

# img = blend(obj_img, bg_img, (0, 0), blending_type="mixed")

# cv2.imwrite('./output/test/tmp.jpg', img)
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import cv2


def affine_elastic_transform(image, mask=None, alpha=100, sigma=11,
                             alpha_affine=40, random_state=None):
    image = np.array(image)
    if mask is not None:
        mask = np.array(mask)
        assert image.shape == mask.shape

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    maskB = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    image_elastic = Image.fromarray(imageC.astype('uint8'))

    maskC = map_coordinates(maskB, indices, order=1, mode='constant').reshape(shape)
    mask_elastic = Image.fromarray(maskC.astype('uint8'))
    if mask is not None:
        return image_elastic, mask_elastic

    return image_elastic


def elastic_transform(image, mask=None, alpha=100, sigma=11, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    image = np.array(image)

    if mask is not None:
        mask = np.array(mask)
        assert image.shape == mask.shape
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    image_elastic = map_coordinates(image, indices, order=1).reshape(shape)
    image_elastic = Image.fromarray(image_elastic.astype('uint8'))

    if mask is not None:
        mask_elastic = map_coordinates(mask, indices, order=1).reshape(shape)
        mask_elastic = Image.fromarray(mask_elastic.astype('uint8'))
        return image_elastic, mask_elastic

    return image_elastic


if __name__ == '__main__':
    img_ori = Image.open('/home/gy/ultrasound_dataset/T_BUSIS/test_gray/4B_60.bmp')
    img_elastic = affine_elastic_transform(img_ori, alpha=20, sigma=11)
    img_elastic.show()

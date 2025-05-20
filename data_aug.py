import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

sometimes = lambda aug: iaa.Sometimes(0.7, aug)

seq = iaa.Sequential([
    iaa.Flipud(0.5),
    sometimes(
        iaa.Sequential([
            iaa.Affine(scale=(0.8, 1.2), rotate=(-30, 30)),
            iaa.CLAHE(clip_limit=4.0),
        ])
    )
])

def data_aug(imgs, masks):
    masks = SegmentationMapsOnImage(masks, shape=imgs.shape)
    imgs_aug, masks_aug = seq(image=imgs, segmentation_maps=masks)
    return imgs_aug.copy(), masks_aug.get_arr().copy()

    
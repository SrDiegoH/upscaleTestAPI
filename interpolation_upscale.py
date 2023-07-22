from enum import Enum

import cv2

from .blur import apply_blur
from .denoise import apply_denoise

class InterpolationType(Enum):
    NEAREST_NEIGHBOR = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    NEAREST_NEIGHBOR_EXACT = cv2.INTER_NEAREST_EXACT
    BILINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    LANCZOS4 = cv2.INTER_LANCZOS4
    BITS2 = cv2.INTER_BITS2

def apply_upscale(interpolation_type, image, scale_factor=4, denoise_intensity=0, blur_intensity=0, blur_type=SIMPLE_BLUR):
    (image_height, image_width) = image.shape[:2]

    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)

    new_image = cv2.resize(image, (new_width, new_height), interpolation=InterpolationType[interpolation_type].value)

    denoised_image = apply_denoise(new_image, denoise_intensity)

    blurred_image = apply_blur(blur_type, denoised_image, blur_intensity)

    return blurred_image
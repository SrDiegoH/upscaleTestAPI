from enum import Enum

import cv2

class BlurType:
    def blur(self, blut_type, image, intensity):
        return getattr(self, f'_{str(blut_type)}', lambda image, intensity: image)(image, intensity)

    def _GAUSSIAN_BLUR(self, image, intensity):
        return cv2.GaussianBlur(image, (intensity, intensity), 0)

    def _MEDIAN_BLUR(self, image, intensity):
        return cv2.medianBlur(image, intensity)

    def _SIMPLE_BLUR(self, image, intensity):
        return cv2.blur(image, (intensity, intensity))

    def _BILATERAL_FILTER(self, image, intensity):
        return cv2.bilateralFilter(image, intensity, 100, 100)

def apply_blur(blur_type, image, intensity):
    return BlurType().blur(blur_type, image, intensity)
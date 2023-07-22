import cv2

def apply_denoise(image, intensity, template_window=7, search_window=21):
    return cv2.fastNlMeansDenoisingColored(image, None, intensity, intensity, template_window, search_window)
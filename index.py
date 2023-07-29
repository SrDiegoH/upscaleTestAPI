from enum import Enum

import cv2
#from .interpolation_upscale import apply_upscale

from flask import Flask, request, send_file

app = Flask(__name__)

'''
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


def apply_denoise(image, intensity, template_window=7, search_window=21):
    return cv2.fastNlMeansDenoisingColored(image, None, intensity, intensity, template_window, search_window)

'''

class InterpolationType(Enum):
    NEAREST_NEIGHBOR = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    NEAREST_NEIGHBOR_EXACT = cv2.INTER_NEAREST_EXACT
    BILINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    LANCZOS4 = cv2.INTER_LANCZOS4
    BITS2 = cv2.INTER_BITS2

def apply_upscale(interpolation_type, image, scale_factor=4, denoise_intensity=0, blur_intensity=0, blur_type='SIMPLE_BLUR'):
    (image_height, image_width) = image.shape[:2]

    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)

    new_image = cv2.resize(image, (new_width, new_height), interpolation=InterpolationType[interpolation_type].value)

    #denoised_image = apply_denoise(new_image, denoise_intensity)

    #blurred_image = apply_blur(blur_type, denoised_image, blur_intensity)

    #return blurred_image
    return new_image

@app.route('/')
def root():
    return 'Working'

@app.route('/', methods=['POST'])
def upscale():
    #return f'Request values: {request.values}, Request form: {request.form}, Request files: {request.files}'
    image = request.files.get('image')

    upscale_type = request.values.get('upscale_type', default='SIMPLE_BLUR').encode("UTF-8")
    scale_factor = int(request.values.get('scale_factor', default=4).encode("UTF-8"))

    denoise_intensity = int(request.values.get('denoise_intensity', default=0).encode("UTF-8"))

    blur_intensity = int(request.values.get('blur_intensity').encode("UTF-8"))
    blur_type = request.values.get('blur_type').encode("UTF-8")

    #if upscale_type in dir(InterpolationType):
    #upscaled_image = apply_upscale(upscale_type, image, scale_factor, denoise_intensity, blur_intensity, blur_type)

    #return send_file(upscaled_image)
    return f'Upscale type: {upscale_type}, Scale factor: {scale_factor}, Denoise intensity: {denoise_intensity}, Blur intensity: {blur_intensity}, Blur type: {blur_type}, Image: {image}'

if __name__ == '__main__':
    app.run()
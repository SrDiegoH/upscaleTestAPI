from datetime import datetime
from enum import Enum

import numpy as np
import cv2
#from .interpolation_upscale import apply_upscale

from flask import Flask, request, send_file, Response


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

'''

def apply_denoise(image, intensity, template_window=7, search_window=21):
    return cv2.fastNlMeansDenoisingColored(image, None, intensity, intensity, template_window, search_window)


class InterpolationType(Enum):
    NEAREST_NEIGHBOR = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    NEAREST_NEIGHBOR_EXACT = cv2.INTER_NEAREST_EXACT
    BILINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    LANCZOS4 = cv2.INTER_LANCZOS4
    BITS2 = cv2.INTER_BITS2

def apply_upscale(interpolation_type, image_bytes, scale_factor=4, denoise_intensity=0, blur_intensity=0, blur_type='SIMPLE_BLUR'):
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    (image_height, image_width) = image.shape[:2]

    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)

    new_image = cv2.resize(image, (new_width, new_height), interpolation=InterpolationType[interpolation_type].value)

    denoised_image = apply_denoise(new_image, denoise_intensity)

    #blurred_image = apply_blur(blur_type, denoised_image, blur_intensity)

    #return blurred_image
    return denoised_image

@app.route('/')
def root():
    return 'Working'

@app.route('/', methods=['POST'])
def upscale():
    #return f''
    image = request.files.get('image')

    if image is None:
        return Response("Image Not Found", status=400)
 
    image_bytes = np.fromfile(image, np.uint8)

    upscale_type = request.values.get('upscale_type')

    raw_scale_factor = request.values.get('scale_factor')
    scale_factor = int(raw_scale_factor.strip()) if raw_scale_factor else None

    raw_denoise_intensity = request.values.get('denoise_intensity')
    denoise_intensity = int(raw_denoise_intensity.strip()) if raw_denoise_intensity else None

    raw_blur_intensity = request.values.get('blur_intensity')
    blur_intensity = int(raw_blur_intensity.strip()) if raw_blur_intensity else None

    raw_blur_type = request.values.get('blur_type')
    blur_type = raw_blur_type.strip() #if raw_blur_type and raw_blur_type.strip() in dir(InterpolationType) else None

    if upscale_type in dir(InterpolationType):
        upscaled_image = apply_upscale(upscale_type, image_bytes, scale_factor, denoise_intensity, blur_intensity, blur_type)
    else:
        return Response("Upscale Type Not Found", status=400)

    file_name = f'{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}{image.filename}'
    #return f'Request values: {request.values}\nRequest form: {request.form}\nRequest files: {request.files}\nUpscale type: {upscale_type}\nScale factor: {scale_factor}\nDenoise intensity: {denoise_intensity}\nBlur intensity: {blur_intensity}\nBlur type: {blur_type}\nImage: {image}\nImage Bytes: {image_bytes}\nUpscaled Image: {upscaled_image}'
    return f'Image: {image}, {dir(image)}, {image.filename}, {datetime.now()}, {datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}, {image.filename}{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    #return send_file(upscaled_image.tofile(f'upscaled_image_{datetime.now()}'), mimetype='image/png')


if __name__ == '__main__':
    app.run()
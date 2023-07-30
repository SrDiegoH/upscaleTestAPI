import base64
from enum import Enum
import os
from urllib.request import urlretrieve

import cv2
#from basicsr.archs.rrdbnet_arch import RRDBNet
from flask import Flask, render_template, request, Response
import numpy as np
#from realesrgan import RealESRGANer
#from realesrgan.archs.srvgg_arch import SRVGGNetCompact


app = Flask(__name__)

class BlurType:
    def blur(self, image, blut_type, intensity):
        return getattr(self, f'_{str(blut_type)}', lambda image, intensity: image)(image, intensity)

    def _GAUSSIAN_BLUR(self, image, intensity):
        return cv2.GaussianBlur(image, (intensity, intensity), 0)

    def _MEDIAN_BLUR(self, image, intensity):
        return cv2.medianBlur(image, intensity)

    def _SIMPLE_BLUR(self, image, intensity):
        return cv2.blur(image, (intensity, intensity))

    def _BILATERAL_FILTER(self, image, intensity):
        return cv2.bilateralFilter(image, intensity, 100, 100)

def apply_blur(image, blur_type, intensity):
    odd_intensity = intensity if intensity % 2 == 1 else intensity + 1
    return BlurType().blur(image, blur_type, odd_intensity)


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

def apply_upscale(interpolation_type, image_bytes, scale_factor, denoise_intensity=0, blur_intensity=0, blur_type='SIMPLE_BLUR'):
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    (image_height, image_width) = image.shape[:2]

    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)

    new_image = cv2.resize(image, (new_width, new_height), interpolation=InterpolationType[interpolation_type].value)

    denoised_image = apply_denoise(new_image, denoise_intensity)

    blurred_image = apply_blur(denoised_image, blur_type, blur_intensity)

    return cv2.imencode('.png', blurred_image)[1]


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

class SuperResolutionType:
    def super_resolution(self, super_resolution_type, image, scale_factor=4, denoise_intensity=0):
        self.super_resolution_network = cv2.dnn_superres.DnnSuperResImpl_create()
        self.base_url = 'https://raw.githubusercontent.com/SrDiegoH/upscaleTestResources/main/resources/models'

        return getattr(self, f'_{str(super_resolution_type)}', lambda image, scale_factor: image)(image, int(scale_factor), denoise_intensity)

    def _EDSR(self, image, scale_factor, denoise_intensity):
        model_name = f'EDSR_x{scale_factor}.pb'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/EDSR/{model_name}', model_path)

        self.super_resolution_network.readModel(model_path)
        self.super_resolution_network.setModel("edsr", scale_factor)

        new_image = self.super_resolution_network.upsample(image)

        denoised_image = apply_denoise(new_image, denoise_intensity)

        delete_file(model_path)

        return denoised_image

    def _ESPCN(self, image, scale_factor, denoise_intensity):
        model_name = f'ESPCN_x{scale_factor}.pb'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESPCN/{model_name}', model_path)

        self.super_resolution_network.readModel(model_path)
        self.super_resolution_network.setModel("espcn", scale_factor)

        new_image = self.super_resolution_network.upsample(image)

        denoised_image = apply_denoise(new_image, denoise_intensity)

        delete_file(model_path)

        return denoised_image

    def _FSRCNN_SMALL(self, image, scale_factor, denoise_intensity):
        model_name = f'FSRCNN-small_x{scale_factor}.pb'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/FSRCNN/{model_name}', model_path)

        self.super_resolution_network.readModel(model_path)
        self.super_resolution_network.setModel("fsrcnn", scale_factor)

        new_image = self.super_resolution_network.upsample(image)

        denoised_image = apply_denoise(new_image, denoise_intensity)

        delete_file(model_path)

        return denoised_image

    def _FSRCNN(self, image, scale_factor, denoise_intensity):
        model_name = f'FSRCNN_x{scale_factor}.pb'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/FSRCNN/{model_name}', model_path)

        self.super_resolution_network.readModel(model_path)
        self.super_resolution_network.setModel("fsrcnn", scale_factor)

        new_image = self.super_resolution_network.upsample(image)

        denoised_image = apply_denoise(new_image, denoise_intensity)

        delete_file(model_path)

        return denoised_image

    def _LAPSRN(self, image, scale_factor, denoise_intensity):
        model_name = f'LapSRN_x{scale_factor}.pb'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/LapSRN/{model_name}', model_path)

        self.super_resolution_network.readModel(model_path)
        self.super_resolution_network.setModel("lapsrn", scale_factor)

        new_image = self.super_resolution_network.upsample(image)

        denoised_image = apply_denoise(new_image, denoise_intensity)

        delete_file(model_path)

        return denoised_image

    '''
    def _REAL_ESRGAN_PLUS(self, image, scale_factor, denoise_intensity):
        model_name = f'RealESRGAN_x{scale_factor}plus.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale_factor)
        new_image, _ = RealESRGANer(scale=scale_factor, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=scale_factor)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)

    def _REAL_ESRGAN_PLUS_ANIME_6B(self, image, scale_factor, denoise_intensity):
        model_name = f'RealESRGAN_x4plus_anime_6B.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        new_image, _ = RealESRGANer(scale=4, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=4)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)

    def _REAL_ESRGAN_NET(self, image, scale_factor, denoise_intensity):
        model_name = f'RealESRNet_x4plus.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        new_image, _ = RealESRGANer(scale=4, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=4)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)

    def _REAL_ESR_ANIME_VIDEO(self, image, scale_factor, denoise_intensity):
        model_name = f'RealESRGANv2-animevideo-xsx{scale_factor}.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale_factor, act_type='prelu')
        new_image, _ = RealESRGANer(scale=4, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=scale_factor)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)

    def _REAL_ESR_ANIME_VIDEO_V3(self, image, scale_factor, denoise_intensity):
        model_name = f'realesr-animevideov3.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        new_image, _ = RealESRGANer(scale=4, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=4)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)

    def _REAL_ESR_GENERAL(self, image, scale_factor, denoise_intensity):
        model_name = f'realesr-general-x4v3.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        new_image, _ = RealESRGANer(scale=4, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=4)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)

    def _REAL_ESR_GENERAL_WDN(self, image, scale_factor, denoise_intensity):
        model_name = f'realesr-general-wdn-x4v3.pth'
        model_path = f'/tmp/{model_name}'

        urlretrieve(f'{self.base_url}/ESRGAN/{model_name}', model_path)

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        new_image, _ = RealESRGANer(scale=4, model_path=model_path, dni_weight=denoise_intensity/10, model=model).enhance(image, outscale=4)

        delete_file(model_path)

        return np.ascontiguousarray(new_image, dtype=np.uint8)
    '''

def apply_super_resolution(super_resolution_type, image, scale_factor=4, denoise_intensity=0, blur_intensity=0, blur_type='SIMPLE_BLUR'):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    upscaled_image = SuperResolutionType().super_resolution(super_resolution_type, rgb_image, scale_factor, denoise_intensity)

    blurred_image = apply_blur(upscaled_image, blur_type, blur_intensity)

    return blurred_image


def upscale():
    image = request.files.get('image')

    if not image:
        return 'Imagem não enviada', 400
 
    image_bytes = np.fromfile(image, np.uint8)

    raw_scale_factor = request.values.get('scale_factor')

    if not raw_scale_factor:
        return 'Fator de crescimento não enviado', 400

    scale_factor = int(raw_scale_factor.strip())

    raw_denoise_intensity = request.values.get('denoise_intensity')
    denoise_intensity = int(raw_denoise_intensity.strip()) if raw_denoise_intensity else None

    raw_blur_intensity = request.values.get('blur_intensity')
    blur_intensity = int(raw_blur_intensity.strip()) if raw_blur_intensity else None

    raw_blur_type = request.values.get('blur_type')
    blur_type = raw_blur_type.strip() if raw_blur_type and f'_{raw_blur_type.strip()}' in dir(BlurType) else None

    upscale_type = request.values.get('upscale_type')

    if not upscale_type:
        return 'Tipo de aumento não enviado', 400

    if upscale_type in dir(InterpolationType):
        upscaled_image = apply_upscale(upscale_type, image_bytes, scale_factor, denoise_intensity, blur_intensity, blur_type)
    elif upscale_type and f'_{upscale_type.strip()}' in dir(SuperResolutionType):
        upscaled_image = apply_super_resolution(upscale_type, image_bytes, scale_factor, denoise_intensity, blur_intensity, blur_type)
    else:
        return 'Tipo de aumento não conhecido', 400

    upscaled_image_bytes = np.array(upscaled_image).tobytes()
    upscaled_image_base64 = base64.b64encode(upscaled_image_bytes).decode("utf-8")
    return upscaled_image_base64, 200


@app.route('/')
def root():
    return render_template('index.html', image='', show_image='block', error_message='', show_error='block')

@app.route('/', methods=['POST'])
def show_upscaled_image():
    response, code = upscale()

    if code == 400:
        return render_template('index.html', image='', show_image='block', error_message=response, show_error='inline')

    return render_template('index.html', image=response, show_image='inline', error_message='', show_error='block')

@app.route('/upscale', methods=['POST'])
def return_upscaled_image():
    response, code = upscale()

    return Response(response if code == 400 else f'<img src="data:image/png;base64,{response}">', status=code)

if __name__ == '__main__':
    is_debug = os.getenv('IS_DEBUG', False)
    app.run(debug=is_debug)
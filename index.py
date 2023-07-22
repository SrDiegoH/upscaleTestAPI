from flask import Flask, request, send_file

from .interpolation_upscale import apply_upscale, InterpolationType

app = Flask(__name__)

@app.route('/')
def root():
    return 'Working'

@app.route('/', methods=['POST'])
def upscale():
    image = request.files('image')

    upscale_type = request.values.get('upscale_type').encode("UTF-8")
    scale_factor = int(equest.values.get('scale_factor').encode("UTF-8"))

    denoise_intensity = int(request.values.get('denoise_intensity').encode("UTF-8"))

    blur_intensity = int(request.values.get('blur_intensity').encode("UTF-8"))
    blur_type = request.values.get('blur_type').encode("UTF-8")

    if upscale_type in dir(InterpolationType):
        upscaled_image = apply_upscale(upscale_type, image, scale_factor, denoise_intensity, blur_intensity, blur_type)

    return send_file(upscaled_image)

if __name__ == '__main__':
    app.run()
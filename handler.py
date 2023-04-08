import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

import os
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

INPUT_SCHEMA = {
    'image_url': {
        'type': str,
        'required': True
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'RealESRGAN_x4plus',
        'constraints': lambda model: model in [
            'RealESRGAN_x4plus',
            'RealESRNet_x4plus',
            'RealESRGAN_x4plus_anime_6B',
            'RealESRGAN_x2plus',
        ]
    },
    'scale': {
        'type': float,
        'required': False,
        'default': 4,
        'constraints': lambda scale: 0 < scale < 16
    },
    'tile': {
        'type': int,
        'required': False,
        'default': 0,
    },
    'tile_pad': {
        'type': int,
        'required': False,
        'default': 10,
    },
    'pre_pad': {
        'type': int,
        'required': False,
        'default': 0,
    },
}


# handler handles upscale request
def handler(job):
    try:
        # get input
        job_input = job['input']
        # validate input
        validated_input = validate(job_input, INPUT_SCHEMA)
        if 'errors' in validated_input:
            raise validated_input['errors']

        validated_input = validated_input['validated_input']

        # Download input objects
        remote_file = rp_download.file(validated_input.get('image_url', None))
        image_path = remote_file["file_path"]

        # check model
        if validated_input["model"] == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif validated_input["model"] == 'RealESRNet_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif validated_input["model"] == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif validated_input["model"] == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            raise "model not found"

    except Exception as e:
        return {"error": e}

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=f'weights/{validated_input["model"]}.pth',
        dni_weight=None,
        model=model,
        tile=validated_input['tile'],
        tile_pad=validated_input['tile_pad'],
        pre_pad=validated_input['pre_pad'],
        half=False,
        gpu_id=None)

    imgname, extension = os.path.splitext(os.path.basename(image_path))
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    image_url = ""
    try:
        output, _ = upsampler.enhance(img, outscale=validated_input["scale"])
    except RuntimeError:
        raise "runtime error"
    else:
        extension = extension[1:]
        if img_mode == 'RGBA':
            extension = 'png'
        save_path = os.path.join("", f'{imgname}.{extension}')

        cv2.imwrite(save_path, output)

        image_url = rp_upload.upload_image(job['id'], save_path)

    return image_url

# start pod
runpod.serverless.start({
    "handler": handler
})

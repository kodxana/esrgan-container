import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

import os
import requests

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer

INPUT_SCHEMA = {
    'image_url': {
        'type': str,
        'required': True
    },
}


def handler(job):
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    outscale = 4
    model_path = "weights/RealESRGAN_x4plus.pth"

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None)

    # Download input objects
    image_file_path = rp_download.download_input_objects(
        [validated_input.get('image_url', None)]
    )  # pylint: disable=unbalanced-tuple-unpacking
    image = image_file_path[0]

    imgname, extension = os.path.splitext(os.path.basename(image))
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None


    image_url = ""
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        extension = extension[1:]
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        save_path = os.path.join("", f'{imgname}.{extension}')

        cv2.imwrite(save_path, output)

        image_url = rp_upload.upload_image(job['id'], save_path)

    return image_url


runpod.serverless.start({
    "handler": handler
})
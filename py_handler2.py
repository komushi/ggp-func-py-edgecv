import json
import logging
import time
import datetime
import sys
import os

import onnxruntime as ort

from insightface.app import FaceAnalysis

# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def function_handler(event, context):
    logger.info('function_handler event: ' + repr(event))

    init_onnx()

    # init_face_app()


def init_onnx():
    try:
        print(f"onnxruntime device: {ort.get_device()}")

        print(f'ort avail providers: {ort.get_available_providers()}')

        ort_session = ort.InferenceSession('/etc/insightface/models/buffalo_sc/det_500m.onnx', providers=["CUDAExecutionProvider"])

        print(ort_session.get_providers())
    except Exception as e:
        logger.info('Caught init_onnx Othert Exception')
        logger.info(e)

def init_face_app():
    try:
        logger.info('init_face_app begin')
        app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection',], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='/etc/insightface')

        logger.info('init_face_app FaceAnalysis initialized')
        app.prepare(ctx_id=0, det_size=(640, 640))#ctx_id=0 CPU

        logger.info('init_face_app FaceAnalysis prepared')
        return app
    except Exception as e:
        logger.info('Caught init_face_app Othert Exception')
        logger.info(e)

if __name__ == "__main__":
    function_handler(event={"method": "aaa", "action": "bbb"}, context=None)

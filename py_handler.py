import json
import logging
import time
import datetime
import sys
import os
from threading import Thread
from multiprocessing.connection import Listener, Client

import greengrasssdk
import socket

import numpy as np
import PIL.Image
from insightface.app import FaceAnalysis

import io
import base64


# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


face_app = init_face_app()
client = greengrasssdk.client("iot-data")

def get_local_ip():
    # Connect to an external host, in this case, Google's DNS server
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    return local_ip

def init_face_app():
    app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='/etc/insightface')
    app.prepare(ctx_id=0, det_size=(640, 640))#ctx_id=0 CPU
    return app

def read_base64_face(base64_string):
    # Step 1: Decode the Base64 string
    image_data = base64.b64decode(base64_string)
    
    # Step 2: Open the image using PIL
    image = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Step 3: Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Step 4: Reorder the channels from RGB to BGR
    image_bgr = image_array[:, :, [2, 1, 0]]  # RGB to BGR
    
    return image_bgr


def function_handler(event, context):
    logger.info('function_handler context: ' + repr(context))
    if context.clientContext.Custom.subject == "gocheckin/req_face_embeddings":
        
        logger.info('function_handler req_face_embeddings event: ' + repr(event))

        image_bgr = read_base64_face(event.faceImgBase64)

        reference_faces = face_app.get(image_bgr)
        return 

        data = {
            "reservationCode": event.reservationCode,
            "memberNo": event.memberNo,
            "faceEmbedding": reference_faces[0].embedding
        }
        
        print(json.dumps(data))

        client.publish(
            topic="gocheckin/res_face_embeddings",
            payload=json.dumps(data)
        )
        sys.exit(0)

    elif context.clientContext.Custom.subject == f"gocheckin/{AWS_IOT_THING_NAME}/init_scanner":
        logger.info('function_handler init_scanner')

        data = {
            "equipmentId": os.environ['AWS_IOT_THING_NAME'],
            "equipmentName": os.environ['AWS_IOT_THING_NAME'],
            "localIp": get_local_ip()
        }
        
        # print(json.dumps(data))
        client.publish(
            topic="gocheckin/scanner_detected",
            payload=json.dumps(data)
        )
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) == 6:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2], "rtsp_src": sys.argv[3], "codec": sys.argv[4], "framerate": sys.argv[5]}, context=None)
    elif len(sys.argv) == 7:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2], "rtsp_src": sys.argv[3], "codec": sys.argv[4], "framerate": sys.argv[5], "face_file": sys.argv[6]}, context=None)
    elif len(sys.argv) == 3:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2]}, context=None)

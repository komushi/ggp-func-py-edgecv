import json
import logging
import time
import datetime
import sys
import os
import numpy as np

import greengrasssdk
iotClient = greengrasssdk.client("iot-data")

# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def get_local_ip():
    import socket

    # Connect to an external host, in this case, Google's DNS server
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    return local_ip

def init_face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='/etc/insightface')
    app.prepare(ctx_id=0, det_size=(640, 640))#ctx_id=0 CPU
    return app

def read_picture_from_url(url):
    import PIL.Image
    import io
    import requests

    # Download the image
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    
    # Open the image from the downloaded content
    image = PIL.Image.open(io.BytesIO(response.content)).convert("RGB")
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Rearrange the channels from RGB to BGR
    image_bgr = image_array[:, :, [2, 1, 0]]
    
    return image_bgr

def start_http_server():
    import http.server
    import socketserver
    import json

    PORT = 8888

    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/recognise':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # Process the POST data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)


                logger.info('/recognise POST: ' + post_data)

                event = json.loads(post_data)


                image_bgr = read_picture_from_url(event['faceImgUrl'])

                reference_faces = face_app.get(image_bgr)

                data = {
                    "reservationCode": event['reservationCode'],
                    "memberNo": event['memberNo'],
                    "faceEmbedding": reference_faces[0].embedding.tolist()
                }
        
                iotClient.publish(
                    topic="gocheckin/res_face_embeddings",
                    payload=json.dumps(data)
                )

                # Example response
                response = {'message': 'Recognition completed'}

                # Send the response
                self.wfile.write(json.dumps(response).encode())

                logger.info('/recognise POST finished')

            elif self.path == '/detect':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # Process the POST data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                # Example response
                response = {'message': 'Detection completed', 'data': json.loads(post_data)}

                # Send the response
                self.wfile.write(json.dumps(response).encode())

            else:
                self.send_error(404, 'Path Not Found: %s' % self.path)

        def address_string(self):  # Limit access to local network requests
            host, port = self.client_address[:2]
            return host

    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print("Serving at port", PORT)
        httpd.serve_forever()

def function_handler(event, context):

    context_vars = vars(context)
    topic = context_vars['client_context'].custom['subject']

    logger.info('function_handler topic: ' + topic)

    if topic == "gocheckin/req_face_embeddings":

        import greengrasssdk
        
        logger.info('function_handler req_face_embeddings event: ' + repr(event))

        image_bgr = read_picture_from_url(event['faceImgUrl'])

        face_app = init_face_app()

        reference_faces = face_app.get(image_bgr)

        data = {
            "reservationCode": event['reservationCode'],
            "memberNo": event['memberNo'],
            "faceEmbedding": reference_faces[0].embedding.tolist()
        }
        
        logger.info('function_handler payload with faceEmbedding: ' + json.dumps(data))

        client = greengrasssdk.client("iot-data")
        client.publish(
            topic="gocheckin/res_face_embeddings",
            payload=json.dumps(data)
        )
        sys.exit(0)

    elif topic == f"gocheckin/{os.environ['AWS_IOT_THING_NAME']}/init_scanner":        
        logger.info('function_handler init_scanner')

        import greengrasssdk

        client = greengrasssdk.client("iot-data")

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


face_app = init_face_app()
start_http_server()
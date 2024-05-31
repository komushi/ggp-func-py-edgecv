import json
import logging
import time
import datetime
import sys
import os
import numpy as np

import PIL.Image
import io
import base64

import http.server
import socketserver

import boto3
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime

from threading import Thread

# import greengrasssdk
# iotClient = greengrasssdk.client("iot-data")

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

    global face_app

    if face_app is None:
        face_app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='/etc/insightface')
        face_app.prepare(ctx_id=0, det_size=(640, 640))#ctx_id=0 CPU
    
    return

def read_picture_from_url(url):
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
    
    return image_bgr, image

def start_http_server():

    PORT = 8888

    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):
            # if self.client_address[0] != '127.0.0.1':
            #     self.send_error(403, "Forbidden: Only localhost is allowed")
            #     return

            if self.path == '/recognise':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # Process the POST data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                event = json.loads(post_data)

                logger.info('/recognise POST: ' + json.dumps(event))


                image_bgr, org_image = read_picture_from_url(event['faceImgUrl'])

                init_face_app()
                reference_faces = face_app.get(image_bgr)

                print('reference_faces[0].embedding:')
                print(type(reference_faces[0].embedding))

                event['faceEmbedding'] = reference_faces[0].embedding.tolist()

                print('event[faceEmbedding]:')
                print(type(event['faceEmbedding']))

                # data = {
                #     "reservationCode": event['reservationCode'],
                #     "memberNo": event['memberNo'],
                #     "faceEmbedding": reference_faces[0].embedding.tolist()
                # }
        
                # iotClient.publish(
                #     topic="gocheckin/res_face_embeddings",
                #     payload=json.dumps(data)
                # )

                bbox = reference_faces[0].bbox.astype(np.int).flatten()
                cropped_face = org_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Convert the image to bytes
                buffered = io.BytesIO()
                cropped_face.save(buffered, format="JPEG")
                cropped_face_bytes = buffered.getvalue()

                event['faceImgBase64'] = base64.b64encode(cropped_face_bytes).decode('utf-8')

                # Send the response
                self.wfile.write(json.dumps(event).encode())

                logger.info('/recognise POST finished')

            elif self.path == '/detect':

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # Process the POST data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                event = json.loads(post_data)

                print(str("/detect motion:" + "{}").format(event['motion']))

                if event['motion'] is True:
                    active_members = get_active_members()

                # Example response
                response = {'message': event}

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

def get_active_reservations():

    # Initialize the DynamoDB resource
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=os.environ['DDB_ENDPOINT'],
        region_name='us-west-1',
        aws_access_key_id='fakeMyKeyId',
        aws_secret_access_key='fakeSecretAccessKey'
    )

    # Specify the table name
    tbl_reservation = os.environ['TBL_RESERVATION']

    # Get the table
    table = dynamodb.Table(tbl_reservation)

    # Get the current date in 'YYYY-MM-DD' format
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Create the filter expression
    filter_expression = Attr('checkInDate').lte(current_date) & Attr('checkOutDate').gte(current_date)

    # Scan the table with the filter expression
    response = table.scan(
        # FilterExpression=filter_expression,
        ProjectionExpression='reservationCode'
    )

    # Get the items from the response
    items = response.get('Items', [])

    for item in items:
        print(item)

    return items

def get_active_members():
    # Initialize the DynamoDB resource
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=os.environ['DDB_ENDPOINT'],
        region_name='us-west-1',
        aws_access_key_id='fakeMyKeyId',
        aws_secret_access_key='fakeSecretAccessKey'
    )
    # Specify the table name
    tbl_member = os.environ['TBL_MEMBER']

    # List of reservation codes to query
    active_reservations = get_active_reservations()

    # Define the list of attributes to retrieve
    attributes_to_get = ['reservationCode', 'memberNo', 'faceEmbedding']

    # Initialize an empty list to store the results
    results = []

    # Iterate over each reservation code and query DynamoDB
    for active_reservation in active_reservations:
        table = dynamodb.Table(tbl_member)
        
        # Query DynamoDB using the partition key (reservationCode)
        response = table.query(
            KeyConditionExpression='reservationCode = :code',
            ProjectionExpression=', '.join(attributes_to_get),
            ExpressionAttributeValues={
                ':code': active_reservation['reservationCode']
            }
        )
        
        # Add the query results to the results list
        results.extend(response['Items'])

    for item in results:
        item['faceEmbedding'] = np.array(item['faceEmbedding'])
        print(item)

    return results

def function_handler(event, context):

    context_vars = vars(context)
    topic = context_vars['client_context'].custom['subject']

    logger.info('function_handler topic: ' + topic)

    if topic == f"gocheckin/{os.environ['AWS_IOT_THING_NAME']}/init_scanner":        
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


def fetch_members():
    current_date = datetime.now()

    global active_members
    global last_fetch_time

    if last_fetch_time is None or last_fetch_time < current_date:
        last_fetch_time = current_date
        active_members = get_active_members()


active_members = []
last_fetch_time = None

face_app = None

t = Thread(target=start_http_server, daemon=True)
t.start()



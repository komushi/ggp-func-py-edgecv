import json
import logging
import time
import datetime
import sys
import os
from threading import Thread
from multiprocessing.connection import Listener, Client

# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# from libs import face_detector as fdm
detector = None

def motion(params):
    try:
        logger.info("motion start params:" + repr(params))

        import motion_detector as mdm

        global detector

        detector = mdm.MotionDetector(params)

        detector.start_detector()

    except Exception as e:
        logger.error("MotionDetector failure: " + repr(e))

def face(params):
    try:
        logger.info("face start params:" + repr(params))

        import face_detector as fdm

        global detector

        detector = fdm.FaceDetector(params)

        detector.start_detector()
        
    except Exception as e:
        logger.error("FaceDetector failure: " + repr(e))

def recognition(params):
    try:
        logger.info("recognition start params:" + repr(params))

        import face_recognition as fdm

        global detector

        detector = fdm.FaceRecognition(params)

        detector.start_detector()
        
    except Exception as e:
        logger.error("FaceRecognition failure: " + repr(e))

def function_handler(event, context):
    logger.info('function_handler event: ' + repr(event))

    if 'method' in event and 'action' in event:
        possibles = globals().copy()
        possibles.update(locals())

        method = possibles.get(event['method'])
        action = event['action']

        params = {}
        
        if 'rtsp_src' in event:
            params['rtsp_src'] = event['rtsp_src']

        if 'codec' in event:
            params['codec'] = event['codec']

        if 'framerate' in event:
            params['framerate'] = event['framerate']

        if 'face_file' in event:
            params['face_file'] = event['face_file']

        logger.info("function_handler params:" + repr(params))

        if not method:
            raise NotImplementedError("Method %s not implemented" % event['method'])
            sys.exit(0)
        else:
            index = list(possibles.keys()).index(event['method'])
            address = ('localhost', 6000 + index)

            if action == 'start':
                with Listener(address, authkey=b'secret password') as listener:                
                    print('Listener Started')
                    t = Thread(target=method, args=(params,))
                    t.start()
                    print('Prepare to accept')
                    with listener.accept() as conn:
                        print('connection accepted from', listener.last_accepted)
                        conn.send_bytes(b'hello')
                        global detector
                        detector.stop_event.set()
                        sys.exit()

            elif action == 'stop':
                with Client(address, authkey=b'secret password') as conn:
                    print(conn.recv_bytes())
                    sys.exit()

    else:
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) == 6:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2], "rtsp_src": sys.argv[3], "codec": sys.argv[4], "framerate": sys.argv[5]}, context=None)
    elif len(sys.argv) == 7:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2], "rtsp_src": sys.argv[3], "codec": sys.argv[4], "framerate": sys.argv[5], "face_file": sys.argv[6]}, context=None)
    elif len(sys.argv) == 3:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2]}, context=None)

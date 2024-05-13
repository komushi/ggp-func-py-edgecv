import json
import logging
import time
import datetime
import sys
import os
import threading
import queue
import traceback

import gstreamer_threading as gst

# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class MotionDetector(object):
    def __init__(self, params):
        #Current Cam
        self.running = True
        self.camProcess = None
        self.cam_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.camlink = params['rtsp_src']

        if params['codec'] == 'h264':
            self.pipeline_str = """rtspsrc name=m_rtspsrc ! rtph264depay name=m_rtph264depay ! avdec_h264 name=m_avdec ! videoconvert
                ! motioncells name=m_motioncells sensitivity=0.7 ! videoconvert name=m_videoconvert ! videorate name=m_videorate ! appsink name=m_appsink"""
        elif params['codec'] == 'h265':
            self.pipeline_str = """rtspsrc name=m_rtspsrc ! rtph265depay name=m_rtph265depay 
                ! avdec_h265 name=m_avdec ! videoconvert
                ! motioncells name=m_motioncells sensitivity=0.7 ! videoconvert name=m_videoconvert ! videorate name=m_videorate ! appsink name=m_appsink"""
        elif params['codec'] == 'webcam':
            self.pipeline_str = """avfvideosrc device-index=0 ! video/x-raw,width=1280,height=720
                ! queue ! videoconvert name=m_videoconvert ! videoscale ! videorate name=m_videorate
                ! motioncells name=m_motioncells sensitivity=0.7 ! videoconvert
                ! queue ! appsink name=m_appsink"""
        
        if params['framerate'] is not None:
            self.framerate = int(params['framerate'])
        else:
            self.framerate = 15

        self.motion_detected = False

    def start_detector(self):
        if threading.current_thread() == threading.main_thread():
            import cv2
                    
        #get all cams
        time.sleep(1)

        self.camProcess = gst.StreamCapture(self.camlink,
                            self.pipeline_str,
                            self.stop_event,
                            self.cam_queue,
                            self.framerate)
        self.camProcess.start()

        try:
            # while self.running:
            while not self.stop_event.is_set():

                if not self.cam_queue.empty():
                    # print('Got frame')
                    cmd, val = self.cam_queue.get(False)

                    if cmd == gst.StreamCommands.FRAME:
                        if val is not None:
                            if threading.current_thread() == threading.main_thread():
                                cv2.imshow('Cam: ' + self.camlink, val)
                                cv2.waitKey(1)
                    elif cmd == gst.StreamCommands.MOTION_BEGIN:
                        print('MOTION DETECTED')
                        self.motion_detected = True
                    elif cmd == gst.StreamCommands.MOTION_END:
                        print('MOTION DETECTION STOPPED')
                        self.motion_detected = False

        except KeyboardInterrupt:
            logger.info('Caught Keyboard interrupt')

        except:
            e = sys.exc_info()
            logger.info('Caught Main Exception')
            logger.info(e)

        self.stop_detector()

        if threading.current_thread() == threading.main_thread():
            cv2.destroyAllWindows()
        logger.info('start_detector after destroyAllWindows')


    def stop_detector(self):
        logger.info('in stop_detector')
        try:
            if self.stop_event is not None:
                self.stop_event.set()
                while not self.cam_queue.empty():
                    try:
                        _ = self.cam_queue.get(false)
                        logger.info('in stop_detector cleanup')
                    except:
                        break
                    self.cam_queue.join()

                self.camProcess.join()
                # self.camProcess.close()
                logger.info('After close')
        except queue.Empty as ex:
            logger.info('Caught stop_detctor Queue.Empty Exception')
            logger.info(e)
            traceback.print_exc()
        except Exception as e:
            logger.info('Caught stop_detector Othert Exception')
            logger.info(e)
            traceback.print_exc()

def function_handler(event, context):
    logger.info('function_handler event: ' + repr(event))

    detector = MotionDetector({"rtsp_src": event['rtsp_src'], "codec": event['codec'], "framerate": event['framerate']})

    if sys.platform == 'darwin' or sys.platform == 'win32':
        detector.start_detector()
    elif sys.platform == 'linux':
        t = threading.Thread(target=detector.start_detector)
        t.start()

if __name__ == "__main__":
    if len(sys.argv) == 6:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2], "rtsp_src": sys.argv[3], "codec": sys.argv[4], "framerate": sys.argv[5]}, context=None)
    elif len(sys.argv) == 3:
        function_handler(event={"method": sys.argv[1], "action": sys.argv[2]}, context=None)

# Importing libraries

import aiohttp_jinja2
import jinja2
import numpy as np
import cv2
import mediapipe as mp

import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaRelay

app = web.Application() # Initialization
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('./templates'))
routes = web.RouteTableDef()
mp_drawing = mp.solutions.drawing_utils # Drawing Utilites
mp_pose = mp.solutions.pose # Pose
mp_holistic = mp.solutions.holistic # Holistic

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        if play[0]:
            frame = squatsRightMonitor(frame)
        elif play[1]:
            frame = squatsLeftMonitor(frame)
        elif play[2]:
            frame = bicepCurlRightMonitor(frame)
        elif play[3]:
            frame = bicepCurlLeftMonitor(frame)
        elif play[4]:
            frame = flamingoRightMonitor(frame)
        elif play[5]:
            frame = flamingoLeftMonitor(frame)
        elif play[6]:
            frame = frontSplitsMonitor(frame)
        elif play[7]:
            frame = splitsRightMonitor(frame)
        elif play[8]:
            frame = splitsLeftMonitor(frame)
        return frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False # Image is no longer writeable
    results = model.process(image) # Make prediction
    image.flags.writeable = True # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks_bicep_curl_right(image, results):
    # Draw right pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[18], list(mp_holistic.POSE_CONNECTIONS)[4], list(mp_holistic.POSE_CONNECTIONS)[30]], 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_bicep_curl_left(image,results):
    # Draw left pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[22], list(mp_holistic.POSE_CONNECTIONS)[34], list(mp_holistic.POSE_CONNECTIONS)[34]],
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=3)
                             )
                            
def draw_styled_landmarks_squat_right(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[7], list(mp_holistic.POSE_CONNECTIONS)[29], list(mp_holistic.POSE_CONNECTIONS)[13], list(mp_holistic.POSE_CONNECTIONS)[18]], 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_squat_left(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[33], list(mp_holistic.POSE_CONNECTIONS)[27], list(mp_holistic.POSE_CONNECTIONS)[5], list(mp_holistic.POSE_CONNECTIONS)[8]], 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_front_splits(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[7], list(mp_holistic.POSE_CONNECTIONS)[29], list(mp_holistic.POSE_CONNECTIONS)[13], list(mp_holistic.POSE_CONNECTIONS)[10], list(mp_holistic.POSE_CONNECTIONS)[5], list(mp_holistic.POSE_CONNECTIONS)[27], list(mp_holistic.POSE_CONNECTIONS)[33], list(mp_holistic.POSE_CONNECTIONS)[18], list(mp_holistic.POSE_CONNECTIONS)[8]], 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_side_splits_right(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[7], list(mp_holistic.POSE_CONNECTIONS)[29], list(mp_holistic.POSE_CONNECTIONS)[13], list(mp_holistic.POSE_CONNECTIONS)[10], list(mp_holistic.POSE_CONNECTIONS)[5], list(mp_holistic.POSE_CONNECTIONS)[27], list(mp_holistic.POSE_CONNECTIONS)[33]], 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_side_splits_left(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[7], list(mp_holistic.POSE_CONNECTIONS)[29], list(mp_holistic.POSE_CONNECTIONS)[13], list(mp_holistic.POSE_CONNECTIONS)[10], list(mp_holistic.POSE_CONNECTIONS)[5], list(mp_holistic.POSE_CONNECTIONS)[27], list(mp_holistic.POSE_CONNECTIONS)[33], list(mp_holistic.POSE_CONNECTIONS)[8]], 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_flamingo_right(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[7], list(mp_holistic.POSE_CONNECTIONS)[29], list(mp_holistic.POSE_CONNECTIONS)[13]], 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=3, circle_radius=3)
                             )

def draw_styled_landmarks_flamingo_left(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, [list(mp_holistic.POSE_CONNECTIONS)[33], list(mp_holistic.POSE_CONNECTIONS)[27], list(mp_holistic.POSE_CONNECTIONS)[5]], 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=3, circle_radius=3)
                             )

class Angle:
    def __init__(self,a,b,c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.radians = np.arctan2(self.c[1]-self.b[1], self.c[0]-self.b[0]) - np.arctan2(self.a[1]-self.b[1], self.a[0]-self.b[0])
        self.angle = abs(self.radians*180.0/np.pi)
        
    def calculate_angle(self):
        if self.angle > 180.0:
            self.angle = 360-self.angle
    
        return self.angle

class Rating(Angle):
    def __init__(self,a,b,c):
        super().__init__(a,b,c)
        
    def calculate_rating_bicep_curl(self):
        if self.angle > 180.0:
            self.angle = 360-self.angle
        
        if self.angle >= 11.47:
            percentage = 100 * (11.47/self.angle)
        else:
            self.angle = 22.94 - self.angle
            percentage = 100 * (11.47/self.angle)
        
        return percentage
    
    def calculate_rating_squat(self):
        if self.angle >= 79.09:
            percentage = 100 * (79.09/self.angle)
        else:
            self.angle = 158.18 - self.angle
            percentage = 100 * (79.09/self.angle)
        
        return percentage
    
    def calculate_rating_squat_back(self):
        if self.angle >= 67.62:
            percentage = 100 * (67.62/self.angle)
        else:
            self.angle = 135.24 - self.angle
            percentage = 100 * (67.62/self.angle)
        
        return percentage
    
    def calculate_rating_splits(self):
        if self.angle >= 180:
            percentage = 100 * (180/self.angle)
        else:
            self.angle = 360 - self.angle
            percentage = 100 * (180/self.angle)
        
        return percentage

    def calculate_rating_splits_back(self):
        if self.angle >= 90:
            percentage = 100 * (90/self.angle)
        else:
            self.angle = 180 - self.angle
            percentage = 100 * (90/self.angle)
        
        return percentage

    def calculate_rating_flamingo(self):
        if self.angle >= 34.53 :
            percentage = 100 * (34.53/self.angle)
        else:
            self.angle = 69.06 - self.angle
            percentage = 100 * (34.53/self.angle)
        
        return percentage

    def message_right(self):
        if self.a[0] > self.c[0]:
            self.angle = -self.angle
        
        if self.angle > 11.47:
            percentage = 100 * (11.47/self.angle)
        else:
            self.angle = 22.94 - self.angle
            percentage = 100 * (11.47/self.angle)
            self.angle = -self.angle + 22.94
            
        if self.angle > 11.47 and percentage < 80:
            message = "Move your arm backward"
        elif self.angle < 11.47 and percentage < 80:
            message = "Move your arm forward"
        else:
            message = "Perfect!"
            
        return message
    
    def message_left(self):
        if self.a[0] < self.c[0]:
            self.angle = -self.angle
        
        if self.angle > 11.47:
            percentage = 100 * (11.47/self.angle)
        else:
            self.angle = 22.94 - self.angle
            percentage = 100 * (11.47/self.angle)
            self.angle = -self.angle + 22.94
            
        if self.angle > 11.47 and percentage < 80:
            message = "Move your arm backward"
        elif self.angle < 11.47 and percentage < 80:
            message = "Move your arm forward"
        else:
            message = "Perfect!"
            
        return message

def bicepCurlRightMonitor(frame):
    # Right Arm

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            
        cnt = 0
        cnt_copy = 0
        curl_stage = None
        set_cnt = 0

        # Set mediapipe model 

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_bicep_curl_right(image, results)

        try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        except Exception as e:
            print(e)
        
        try:
            
            # Instantiate objects            
            rating = Rating(hip,shoulder,elbow)
            message = rating.message_right()
            percentage = rating.calculate_rating_bicep_curl()
            
            # Display Green or Red Message
            if percentage >= 80:
                cv2.putText(image, message,
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 2, cv2.LINE_AA
                                    )
            else:
                cv2.putText(image, message,
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
                
            angle = Angle(shoulder,elbow,wrist)
            angle = angle.calculate_angle()
            
            if angle > 150 and percentage > 80:
                curl_stage = 'down'
            if angle < 40 and curl_stage == 'down':
                curl_stage = 'up'
                cnt += 1
                cnt_copy += 1
                
            if cnt_copy == 10:
                set_cnt += 1
                cnt_copy = 0
                
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Accuarcy
            cv2.putText(image, 'Accuarcy', (25,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (25,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
            
            # Curl Stage data
            cv2.putText(image, 'Curl Stage', (175,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, curl_stage, 
                        (175,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Set
            cv2.putText(image, 'Set', (325,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_cnt), 
                        (325,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Repetitions
            cv2.putText(image, 'Repetitions', (475,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(cnt), 
                        (475,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        except Exception as e:
            print(e)

        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


def bicepCurlLeftMonitor(frame):
    # Left Arm

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        cnt = 0
        cnt_copy = 0
        curl_stage = None
        set_cnt = 0

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_bicep_curl_left(image, results)

        try:

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        except Exception as e:
            print(e)
        
        try:
            
            # Instantiate objects            
            rating = Rating(hip,shoulder,elbow)
            message = rating.message_left()
            percentage = rating.calculate_rating_bicep_curl()
            
            # Display Green or Red Message
            if percentage >= 80:
                cv2.putText(image, message,
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 2, cv2.LINE_AA
                                    )
            else:
                cv2.putText(image, message,
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
                
            angle = Angle(shoulder,elbow,wrist)
            angle = angle.calculate_angle()
            
            if angle > 150 and percentage > 80:
                curl_stage = 'down'
            if angle < 40 and curl_stage == 'down':
                curl_stage = 'up'
                cnt += 1
                cnt_copy += 1
                
            if cnt_copy == 10:
                set_cnt += 1
                cnt_copy = 0
                
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Accuarcy
            cv2.putText(image, 'Accuarcy', (25,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (25,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
            
            # Curl Stage data
            cv2.putText(image, 'Curl Stage', (175,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, curl_stage, 
                        (175,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Set
            cv2.putText(image, 'Set', (325,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_cnt), 
                        (325,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Repetitions
            cv2.putText(image, 'Repetitions', (475,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(cnt), 
                        (475,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

def squatsRightMonitor(frame):
    # Squats Right
    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 
        cnt = 0
        stage = None

        # Recolor image to RGB
        image = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_squat_right(image, results)
        
        try:

            landmarks = results.pose_landmarks.landmark
                    
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        except Exception as e:
            print(e)

        # Extract landmarks
        try:
            # Instantiate objects
            squat_angle_obj = Angle(ankle,knee,hip)
            squat_angle = squat_angle_obj.calculate_angle()
            
            back_angle_obj = Angle(knee,hip,shoulder)
            back_angle = back_angle_obj.calculate_angle()
            
            squat_obj = Rating(ankle,knee,hip)
            squat_percentage = squat_obj.calculate_rating_squat()
            
            back_obj = Rating(knee,hip,shoulder)
            back_percentage = back_obj.calculate_rating_squat_back()
            
            # Display Green or Red Message
            if squat_percentage >= 80 and back_percentage >= 80:
                cv2.putText(image, "Perfect!",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 2, cv2.LINE_AA
                                )
            elif squat_percentage < 80 and squat_angle > 77:
                cv2.putText(image, "Lower your hip",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            elif squat_percentage < 80 and squat_angle < 77:
                cv2.putText(image, "Raise your hip",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            elif back_percentage < 80 and back_angle < 68:
                cv2.putText(image, "Move your back backward",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            elif back_percentage < 80 and back_angle > 68:
                cv2.putText(image, "Move your back forward",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            
            if squat_percentage > 80 and back_percentage > 80:
                stage = 'down'
            if squat_angle > 90 and stage == 'down':
                stage = 'up'
                cnt += 1
        
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Leg Accuarcy
            cv2.putText(image, 'Leg Accuarcy', (25,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(squat_percentage, 2)) + '%', 
                        (25,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
            
            # Display Back Accuarcy
            cv2.putText(image, 'Back Accuarcy', (175,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                        (175,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
            
            # Curl Stage data
            cv2.putText(image, 'Squat Stage', (325,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (325,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Repetitions
            cv2.putText(image, 'Repetitions', (475,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(cnt), 
                        (475,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

def squatsLeftMonitor(frame):
    # Squats Left

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 

        cnt = 0
        stage = None

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_squat_left(image, results)

        try:

            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        except Exception as e:
            print(e)
        
        # Extract landmarks
        try:
                
            # Instantiate objects
            squat_angle_obj = Angle(ankle,knee,hip)
            squat_angle = squat_angle_obj.calculate_angle()
            
            back_angle_obj = Angle(knee,hip,shoulder)
            back_angle = back_angle_obj.calculate_angle()
            
            squat_obj = Rating(ankle,knee,hip)
            squat_percentage = squat_obj.calculate_rating_squat()
            
            back_obj = Rating(knee,hip,shoulder)
            back_percentage = back_obj.calculate_rating_squat_back()
            
            # Display Green or Red Message
            if squat_percentage >= 80 and back_percentage >= 70:
                cv2.putText(image, "Perfect!",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 2, cv2.LINE_AA
                                )
            elif squat_percentage < 80 and squat_angle > 77:
                cv2.putText(image, "Lower your hip",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            elif squat_percentage < 80 and squat_angle < 77:
                cv2.putText(image, "Raise your hip",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            elif back_percentage < 70 and back_angle < 68:
                cv2.putText(image, "Move your back backward",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            elif back_percentage < 70 and back_angle > 68:
                cv2.putText(image, "Move your back forward",
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )
            
            if squat_percentage >= 80 and back_percentage >= 70:
                stage = 'down'
            if squat_angle > 90 and stage == 'down':
                stage = 'up'
                cnt += 1
        
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Leg Accuarcy
            cv2.putText(image, 'Leg Accuarcy', (25,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(squat_percentage, 2)) + '%', 
                        (25,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
            
            # Display Back Accuarcy
            cv2.putText(image, 'Back Accuarcy', (175,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                        (175,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
            
            # Curl Stage data
            cv2.putText(image, 'Squat Stage', (325,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (325,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display Repetitions
            cv2.putText(image, 'Repetitions', (475,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(cnt), 
                        (475,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

def flamingoRightMonitor(frame):
    # Flamingo Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_flamingo_right(image, results)

        try:

            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
        except Exception as e:
            print(e)
        
        # Extract landmarks
        try:
            
            # Calculate percentage
            obj = Rating(ankle,knee,hip)
            angle = obj.calculate_angle()
            percentage = obj.calculate_rating_flamingo()
            
            if percentage >= 80 or angle < 34.53:
                cv2.putText(image, "Perfect!", 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA
                                )
            else:
                cv2.putText(image, "Please move your leg higher", 
                        tuple(np.multiply(knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                                )
        
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Accuarcy
            cv2.putText(image, 'Flamingo Accuarcy', (250,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (250,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

def flamingoLeftMonitor(frame):
    # Flamingo Left

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_flamingo_left(image, results)

        try:

            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        except Exception as e:
            print(e)
        
        # Extract landmarks
        try:
            
            # Calculate percentage
            obj = Rating(ankle,knee,hip)
            angle = obj.calculate_angle()
            percentage = obj.calculate_rating_flamingo()
            
            if percentage > 80 or angle < 34.53:
                cv2.putText(image, "Perfect!", 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA
                                )
            else:
                cv2.putText(image, "Please move your leg higher", 
                        tuple(np.multiply(knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                                )
        
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Accuarcy
            cv2.putText(image, 'Flamingo Accuarcy', (250,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (250,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

def splitsRightMonitor(frame):
    # Splits Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_side_splits_right(image, results)

        try:

            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        except Exception as e:
            print(e)
        
        # Extract landmarks
        try:
            
            # Calculate percentage
            obj_right_1 = Rating(right_knee,right_hip,left_hip)
            right_percentage_1 = obj_right_1.calculate_rating_splits()

            obj_right_2 = Rating(right_ankle,right_knee,right_hip)
            right_percentage_2 = obj_right_2.calculate_rating_splits()
            
            obj_left_1 = Rating(left_knee,left_hip,right_hip)
            left_percentage_1 = obj_left_1.calculate_rating_splits()
            
            percentage = (right_percentage_1 + right_percentage_2 + left_percentage_1)/3
            
            obj = Rating(right_knee,right_hip,left_shoulder)
            back_angle = obj.calculate_angle()
            back_percentage = obj.calculate_rating_splits_back()
            
            # Display Green or Red Message
            if percentage >= 85 and back_percentage >= 80:
                cv2.putText(image, "Perfect!", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA
                                )
            elif back_percentage < 80 and back_angle > 90:
                cv2.putText(image, "Move your back forward", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )
            elif back_percentage < 80 and back_angle < 90:
                cv2.putText(image, "Move your back backward", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )
            else:
                cv2.putText(image, "Lower your legs", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                                )
                    
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
            
            # Display Leg Accuarcy
            cv2.putText(image, 'Split Accuarcy', (200,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (200,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
                    
            # Display Back Accuarcy
            cv2.putText(image, 'Back Accuarcy', (400,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                        (400,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
                

def splitsLeftMonitor(frame):
    # Splits Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_side_splits_right(image, results)

        try:

            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        except Exception as e:
            print(e)
        
        # Extract landmarks
        try:

            # Calculate percentage
            obj_right = Rating(right_knee,right_hip,left_hip)
            right_percentage = obj_right.calculate_rating_splits()
            
            obj_left_1 = Rating(left_knee,left_hip,right_hip)
            left_percentage_1 = obj_left_1.calculate_rating_splits()

            obj_left_2 = Rating(left_ankle, left_knee, left_hip)
            left_percentage_2 = obj_left_2.calculate_rating_splits()
            
            percentage = (right_percentage + left_percentage_1 + left_percentage_2)/3
            
            obj = Rating(left_knee,left_hip,right_shoulder)
            back_angle = obj.calculate_angle()
            back_percentage = obj.calculate_rating_splits_back()
            
            # Display Green or Red Message
            if percentage >= 85 and back_percentage >= 80:
                cv2.putText(image, "Perfect!", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA
                                )
            elif back_percentage < 80 and back_angle > 90:
                cv2.putText(image, "Move your back forward", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )
            elif back_percentage < 80 and back_angle < 90:
                cv2.putText(image, "Move your back backward", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )
            else:
                cv2.putText(image, "Lower your legs", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                                )
                    
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Leg Accuarcy
            cv2.putText(image, 'Split Accuarcy', (200,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (200,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
                    
            # Display Back Accuarcy
            cv2.putText(image, 'Back Accuarcy', (400,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                        (400,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
                

def frontSplitsMonitor(frame):
    # Splits Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model

        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks_side_splits_right(image, results)

        try:

            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        except Exception as e:
            print(e)
        
        # Extract landmarks
        try:

            # Calculate percentage
            obj_right = Rating(right_knee,right_hip,left_hip)
            right_percentage = obj_right.calculate_rating_splits()
            
            obj_left = Rating(left_knee,left_hip,right_hip)
            left_percentage = obj_left.calculate_rating_splits()
            
            percentage = (right_percentage + left_percentage)/2
            
            obj_right = Rating(right_knee,right_hip,right_shoulder)
            obj_left = Rating(left_knee,left_hip,left_shoulder)
            back_right_angle = obj_right.calculate_angle()
            back_left_angle = obj_left.calculate_angle()
            back_angle = (back_right_angle + back_left_angle) / 2
            back_percentage = (obj_right.calculate_rating_splits_back() + obj_left.calculate_rating_splits_back())/2
            
            # Display Green or Red Message
            if percentage >= 80 and back_percentage >= 80:
                cv2.putText(image, "Perfect!", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA
                                )
            elif back_percentage < 80 and back_angle > 90:
                cv2.putText(image, "Move your back forward", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )
            elif back_percentage < 80 and back_angle < 90:
                cv2.putText(image, "Move your back backward", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                            )
            else:
                cv2.putText(image, "Lower your legs", 
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                                )
                    
            # Rectangle Setup on Top
            cv2.rectangle(image, (0,0), (1000,80), (245,117,16), -1)
            
            # Display Leg Accuarcy
            cv2.putText(image, 'Split Accuarcy', (200,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(percentage, 2)) + '%', 
                        (200,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)
                    
            # Display Back Accuarcy
            cv2.putText(image, 'Back Accuarcy', (400,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                        (400,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,250), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

play = [False] * 9

@routes.get('/')
async def launch(request):
    content = open(os.path.join(ROOT, "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

@routes.get('/index.html')
async def launch(request):
    content = open(os.path.join(ROOT, "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

@routes.get('/bicepcurlmonitor.html')
def launchBicepCurlMonitor(request):
    content = open(os.path.join(ROOT, "templates/bicepcurlmonitor.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

@routes.get('/squatsmonitor.html')
def launchSquatsMonitor(request):
    content = open(os.path.join(ROOT, "templates/squatsmonitor.html"), "r").read()
    play[0] = False
    return web.Response(content_type="text/html", text=content)

@routes.get('/flamingomonitor.html')
def launchFlamingoMonitor(request):
    content = open(os.path.join(ROOT, "templates/flamingomonitor.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

@routes.get('/splitsmonitor.html')
def launchSplitsMonitor(request):
    content = open(os.path.join(ROOT, "templates/splitsmonitor.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

@routes.get('/squatsmonitorright.html')
def launchSquatsMonitorRight(request):
    content = open(os.path.join(ROOT, "templates/squatsmonitorright.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[0] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/squatsmonitorleft.html')
def launchSquatsMonitorLeft(request):
    content = open(os.path.join(ROOT, "templates/squatsmonitorleft.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[1] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/bicepcurlmonitorright.html')
def launchBicepCurlMonitorRight(request):
    content = open(os.path.join(ROOT, "templates/bicepcurlmonitorright.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[2] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/bicepcurlmonitorleft.html')
def launchBicepCurlMonitorLeft(request):
    content = open(os.path.join(ROOT, "templates/bicepcurlmonitorleft.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[3] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/flamingomonitorright.html')
def launchFlamingoMonitorRight(request):
    content = open(os.path.join(ROOT, "templates/flamingomonitorright.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[4] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/flamingomonitorleft.html')
def launchFlamingoMonitorLeft(request):
    content = open(os.path.join(ROOT, "templates/flamingomonitorleft.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[5] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/frontsplitsmonitor.html')
def launchFrontSplitsMonitor(request):
    content = open(os.path.join(ROOT, "templates/frontsplitsmonitor.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[6] = True
    return web.Response(content_type="text/html", text=content)

@routes.get('/splitsmonitorright.html')
def launchSplitsMonitorRight(request):
    content = open(os.path.join(ROOT, "templates/splitsmonitorright.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[7] = True
    return web.Response(content_type="text/html", text=content)
    
@routes.get('/splitsmonitorleft.html')
def launchSplitsMonitorLeft(request):
    content = open(os.path.join(ROOT, "templates/splitsmonitorleft.html"), "r").read()
    for i in range(9):
        play[i] = False
    play[8] = True
    return web.Response(content_type="text/html", text=content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    STATIC_PATH = os.path.join(os.path.dirname(__file__), "static")    
    app.router.add_static('/static/', STATIC_PATH, name='static')
    app.add_routes(routes)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
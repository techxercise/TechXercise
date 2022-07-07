# Importing libraries
# testing
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, app, render_template, Response, request

mp_drawing = mp.solutions.drawing_utils # Drawing Utilites
mp_pose = mp.solutions.pose # Pose
mp_holistic = mp.solutions.holistic # Holistic
app = Flask(__name__) # Initialization

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

def bicepCurlRightMonitor(camera=False):
    # Right Arm

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        if camera == True:
            cap = cv2.VideoCapture(0)
            cnt = 0
            cnt_copy = 0
            curl_stage = None
            set_cnt = 0
            exit = 0
            # Set mediapipe model 
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                if ret:
                    if exit:
                        cap.release()
                        cv2.destroyAllWindows()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Accuarcy
                    cv2.putText(image, 'Accuarcy', (150,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(percentage, 2)) + '%', 
                                (150,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                    
                    # Curl Stage data
                    cv2.putText(image, 'Curl Stage', (450,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, curl_stage, 
                                (450,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display Set
                    cv2.putText(image, 'Set', (750,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(set_cnt), 
                                (750,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display Repetitions
                    cv2.putText(image, 'Repetitions', (1050,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(cnt), 
                                (1050,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return


def bicepCurlLeftMonitor(camera=False):
    # Left Arm

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        if camera == True:
            cap = cv2.VideoCapture(0)
            cnt = 0
            cnt_copy = 0
            curl_stage = None
            set_cnt = 0
            exit = 0
            # Set mediapipe model 
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                if ret:
                    if exit:
                        cap.release()
                        cv2.destroyAllWindows()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Accuarcy
                    cv2.putText(image, 'Accuarcy', (150,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(percentage, 2)) + '%', 
                                (150,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                    
                    # Curl Stage data
                    cv2.putText(image, 'Curl Stage', (450,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, curl_stage, 
                                (450,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display Set
                    cv2.putText(image, 'Set', (750,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(set_cnt), 
                                (750,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display Repetitions
                    cv2.putText(image, 'Repetitions', (1050,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(cnt), 
                                (1050,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return

def squatsRightMonitor(camera=False):
    # Squats Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 
        if camera == True:
            cap = cv2.VideoCapture(0)
            cnt = 0
            stage = None
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make Detections
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                draw_styled_landmarks_squat_right(image, results)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    
                    # Instantiate objects
                    squat_angle_obj = Angle(ankle,knee,hip)
                    squat_angle = squat_angle_obj.calculate_angle()
                    
                    back_angle_obj = Angle(ankle,knee,hip)
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
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Leg Accuarcy
                    cv2.putText(image, 'Leg Accuarcy', (150,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(squat_percentage, 2)) + '%', 
                                (150,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                    
                    # Display Back Accuarcy
                    cv2.putText(image, 'Back Accuarcy', (450,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                                (450,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                    
                    # Curl Stage data
                    cv2.putText(image, 'Squat Stage', (750,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (750,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display Repetitions
                    cv2.putText(image, 'Repetitions', (1050,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(cnt), 
                                (1050,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return

def squatsLeftMonitor(camera=False):
    # Squats Left

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 
        if camera == True:
            cap = cv2.VideoCapture(0)
            cnt = 0
            stage = None
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make Detections
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                draw_styled_landmarks_squat_left(image, results)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        
                    # Instantiate objects
                    squat_angle_obj = Angle(ankle,knee,hip)
                    squat_angle = squat_angle_obj.calculate_angle()
                    
                    back_angle_obj = Angle(ankle,knee,hip)
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
                    
                    if squat_percentage >= 80 and back_percentage >= 80:
                        stage = 'down'
                    if squat_angle > 90 and stage == 'down':
                        stage = 'up'
                        cnt += 1
                
                    # Rectangle Setup on Top
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Leg Accuarcy
                    cv2.putText(image, 'Leg Accuarcy', (150,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(squat_percentage, 2)) + '%', 
                                (150,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                    
                    # Display Back Accuarcy
                    cv2.putText(image, 'Back Accuarcy', (450,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                                (450,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                    
                    # Curl Stage data
                    cv2.putText(image, 'Squat Stage', (750,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (750,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display Repetitions
                    cv2.putText(image, 'Repetitions', (1050,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(cnt), 
                                (1050,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return

def flamingoRightMonitor(camera=False):
    # Flamingo Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 
        if camera == True:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make Detections
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                draw_styled_landmarks_flamingo_right(image, results)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
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
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Accuarcy
                    cv2.putText(image, 'Flamingo Accuarcy', (575,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(percentage, 2)) + '%', 
                                (575,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                
                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return

def flamingoLeftMonitor(camera=False):
    # Flamingo Left

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model 
        if camera == True:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make Detections
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                draw_styled_landmarks_flamingo_left(image, results)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
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
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Accuarcy
                    cv2.putText(image, 'Flamingo Accuarcy', (575,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(percentage, 2)) + '%', 
                                (575,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                
                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return

def splitsRightMonitor(camera=False):
    # Splits Right

    # Access mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set mediapipe model
        if camera == True:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make Detections
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                draw_styled_landmarks_side_splits_right(image, results)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    # Calculate percentage
                    obj_right = Rating(right_knee,right_hip,left_hip)
                    right_percentage = obj_right.calculate_rating_splits()
                    
                    obj_left = Rating(left_knee,left_hip,right_hip)
                    left_percentage = obj_left.calculate_rating_splits()
                    
                    percentage = (right_percentage + left_percentage)/2
                    
                    obj = Rating(right_knee,right_hip,right_shoulder)
                    back_angle = obj.calculate_angle()
                    back_percentage = obj.calculate_rating_splits_back()
                    
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
                    cv2.rectangle(image, (0,0), (10000,80), (245,117,16), -1)
                    
                    # Display Leg Accuarcy
                    cv2.putText(image, 'Split Accuarcy', (300,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(percentage, 2)) + '%', 
                                (300,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)
                            
                    # Display Back Accuarcy
                    cv2.putText(image, 'Back Accuarcy', (800,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(back_percentage, 2)) + '%', 
                                (800,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,127,250), 2, cv2.LINE_AA)

                    # Show to screen
                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except AttributeError:
                    print("Attribute Error")
        else:
            return

conditions = [False] * 9

@app.route('/')
def launch():
    bicepCurlRightMonitor(False)
    bicepCurlLeftMonitor(False)
    squatsRightMonitor(False)
    squatsLeftMonitor(False)
    flamingoRightMonitor(False)
    flamingoLeftMonitor(False)
    splitsRightMonitor(False)
    #splitsLeftMonitor(False)
    #frontSplitsMonitor(False)
    return app.send_static_file('index.html')

@app.route('/index.html')
def index():
    bicepCurlRightMonitor(False)
    bicepCurlLeftMonitor(False)
    squatsRightMonitor(False)
    squatsLeftMonitor(False)
    flamingoRightMonitor(False)
    flamingoLeftMonitor(False)
    splitsRightMonitor(False)
    #splitsLeftMonitor(False)
    #frontSplitsMonitor(False)
    return app.send_static_file('index.html')

@app.route('/bicepcurlmonitor.html')
def launchBicepCurlMonitor():
    return app.send_static_file('bicepcurlmonitor.html')

@app.route('/squatsmonitor.html')
def launchSquatsMonitor():
    return app.send_static_file('squatsmonitor.html')

@app.route('/flamingomonitor.html')
def launchFlamingoMonitor():
    return app.send_static_file('flamingomonitor.html')

@app.route('/splitsmonitor.html')
def launchSplitsMonitor():
    return app.send_static_file('splitsmonitor.html')

@app.route('/bicepcurlmonitorright.html/camera=False')
def launchBicepCurlMonitorRightOff():
    conditions[2] = False
    return render_template('bicepcurlmonitorright.html')

@app.route('/bicepcurlmonitorright.html/camera=True')
def launchBicepCurlMonitorRightOn():
    conditions[2] = True
    return render_template('bicepcurlmonitorright.html')

@app.route('/bicepcurlmonitorleft.html/camera=False')
def launchBicepCurlMonitorLeftOff():
    conditions[3] = False
    return render_template('bicepcurlmonitorleft.html')

@app.route('/bicepcurlmonitorleft.html/camera=True')
def launchBicepCurlMonitorLeftOn():
    conditions[3] = True
    return render_template('bicepcurlmonitorleft.html')

@app.route('/squatsmonitorright.html/camera=False')
def launchSquatsMonitorRightOff():
    conditions[0] = False
    return render_template('squatsmonitorright.html')

@app.route('/squatsmonitorright.html/camera=True')
def launchSquatsMonitorRightOn():
    conditions[0] = True
    return render_template('squatsmonitorright.html')

@app.route('/squatsmonitorleft.html/camera=False')
def launchSquatsMonitorLeftOff():
    conditions[1] = False
    return render_template('squatsmonitorleft.html')

@app.route('/squatsmonitorleft.html/camera=True')
def launchSquatsMonitorLeftOn():
    conditions[1] = True
    return render_template('squatsmonitorleft.html')

@app.route('/flamingomonitorright.html/camera=False')
def launchFlamingoMonitorRightOff():
    conditions[7] = False
    return render_template('flamingomonitorright.html')

@app.route('/flamingomonitorright.html/camera=True')
def launchFlamingoMonitorRightOn():
    conditions[7] = True
    return render_template('flamingomonitorright.html')

@app.route('/flamingomonitorleft.html/camera=False')
def launchFlamingoMonitorLeftOff():
    conditions[8] = False
    return render_template('flamingomonitorleft.html')

@app.route('/flamingomonitorleft.html/camera=True')
def launchFlamingoMonitorLeftOn():
    conditions[8] = True
    return render_template('flamingomonitorleft.html')

@app.route('/frontsplitsmonitor.html/camera=False')
def launchFrontSplitsMonitorOff():
    conditions[4] = False
    return render_template('frontsplitsmonitor.html')

@app.route('/frontsplitsmonitor.html/camera=True')
def launchFrontSplitsMonitorOn():
    conditions[4] = True
    return render_template('frontsplitsmonitor.html')

@app.route('/splitsmonitorright.html/camera=False')
def launchSplitsMonitorRightOff():
    conditions[5] = False
    return render_template('splitsmonitorright.html')
    
@app.route('/splitsmonitorright.html/camera=True')
def launchSplitsMonitorRightOn():
    conditions[5] = True
    return render_template('splitsmonitorright.html')

@app.route('/splitsmonitorleft.html/camera=False')
def launchSplitsMonitorLeftOff():
    conditions[6] = False
    return render_template('splitsmonitorleft.html')

@app.route('/splitsmonitorleft.html/camera=True')
def launchSplitsMonitorLeftOn():
    conditions[6] = True
    return render_template('splitsmonitorleft.html')

@app.route('/bicepcurlrightvideo')
def bicepCurlRightVideo():
    return Response(bicepCurlRightMonitor(conditions[2]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bicepcurlleftvideo')
def bicepCurlLeftVideo():
    return Response(bicepCurlLeftMonitor(conditions[3]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squatsrightvideo')
def squatsRightVideo(): 
    return Response(squatsRightMonitor(conditions[0]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squatsleftvideo')
def squatsLeftVideo():
    return Response(squatsLeftMonitor(conditions[1]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/flamingorightvideo')
def flamingoRightVideo():
    return Response(flamingoRightMonitor(conditions[7]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/flamingoleftvideo')
def flamingoLeftVideo():
    return Response(flamingoRightMonitor(conditions[8]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frontsplitsvideo')
def frontSplitsVideo():
    return Response(frontSplitsMonitor(conditions[4]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/splitsrightvideo')
def splitsRightVideo():
    return Response(splitsRightMonitor(conditions[5]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/splitsleftvideo')
def splitsLeftVideo():
    return Response(splitsLeftMonitor(conditions[6]), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

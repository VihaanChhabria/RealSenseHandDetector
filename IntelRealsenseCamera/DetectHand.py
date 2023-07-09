import cv2
import mediapipe as mp
import imutils

import pyrealsense2 as rs
import numpy as np

import threading
import queue

import time

"""class HandDetection:
    mpHands = mp.solutions.hands # MP hand solution
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils # Draws the lines of hand joints
    
    def __init__(self):
        self.colorImage = None
        self.depthColorMap = None
        self.results = None

    def RunHandDetection(self, frameSupplier):

        while True:
            self.images = frameSupplier.get()
            self.colorImage = self.images[0]
            self.depthColorMap = self.images[1]

            self.ProcessImage()
            self.DrawHandConnections()

            self.images = np.hstack((self.colorImage, self.depthColorMap))
            
            # Displaying the output
            cv2.imshow("Hand tracker", self.images)

            # Program terminates when q key is pressed
            if cv2.waitKey(1) == ord('q'):
                self.pipeline.stop()
                cv2.destroyAllWindows()

    # Processing the input image
    def ProcessImage(self):
        # Converting the input to grayscale
        gray_image = cv2.cvtColor(self.colorImage, cv2.COLOR_BGR2RGB)
        self.results = HandDetection.hands.process(gray_image)

    # Drawing landmark connections
    def DrawHandConnections(self):
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = self.colorImage.shape
                    
                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Creating a circle around each landmark
                    cv2.circle(self.colorImage, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    cv2.circle(self.depthColorMap, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    # Drawing the landmark connections
                    HandDetection.mpDraw.draw_landmarks(self.colorImage, handLms,
                                        self.mpHands.HAND_CONNECTIONS)
                    HandDetection.mpDraw.draw_landmarks(self.depthColorMap, handLms,
                                        self.mpHands.HAND_CONNECTIONS)

        return self.colorImage, self.depthColorMap

            #pointer_finger = self.results.multi_hand_landmarks[0].landmark[8]
            #middle_finger = self.results.multi_hand_landmarks[0].landmark[2]
            #return [min(int(pointer_finger.x* w), 639), min(int(pointer_finger.y* h), 479)], [min(int(middle_finger.x* w), 639), min(int(middle_finger.y* h), 479)]
"""

class Camera_Feed:

    def __init__(self):
        self.mpHands = mp.solutions.hands # MP hand solution
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils # Draws the lines of hand joints
        self.status = True

        self.pipeline = None
        self.frames = None
        self.alignFunction = None
        self.alignedFrames = None

        self.depthFrame = None
        self.depthData = None
        
        self.depthColorMap = None
        self.depthColorMapDim = (0, 0, 0)

        self.colorFrame = None
        self.colorImage = None

    def CameraFeed(self):
        self._ReadyRealsense()
    
        while self.status:
            self.frames = self.pipeline.wait_for_frames()
            self.alignFunction = rs.align(rs.stream.color)
            self.alignedFrames = self.alignFunction.process(self.frames)

            self.depthFrame = self.alignedFrames.get_depth_frame()
            self.depthData = np.asanyarray(self.depthFrame.get_data())
            self.depthColorMap = cv2.applyColorMap(cv2.convertScaleAbs(self.depthData, alpha=0.03), cv2.COLORMAP_JET)
            self.depthColorMapDim = self.depthColorMap.shape

            self.colorFrame = self.alignedFrames.get_color_frame()
            self.colorImage = np.asanyarray(self.colorFrame.get_data())
            self.colorImage = cv2.resize(self.colorImage, dsize=(self.depthColorMapDim[1], self.depthColorMapDim[0]), interpolation=cv2.INTER_AREA)


            if (not self.colorFrame) or (not self.depthFrame):
                continue

    
    def _ReadyRealsense(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

    def ReturnFrame(self):
        try:
            self.ProcessImage()
            self.DrawHandConnections()
        except:
            pass

        return (self.colorImage, self.depthColorMap)
    
    def EndProcess(self):
        self.status = False

    # Processing the input image
    def ProcessImage(self):
        # Converting the input to grayscale
        gray_image = cv2.cvtColor(self.colorImage, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(gray_image)

    # Drawing landmark connections
    def DrawHandConnections(self):
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = self.colorImage.shape
                    
                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Creating a circle around each landmark
                    cv2.circle(self.colorImage, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    cv2.circle(self.depthColorMap, (cx, cy), 10, (0, 255, 0),
                            cv2.FILLED)
                    # Drawing the landmark connections
                    self.mpDraw.draw_landmarks(self.colorImage, handLms,
                                        self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(self.depthColorMap, handLms,
                                        self.mpHands.HAND_CONNECTIONS)

        return self.colorImage, self.depthColorMap

def main():

    cameraFeedProcess = Camera_Feed()

    #cameraFeedProcess.GetCameraFeed(frameSupplier)
    #handDetectionProcess.RunHandDetection(frameSupplier)

    cameraThread = threading.Thread(target=cameraFeedProcess.CameraFeed)
    cameraThread.start()

    prev_frame_time = 0
    
    new_frame_time = 0

    #avg_fps = []


    while True:
        colorImage, depthColorMap = cameraFeedProcess.ReturnFrame()

        images = np.hstack((colorImage, depthColorMap))
        
        try:
            if (colorImage == None) or (depthColorMap == None):
                continue
        except:
            pass

        new_frame_time = time.time()

        fps = 1/(new_frame_time-prev_frame_time)
        #avg_fps.append(fps)
        prev_frame_time = new_frame_time

        avg_fps_num = 0

        #for fps_past in avg_fps:
        #    avg_fps_num = fps_past + avg_fps_num

        #avg_fps_num = avg_fps_num/len(avg_fps)

        #cv2.putText(colorImage, str(int(avg_fps_num)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        print(fps)

        # Displaying the output
        cv2.imshow("Hand tracker", images)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cameraFeedProcess.EndProcess()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
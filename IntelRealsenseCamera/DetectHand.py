import cv2
import mediapipe as mp
import imutils

import pyrealsense2 as rs
import numpy as np

class HandDetection:
    mpHands = mp.solutions.hands # MP hand solution
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils # Draws the lines of hand joints
    
    def __init__(self, _colorImage, _depthColormap):
        self.colorImage = _colorImage
        self.depthColorMap = _depthColormap
        self.results = None

    # Processing the input image
    def process_image(self):
        # Converting the input to grayscale
        gray_image = cv2.cvtColor(self.colorImage, cv2.COLOR_BGR2RGB)
        self.results = HandDetection.hands.process(gray_image)

    # Drawing landmark connections
    def draw_hand_connections(self):
        
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
        
def main():

    pipeline = ready_realsense()
    stable_point_found = 0

    while True:
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)


        if (not color_frame) or (not depth_frame):
            continue

        handProcess = HandDetection(color_image, depth_colormap)

        handProcess.process_image()

        color_image, depth_colormap = handProcess.draw_hand_connections()

        images = np.hstack((color_image, depth_colormap))

        # Displaying the output
        cv2.imshow("Hand tracker", images)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            pipeline.stop()
            cv2.destroyAllWindows()

def click_mouse(depth_image, pointer_coord, middle_coord, depth_colormap_dim, stable_point):
    if depth_image == [0, 0]:
        return None
    #pointer_depth = depth_image[pointer_coord[1], pointer_coord[0]]
    #middle_depth = depth_image[middle_coord[1], middle_coord[0]]

    #print("pointer_depth:", pointer_depth)
    #print("middle_depth:", middle_depth)
    #print(depth_image[0, 0])

    #stable_point = depth_image[0, 0]

    for y in range(depth_colormap_dim[1]):
        for x in range(depth_colormap_dim[0]):
            depth = 0
            try:
                depth = depth_image[y, x]
            except:
                continue
            if depth-20 > stable_point:
                cv2.circle(depth_image, (x, y), 5, (255, 255, 0), -1)
                print(stable_point)
                print(depth, x, y)
                print("in")
                return

def ready_realsense():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    return pipeline

if __name__ == "__main__":
    main()
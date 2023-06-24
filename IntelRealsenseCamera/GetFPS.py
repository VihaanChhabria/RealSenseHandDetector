import cv2
import mediapipe as mp
import imutils

import pyrealsense2 as rs
import numpy as np

import time

mpHands = mp.solutions.hands # MP hand solution
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # Draws the lines of hand joints



# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

# Drawing landmark connections
def draw_hand_connections(color_image, depth_colormap, results):
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = color_image.shape
                
                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Creating a circle around each landmark
                cv2.circle(color_image, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                cv2.circle(depth_colormap, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(color_image, handLms,
                                      mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(depth_colormap, handLms,
                                      mpHands.HAND_CONNECTIONS)

        pointer_finger = results.multi_hand_landmarks[0].landmark[8]
        middle_finger = results.multi_hand_landmarks[0].landmark[2]
        return [min(int(pointer_finger.x* w), 639), min(int(pointer_finger.y* h), 479)], [min(int(middle_finger.x* w), 639), min(int(middle_finger.y* h), 479)]
    


def main():

    pipeline = ready_realsense()

    prev_frame_time = 0
    
    new_frame_time = 0

    avg_fps = []

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
    
        results = process_image(color_image)
        draw_hand_connections(color_image, depth_colormap, results)

        new_frame_time = time.time()

        fps = 1/(new_frame_time-prev_frame_time)
        avg_fps.append(fps)
        prev_frame_time = new_frame_time

        avg_fps_num = 0

        for fps_past in avg_fps:
            avg_fps_num = fps_past + avg_fps_num

        avg_fps_num = avg_fps_num/len(avg_fps)

        cv2.putText(color_image, str(int(avg_fps_num)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        print(fps)

        images = np.hstack((color_image, depth_colormap))

        # Displaying the output
        cv2.imshow("Hand tracker", images)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            pipeline.stop()
            cv2.destroyAllWindows()

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
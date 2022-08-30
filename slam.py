import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def displayVideo():
    
    #Initialize Capture and Feature objects
    cap = cv.VideoCapture('drive.mp4')
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    prev_frame = None
    prev_des = None
    prev_kp = None
    while(cap.isOpened()):

        # Read every video frame with cap.read()
        ret, frame = cap.read()
        if frame is None:
            break
        
        # Detect good features with Shi-Tomasi Corner Detector
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray_frame, 1000, 0.01, 3) 
        kp = corners_to_keypoints(corners)
        kp, des = orb.compute(frame, kp)
        
        # Match frame to frame features
        if prev_des is not None:
            matches = bf.match(des, prev_des)
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw features and matches on image
            for keypoint in kp:
                x, y = keypoint.pt
                cv.circle(frame, (int(x), int(y)), radius=2, color=(0, 255, 0))
                



        # Display video frame
        cv.imshow('frame', frame)

        # Wait for q key from user to exit video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
       
        prev_frame = frame.copy()
        prev_des = des.copy()
        prev_kp = kp

    cap.release()
    cv.destroyAllWindows()

def corners_to_keypoints(corners):
	if corners is None: 
		keypoints = []
	else:
		keypoints = [cv.KeyPoint(kp[0][0], kp[0][1], 20) for kp in corners]

	return keypoints


if __name__ == '__main__':
    displayVideo()
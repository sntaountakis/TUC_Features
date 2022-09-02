import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform



# Define camera intrinsic matrix K
F = 90 #or 90?
WIDTH_CENTER = 1280 // 2
HEIGHT_CENTER = 720 // 2
K = np.array([[F, 0, WIDTH_CENTER], [0, F, HEIGHT_CENTER], [0, 0, 1]])
K_INV = np.linalg.inv(K)

def displayVideo():
    
    #Initialize Capture and Feature objects
    cap = cv.VideoCapture('drive.mp4')
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    prev_frame = None
    prev_des = None
    prev_kp = None
    matched_frame = None
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

            # Filter matches with Fundamental Matrix
            matches = fundamental_matrix(prev_kp, kp, matches)

            # Draw features and matches on image
            matched_frame = cv.drawMatches(frame, kp, prev_frame, 
                            prev_kp, matches, None,
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            '''
            for keypoint in kp:
                x, y = keypoint.pt
                cv.circle(frame, (int(x), int(y)), radius=2, color=(0, 255, 0))
            ''' 
            cv.imshow('frame', matched_frame)

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

#f_avg = 0
#f_est = []
def fundamental_matrix(prev_kp, kp, matches):
    ret = []
    for m in matches:
        kp1 = prev_kp[m.trainIdx].pt
        kp2 = kp[m.queryIdx].pt
        ret.append([kp1, kp2])

    ret = np.array(ret)

    ret[:, 0, :] = normalize(ret[:, 0, :])
    ret[:, 1, :] = normalize(ret[:, 1, :])

    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=100)
    matches = np.array(matches)
    
    Rt = getRtMatrix(model.params)

    #s,v,d = np.linalg.svd(model.params)
    #f_est.append(np.sqrt(2)/((v[0]+v[1])/2))
    #f_avg = np.median(f_est)
    print(Rt)

    return matches[inliers]

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(pts):
    return np.dot(K_INV, add_ones(pts).T).T[:, 0:2]

def decalibrate():
    return

def getRtMatrix(E):
	W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
	U,d,Vt = np.linalg.svd(E)
	assert np.linalg.det(U) > 0
	if np.linalg.det(Vt) < 0:
		Vt *= -1.0
	R = np.dot(np.dot(U, W), Vt)
	if np.sum(R.diagonal()) < 0:
		R = np.dot(np.dot(U, W.T), Vt)
	t = U[:, 2]
	ret = np.eye(4)
	ret[:3, :3] = R
	ret[:3, 3] = t
	return ret


if __name__ == '__main__':
    displayVideo()

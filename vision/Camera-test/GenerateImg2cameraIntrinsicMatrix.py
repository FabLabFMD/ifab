import os

import cv2
import cv2 as cv
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras(cv2.CAP_V4L2):
    print(f'{camera_info.index}: {camera_info.name}')

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

frames = []
os.makedirs("logitec", exist_ok=True)

while len(objpoints) < 10:
    ret, frame = cam.read()

    # Write the frame to the output file
    # out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('k'):
        frames.append(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.imwrite(f'logitec/frame_{len(objpoints)}.png', frame)

            # Draw and display the corners
            cv.drawChessboardCorners(frame, (7, 6), corners2, ret)
            cv.imshow('img', frame)
            cv.waitKey(500)
        print(f"Captured frame {len(objpoints)}")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

cv2.waitKey(0)

cv.destroyAllWindows()

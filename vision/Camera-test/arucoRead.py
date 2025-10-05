import cv2
import pickle
import numpy as np

def generateCamera(index: int = 0) -> tuple:
    """
    Detect the index of the camera.
    :return: Camera index
    """
    # Open the default camera
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980);
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024);

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Default dimensions for cv2.VideoCapture is 640x480. Switch to a 16:9 aspect ratio to cover the whole field
    return cam, frame_width, frame_height

def frameRemap(frameOriginal, mtx, dist):
    # cv2.imshow('Camera_original', frameOriginal)
    h,  w = frameOriginal.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort with remapping
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(frameOriginal, mapx, mapy, cv2.INTER_LINEAR)
    
    # crop the image
    x, y, w, h = roi
    frame = dst[y:y+h, x:x+w]
    return frame


if __name__ == '__main__':

    # Carichiamo da file i dati di calibrazione intrisechi della camera, genrati con "generateInstrinsic.py"
    with open('logitec/calib_data_logitec.pkl', 'rb') as f:
        calib_data = pickle.load(f)

    mtx = calib_data['camera_matrix']
    dist = calib_data['dist_coeff']
    rvecs = calib_data['rvecs']
    tvecs = calib_data['tvecs']

    cam, frame_width, frame_height = generateCamera(0)
    while True:
        ret, frame_cam = cam.read()
        frame = frameRemap(frame_cam, mtx, dist)

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        # Print the detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                # Stampa le coordinate dei marker
                pointList = str(corners[i]).replace('\n', '')
                print(f'Marker ID: {ids[i][0]}\tCoordinates: {pointList}')
            print("-" * 60)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()

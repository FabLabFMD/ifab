import cv2


def generateCamera(index: int = 0) -> tuple:
    """
    Detect the index of the camera.
    :return: Camera index
    """
    # Open the default camera
    cam = cv2.VideoCapture(index)
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cam, frame_width, frame_height


if __name__ == '__main__':

    cam, frame_width, frame_height = generateCamera(0)
    while True:
        ret, frame = cam.read()

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

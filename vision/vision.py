import atexit
import signal
import sys
import tkinter as tk
from typing import Callable, Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pickle

from mergedeep import merge

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

class ArUcoDetector:
    """Handles ArUco marker detection and related operations."""

    def __init__(self, aruco_dict_type: int = cv2.aruco.DICT_6X6_250):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_markers(self, frame: np.ndarray) -> Tuple[List, Optional[np.ndarray], List]:
        """Detects ArUco markers in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMarkers(gray)


class PerspectiveTransformer:
    """Handles perspective transformation operations."""

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Orders points in [top-left, top-right, bottom-right, bottom-left] order."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top_Left                Top_right
        #   (0,0) --------------- (X_max,0)
        #     |                       |
        #     |                       |
        # (0,Y_max) ------------(X_max,Y_max)
        # Bottom_left             Bottom_right

        # La somma delle coordinate (x+y) è minima nell'angolo in alto a sinistra
        # e massima nell'angolo in basso a destra
        s = pts.sum(axis=1)
        tl_idx = np.argmin(s)  # Top-left
        br_idx = np.argmax(s)  # Bottom-right
        rect[0] = pts[tl_idx]
        rect[2] = pts[br_idx]

        # Trova i due punti rimanenti
        remaining_indices = [i for i in range(len(pts)) if i != tl_idx and i != br_idx]
        remaining_points = pts[remaining_indices]

        # Il punto top-right ha X maggiore e Y minore tra i due rimasti
        if remaining_points[0][0] > remaining_points[1][0] and remaining_points[0][1] < remaining_points[1][1]:
            rect[1] = remaining_points[0]  # Top-right
            rect[3] = remaining_points[1]  # Bottom-left
        elif remaining_points[1][0] > remaining_points[0][0] and remaining_points[1][1] < remaining_points[0][1]:
            rect[1] = remaining_points[1]  # Top-right
            rect[3] = remaining_points[0]  # Bottom-left
        else:
            # Se la condizione ideale non è soddisfatta, usa il punto con X maggiore come top-right
            tr_idx = np.argmax(remaining_points[:, 0])
            rect[1] = remaining_points[tr_idx]
            rect[3] = remaining_points[1 - tr_idx]
        return rect

    @staticmethod
    def transform_perspective(frame: np.ndarray, corners: np.ndarray, wrapped_output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Applies perspective transform to the frame based on detected corners."""

        if corners is None:
            raise ValueError("Corners are None")
        if len(corners) != 4:
            raise ValueError(f"Corners are less than 4. ({len(corners)})")

        # Get ordered source points
        src_pts = PerspectiveTransformer.order_points(corners.astype(np.float32))

        def is_collinear(p1, p2, p3, epsilon=1e-5):
            # Calcola l'area del triangolo formato dai tre punti
            # Se l'area è (quasi) zero, i punti sono collineari
            area = abs((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1]))
            return area < epsilon

        # Controlla che nessun gruppo di 3 punti sia collineare
        if (is_collinear(src_pts[0], src_pts[1], src_pts[2]) or
                is_collinear(src_pts[0], src_pts[1], src_pts[3]) or
                is_collinear(src_pts[0], src_pts[2], src_pts[3]) or
                is_collinear(src_pts[1], src_pts[2], src_pts[3])):
            raise ValueError("I punti di origine sono collineari e non possono definire una trasformazione prospettica")

        # Define destination points for the transform
        dst_pts = np.array([
            [0, 0],
            [wrapped_output_size[0], 0],
            [wrapped_output_size[0], wrapped_output_size[1]],
            [0, wrapped_output_size[1]]
        ], dtype=np.float32)

        # Verifica presenza di NaN o infiniti
        if np.any(np.isnan(src_pts)) or np.any(np.isnan(dst_pts)):
            raise ValueError("I punti contengono valori NaN")
        if np.any(np.isinf(src_pts)) or np.any(np.isinf(dst_pts)):
            raise ValueError("I punti contengono valori infiniti")

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply perspective transform
        warped = cv2.warpPerspective(frame, matrix, wrapped_output_size)

        return warped, matrix


class MarkerPoseCalculator:
    """Calculates poses of ArUco markers in the transformed space."""

    def __init__(self, width: float, height: float, output_size: Tuple[int, int]):
        self.width = width
        self.height = height
        self.output_size = output_size
        self.scale_x = self.width / self.output_size[0]
        self.scale_y = self.height / self.output_size[1]

    def calculate_marker_pose(self, current_marker_corners: np.ndarray, matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Calculates the transformed position (cm) and angle (radians) of a marker."""
        # Calculate center of the marker in original frame

        center_x = np.mean(current_marker_corners[:, 0])
        center_y = np.mean(current_marker_corners[:, 1])
        marker_center_orig = np.array([center_x, center_y, 1.0])

        # Transform the center point to the warped coordinate system
        transformed_center = np.matmul(matrix, marker_center_orig)
        transformed_x_px = transformed_center[0] / transformed_center[2]
        transformed_y_px = transformed_center[1] / transformed_center[2]

        # Convert pixel coordinates to cm from top-left of warped image
        pos_x = (transformed_x_px) * self.scale_x
        pos_y = self.height - ((transformed_y_px) * self.scale_y)  # Invertiamo l'asse y per ottenere un sistema di riferimento destro (origine in basso)

        # Calculate the orientation angle of the marker in the original frame
        v_orientation = current_marker_corners[1] - current_marker_corners[0]
        angle_rad_orig = np.arctan2(v_orientation[1], v_orientation[0])

        # To find the transformed angle, transform two points along the orientation vector
        p1_orig = np.array([center_x, center_y, 1.0])
        p2_orig = np.array([center_x + np.cos(angle_rad_orig) * 20,
                            center_y + np.sin(angle_rad_orig) * 20, 1.0])

        # Transform both points
        tp1 = np.matmul(matrix, p1_orig)
        tp2 = np.matmul(matrix, p2_orig)

        # Normalize the transformed points (perspective divide)
        tp1_norm = tp1 / tp1[2]
        tp2_norm = tp2 / tp2[2]

        # Calcola l'angolo invertendo la direzione dell'asse Y per mantenere
        # coerenza con il sistema di riferimento destro (origine in basso)
        # usato per le coordinate
        transformed_angle_rad = np.arctan2(-(tp2_norm[1] - tp1_norm[1]), tp2_norm[0] - tp1_norm[0])

        return pos_x, pos_y, transformed_angle_rad, transformed_x_px, transformed_y_px

    @staticmethod
    def apply_offset(x: float, y: float, angle_rad: float, config: Dict[str, Any]) -> Tuple[float, float, float]:
        """Applies configured offset to a pose."""
        offset_x = config.get("x_offset", 0.0)
        offset_y = config.get("y_offset", 0.0)
        offset_theta = np.deg2rad(config.get("theta_offset", 0.0))

        # Rotate offset vector by marker angle and add to position
        final_x = x + offset_x * np.cos(angle_rad) - offset_y * np.sin(angle_rad)
        final_y = y + offset_x * np.sin(angle_rad) + offset_y * np.cos(angle_rad)

        # Add angle offset and normalize to [-pi, pi]
        final_angle = (angle_rad + offset_theta + 2 * np.pi) % (2 * np.pi)
        if final_angle > np.pi:
            final_angle -= 2 * np.pi
        return final_x, final_y, final_angle


class Visualizer:
    """Handles visualization of detected markers and transformed views."""

    @staticmethod
    def draw_quadrilateral(frame: np.ndarray, marker_centers: Dict[int, Tuple[float, float]]) -> np.ndarray:
        """Draws the quadrilateral connecting marker centers."""
        display_frame = frame.copy()

        if len(marker_centers) > 1:
            sorted_indices = sorted(marker_centers.keys())
            for i in range(len(sorted_indices)):
                pt1_idx = sorted_indices[i]
                pt2_idx = sorted_indices[(i + 1) % len(sorted_indices)]

                if pt1_idx in marker_centers and pt2_idx in marker_centers:
                    pt1 = tuple(map(int, marker_centers[pt1_idx]))
                    pt2 = tuple(map(int, marker_centers[pt2_idx]))
                    cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

        return display_frame

    @staticmethod
    def draw_marker_info(warped: np.ndarray, marker_id: int,
                         pos_x: float, pos_y: float, angle_rad: float,
                         trans_x_px: float, trans_y_px: float,
                         # target_x: float, target_y: float, target_theta: float,
                         robot_config: Dict[str, Any], macchinari_id_to_key: Dict[int, str]) -> np.ndarray:
        """Draws marker information on the warped image."""
        angle_deg = np.degrees(angle_rad)

        # Draw center
        center_px = (int(trans_x_px), int(trans_y_px))
        cv2.circle(warped, center_px, 5, (0, 0, 255), -1)

        # # Draw target position
        # target_px = (int(target_x * 100), int(target_y * 100))
        # cv2.circle(warped, target_px, 3, (255, 0, 0), -1)

        # Draw orientation line
        line_length = 30
        end_x = int(trans_x_px + line_length * np.cos(angle_rad))
        end_y = int(trans_y_px + line_length * np.sin(-angle_rad))  # Inverti Y per via del sistema Sinistro in visualizzazione
        cv2.line(warped, center_px, (end_x, end_y), (0, 255, 0), 2)

        # # Draw target orientation line
        # line_end_length = 15
        # target_end_x = int(target_x * 100 + line_end_length * np.cos(target_theta))
        # target_end_y = int(target_y * 100 + line_end_length * np.sin(target_theta))

        # Add text labels
        display_name = f"ID: {marker_id}"
        if marker_id == robot_config.get("aruco"):
            display_name = f"Robot ({marker_id})"
        elif marker_id in macchinari_id_to_key:
            display_name = f"{macchinari_id_to_key[marker_id]} ({marker_id})"

        pos_text = f"Pos: ({pos_x * 100:.1f}, {pos_y * 100:.1f})cm"
        angle_text = f"Ang: {angle_deg:.1f} deg; {angle_rad:.2f} rad"

        cv2.putText(warped, display_name, (center_px[0] + 10, center_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(warped, pos_text, (center_px[0] + 10, center_px[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(warped, angle_text, (center_px[0] + 10, center_px[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return warped


class Vision:
    """
    Detects a quadrilateral defined by four specific ArUco markers,
    performs perspective transformation on the camera feed based on these markers,
    and detects other ArUco markers within the transformed view, calculating their
    position and orientation in centimeters.
    """

    def __init__(self,
                 camera_index: int = 0,
                 aruco_dict_type: int = cv2.aruco.DICT_6X6_250,
                 marker_corners_ids=None,
                 px_windows_height: int = 600,
                 width: float = 30.0,
                 height: float = 30.0,
                 robot=None,
                 targets=None,
                 visionStateUpdate: Optional[Callable[[Dict[str, Any]], None]] = None,
                 display: bool = True,
                 calib_camera=None):
        """Initializes the ArUcoQuadrilateralTransformer."""
        # Real Fields parameters
        if marker_corners_ids is None:
            marker_corners_ids = [10, 12, 14, 16]
        if robot is None:
            robot = {"aruco": 18, "x_offset": 0, "y_offset": 0, "theta_offset": 0}
        if targets is None:
            targets = {
                "3d": {"aruco": 20, "x_offset": 0, "y_offset": 0, "theta_offset": 0},
                "laser": {"aruco": 21, "x_offset": 0, "y_offset": 0, "theta_offset": 0}
            }
        self.width = width
        self.height = height
        self.field_marker_centers = {}

        # OpenCV display data
        self.display = display  # Enable disable frame view
        root = tk.Tk()  # Get current monitor information using tkinter
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()

        # Camera settings
        if calib_camera: # Intrinsic calibration matrix
            self.mtx = calib_camera['camera_matrix']
            self.dist = calib_camera['dist_coeff']
            self.rvecs = calib_camera['rvecs']
            self.tvecs = calib_camera['tvecs']
            self.IntrinsicLoad=True
        else:
            self.IntrinsicLoad=False
        self.camera_index = camera_index
        self.cam = cv2.VideoCapture(self.camera_index)
        if not self.cam.isOpened():
            raise IOError(f"Cannot open camera with index {self.camera_index}")

        # Default dimensions for cv2.VideoCapture is 640x480. Switch to a 16:9 aspect ratio to cover the whole field
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
        # Window names
        self.cameraView_windowsName = 'ArUco Detection View'
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera frame dimensions: {self.frame_width}x{self.frame_height}");

        self.warped_windowName = 'Transformed View'
        self.warped_output_size = (int(px_windows_height * (self.width / self.height)), px_windows_height)
        self.last_good_warped = None

        # Initialize helper components
        self.aruco_detector = ArUcoDetector(aruco_dict_type)
        self.pose_calculator = MarkerPoseCalculator(self.width, self.height, self.warped_output_size)

        # Expected marker IDs for the four corners
        if len(set(marker_corners_ids)) != 4:
            raise ValueError("marker_corners_ids must be 4 unique IDs.")
        self.marker_corners_ids = marker_corners_ids

        # Store configuration for robot and machines
        self.robot_config = robot
        self.macchinari_config = targets
        self.macchinari_id_to_key = {mData["aruco"]: key for key, mData in targets.items()}

        self.sendToRobot = visionStateUpdate
        self.last_good_corners = None

    def find_quadrilateral(self, frame: np.ndarray, display: bool = True) -> Optional[np.ndarray]:
        """Finds the quadrilateral defined by the specified ArUco corner markers."""
        corners, ids, rejected = self.aruco_detector.detect_markers(frame)

        if ids is None:
            if display:
                cv2.imshow(self.cameraView_windowsName, frame)
            raise ValueError("No ArUco markers detected.")

        # Map detected markers to their expected positions
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_corners_ids:
                field_marker_corners_raw = corners[i][0]
                center_x = np.mean(field_marker_corners_raw[:, 0])
                center_y = np.mean(field_marker_corners_raw[:, 1])
                # Update marker_centers stored information with the new found center
                self.field_marker_centers[marker_id] = [center_x, center_y]

        # Draw detected markers if display is enabled
        if display:
            display_frame = frame.copy()
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            if len(self.field_marker_centers) > 1:
                display_frame = Visualizer.draw_quadrilateral(display_frame, self.field_marker_centers)
            cv2.imshow(self.cameraView_windowsName, display_frame)

        # Return corners only if all four are found
        field_center_corners = np.zeros((4, 2), dtype=np.float32)
        for index, cornerIds in enumerate(self.marker_corners_ids):  # format return information as matrix vector
            field_center_corners[index] = self.field_marker_centers.get(cornerIds, [0, 0])
        return field_center_corners if len(self.field_marker_centers) == 4 else None

    def get_frame(self) -> np.ndarray:
        """Captures and returns a frame from the camera."""
        ret, frame = self.cam.read()
        if not ret:
            raise IOError("Error: Could not read frame from camera.")
        if self.IntrinsicLoad: # Se falso, NON ESISTONO self.mtx, self.dist !!!
            return frameRemap(frame, self.mtx, self.dist)
        else:
            return frame

    def process_frame(self, frame: Optional[np.ndarray] = None, display: bool = True) -> None:
        """
        Processes a frame to find the quadrilateral and internal markers.
        Can optionally display visualization and return the processed frame.
        
        Args:
            frame: The input camera frame. If None, captures a new frame.
            display: Whether to display visualization windows.
            
        Returns:
            Nothing, to share information we use callback
        """
        # Capture frame if not provided
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame provided.")

        # Calculate the perspective transform matrix
        try:
            # Find quadrilateral field_center_corners
            field_center_corners = self.find_quadrilateral(frame, display=display)
            warped, matrix = PerspectiveTransformer.transform_perspective(frame, field_center_corners, self.warped_output_size)
            # Save the last good warped frame
            self.last_good_warped = warped.copy()
        except Exception as e:
            if display:
                if self.last_good_warped:
                    # Show the last good warped frame if available
                    cv2.imshow(self.warped_windowName, self.last_good_warped)
                else:
                    # Add error text to original frame
                    error_frame = frame.copy()
                    text = "No enough Corner available to generate field projection"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5  # Dimensione più grande
                    thickness = 3

                    # Calcola dimensioni del testo per centrarlo
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = (error_frame.shape[1] - text_size[0]) // 2
                    text_y = (error_frame.shape[0] + text_size[1]) // 2

                    # Aggiunge contorno nero al testo per massimo contrasto
                    cv2.putText(error_frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                    # Aggiunge testo in giallo per alta visibilità
                    cv2.putText(error_frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

                    cv2.imshow(self.warped_windowName, error_frame)
            return

        all_corners, all_ids, rejected = self.aruco_detector.detect_markers(frame)
        if all_ids is None:
            raise ValueError("No markers detected.")

        # Prepare result structure
        result = {
            'markers': {},
            'robot': None
        }
        newData = False

        # Process all detected markers
        for i, marker_id_np in enumerate(all_ids.flatten()):
            marker_id = int(marker_id_np)

            # Skip corner markers
            if marker_id in self.marker_corners_ids:
                continue

            newData = True
            current_aruco_corners = all_corners[i][0]

            # Calculate base pose
            try:
                pos_x, pos_y, angle_rad, trans_x_px, trans_y_px = self.pose_calculator.calculate_marker_pose(current_aruco_corners, matrix)
            except Exception as e:
                print(f"Error calculating pose for marker {marker_id}: {e}")
                continue

            # Apply offsets based on marker type
            final_x, final_y, final_angle_rad = pos_x, pos_y, angle_rad
            is_robot = False

            if marker_id == self.robot_config.get("aruco"):
                final_x, final_y, final_angle_rad = MarkerPoseCalculator.apply_offset(pos_x, pos_y, angle_rad, self.robot_config)
                is_robot = True
            elif marker_id in self.macchinari_id_to_key:
                machine_config = self.macchinari_config.get(self.macchinari_id_to_key[marker_id], {})
                final_x, final_y, final_angle_rad = MarkerPoseCalculator.apply_offset(pos_x, pos_y, angle_rad, machine_config)

            # Store marker data
            marker_data = {
                'id': marker_id,
                'position': [float(final_x), float(final_y)],
                'angle': float(final_angle_rad),
                'position_px': [float(trans_x_px), float(trans_y_px)],
            }

            # Assign to the correct category in the result
            if is_robot:
                result['robot'] = marker_data
            else:
                marker_key = self.macchinari_id_to_key.get(marker_id, f"unknown_{marker_id}")
                result['markers'][marker_key] = marker_data

            # Draw marker information if display is enabled
            if display:
                # Show marker ID and position and associated target position (offsets)
                # TODO: Mostrare anche l'offset calcolando la posizione del target a schermo
                warped = Visualizer.draw_marker_info(warped, marker_id,
                                                     pos_x, pos_y, angle_rad, trans_x_px, trans_y_px,
                                                     # final_x, final_y, final_angle_rad,
                                                     self.robot_config, self.macchinari_id_to_key)

        # Print final result of warped image with HUD information
        if display:
            cv2.imshow(self.warped_windowName, warped)

        # Send data if new data was processed and callback exists
        if newData and self.sendToRobot:
            try:
                self.sendToRobot(result)  # send infornation using callback
            except Exception as e:
                print(f"Error calling sendToRobot callback: {e}")

    def setup_windows(self):
        """Initializes and positions the OpenCV windows."""
        # Finestra per la visualizzazione della camera
        cv2.namedWindow(self.cameraView_windowsName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.cameraView_windowsName, 20, 20)  # Prima finestra a sinistra
        cv2.resizeWindow(self.cameraView_windowsName, 640, 480)

        # Finestra per la visualizzazione della trasformazione prospettica
        cv2.namedWindow(self.warped_windowName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.warped_windowName, self.screen_width - self.warped_output_size[0] - 40, 20)  # Finestra allineata a destra
        cv2.resizeWindow(self.warped_windowName, self.warped_output_size[0], self.warped_output_size[1])

        # Assicuriamoci che le finestre siano visibili
        cv2.waitKey(100)  # Piccola pausa per assicurarci che l'interfaccia grafica si aggiorni

    def run(self):
        """Starts the main processing loop."""
        try:
            # Inizializza le finestre nel thread principale
            if self.display:
                self.setup_windows()
            while True:
                try:
                    frame = self.get_frame()
                    self.process_frame(frame, display=self.display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Exit key 'q' pressed.")
                        break
                except (IOError, ValueError) as e:
                    print(f"Error getting frame: {type(e).__name__} - {e}")
                    continue
                except KeyboardInterrupt:
                    print("Interruzione da tastiera rilevata nel ciclo principale.")
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Releases the camera and destroys all OpenCV windows."""
        try:
            print("Releasing camera and closing windows...")
            if self.cam is not None and self.cam.isOpened():
                self.cam.release()
            cv2.destroyAllWindows()
            print("Cleanup complete.")
        except Exception as e:
            print(f"Errore durante la pulizia del sottosistema di visione: {e}")


# Setup del sottosistema di visione, avvia un thread per la visione della camera e ritorna il riferimento alla classe
def vision_setup(conf: dict, visionStateUpdate: Optional[Callable[[Dict[str, Any]], None]] = None) -> Vision:
    table = conf['table']
    aruco = table['aruco']

    with open(conf['cameraIntrinsic_pikleFile'], 'rb') as f:
        calib_data = pickle.load(f)
     
    corners_ids = [
        aruco['top-left'], aruco['top-right'],
        aruco['bottom-right'], aruco['bottom-left']]
    targetMachines = merge({}, conf['workZone'], conf['macchinari'])

    print("Avvio sottosistema di visione")
    transformer = Vision(camera_index=conf["cameraIndex"],
                         marker_corners_ids=corners_ids,
                         width=table['width'], height=table['height'],
                         robot=conf['robot'], targets=targetMachines,
                         visionStateUpdate=visionStateUpdate,
                         display=True,
                         calib_camera=calib_data)

    # Registra la funzione di cleanup con atexit
    atexit.register(transformer.cleanup)

    # Configura il gestore di segnali per Ctrl+C
    def signal_handler(sig, frame):
        print("\nRilevata interruzione da tastiera (Ctrl+C)...")
        transformer.cleanup()
        print("Pulizia completata. Uscita in corso...")
        sys.exit(0)

    # Registra il gestore per SIGINT (Ctrl+C) e SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("└─▶ Starting vision subsystem")
    return transformer


if __name__ == '__main__':
    # Esegui il server temporaneo per testare la pagina di benvenuto
    vision = vision_setup({
        "cameraIndex": 0,
        "table": {
            "width": 30.0,
            "height": 30.0,
            "aruco": {
                "top-left": 10,
                "top-right": 12,
                "bottom-right": 14,
                "bottom-left": 16
            }
        },
        "robot": {
            "aruco": 18,
            "x_offset": 0,
            "y_offset": 0,
            "theta_offset": 0
        },
        "workZone": {},
        "macchinari": {}
    })
    vision.run()
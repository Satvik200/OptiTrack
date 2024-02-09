# Import required python Packages
from __future__ import division
import os
import cv2
import dlib

# Import from other defined files
from .eye import Eye
from .calibration import Calibration

# This class tracks the user's gaze
# It provides useful information like the position of the eyes and pupils and allows to know if the eyes are open or closed
class OpticalTracking(object):
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()
        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def locatedPupils(self):
        # Check that the pupils have been located
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def analyzer(self):
        # Detects the face and initialize Eye objects
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)
        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)
        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        # Refreshes the frame and analyzes it.
        self.frame = frame
        self.analyzer()

    def leftPupilCoords(self):
        # Returns the coordinates of the left pupil
        if self.locatedPupils:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def rightPupilCoords(self):
        # Returns the coordinates of the right pupil
        if self.locatedPupils:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontalRatio(self):
        # Indicates the horizontal direction of the gaze
        # 0.0 -> extreme right
        # 0.5 -> center
        # 1.0 -> extreme left
        if self.locatedPupils:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def verticalRatio(self):
        # Indicates the vertical direction of the gaze
        # 0.0 -> extreme top
        # 0.5 -> center
        # 1.0 -> extreme bottom
        if self.locatedPupils:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def checkRight(self):
        # Return true if the user is looking to the right
        if self.locatedPupils:
            return self.horizontal_ratio() <= 0.35

    def checkLeft(self):
        # Return true if the user is looking to the left
        if self.locatedPupils:
            return self.horizontal_ratio() >= 0.65

    def checkCenter(self):
        # Return true if the user is looking to the center
        if self.locatedPupils:
            return self.checkRight() is not True and self.checkLeft() is not True

    def checkBlinking(self):
        # Return true if the user closes his eyes
        if self.locatedPupils:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotatedFrame(self):
        # Return the main frame with pupils highlighted
        frame = self.frame.copy()
        if self.locatedPupils:
            color = (0, 255, 0)
            x_left, y_left = self.leftPupilCoords()
            x_right, y_right = self.rightPupilCoords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        return frame
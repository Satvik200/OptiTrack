# Import required python Packages
import cv2
from OptiTrack import opticalTracking

gaze = opticalTracking()
webcam = cv2.VideoCapture(0)

while True:
    # Get a new frame from the webcam
    _, frame = webcam.read()

    # Send the frame to OpticalTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.checkBlinking():
        text = "Eyes are blinking"
    elif gaze.checkRight():
        text = "Eyes are looking towards right"
    elif gaze.checkLeft():
        text = "Eyes are looking towards left"
    elif gaze.checkCenter():
        text = "Eyes are Looking at the center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
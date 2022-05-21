#from facial_emotion_recognition import EmotionRecognition

import facial_emotion_recognition
import cv2
emotion = facial_emotion_recognition.EmotionRecognition(device="cpu")

cam = cv2.VideoCapture(1)

while True:
    success, frame = cam.read()
    frame = emotion.recognise_emotion(frame, return_type ="BGR")
    cv2.imshow("Hello", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
#cv2.release()
cv2.destroyAllWindows()

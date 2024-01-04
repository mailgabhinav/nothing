import cv2
import mediapipe as m

cap=cv2.VideoCapture(0)

m_drawing=m.solutions.drawing_utils
m_hands=m.solutions.hands

hands = m_hands.Hands(min_detection_confidence =0.8,min_tracking_confidence =0.5)

def drawhandlandmarks(image,hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            m_drawing.draw_landmarks(image,landmarks,m_hands.HAND_CONNECTIONS)

while True:
    succes,image=cap.read()
    image=cv2.flip(image,1)
    results=hands.process(image)
    hand_landmarks=results.multi_hand_landmarks
    drawhandlandmarks(image,hand_landmarks)
    cv2.imshow("window",image)
    key=cv2.waitKey(1)
    if key==32:
        break

cv2.destroyAllWindows()




























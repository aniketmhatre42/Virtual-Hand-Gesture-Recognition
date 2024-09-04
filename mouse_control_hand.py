import cv2
import mediapipe as mp
import pyautogui

capture_hands = mp.solutions.hands.Hands()
drawing_option = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)
dragging = False

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks
    
    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                
                if id == 8:
                    mouse_x = int(screen_width / image_width * x)
                    mouse_y = int(screen_height / image_height * y)
                    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1 = x
                    y1 = y
                    
                if id == 4:
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
            
            dist = abs(y2 - y1)
            
            if dist < 20:
                pyautogui.click()
                
            elif 20 <= dist < 50:
                pyautogui.click(button='right')
                
            elif 50 <= dist < 100 and not dragging:
                pyautogui.mouseDown()
                dragging = True
                
            elif dist >= 100 and dragging:
                pyautogui.mouseUp()
                dragging = False
    
    cv2.imshow("Hand Movement Video Capture", image)
    key = cv2.waitKey(100)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

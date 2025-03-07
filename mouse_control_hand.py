import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

capture_hands = mp.solutions.hands.Hands()
drawing_option = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)
dragging = False

prev_positions = []
max_positions = 10
cooldown_time = 1.0
last_action_time = 0

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
            
            x1, y1, x2, y2 = 0, 0, 0, 0
            
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

                    if len(prev_positions) >= max_positions:
                        prev_positions.pop(0)
                    prev_positions.append((x1, y1))
                    
                if id == 4:
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
            
            dist = abs(y2 - y1)
            
            current_time = time.time()
            if current_time - last_action_time > cooldown_time:
                if dist < 20:
                    pyautogui.click()
                    last_action_time = current_time
                elif 20 <= dist < 50:
                    pyautogui.click(button='right')
                    last_action_time = current_time
                elif 50 <= dist < 100 and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    last_action_time = current_time
                elif dist >= 100 and dragging:
                    pyautogui.mouseUp()
                    dragging = False
                    last_action_time = current_time

            if len(prev_positions) == max_positions:
                dx = [prev_positions[i+1][0] - prev_positions[i][0] for i in range(max_positions - 1)]
                dy = [prev_positions[i+1][1] - prev_positions[i][1] for i in range(max_positions - 1)]
                sum_dx = sum(dx)
                sum_dy = sum(dy)
                
                if sum_dx > 50 and sum_dy < -50:
                    cv2.putText(image, "Clockwise", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    
                elif sum_dx < -50 and sum_dy > 50:
                    cv2.putText(image, "Anticlockwise", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow("Hand Movement Video Capture", image)
    key = cv2.waitKey(100)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

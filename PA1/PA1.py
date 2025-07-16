'''
Demo code
'''

import numpy as np
import cv2 as cv
import sys

NUM_OBJECTS = 3
SPACE_BAR = 32      # Space bar key code = 32
GREEN = (0, 255, 0)

templates = {}

method_mapping = {
    'TM_SQDIFF': cv.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv.TM_SQDIFF_NORMED,
    'TM_CCORR': cv.TM_CCORR,
    'TM_CCORR_NORMED': cv.TM_CCORR_NORMED,
    'TM_CCOEFF': cv.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv.TM_CCOEFF_NORMED,
}

def capture_template(frame):
    roi = cv.selectROI('Select ROI for Template', frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi
    template = frame[y:y+h, x:x+w]

    return template

def main():
    use_grayscale = False
    method = sys.argv[1]

    if len(sys.argv) == 3:
        use_grayscale = True
    
    method = method_mapping.get(method)
    print(f'Selected method: {method}, Grayscale mode: {use_grayscale}')
    
    if method is None:
        print('Method not recognized, terminating program')
        exit()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()

    print('Teaching about 3 objects')
    for i in range(NUM_OBJECTS):
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot receive frame. Exiting ...')
                break
            
            if use_grayscale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow('teaching', frame)

            if cv.waitKey(1) == SPACE_BAR:
                template = capture_template(frame)

                object_name = input('Enter name of this object: ')
                
                templates[object_name] = template
                print(f'Template {object_name} captured successfully\n')
                break

    print('Teaching completed. Switching to recognition mode.')
    print('Press \"q\" to exit')
    min_threshold = 0.3
    max_threshold = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Cannot receive frame. Exiting ...')
            break

        if use_grayscale:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for object_name, template in templates.items():
            result = cv.matchTemplate(frame, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc

                print(f'{object_name} - Method {method} - min {min_val}')

                if min_val < min_threshold:
                    top_left_x_coord = top_left[0]
                    top_left_y_coord = top_left[1]
                    template_width = template.shape[1]
                    template_height = template.shape[0]

                    bottom_right = (top_left_x_coord + template_width, top_left_y_coord + template_height)

                    cv.rectangle(frame, top_left, bottom_right, GREEN, 2)
                    cv.putText(frame, f'{object_name} - Method {method}', (top_left[0], top_left[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                else:
                    # Hide the rectangle if max_val is below the threshold
                    cv.rectangle(frame, (0, 0), (0, 0), GREEN, 2)
            else:
                top_left = max_loc

                print(f'{object_name} - Method {method} - max {max_val}')

                if max_val > max_threshold:
                    top_left_x_coord = top_left[0]
                    top_left_y_coord = top_left[1]
                    template_width = template.shape[1]
                    template_height = template.shape[0]

                    bottom_right = (top_left_x_coord + template_width, top_left_y_coord + template_height)

                    cv.rectangle(frame, top_left, bottom_right, GREEN, 2)
                    cv.putText(frame, f'{object_name} - Method {method}', (top_left[0], top_left[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                else:
                    # Hide the rectangle if max_val is below the threshold
                    cv.rectangle(frame, (0, 0), (0, 0), GREEN, 2)

        cv.imshow("object recognition", frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('m'):
            new_method = input('Enter new method: ')
            if new_method in method_mapping:
                method = method_mapping[new_method]
                print(f'Method changed to {new_method}')
        elif key == ord('t'):
            old_min = min_threshold
            min_threshold = float(input('Enter new min threshold: '))
            print(f'Min threshold changed from {old_min} to {min_threshold}')

            old_max = max_threshold
            max_threshold = float(input('Enter new max threshold: '))
            print(f'Min threshold changed from {old_max} to {max_threshold}')

    cap.release()
    cv.destroyAllWindows()

main()
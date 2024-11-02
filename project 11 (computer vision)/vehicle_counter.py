import cv2, cvzone
import numpy as np
from ultralytics import YOLO
from sort import Sort
from time import time
from utils import find_closest_coords, get_text_dimensions, create_transparent_polygon

vehicle_detector = YOLO("yolo11l.pt")
mot_detector = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
class_dict = vehicle_detector.names

cap = cv2.VideoCapture("C:/Users/asus/Downloads/sample_for_cv_1.mp4")

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

counters_left = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
counters_right = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}

car_ids_right = []
previous_ids_type_right = {v_type: set() for v_type in counters_right.keys()}

car_ids_left = []
previous_ids_type_left = {v_type: set() for v_type in counters_left.keys()}



# line left
LINE_Y1 = 600
LINE_X1_START, LINE_X1_END = 230, 760

# line right_1
LINE_Y2 = 600
LINE_X2_START, LINE_X2_END = 800, 1080

# line right_2
LINE_Y3_START = 600
LINE_Y3_END = 800
LINE_X3_START, LINE_X3_END = 1300, 1700

DELTA_THRESHOLD = 60
# THRESHOLD = 30
OFFSET = 5
BRIGHTNESS_VALUE = -15
CONTRAST_VALUE = 0.8

POLYGON_POINTS = np.array([[305.118,509.711,766.342,528.607,750.783,580.849,262.886,559.73],
                  [810.797,530.83,1057.524,534.165,1071.972,585.295,815.243,580.849],
                  [1282.764,564.475,1214.229,591.964,1808.262,839.837,1868.276,803.156]], np.int32).reshape((-1,4,1,2))

w1, h1 = get_text_dimensions("LEFT LANE", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
w2, h2 = get_text_dimensions("RIGHT LANE", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
w3, h3 = get_text_dimensions("TOTAL = ", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
h3_2 = 3*((225+h3) - (185+h1))//4
h3_2 = 185 + h1 + h3_2


mask = cv2.imread("C:/Users/asus/Downloads/video_mask_1.png")
previous_time = time()

while True:
    status, frame = cap.read()
    frame_copy = frame.copy()
    framemasked = cv2.bitwise_and(frame, mask)
    brightness_value = 50  # You can adjust this value
    framemasked = cv2.convertScaleAbs(framemasked, alpha=CONTRAST_VALUE, beta=BRIGHTNESS_VALUE)
    
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    text_1, text_2 = "LEFT LANE",  "RIGHT LANE"

    cv2.rectangle(frame_copy, (20,20),(160, 180+h1),(255,0,255), thickness=cv2.FILLED)
    cv2.putText(frame_copy, text_1, (90-w1//2, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.rectangle(frame_copy, (180,20),(320, 180+h1),(255,0,255), thickness=cv2.FILLED)
    cv2.putText(frame_copy, text_2, (250-w2//2, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.rectangle(frame_copy, (20, 185+h1),(320, 225+h3),(255,0,255), thickness=cv2.FILLED)
    # cv2.circle(frame_copy, (25, h3_2), 2, (255, 0, 0), thickness=cv2.FILLED)
    # create counter area
    frame_copy = create_transparent_polygon(frame_copy, polygons=POLYGON_POINTS, alpha=0.3)

    vehicle_detections = np.empty((0,5))
    class_detection_map = {}
    
    results_vehicle = vehicle_detector.predict(framemasked, conf=0.45, imgsz=640)
    
    for result in results_vehicle:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])            
            class_id = int(box.cls[0])
            category = class_dict.get(class_id)
            confidence = float(box.conf[0])

            if category in ["car", "bus", "truck", "motorcycle"]:         
                detect_box = np.array([x1, y1, x2, y2, confidence])
                vehicle_detections = np.vstack((vehicle_detections, detect_box))
                coords_x_class = (x1, y1, x2, y2)
                class_detection_map[coords_x_class] = category
                cvzone.cornerRect(frame_copy, (x1, y1, x2 - x1, y2 - y1), rt=0)

    tracker_results = mot_detector.update(vehicle_detections)

    current_ids_left = set()
    current_ids_type_left = {v_type: set() for v_type in counters_left.keys()}

    current_ids_right = set()
    current_ids_type_right = {v_type: set() for v_type in counters_right.keys()}
    
    for o in tracker_results:
        x1, y1, x2, y2, id = map(int, o)
        coords = (x1,y1,x2,y2)
        center = (x1 + (x2-x1)//2, y1 + (y2-y1)//2)
        
        cvzone.putTextRect(frame_copy, f"{id}", (x1+1, y2), scale=1, thickness=2, offset=3)

        condition_left = cv2.pointPolygonTest(POLYGON_POINTS[0], center, False) == 1 or cv2.pointPolygonTest(POLYGON_POINTS[0], center, False) == 0
        condition_right_1 = cv2.pointPolygonTest(POLYGON_POINTS[1], center, False) == 1 or cv2.pointPolygonTest(POLYGON_POINTS[1], center, False) == 0
        condition_right_2 = cv2.pointPolygonTest(POLYGON_POINTS[2], center, False) == 1 or cv2.pointPolygonTest(POLYGON_POINTS[2], center, False) == 0

        if condition_left:

            class_coords = find_closest_coords(coords,  list(class_detection_map.keys()))
            class_type = class_detection_map.get(class_coords)

            frame_copy = create_transparent_polygon(frame_copy, polygons=[POLYGON_POINTS[0]], alpha=0.3,color=(0,0,255))

            if class_type and id not in car_ids_left:
                current_ids_type_left[class_type].add(id)
                current_ids_left.add(id)
                car_ids_left.append(id)

        if condition_right_1 :

            class_coords = find_closest_coords(coords,  list(class_detection_map.keys()))
            class_type = class_detection_map.get(class_coords)

            frame_copy = create_transparent_polygon(frame_copy, polygons=[POLYGON_POINTS[1]], alpha=0.3,color=(0,0,255))

            if class_type and id not in car_ids_right:
                current_ids_type_right[class_type].add(id)
                current_ids_right.add(id)
                car_ids_right.append(id)

        if condition_right_2 :

            class_coords = find_closest_coords(coords,  list(class_detection_map.keys()))
            class_type = class_detection_map.get(class_coords)

            frame_copy = create_transparent_polygon(frame_copy, polygons=[POLYGON_POINTS[2]], alpha=0.3,color=(0,0,255))

            if class_type and id not in car_ids_right:
                current_ids_type_right[class_type].add(id)
                current_ids_right.add(id)
                car_ids_right.append(id)

            


    left_cond = len(current_ids_left) != 0
    right_cond = len(current_ids_right) != 0
    
    if left_cond:

        for v_type in counters_left.keys():
                new_objects = current_ids_type_left[v_type]
                counters_left[v_type] += len(new_objects)
    
    elif right_cond:

        for v_type in counters_right.keys():
                new_objects = current_ids_type_right[v_type]
                counters_right[v_type] += len(new_objects)
    
    total_count_left = sum(counters_left.values())
    total_count_right = sum(counters_right.values())
    total = total_count_left + total_count_right

    display_texts_left = [
        f"Car = {counters_left['car']}",
        f"Truck = {counters_left['truck']}",
        f"Bus = {counters_left['bus']}",
        f"Motorcycle = {counters_left['motorcycle']}"
    ]

    display_texts_right = [
        f"Car = {counters_right['car']}",
        f"Truck = {counters_right['truck']}",
        f"Bus = {counters_right['bus']}",
        f"Motorcycle = {counters_right['motorcycle']}"
    ]

    for idx, (text_l, text_r) in enumerate(zip(display_texts_left, display_texts_right)):
        x_pos_l, x_post_r = 25, 185
        y_pos = 40 + idx * (30 + OFFSET)
        cv2.putText(frame_copy, text_l, (x_pos_l, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(frame_copy, text_r, (x_post_r, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.putText(frame_copy, f"Total = {total}", (170 - w3//2, h3_2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


    
    cv2.imshow("Video", frame_copy)
    # cv2.imshow("Video_adjusted", frame)
    delta_time = time() - previous_time

    if delta_time >= DELTA_THRESHOLD:
        car_ids_left = car_ids_left[10:]
        car_ids_right = car_ids_right[10:]
        previous_time = time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
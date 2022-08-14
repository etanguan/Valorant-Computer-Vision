import cv2, queue, threading, time
import numpy as np
from mss import mss
from PIL import Image




INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

def click(x,y):
    print(x, y)
    px, py = pyautogui.position()
    print(px, py)

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


def format_yolov5(source):
    # put the image in square big enough
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source

    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1 / 255.0, (640, 640), swapRB=True)

    return result


def draw_label(im, label, x, y):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # cv2.rectangle(im, (x,y), (x+dim[0], y+dim[1]+ baseline), (0,0,0), cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1)
    net.setInput(blob)

    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs


def post_process(input_image, outputs):
    class_ids = []
    confidences = []
    boxes = []

    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if (classes_scores[class_id]) > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, THICKNESS)

        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        if class_ids[i] == 1:
            draw_label(input_image, label, left, top - 20)
        else:
            draw_label(input_image, label, left, top)
            click(top+width//2, left+height//2)



    return input_image


classes = ['Valorant-enemy', 'enemy', 'enemy-head']
net = cv2.dnn.readNet('yolov5/best.onnx')

mon = {'left': 0, 'top': 0, 'width': SCREEN_WIDTH, 'height': SCREEN_HEIGHT}

with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB',
            (screenShot.width, screenShot.height),
            screenShot.rgb,
        )

        frame = np.array(img)
        frame = frame[:, :, [2, 1, 0]]

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        detections = pre_process(frame, net)
        img = post_process(frame.copy(), detections)
        t, _ = net.getPerfProfile()
        label = 'Inference timeL %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        print(label)
        cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
        cv2.imshow('Output', img)
        cv2.waitKey(1)

cv2.destroyAllWindows()



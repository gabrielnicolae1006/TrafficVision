import cv2
from ultralytics import YOLO
import easyocr
import re
from collections import Counter
import math

model = YOLO("CarPlate.pt")
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
video_path = 'Traffic.mp4'
cap = cv2.VideoCapture(video_path)

PATTERNS = {
    'ROMANIA': r'^[A-Z]{1,2}[0-9]{2,3}[A-Z]{3}$',
    'MOLDOVA': r'^[A-Z]{3}[0-9]{3}$',
    'GERMANY': r'^[A-Z]{1,3}[A-Z]{1,2}[0-9]{1,4}$',
    'FRANCE_ITALY': r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$',
    'UK': r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',
    'BULGARIA': r'^[ABEIKMOPCTXY]{1,2}[0-9]{4}[ABEIKMOPCTXY]{2}$',
    'UKRAINE': r'^[A-Z]{2}[0-9]{4}[A-Z]{2}$',
    'HUNGARY_NEW': r'^[A-Z]{4}[0-9]{3}$',
    'NETHERLANDS': r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$|^[0-9]{2}[A-Z]{2}[0-9]{2}$|^[A-Z]{1}[0-9]{3}[A-Z]{2}$|^[A-Z]{2}[0-9]{4}$|^[0-9]{2}[A-Z]{2}[A-Z]{2}$',
    'POLAND': r'^[A-Z]{2,3}[A-Z0-9]{4,5}$',
    'USA_GENERIC': r'^[A-Z0-9]{6,7}$'
}

dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '7': 'Z', '8': 'B'}
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '7', 'B': '8', 'Q': '0', 'D': '0'}

history = {}
finalized_numbers = set()
previous_centers = []
next_id = 0


def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def repair_and_identify(text):
    text_list = list(text)
    original_len = len(text)

    if 6 <= original_len <= 7:
        if text[0].isalpha() and text[2].isdigit():
            for i in range(original_len - 3, original_len):
                if text_list[i] in dict_int_to_char:
                    text_list[i] = dict_int_to_char[text_list[i]]

    if original_len == 6:
        if text[0].isalpha() and text[1].isalpha():
            if text_list[2] in dict_int_to_char: text_list[2] = dict_int_to_char[text_list[2]]
        for i in range(3, 6):
            if text_list[i] in dict_char_to_int: text_list[i] = dict_char_to_int[text_list[i]]

    if original_len == 7:
        if text_list[2] in dict_char_to_int: text_list[2] = dict_char_to_int[text_list[2]]
        if text_list[3] in dict_char_to_int: text_list[3] = dict_char_to_int[text_list[3]]

    repaired_text = "".join(text_list)

    if re.match(PATTERNS['ROMANIA'], repaired_text): return repaired_text, "ROMANIA"
    if re.match(PATTERNS['MOLDOVA'], repaired_text): return repaired_text, "MOLDOVA"
    if re.match(PATTERNS['BULGARIA'], repaired_text): return repaired_text, "BULGARIA"
    if re.match(PATTERNS['UKRAINE'], repaired_text): return repaired_text, "UKRAINE"
    if re.match(PATTERNS['HUNGARY_NEW'], repaired_text): return repaired_text, "HUNGARY"
    if re.match(PATTERNS['FRANCE_ITALY'], repaired_text): return repaired_text, "FRANCE / ITALY"
    if re.match(PATTERNS['UK'], repaired_text): return repaired_text, "UNITED KINGDOM"
    if re.match(PATTERNS['GERMANY'], repaired_text): return repaired_text, "GERMANY"
    if re.match(PATTERNS['NETHERLANDS'], repaired_text): return repaired_text, "NETHERLANDS"
    if re.match(PATTERNS['POLAND'], repaired_text): return repaired_text, "POLAND"
    if re.match(PATTERNS['USA_GENERIC'], repaired_text):
        return repaired_text, "POSSIBLE USA/OTHER"

    return repaired_text, "UNKNOWN"


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    return contrast


print("--- START ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (1020, 500))
    display_frame = frame.copy()
    current_centers = []

    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            if area < 1500: continue

            if conf > 0.25:
                best_id = -1
                min_dist = 9999
                for pid, px, py in previous_centers:
                    dist = math.hypot(cx - px, cy - py)
                    if dist < 100:
                        if dist < min_dist:
                            min_dist = dist
                            best_id = pid

                if best_id == -1:
                    best_id = next_id
                    next_id += 1

                current_centers.append((best_id, cx, cy))

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

                if best_id not in finalized_numbers:
                    try:
                        p = 6
                        h, w, _ = frame.shape
                        roi = frame[max(0, y1 - p):min(h, y2 + p), max(0, x1 - p):min(w, x2 + p)]
                        roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        roi_proc = process_image(roi)

                        ocr_res = reader.readtext(roi_proc, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                        if ocr_res:
                            text_raw = "".join(ocr_res)
                            text_clean = clean_text(text_raw)

                            if len(text_clean) >= 6:
                                if best_id not in history: history[best_id] = []
                                history[best_id].append(text_clean)

                                if len(history[best_id]) >= 5:
                                    most_common, count = Counter(history[best_id]).most_common(1)[0]
                                    text_final, country = repair_and_identify(most_common)

                                    if country in ['ROMANIA', 'MOLDOVA', 'BULGARIA', 'UKRAINE', 'HUNGARY_NEW',
                                                   'FRANCE_ITALY', 'UK', 'GERMANY', 'NETHERLANDS', 'POLAND', 'USA_GENERIC']:
                                        print(f"ID {best_id}: {text_final} ({country})")
                                        finalized_numbers.add(best_id)

                                    elif len(history[best_id]) > 12 and count > 6:
                                        print(f"ID {best_id}: {most_common} (Unknown/Foreign)")
                                        finalized_numbers.add(best_id)

                    except Exception as e:
                        continue
                else:
                    cv2.putText(display_frame, "OK", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    previous_centers = current_centers
    cv2.imshow('Traffic.mp4', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
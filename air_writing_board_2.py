import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

canvas = None
prev_x, prev_y = 0, 0

PEN_COLOR = (255, 255, 0)     # Bright Cyan
PEN_SIZE = 8

ERASER_SIZE = 40


def is_fist(hand_landmarks):
    """
    Returns True if all fingers are folded.
    """

    tips = [8, 12, 16, 20]

    folded = 0

    for tip in tips:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            folded += 1

    return folded >= 4


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    mode = "DRAW"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape

        index_tip = hand.landmark[8]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # Detect fist
        fist_mode = is_fist(hand)

        if fist_mode:

            mode = "ERASER"

            cv2.circle(frame, (x, y), 25, (0, 0, 255), 3)

            if prev_x != 0 and prev_y != 0:
                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    (0, 0, 0),
                    ERASER_SIZE
                )

        else:

            mode = "DRAW"

            cv2.circle(frame, (x, y), 10, PEN_COLOR, -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (x, y),
                PEN_COLOR,
                PEN_SIZE
            )

        prev_x, prev_y = x, y

    else:
        prev_x, prev_y = 0, 0

    frame = cv2.add(frame, canvas)

    cv2.putText(
        frame,
        f"MODE : {mode}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Fist = Eraser | C = Clear | S = Save",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Air Writing Board", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros_like(frame)

    elif key == ord('s'):
        cv2.imwrite("air_drawing.png", canvas)
        print("Saved!")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
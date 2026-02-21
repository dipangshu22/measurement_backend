import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None, None

    image_height, image_width, _ = image.shape

    # -------- HEIGHT DETECTION --------
    height_cm = None
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            head_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y
            right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y
            heel_y = max(left_heel, right_heel)

            pixel_height = abs(heel_y - head_y) * image_height

            reference_pixel_height = image_height * 0.2
            cm_per_pixel = 29.7 / reference_pixel_height

            height_cm = round(pixel_height * cm_per_pixel, 2)

    # -------- HAND GESTURE DETECTION --------
    gesture = "No hand detected"

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        hand_results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if hand_results.multi_hand_landmarks:
            landmarks = hand_results.multi_hand_landmarks[0].landmark

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            # 👍 GOOD (Thumb up)
            if thumb_tip.y < index_tip.y:
                gesture = "GOOD"

            # 👎 BAD (Thumb down)
            elif thumb_tip.y > index_tip.y:
                gesture = "BAD"

            # 👌 OK (Thumb touching index)
            distance = ((thumb_tip.x - index_tip.x) ** 2 +
                        (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            if distance < 0.05:
                gesture = "OK"

    return height_cm, gesture
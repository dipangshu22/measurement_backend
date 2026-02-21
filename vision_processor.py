import cv2

def process_image(image_path):
    import mediapipe as mp   # 🔥 moved inside function

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image_height, _, _ = image.shape

    height_cm = None
    gesture = "No hand detected"

    # -------- POSE --------
    with mp_pose.Pose(static_image_mode=True, model_complexity=0) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            head_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            heel_y = max(
                landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,
            )
            pixel_height = abs(heel_y - head_y) * image_height
            cm_per_pixel = 29.7 / (image_height * 0.2)
            height_cm = round(pixel_height * cm_per_pixel, 2)

    # -------- HANDS --------
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        hand_results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if hand_results.multi_hand_landmarks:
            gesture = "Hand detected"

    return height_cm, gesture

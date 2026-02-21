import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def estimate_height(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (640, 480))
    h, w, _ = image.shape

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.7
    ) as pose:

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # Draw skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Get top & bottom of body
        y_values = [lm.y for lm in landmarks]

        top_pixel = min(y_values) * h
        bottom_pixel = max(y_values) * h

        pixel_height = bottom_pixel - top_pixel

        # Temporary proportional scaling (adjustable)
        reference_frame_height_cm = 170
        cm_per_pixel = reference_frame_height_cm / (h * 0.8)

        height_cm = pixel_height * cm_per_pixel

        # Draw bounding box
        x_vals = [lm.x for lm in landmarks]
        x_min = int(min(x_vals) * w)
        x_max = int(max(x_vals) * w)
        y_min = int(min(y_values) * h)
        y_max = int(max(y_values) * h)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw height line
        cv2.line(
            image,
            (int(w / 2), int(top_pixel)),
            (int(w / 2), int(bottom_pixel)),
            (255, 0, 0),
            3
        )

        cv2.putText(
            image,
            f"Height: {int(height_cm)} cm",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imwrite("output_detected.jpg", image)

        return height_cm
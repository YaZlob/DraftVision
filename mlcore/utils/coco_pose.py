COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

KEYPOINT_COLORS = [
    (255, 100, 100),
    (255, 150, 100),
    (255, 150, 100),
    (255, 200, 100),
    (255, 200, 100),  # Head
    (200, 255, 100),
    (100, 255, 100),  # Shoulders
    (100, 255, 200),
    (100, 200, 255),  # Elbows
    (100, 150, 255),
    (100, 100, 255),  # Wrists
    (200, 100, 255),
    (255, 100, 255),  # Hips
    (255, 100, 200),
    (255, 100, 150),  # Knees
    (255, 100, 100),
    (255, 150, 100),  # Ankles
]

# Color palette for 19 Limbs (BGR format)
LIMB_COLORS = [
    (200, 200, 100),
    (200, 200, 100),  # Left Leg (Ankle->Knee->Hip)
    (100, 200, 200),
    (100, 200, 200),  # Right Leg
    (150, 150, 150),  # Pelvis (Hip->Hip)
    (200, 255, 100),
    (100, 255, 200),  # Torso (Hip->Shoulder)
    (150, 255, 150),  # Chest (Shoulder->Shoulder)
    (100, 255, 200),
    (100, 200, 255),  # Right Arm
    (200, 255, 100),
    (255, 255, 100),  # Left Arm
    (255, 150, 150),
    (255, 100, 100),
    (255, 100, 100),  # Head/Neck
    (255, 200, 100),
    (255, 200, 100),  # Face sides
    (200, 255, 100),
    (100, 255, 200),  # Neck to Shoulders
]

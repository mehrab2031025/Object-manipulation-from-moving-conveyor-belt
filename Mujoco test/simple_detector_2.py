import cv2
import numpy as np


class ColorDetector:
    """
    Improved color-based detector for red box.
    """

    def __init__(self):
        # Wider red color range in HSV
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([20, 255, 255])
        self.lower_red2 = np.array([160, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # Also detect bright red (RGB)
        self.min_brightness = 50

    def detect(self, image):
        """
        Detect red object in image.
        Returns: (x, y, w, h) in pixels, or None
        """
        # Check if image is valid
        if image is None or image.size == 0:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create mask for red
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find all contours with reasonable area
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:  # Very small threshold
                valid_contours.append(cnt)

        if not valid_contours:
            return None

        # Find largest
        largest = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Return center
        center_x = x + w / 2
        center_y = y + h / 2

        return (center_x, center_y, w, h)

    def get_object_position(self, image):
        """
        Convert 2D detection to 3D position.
        """
        detection = self.detect(image)
        if detection is None:
            return None

        x_pixel, y_pixel, w, h = detection

        # Map x pixel to x position
        image_width = image.shape[1]
        x_norm = x_pixel / image_width
        x_pos = -0.8 + x_norm * 1.3

        # Fixed y and z
        y_pos = -0.8
        z_pos = 0.12

        return (x_pos, y_pos, z_pos)


# Test with debugging
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ur5_conveyor_env import UR5ConveyorEnv
    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_PATH = os.path.join(CURRENT_DIR, "ur5e_robot", "ur5e_with_conveyor.xml")

    env = UR5ConveyorEnv(XML_PATH)
    obs, info = env.reset()

    # Move object to center
    for _ in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Get image
    image = env.get_camera_image()
    print(f"Image shape: {image.shape}")
    print(f"Image mean color: {image.mean(axis=(0, 1))}")  # RGB means

    # Show image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    # Detect
    detector = ColorDetector()
    detection = detector.detect(image)

    # Show mask
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, detector.lower_red1, detector.upper_red1)
    mask2 = cv2.inRange(hsv, detector.lower_red2, detector.upper_red2)
    mask = mask1 + mask2

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Red Mask")

    if detection:
        x, y, w, h = detection
        print(f"Detected: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")

        # Draw
        img_display = image.copy()
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.subplot(1, 3, 3)
        plt.imshow(img_display)
        plt.title(f"Detection: ({x:.0f}, {y:.0f})")
    else:
        print("No detection")
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.title("No Detection")

    plt.tight_layout()
    plt.savefig("detection_debug.png", dpi=150)
    print("Saved to detection_debug.png")
    plt.show()

    env.close()
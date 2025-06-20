import cv2
import numpy as np
import os

def resize_image(img, max_width=512):
    """
    Resize the image to a maximum width while keeping aspect ratio.
    """
    h, w = img.shape[:2]
    if w > max_width:
        scale_ratio = max_width / w
        new_dim = (int(w * scale_ratio), int(h * scale_ratio))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        return resized
    return img

def reduce_intensity_levels(img, levels):
    """
    Reduces the number of grayscale intensity levels in an image.
    """
    if levels not in [2 ** i for i in range(1, 9)]:
        raise ValueError("Levels must be a power of 2 between 2 and 256.")
    factor = 256 // levels
    reduced_img = (img // factor) * factor
    return reduced_img

def add_topic_label(img, text):
    """
    Adds a topic label at the top of the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_color = 255  # White for grayscale
    label_height = 40

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    label = np.full((label_height, img.shape[1]), 0, dtype=np.uint8)
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2 - 5

    cv2.putText(label, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return np.vstack((label, img))

if __name__ == "__main__":
    try:
        # User Inputs 
        image_path = input("Enter path to grayscale image (e.g., image.jpg): ").strip()
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image not found. Please check the path.")

        levels = int(input("Enter desired number of intensity levels (power of 2 between 2 and 256): "))
        if levels not in [2 ** i for i in range(1, 9)]:
            raise ValueError("Invalid input. Levels must be a power of 2 between 2 and 256 (e.g., 2, 4, 8, ..., 256).")

        # Load grayscale image 
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to read image. Make sure it's a valid grayscale image.")

        # Resize image to a smaller size 
        img = resize_image(img, max_width=512)

        # Reduce intensity and label 
        reduced_img = reduce_intensity_levels(img, levels)
        labeled_original = add_topic_label(img, "Original Image")
        labeled_reduced = add_topic_label(reduced_img, f"Reduced to {levels} Levels")

        # Combine, show, save 
        combined = np.hstack((labeled_original, labeled_reduced))
        cv2.imshow("Original vs Reduced", combined)

        output_path = f"combined_reduced_{levels}.png"
        cv2.imwrite(output_path, combined)
        print(f"Combined image saved as: {output_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

import cv2
import numpy as np
import os

def rotate_image(image, angle):
    """
    Rotates an image by a given angle while keeping the entire image visible.
    
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new dimensions to avoid cropping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation to keep the image centered
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated


if __name__ == "__main__":
    try:
        path = input("Enter the path to the image: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError("Image path does not exist.")

        image = cv2.imread(path)
        if image is None:
            raise ValueError("Failed to load image. Ensure it is a valid image file.")

        # Angles to rotate the image
        angles = [45, 90]

        # Show original image
        cv2.imshow("Original Image", image)

        # Prepare output folder
        output_dir = "rotated_outputs"
        os.makedirs(output_dir, exist_ok=True)

        base_name, ext = os.path.splitext(os.path.basename(path))

        # Rotate and save images
        for angle in angles:
            rotated_img = rotate_image(image, angle)
            window_name = f"Rotated {angle} Degrees"
            cv2.imshow(window_name, rotated_img)

            save_path = os.path.join(output_dir, f"{base_name}_rotated_{angle}{ext}")
            cv2.imwrite(save_path, rotated_img)
            print(f"Saved rotated image: {save_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

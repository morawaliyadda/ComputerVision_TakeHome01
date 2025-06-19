import cv2
import numpy as np
import os

def reduce_intensity_levels(image_path, levels):
    """
    Reduces the number of grayscale intensity levels in an image.

    """
    if levels not in [2 ** i for i in range(1, 9)]:
        raise ValueError("Levels must be a power of 2 between 2 and 256.")

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    # Calculate the quantization factor
    factor = 256 // levels

    # Reduce the intensity levels
    reduced_img = (img // factor) * factor

    return img, reduced_img


if __name__ == "__main__":
    try:
        # Get image path from user
        image_path = input("Enter the path to the image: ").strip()

        if not os.path.exists(image_path):
            raise FileNotFoundError("Image path does not exist.")

        # Get number of intensity levels
        levels = int(input("Enter the number of intensity levels (power of 2 between 2 and 256): "))

        # Process the image
        original, reduced = reduce_intensity_levels(image_path, levels)

        # Display both images
        cv2.imshow("Original Image", original)
        cv2.imshow(f"Reduced to {levels} Levels", reduced)

        # Save reduced image
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_reduced_{levels}{ext}"
        cv2.imwrite(output_path, reduced)
        print(f"Reduced image saved as: {output_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

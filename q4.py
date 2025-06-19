import cv2
import numpy as np
import os

def blockwise_average(img, block_size):
    """
    Applies block-wise averaging to the grayscale image.

    """
    h, w = img.shape
    result = img.copy()

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img[i:i + block_size, j:j + block_size]
            avg = int(np.mean(block))
            result[i:i + block_size, j:j + block_size] = avg

    return result


if __name__ == "__main__":
    try:
        # Input image path
        path = input("Enter the path to a grayscale image: ").strip()

        if not os.path.exists(path):
            raise FileNotFoundError("Image path does not exist.")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Failed to load image. Ensure it's a valid grayscale image.")

        # Define block sizes to apply
        block_sizes = [3, 5, 7]

        # Show original image
        cv2.imshow("Original Image", img)

        # Output directory for saving results
        output_dir = "blockwise_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Base name for saving
        base_name, ext = os.path.splitext(os.path.basename(path))

        # Process and save each block-size averaged image
        for size in block_sizes:
            averaged_img = blockwise_average(img, size)
            window_title = f"{size}x{size} Block Averaged"
            cv2.imshow(window_title, averaged_img)

            save_path = os.path.join(output_dir, f"{base_name}_block_{size}x{size}{ext}")
            cv2.imwrite(save_path, averaged_img)
            print(f"Saved: {save_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

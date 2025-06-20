import cv2
import numpy as np
import os

def resize_image(img, max_width=400):
    h, w = img.shape
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

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

def add_label(img, text, height=40):
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = np.zeros((height, img.shape[1]), dtype=np.uint8)
    text_size = cv2.getTextSize(text, font, 0.6, 1)[0]
    x = (img.shape[1] - text_size[0]) // 2
    y = (height + text_size[1]) // 2 - 5
    cv2.putText(label, text, (x, y), font, 0.6, 255, 1, cv2.LINE_AA)
    return np.vstack((label, img))

if __name__ == "__main__":
    try:
        # Read and validate input image
        path = input("Enter the path to a grayscale image: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError("Image path does not exist.")
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image. Ensure it's a valid grayscale image.")

        # Resize the original image for uniform display
        img = resize_image(img)

        # Prepare output directory
        output_dir = "blockwise_outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name, ext = os.path.splitext(os.path.basename(path))

        # Create images with labels: original and block-wise averaged versions
        images = [
            add_label(img, "Original Image"),
            add_label(blockwise_average(img, 3), "3x3 Block Averaged"),
            add_label(blockwise_average(img, 5), "5x5 Block Averaged"),
            add_label(blockwise_average(img, 7), "7x7 Block Averaged")
        ]

        # Ensure all images have the same size by padding
        max_h = max(i.shape[0] for i in images)
        max_w = max(i.shape[1] for i in images)
        for i in range(len(images)):
            h, w = images[i].shape
            pad_h = max_h - h
            pad_w = max_w - w
            images[i] = cv2.copyMakeBorder(images[i], 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Arrange the four images into a 2x2 grid
        top_row = np.hstack((images[0], images[1]))
        bottom_row = np.hstack((images[2], images[3]))
        combined = np.vstack((top_row, bottom_row))

        # Save and display the combined image
        final_path = os.path.join(output_dir, f"{base_name}_block_combined_grid{ext}")
        cv2.imwrite(final_path, combined)
        print(f"Combined image saved as: {final_path}")
        cv2.imshow("Block-wise Averaging Grid", combined)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

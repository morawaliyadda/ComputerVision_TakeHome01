import cv2
import numpy as np
import os

def resize_image(img, max_width=512):
    """
    Resize image to a maximum width while keeping the aspect ratio.
    """
    h, w = img.shape
    if w > max_width:
        scale_ratio = max_width / w
        new_dim = (int(w * scale_ratio), int(h * scale_ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    return img

def add_label(img, text, label_height=40):
    """
    Add a text label above the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_color = 255

    label = np.zeros((label_height, img.shape[1]), dtype=np.uint8)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2 - 5

    cv2.putText(label, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return np.vstack((label, img))

def apply_average_blur(image_path):
    """
    Apply average filters and display results in a 2x2 grid with labels.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found.")
    img = resize_image(img)

    kernel_sizes = [(3, 3), (10, 10), (20, 20)]
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    output_dir = "blurred_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare all labeled images
    labeled_images = []
    labeled_images.append(add_label(img, "Original Image"))

    for ksize in kernel_sizes:
        blurred = cv2.blur(img, ksize)
        label = f"Average Filter {ksize[0]}x{ksize[1]}"
        labeled = add_label(blurred, label)
        labeled_images.append(labeled)

        # Save individual blurred image
        save_path = os.path.join(output_dir, f"{base_name}_blur_{ksize[0]}x{ksize[1]}{ext}")
        cv2.imwrite(save_path, blurred)
        print(f"Saved blurred image: {save_path}")

    # Stack images into 2x2 grid
    row1 = np.hstack((labeled_images[0], labeled_images[1]))  # Original & 3x3
    row2 = np.hstack((labeled_images[2], labeled_images[3]))  # 10x10 & 20x20
    grid = np.vstack((row1, row2))

    # Show and save
    cv2.imshow("Average Blur - 2x2 Grid", grid)
    combined_output = os.path.join(output_dir, f"{base_name}_blur_grid.png")
    cv2.imwrite(combined_output, grid)
    print(f"Saved 2x2 grid image as: {combined_output}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        path = input("Enter the path to the image: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError("Image path does not exist.")
        apply_average_blur(path)
    except Exception as e:
        print("Error:", e)

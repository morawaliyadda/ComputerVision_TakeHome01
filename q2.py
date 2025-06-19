import cv2
import os

def apply_average_blur(image_path):
    """
    Apply average blur filters with different kernel sizes to a grayscale image.

    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    # List of kernel sizes to apply
    kernel_sizes = [(3, 3), (10, 10), (20, 20)]

    # Display the original image
    cv2.imshow("Original Image", img)

    base_name, ext = os.path.splitext(os.path.basename(image_path))
    output_dir = "blurred_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Apply and display average blur for each kernel size
    for ksize in kernel_sizes:
        blurred = cv2.blur(img, ksize)
        window_name = f"Average Filter {ksize[0]}x{ksize[1]}"
        cv2.imshow(window_name, blurred)

        # Save the blurred image
        save_path = os.path.join(output_dir, f"{base_name}_blur_{ksize[0]}x{ksize[1]}{ext}")
        cv2.imwrite(save_path, blurred)
        print(f"Saved blurred image: {save_path}")

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

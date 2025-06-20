import cv2
import numpy as np
import os

def resize(img, max_width=400):
    """
    Resizes the image to a given max width while preserving aspect ratio.
    """
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def rotate(img, angle):
    """
    Rotates the image by a given angle while preserving the full content.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    mat[0, 2] += (new_w / 2) - center[0]
    mat[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, mat, (new_w, new_h))

def label(img, text, height=40):
    """
    Adds a text label above the image as a black bar with white text.
    """
    color = (255, 255, 255)
    label_bar = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    x = (img.shape[1] - size[0]) // 2
    y = (height + size[1]) // 2 - 5
    cv2.putText(label_bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return np.vstack((label_bar, img))

if __name__ == "__main__":
    try:
        # Load input image
        path = input("Enter image path: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError("Image path not found.")
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Could not load image.")

        # Resize and label original image
        img = resize(img)
        images = [label(img, "Original Image")]

        # Rotate, label, and collect rotated versions
        for angle in [45, 90]:
            rot = resize(rotate(img, angle))
            images.append(label(rot, f"Rotated {angle} degrees"))

        # Pad all images to have the same height
        h_max = max(i.shape[0] for i in images)
        for i in range(len(images)):
            if images[i].shape[0] < h_max:
                pad_h = h_max - images[i].shape[0]
                pad = np.zeros((pad_h, images[i].shape[1], 3), dtype=np.uint8)
                images[i] = np.vstack((images[i], pad))

        # Combine images horizontally
        combined = np.hstack(images)

        # Show and save the final combined image
        cv2.imshow("Rotated Images", combined)
        os.makedirs("rotated_outputs", exist_ok=True)
        out_path = os.path.join("rotated_outputs", f"{os.path.splitext(os.path.basename(path))[0]}_rotated_combined.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved to {out_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

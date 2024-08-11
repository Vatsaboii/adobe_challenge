import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import rotate
from scipy.fftpack import fft

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Calculate the centroid of the binary image
    M = cv2.moments(binary_image)
    if M["m00"] == 0:
        raise ValueError("The centroid cannot be computed.")
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return binary_image, (cx, cy)

def flip_and_compare(image, axis, tolerance=1):
    if axis == 'vertical':
        flipped_image = cv2.flip(image, 0)  # Flip vertically
    elif axis == 'horizontal':
        flipped_image = cv2.flip(image, 1)  # Flip horizontally
    else:
        return False
    
    # Compare the flipped image to the original
    difference = cv2.absdiff(image, flipped_image)
    
    # Check if the difference is minimal (i.e., images are almost the same)
    non_zero_count = np.count_nonzero(difference)
    total_pixels = image.shape[0] * image.shape[1]
    
    # If the difference is within the tolerance, consider it symmetric
    return non_zero_count / total_pixels < tolerance / 100

def check_reflection_symmetry(image, num_axes=36, tolerance=1):
    symmetries = []
    for i in range(num_axes):
        angle = 180.0 / num_axes * i
        rotated_image = rotate(image, angle, reshape=False)
        if flip_and_compare(rotated_image, 'vertical', tolerance):
            symmetries.append(('Vertical', angle))
        if flip_and_compare(rotated_image, 'horizontal', tolerance):
            symmetries.append(('Horizontal', angle))
    return symmetries

def fourier_descriptors(contour, num_descriptors=10):
    contour_array = contour[:, 0, :]
    contour_complex = contour_array[:, 0] + 1j * contour_array[:, 1]
    fourier_result = fft(contour_complex)
    
    # Retain only the first few descriptors
    descriptors = np.zeros(fourier_result.shape, dtype=complex)
    descriptors[:num_descriptors] = fourier_result[:num_descriptors]
    
    return descriptors

def rotational_symmetry_with_phase_correlation(image, angles=[90, 180, 270], tolerance=1):
    rows, cols = image.shape
    for angle in angles:
        rotated_image = rotate(image, angle, reshape=False)
        correlation = cv2.phaseCorrelate(image.astype(np.float32), rotated_image.astype(np.float32))
        if np.linalg.norm(correlation[0]) < tolerance:
            return True, angle
    return False, None

def visualize_symmetry(image, symmetries, rotational_symmetry_detected, angle):
    num_symmetries = len(symmetries) + (1 if rotational_symmetry_detected else 0)
    num_plots = min(num_symmetries, 4)  # Limit to 4 subplots for a 2x2 grid

    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    # Visualize reflection symmetry
    for i, (symmetry_type, angle) in enumerate(symmetries[:num_plots-1]):
        plt.subplot(2, 2, i+2)
        rotated_image = rotate(image, angle, reshape=False)
        plt.imshow(rotated_image, cmap='gray')
        if symmetry_type == 'Vertical':
            plt.axhline(image.shape[0] // 2, color='r', linestyle='--', linewidth=2)
        elif symmetry_type == 'Horizontal':
            plt.axvline(image.shape[1] // 2, color='r', linestyle='--', linewidth=2)
        plt.title(f'{symmetry_type} Symmetry at {angle}°')

    # Visualize rotational symmetry
    if rotational_symmetry_detected and num_plots < 4:
        plt.subplot(2, 2, num_plots)
        rotated_image = rotate(image, angle, reshape=False)
        plt.imshow(rotated_image, cmap='gray')
        plt.title(f'Rotational Symmetry at {angle}°')
    
    plt.tight_layout()
    plt.show()

def detect_curves(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) > 5:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                cv2.putText(image, 'Ellipse', (int(ellipse[0][0]), int(ellipse[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                cv2.putText(image, 'Square', (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.drawContours(image, [approx], -1, (255, 0, 0), 2)
                cv2.putText(image, 'Rectangle', (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        elif len(approx) > 4:
            cv2.drawContours(image, [approx], -1, (255, 255, 0), 2)
            cv2.putText(image, 'Curve', (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return image

def detect_symmetry(image):
    binary_image, centroid = preprocess_image(image)
    
    # Test for reflection symmetry along multiple axes
    symmetries = check_reflection_symmetry(binary_image, num_axes=36, tolerance=1)
    
    # Test for rotational symmetry using phase correlation
    rotational_symmetry_detected, angle = rotational_symmetry_with_phase_correlation(binary_image)
    
    # Visualize results
    visualize_symmetry(image, symmetries, rotational_symmetry_detected, angle)
    
    if not symmetries and not rotational_symmetry_detected:
        print("No symmetry detected.")

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Detect and display curves
    image_with_curves = detect_curves(image.copy())
    cv2.imshow('Curves Detected', image_with_curves)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detect and visualize symmetry
    detect_symmetry(image)

if __name__ == "__main__":
    image_path = '/Users/srivatsapalepu/Downloads/problems/occlusion1.jpg'
    main(image_path)

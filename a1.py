import cv2
import numpy as np

def detect_curves(image_path):
    image = cv2.imread(image_path)
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

    result_image_path = 'curves_detected.jpg'
    cv2.imwrite(result_image_path, image)
    return result_image_path

def main(jpg_path):
    result_image_path = detect_curves(jpg_path)
    print(f"Curves detected and saved to {result_image_path}")

if __name__ == "__main__":
    jpg_path = '/Users/srivatsapalepu/Downloads/frag0.jpg'
    main(jpg_path)




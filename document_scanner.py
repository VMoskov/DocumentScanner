import cv2
import imutils
import argparse
import numpy as np
from pathlib import Path
from skimage.filters import threshold_local


def get_paper_contour(contours):
    '''Finds the paper contour in the list of contours'''
    for ctr in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(ctr, True)
        approximated_contour = cv2.approxPolyDP(ctr, 0.02 * perimeter, True)

        # If the approximated contour has 4 points, then we can assume that we have found the paper
        if len(approximated_contour) == 4:
            return approximated_contour

    return None


def order_points(points):
    '''Orders the points in the contour such that the points are ordered clockwise starting from the top-left point'''
    ordered_points = np.zeros((4, 2), dtype='float32')  # top-left, top-right, bottom-left, bottom-right

    coord_sum = points.sum(axis=1)
    ordered_points[0] = points[np.argmin(coord_sum)]  # top-left point has the smallest sum of x and y
    ordered_points[3] = points[np.argmax(coord_sum)]  # bottom-right point has the largest sum of x and y

    coord_diff = np.diff(points, axis=1)
    ordered_points[1] = points[np.argmin(coord_diff)]  # top-right point has the smallest difference of x and y
    ordered_points[2] = points[np.argmax(coord_diff)]  # bottom-left point has the largest difference of x and y

    return ordered_points


def calculate_document_ratio(contour):
    '''Calculates the aspect ratio of the document'''
    top_width = np.linalg.norm(contour[1] - contour[0])
    bottom_width = np.linalg.norm(contour[3] - contour[2])
    left_height = np.linalg.norm(contour[2] - contour[0])
    right_height = np.linalg.norm(contour[3] - contour[1])

    width = (top_width + bottom_width) / 2
    height = (left_height + right_height) / 2

    return width / height


def detect_color(image, threshold=50):
    '''Determines if the document is colored or grayscale'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    return mean_saturation > threshold


def enhance_lightness(image, binary_mask, alpha=1.25, beta=-20):
    '''Enhances the lightness channel of the LAB image'''
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # LAB = Lightness, A (green to red), B (blue to yellow)
    L, A, B = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Contrast Limited Adaptive Histogram Equalization
    L = clahe.apply(L)
    L = cv2.bitwise_and(L, L, mask=binary_mask)  # Keep only the lightness channel where the paper is
    L = cv2.convertScaleAbs(L, alpha=alpha, beta=beta)  # Enhance the lightness channel
    scan = cv2.merge([L, A, B])
    scan = cv2.cvtColor(scan, cv2.COLOR_LAB2BGR)
    return scan


def diagnostic_output():
    '''Shows the image with the title'''
    ### Diagnostic Output 1 ###
    cv2.imshow('Image', image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    ### Diagnostic Output 2 ###
    cv2.imshow('Paper Contour', image_contour)
    cv2.waitKey(0)
    ### Diagnostic Output 3 ###
    cv2.imshow('Original Image', original)
    cv2.imshow('Scan', scan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help='Path to the image to be scanned')
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f'Error: {image_path} not found')
        exit(1)
    
    image = cv2.imread(image_path)
    scale_ratio = image.shape[0] / 500.0  # Keep track of the ratio of the original image to the resized image
    original = image.copy()
    image = imutils.resize(image, height=500)

    # Convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)

    ### Diagnostic Output 1 ###

    # Find all the contours (geometric shapes surrounded by edges) in the edge detected image, and keep the top 5
    contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Get the top 5 contours

    # Find the paper contour
    paper_contour = get_paper_contour(contours)

    if paper_contour is None:
        print('Error: Paper not found')
        exit(1)

    # Show the paper contour
    image_contour = image.copy()
    cv2.drawContours(image_contour, [paper_contour], -1, (0, 255, 0), 2)
    
    ### Diagnostic Output 2 ###

    # Apply a four point perspective transform to get a top-down view of the paper
    paper_contour = paper_contour.reshape(4, 2).astype(np.float32) * scale_ratio
    paper_contour = order_points(paper_contour)
    aspect_ratio = calculate_document_ratio(paper_contour)  # Get the aspect ratio of the document

    # We want the document to be 800 pixels high
    H = 800
    W = int(H * aspect_ratio)

    destination_points = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype='float32')
    perspective_transform = cv2.getPerspectiveTransform(paper_contour, destination_points)
    warped_image = cv2.warpPerspective(original, perspective_transform, (W, H))

    # We give the warped image a paper scan effect
    warped_image_grayscale = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped_image_grayscale, 11, offset=10, method='gaussian')
    binary_mask = (warped_image_grayscale > T).astype('uint8') * 255

    is_colored = detect_color(warped_image)  # Dynamically determine if the document is colored or grayscale

    if not is_colored:
        scan = binary_mask
    else:
        scan = enhance_lightness(warped_image, binary_mask)

    # interpolate the scanned image to increase the resolution
    scan = cv2.resize(scan, (int(original.shape[0] * aspect_ratio), original.shape[0]), interpolation=cv2.INTER_CUBIC)

    ### Diagnostic Output 3 ###
    diagnostic_output()
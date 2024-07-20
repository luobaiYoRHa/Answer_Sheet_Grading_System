# Automated Exam Grading System

This project is an automated exam grading system built using OpenCV. The system processes scanned images of multiple-choice exam sheets, detects filled answer bubbles, compares them against a predefined answer key, and highlights the results.

## Features

- Utilizes OpenCV for edge detection, contour detection, and perspective transformation to isolate answer bubbles.
- Implements thresholding and bitwise operations to determine filled bubbles, then sorts contours for comparison against an answer key.
- Overlays grading results on original images, highlighting correct and incorrect answers with distinct colors, and displays scores directly on the processed images.


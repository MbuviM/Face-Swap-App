# Face Swap Application

This project implements a face swapping application using OpenCV and MediaPipe. The application can perform face swapping in real-time using a webcam, which can be useful for anonymity or entertainment purposes.

## Features

- Real-time face detection and swapping using a webcam.
- Seamless blending of swapped faces for a natural look.
- Resize and adjust images for better face alignment.

## Requirements

- Python 3.7+
- OpenCV 4.5+
- MediaPipe 0.8+
- NumPy 1.19+

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/face-swap-app.git
    cd face-swap-app
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    pip install virtualenv
    virtualenv .venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your source image**:
    - Place your source image in the `swap` directory and name it `source.jpg` (or modify the code to use a different image name).

2. **Run the application**:
    ```sh
    cd swap
    python functionality.py
    ```

3. **Using the application**:
    - The application will open your webcam.
    - The source face will be swapped onto the detected face in the webcam feed.
    - Press `q` to quit the application.

## How It Works

- **Face Detection**: Uses MediaPipe's FaceMesh to detect facial landmarks.
- **Affine Transformation**: Warps the source face to match the geometry of the destination face.
- **Seamless Cloning**: Blends the warped source face with the destination face using OpenCV's `seamlessClone` for a natural look.

## Code Overview

- `functionality.py`: Contains the main logic for face detection, landmark extraction, affine transformation, and seamless cloning.
- `source.jpg`: Default source image used for face swapping.

### Key Functions

- `resize_image(image, width)`: Resizes the image to a specified width while maintaining aspect ratio.
- `detect_faces(image_gray)`: Detects faces in a grayscale image using a pre-trained Haar cascade classifier.
- `get_face_landmarks(image, face_mesh)`: Extracts facial landmarks from an image using MediaPipe's FaceMesh.
- `apply_affine_transform(src, src_tri, dst_tri, size)`: Applies affine transformation to warp triangles from the source face to the destination face.
- `warp_triangle(img1, img2, t1, t2)`: Warps and blends triangular regions between two images.
- `swap_faces_real_time(source_image_path, resize_width=500)`: Main function to perform real-time face swapping using a webcam feed.

## Troubleshooting

- **Issue**: Application does not start or crashes immediately.
  - **Solution**: Ensure your webcam is connected properly and the source image path is correct.

- **Issue**: No face detected or poor swapping quality.
  - **Solution**: Improve lighting conditions and ensure the source image has a clear, frontal view of the face.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision functions.
- [MediaPipe](https://mediapipe.dev/) for face detection and landmark extraction.


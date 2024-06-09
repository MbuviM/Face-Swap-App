import numpy as np
import cv2
import mediapipe as mp
import os

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_size = (width, int(width / aspect_ratio))
    return cv2.resize(image, new_size)

def detect_faces(image_gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    if len(faces) == 0:
        return None
    return faces

def get_face_landmarks(image, face_mesh):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
        return landmarks
    return None

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img2_rect = img2_rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

def swap_faces_real_time(source_image_path, resize_width=500):
    load_source_image = cv2.imread(source_image_path)
    if load_source_image is None:
        print(f"Error: Could not read the image '{source_image_path}'. Check the file path and file integrity.")
        return

    # Resize source image
    load_source_image = resize_image(load_source_image, resize_width)
    source_IMAGE_gray = cv2.cvtColor(load_source_image, cv2.COLOR_BGR2GRAY)
    source_landmarks = get_face_landmarks(load_source_image, face_mesh)

    if source_landmarks is None:
        print("No face landmarks found in source image.")
        return

    source_landmarks = np.array(source_landmarks * [load_source_image.shape[1], load_source_image.shape[0]], dtype=np.int32)

    # Delaunay triangulation for source face
    rect = cv2.boundingRect(source_landmarks)
    subdiv = cv2.Subdiv2D(rect)
    for p in source_landmarks:
        subdiv.insert((int(p[0]), int(p[1])))
    triangles = subdiv.getTriangleList()
    triangles = np.array([[int(pt) for pt in tri] for tri in triangles], dtype=np.int32)

    # Fix triangle indices
    indices = []
    for tri in triangles:
        idx1 = np.where((source_landmarks == tri[0:2]).all(axis=1))[0]
        idx2 = np.where((source_landmarks == tri[2:4]).all(axis=1))[0]
        idx3 = np.where((source_landmarks == tri[4:6]).all(axis=1))[0]
        if len(idx1) > 0 and len(idx2) > 0 and len(idx3) > 0:
            indices.append([idx1[0], idx2[0], idx3[0]])

    # Open webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        destination_IMAGE_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        destination_landmarks = get_face_landmarks(frame, face_mesh)

        if destination_landmarks is not None:
            destination_landmarks = np.array(destination_landmarks * [frame.shape[1], frame.shape[0]], dtype=np.int32)

            # Swap faces in the frame
            for idx in indices:
                t1 = [tuple(source_landmarks[i]) for i in idx]
                t2 = [tuple(destination_landmarks[i]) for i in idx]
                warp_triangle(load_source_image, frame, t1, t2)

        cv2.imshow('Swapped Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Current Working Directory:", os.getcwd())
    source_image = 'images\source.jpg'
    swap_faces_real_time(source_image)

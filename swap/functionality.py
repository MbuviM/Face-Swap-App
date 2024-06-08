import numpy as np
import cv2
import mediapipe as mp
import os

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image '{image_path}'. Check the file path and file integrity.")
    return image

def detect_faces(image_gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    if len(faces) == 0:
        print("No faces found.")
        return None
    return faces

def get_face_landmarks(image, face_mesh):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
        return landmarks
    print("No face landmarks found.")
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
        t1_rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img2_rect = img2_rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

def swap_faces(source_image_path, destination_image_path):
    load_source_image = load_image(source_image_path)
    load_destination_image = load_image(destination_image_path)

    if load_source_image is None or load_destination_image is None:
        return

    source_IMAGE_gray = cv2.cvtColor(load_source_image, cv2.COLOR_BGR2GRAY)
    destination_IMAGE_gray = cv2.cvtColor(load_destination_image, cv2.COLOR_BGR2GRAY)

    source_faces = detect_faces(source_IMAGE_gray)
    destination_faces = detect_faces(destination_IMAGE_gray)

    if source_faces is None or destination_faces is None:
        return

    source_landmarks = get_face_landmarks(load_source_image, face_mesh)
    destination_landmarks = get_face_landmarks(load_destination_image, face_mesh)

    if source_landmarks is None or destination_landmarks is None:
        return

    source_landmarks = np.array(source_landmarks * [load_source_image.shape[1], load_source_image.shape[0]], dtype=np.int32)
    destination_landmarks = np.array(destination_landmarks * [load_destination_image.shape[1], load_destination_image.shape[0]], dtype=np.int32)

    # Delaunay triangulation
    rect = cv2.boundingRect(destination_landmarks)
    subdiv = cv2.Subdiv2D(rect)
    for p in destination_landmarks:
        subdiv.insert((int(p[0]), int(p[1])))
    triangles = subdiv.getTriangleList()
    triangles = np.array([[int(pt) for pt in tri] for tri in triangles], dtype=np.int32)

    for tri in triangles:
        t1 = []
        t2 = []
        for i in range(0, 6, 2):
            t1.append((source_landmarks[tri[i]//2][0], source_landmarks[tri[i]//2][1]))
            t2.append((destination_landmarks[tri[i]//2][0], destination_landmarks[tri[i]//2][1]))
        warp_triangle(load_source_image, load_destination_image, t1, t2)

    return load_destination_image

if __name__ == '__main__':
    source_image = 'source.jpg'
    destination_image = 'destination.jpg'
    swapped_image = swap_faces(source_image, destination_image)
    if swapped_image is not None:
        cv2.imshow('Swapped Faces', swapped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

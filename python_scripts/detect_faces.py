import cv2
import dlib

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_face(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Draw landmarks on face
        for n in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Save and display the result
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    cv2.imshow("Face with Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_path


# Run the function
if __name__ == "__main__":
    image_path = "face.jpg"  # Replace with your image path
    result = detect_face(image_path)
    print(f"Processed image saved at: {result}")

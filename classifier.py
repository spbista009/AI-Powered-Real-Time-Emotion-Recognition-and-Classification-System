import tkinter as tk  # Tkinter is used for creating the GUI.
import cv2  # OpenCV is used for accessing and manipulating video feed.
import PIL.Image  # PIL is used to handle images.
import PIL.ImageTk  # PIL.ImageTk is used to integrate images into Tkinter.
from deepface import DeepFace  # DeepFace is used for emotion recognition.

class SnapshotApp:
    def __init__(self, video_source=0):
        """
        Initializes the main application with video feed and a GUI.
        Args:
            video_source (int or str): The video source for OpenCV. Default is 0 for the primary webcam.
        """
        # Create the main Tkinter window
        self.window = tk.Tk()
        self.window.title("Camera Snapshots with Tkinter")  # Set the window title.

        # Store the video source and initialize the video capture object
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            # Raise an error if the video source cannot be opened
            raise ValueError("Unable to open video source", video_source)

        # Get the dimensions of the video feed
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Create a canvas in the Tkinter window to display the video feed
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()  # Add the canvas to the GUI.

        # Add a "Classifier" button that opens a menu for further options
        self.btn_classifier = tk.Button(self.window, text="Classifier", width=50, command=self.show_classifier_options)
        self.btn_classifier.pack(anchor=tk.CENTER, expand=True)

        # Create a frame to hold the classifier options
        self.classifier_frame = tk.Frame(self.window)
        self.classifier_frame.pack(anchor=tk.CENTER, expand=True)

        # Set a delay (in milliseconds) for updating the video feed
        self.delay = 15
        self.update()  # Start updating the video feed.
        self.window.mainloop()  # Start the Tkinter main loop.

    def __del__(self):
        """
        Ensures the video capture object is released when the application closes.
        """
        if self.vid.isOpened():
            self.vid.release()

    def update(self):
        """
        Continuously captures frames from the video source and displays them on the canvas.
        """
        ret, frame = self.vid.read()  # Read a frame from the video source.
        if ret:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV format) to RGB.
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))  # Convert the frame to a Tkinter-compatible image.
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # Display the image on the canvas.
        self.window.after(self.delay, self.update)  # Schedule the next update.

    def show_classifier_options(self):
        """
        Displays buttons for different classifiers under the "Classifier" menu.
        """
        # Clear any existing widgets in the classifier frame
        for widget in self.classifier_frame.winfo_children():
            widget.destroy()

        # Add a button for Emotion Recognition
        btn_emotion_recognition = tk.Button(self.classifier_frame, text="Emotion Recognition", command=self.open_emotion_recognition)
        btn_emotion_recognition.pack(side=tk.LEFT, padx=10)

        # Add a button for Cat vs Dog classification
        btn_cat_dog = tk.Button(self.classifier_frame, text="Cat vs Dog", command=self.open_cat_vs_dog)
        btn_cat_dog.pack(side=tk.LEFT, padx=10)

        # Add a button for Face Recognition
        btn_face_recognizer = tk.Button(self.classifier_frame, text="Face Recognizer", command=self.open_face_recognizer)
        btn_face_recognizer.pack(side=tk.LEFT, padx=10)

    def open_emotion_recognition(self):
        """
        Stops the current video feed and opens the Emotion Recognition module.
        """
        self.vid.release()  # Release the current video feed.
        self.window.destroy()  # Close the current window.
        EmotionRecognitionApp()  # Open the Emotion Recognition app.

    def open_cat_vs_dog(self):
        """
        Stops the current video feed and opens the Cat vs Dog module.
        """
        self.vid.release()
        self.window.destroy()
        CatVsDogApp()

    def open_face_recognizer(self):
        """
        Stops the current video feed and opens the Face Recognizer module.
        """
        self.vid.release()
        self.window.destroy()
        FaceRecognizerApp()

class EmotionRecognitionApp:
    def __init__(self):
        """
        Opens the Emotion Recognition module using DeepFace for emotion analysis.
        Allows the user to capture and save an image during the process.
        """
        # Initialize a face detector using OpenCV Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)  # Open the default webcam.

        while True:
            ret, frame = self.cap.read()  # Read a frame from the webcam.
            if not ret:
                break

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]  # Extract the region of interest (face).
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)  # Analyze emotions.
                emotion = result[0]['dominant_emotion']  # Get the dominant emotion.

                # Draw a rectangle around the face and display the emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display instructions on the video feed
            instructions = [
                "Press 's' to save a snapshot of the current frame.",
                "Press 'q' to exit."
            ]
            y_offset = 20
            for line in instructions:
                cv2.putText(
                    frame, line, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )
                y_offset += 30  # Move the next line down

            cv2.imshow('Emotion Detection', frame)  # Display the frame with annotations.

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Exit when 'q' is pressed.
                break
            elif key == ord('s'):  # Save a snapshot when 's' is pressed.
                self.save_snapshot(frame)

        self.cap.release()  # Release the video feed.
        cv2.destroyAllWindows()  # Close all OpenCV windows.

    def save_snapshot(self, frame):
        """
        Saves the current frame to the computer.
        Args:
            frame: The frame to save as an image.
        """
        # Generate a unique filename using a timestamp
        import time
        filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)  # Save the frame as a JPEG file.
        print(f"Snapshot saved as {filename}")


class CatVsDogApp:
    def __init__(self):
        """
        Placeholder for the Cat vs Dog classification module.
        """
        print("Cat vs Dog functionality is not implemented yet.")

class FaceRecognizerApp:
    def __init__(self):
        """
        Placeholder for the Face Recognition module.
        """
        print("Face Recognizer functionality is not implemented yet.")

# Entry point of the application
if __name__ == "__main__":
    SnapshotApp()

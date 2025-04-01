# AI-powered-Face-recognition-home-security-camera-using-TinyMl-
AI powered Face recognition home security camera using TinyMl 
# TinyML-Powered Face Recognition Home Security Camera
import cv2
import numpy as np
import os
import time
import datetime
from PIL import Image

# Define paths
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATASET_PATH = 'dataset/'
TRAINER_PATH = 'trainer/'
SECURITY_PATH = 'security_alerts/'

# Create directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(TRAINER_PATH, exist_ok=True)
os.makedirs(SECURITY_PATH, exist_ok=True)

# Load face cascade classifier - optimized for TinyML with frontal face focus
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Initialize video capture with modest resolution to save processing power
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Width - reduced for TinyML
cap.set(4, 240)  # Height - reduced for TinyML

# Function to collect face data for known people
def collect_face_data(face_id, name, num_samples=30):
    print(f"\n[INFO] Initializing face capture for {name}. Look at the camera and wait...")
    count = 0
    
    # Save the name to a file for later reference
    with open(f"{DATASET_PATH}names.txt", "a") as file:
        file.write(f"{face_id}:{name}\n")
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale to reduce computation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use optimized parameters for face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Lower scale factor for better detection
            minNeighbors=5,
            minSize=(20, 20)  # Smaller minimum size
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            
            # Save face image - only the face region to reduce storage
            cv2.imwrite(f"{DATASET_PATH}User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
            
        # Display progress
        cv2.putText(img, f"Samples: {count}/{num_samples}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Face Collection', img)
        
        # Wait for 100 milliseconds
        k = cv2.waitKey(100) & 0xff
        if k == 27:  # ESC to exit
            break
        elif count >= num_samples:
            break

    print(f"\n[INFO] {count} face samples collected for {name}")
    return


# Function to load name dictionary
def load_names():
    names = {0: "Unknown"}  # Default for unrecognized faces
    
    if os.path.exists(f"{DATASET_PATH}names.txt"):
        with open(f"{DATASET_PATH}names.txt", "r") as file:
            for line in file:
                if line.strip():
                    id_name = line.strip().split(":")
                    if len(id_name) == 2:
                        names[int(id_name[0])] = id_name[1]
    
    return names


# Function to train the recognizer with TinyML-optimized parameters
def train_recognizer():
    print("\n[INFO] Training faces. This may take a moment...")
    
    # Get all image paths
    image_paths = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]
    
    if not image_paths:
        print("[ERROR] No training images found. Please collect face data first.")
        return False
    
    face_samples = []
    ids = []
    
    # Use LBPH recognizer - efficient for TinyML applications
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,           # Smaller radius for faster computation
        neighbors=8,        # Standard neighborhood size
        grid_x=8,           # Coarser grid for efficiency
        grid_y=8,           # Coarser grid for efficiency
        threshold=100       # Default threshold
    )
    
    # Process each image efficiently
    for image_path in image_paths:
        try:
            # Convert to grayscale numpy array efficiently
            img_numpy = np.array(Image.open(image_path).convert('L'), 'uint8')
            
            # Extract id from the image path (User.id.count.jpg)
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
            
            # Add face directly - skipping detection for efficiency
            face_samples.append(img_numpy)
            ids.append(face_id)
        except Exception as e:
            print(f"[WARNING] Error processing {image_path}: {e}")
            continue
    
    if not face_samples:
        print("[ERROR] Could not extract any valid faces from the dataset.")
        return False
    
    # Train recognizer
    try:
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(f"{TRAINER_PATH}trainer.yml")
        print(f"\n[INFO] Training complete. Model saved to {TRAINER_PATH}trainer.yml")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return False


# Function to recognize faces with TinyML optimizations
def recognize_faces():
    # Check if trained model exists
    if not os.path.exists(f"{TRAINER_PATH}trainer.yml"):
        print("[ERROR] No trained model found. Please train the recognizer first.")
        return
    
    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"{TRAINER_PATH}trainer.yml")
    
    # Load names dictionary
    names = load_names()
    
    # Set confidence threshold - higher value is more strict
    confidence_threshold = 70
    
    # Initialize security variables
    unknown_face_counter = 0
    alert_triggered = False
    last_save_time = time.time()
    save_interval = 5  # Save every 5 seconds when unknown detected
    
    print("\n[INFO] Face recognition started. Press 'ESC' to exit...")
    
    # Main recognition loop
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Create a working copy at reduced size for faster processing
        small_frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Scale coordinates back to original size
            x2, y2, w2, h2 = x*2, y*2, w*2, h*2
            
            # Draw rectangle around face on original image
            cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
            
            # Try to recognize the face
            try:
                # Resize face region to ensure consistent input size
                face_img = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                face_id, confidence = recognizer.predict(face_img)
                
                # Convert confidence to percentage (0 confidence is 100% match)
                match_percentage = round(100 - confidence)
                
                # Determine if face is known based on confidence threshold
                if confidence < confidence_threshold and face_id in names:
                    name = names[face_id]
                    # Reset unknown counter for known faces
                    unknown_face_counter = 0
                    color = (0, 255, 0)  # Green for known
                else:
                    name = "Unknown"
                    unknown_face_counter += 1
                    color = (0, 0, 255)  # Red for unknown
                    
                    # Save image of unknown person periodically
                    current_time = time.time()
                    if current_time - last_save_time > save_interval:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"{SECURITY_PATH}unknown_{timestamp}.jpg", img)
                        last_save_time = current_time
                        
                        # Trigger alert after several consecutive unknown detections
                        if unknown_face_counter > 10 and not alert_triggered:
                            alert_triggered = True
                            alert_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"\n[ALERT] Unknown person detected at {alert_timestamp}!")
                            # Here you would add code to send email/SMS alerts
                
                # Display name and confidence on the image
                cv2.putText(img, name, (x2+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img, f"{match_percentage}%", (x2+5, y2+h2-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            except Exception as e:
                print(f"[WARNING] Recognition error: {e}")
                continue
        
        # Display security status
        if alert_triggered:
            cv2.putText(img, "ALERT: Unknown Person!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display timestamp
        cv2.putText(img, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the video feed
        cv2.imshow('TinyML Security Camera', img)
        
        # Reset alert if no unknown faces detected for a while
        if alert_triggered and unknown_face_counter == 0:
            alert_triggered = False
        
        # Press 'ESC' to exit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    print("\n[INFO] Security camera stopped.")
    return


# Main function to run the security camera system
def run_security_camera():
    print("\n=======================================================")
    print("  TinyML-Powered Face Recognition Security Camera")
    print("=======================================================")
    
    while True:
        print("\nOptions:")
        print("1. Collect face data for a person")
        print("2. Train face recognizer")
        print("3. Start security camera")
        print("4. View collected faces")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            face_id = input("Enter face ID (1-100): ")
            try:
                face_id = int(face_id)
                if face_id < 1:
                    print("[ERROR] Face ID must be a positive number.")
                    continue
            except ValueError:
                print("[ERROR] Face ID must be a number.")
                continue
                
            name = input("Enter person's name: ")
            collect_face_data(face_id, name)
            
        elif choice == '2':
            train_recognizer()
            
        elif choice == '3':
            recognize_faces()
            
        elif choice == '4':
            # Display list of registered faces
            names = load_names()
            if len(names) <= 1:  # Only Unknown is present
                print("\n[INFO] No faces have been registered yet.")
            else:
                print("\nRegistered faces:")
                for face_id, name in names.items():
                    if face_id > 0:  # Skip the Unknown entry
                        print(f"  ID {face_id}: {name}")
            
        elif choice == '5':
            print("\n[INFO] Exiting program. Thank you!")
            break
            
        else:
            print("\n[ERROR] Invalid choice. Please enter a number between 1 and 5.")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Run the program
if __name__ == "__main__":
    try:
        run_security_camera()
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
    finally:
        # Ensure resources are released
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


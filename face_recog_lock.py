import face_recognition
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT) # servo motor
GPIO.setwarnings(False)

IR_PIN = 3
GPIO.setup(IR_PIN, GPIO.IN)  #ir sensor


p = GPIO.PWM(2, 50) # GPIO 2 for PWM with 50Hz
p.start(8.8) 

video_capture = cv2.VideoCapture(0)

img_neeraj = face_recognition.load_image_file('/home/pi/Desktop/Face_Recognition/images/Neeraj.jpg')
img_neeraj_encoding = face_recognition.face_encodings(img_neeraj)[0]

#img_sagar = face_recognition.load_image_file('/home/pi/Desktop/Face_Recognition/images/sagar.jpg')
#img_sagar_encoding = face_recognition.face_encodings(img_sagar)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [img_neeraj_encoding]
known_face_names = ["Neeraj"]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
try:
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                if GPIO.input (IR_PIN)== True:
                            print("close.............")
                            p.start(8.8)
                            
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    if name =='Neeraj':
                            print("Door open")
                            p.ChangeDutyCycle(6)
                            time.sleep(0.5)
                    else:
                            print("door close")
                            p.start(8.8)

                    
                            
                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 4)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)
            
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        

# Release handle to the webcam
except KeyboardInterrupt:
  p.start(5.8)
  GPIO.cleanup()
  video_capture.release()
  cv2.destroyAllWindows()

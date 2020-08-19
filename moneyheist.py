#module and library required to build Face recognition System

import face_recognition
import cv2

#objective: this code will help you in running face
#recognition on a video file and saving the results to a new
#video file

#Open the input movie file
#'VideoCapture' is a class for video capturing from video
#files, images sequences or cameras

inputVideo = cv2.VideoCapture("moneyheist.mp4")

#"CAP_PROP_FRAME_COUNT": it helps in finding number of frames in a video file.

length = int(inputVideo.get(cv2.CAP_PROP_FRAME_COUNT))

#create an output movie file(make sure resolution/frame rate
#matches input video!)
#So we capture a video, process it frame_by-frame and we want to save that
#video, it only possible by using "VideoWriter" object

#FourCC is a 4-byte code used to specify the video codec. the list
#of available codes can be found in fourcc.org. it is platform dependent

fourcc = cv2.VideoWriter_fourcc('M','P','E','G')

#24.07 - number of frames per second(fps)
#(640, 640) - frame size

output_video = cv2.VideoWriter('output.avi', fourcc, 24.07, (640,640))

#load some sample pictures and learn how to recognize them.
professor = face_recognition.load_image_file("professor.jpg")
professor_face_encoding = face_recognition.face_encodings(professor)[0]

#"face_recognition.face_encodings": it's a face_recognition 
#package which returns a list of 128-dimensional face encodings

lisbon = face_recognition.load_image_file("lisbon.jpg")
lisbon_face_encoding = face_recognition.face_encodings(lisbon)[0]

denver = face_recognition.load_image_file("denver.jpg")
denver_face_encoding = face_recognition.face_encodings(denver)[0]

stockholm = face_recognition.load_image_file("Stockholm.jpg")
stockholm_face_encoding = face_recognition.face_encodings(stockholm)[0]

helsinki = face_recognition.load_image_file("helsinki.jpg")
helsinki_face_encoding = face_recognition.face_encodings(helsinki)[0]

tokyo = face_recognition.load_image_file("tokyo.jpg")
tokyo_face_encoding = face_recognition.face_encodings(tokyo)[0]

rio = face_recognition.load_image_file("rio.jpg")
rio_face_encoding = face_recognition.face_encodings(rio)[0]

palermo = face_recognition.load_image_file("Palermo.jpg")
palermo_face_encoding = face_recognition.face_encodings(palermo)[0]

bogota = face_recognition.load_image_file("Bogota.jpg")
bogota_face_encoding = face_recognition.face_encodings(bogota)[0]

known_faces = [
professor_face_encoding, lisbon_face_encoding,
denver_face_encoding, stockholm_face_encoding, tokyo_face_encoding, rio_face_encoding,
palermo_face_encoding, bogota_face_encoding, helsinki_face_encoding]
#initialize some variables
face_locations = []
face_encondings = []
face_names = []
frame_number = 0

while True:
	#Grab a single frame of video
	ret, frame = inputVideo.read()
	frame_number += 1

	#Quit when the input video file ends
	if not ret:
		break

	#Convert the image from BGR color(which openCV uses) to RGB color
	#(which face_recognition uses)

	rgb_frame = frame[:, :, ::-1]

	#find all faces and face encodings in the current frame of video
	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	face_names = []
	for face_encoding in face_encodings:
		#see if the face is a match for the known faces
		match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

		name = None
		if match[0]:
			name = "Professor"
		elif match[1]:
			name = "Lisbon"
		elif match[2]:
			name = "Denver"
		elif match[3]:
			name = "Stockholm"
		elif match[4]:
			name = "Tokyo"
		elif match[5]:
			name = "Rio"
		elif match[6]:
			name = "Palermo"
		elif match[7]:
			name = "Bogota"
		elif match[8]:
			name = "Helsinki"

		face_names.append(name)
	#Label the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		if not name:
			continue

		#Draw a box around the face
		cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)

		#Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom-24), (right,bottom),(0,0,255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 8, bottom - 8), font, 0.6, (255,255,255),1)

	#write the resulting image to the output video file
	print("Writing frame {}/{}".format(frame_number, length))
	output_video.write(frame)

#All done!
inputVideo.release()
cv2.destroyAllWindows()
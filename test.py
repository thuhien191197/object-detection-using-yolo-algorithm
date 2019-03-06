from models import Model_yolov2
import cv2

model = Model_yolov2()
# image = cv2.imread("images/000004.jpg")
# model.image(image)

#model = Model_yolov2()
# image = cv2.imread("images/oto.mp4")
#model.counting_object()
video_path = "images/bandem.mp4"
cap = cv2.VideoCapture(video_path)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	frame = model.image(frame)

	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
  		break

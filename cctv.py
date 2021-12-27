IMAGE SEGMENTATION (OPTIONAL)

import cv2
cap=cv2.VideoCapture("cctv.mp4")
fgbg=cv2.createBackgroundSubtractorKNN(detectShadows=True)

while True:
	success, img = cap.read()
	fgmask=fgbg.apply(img)
	cv2.imshow("frame",img)
	cv2.imshow("FGMSK",fgmask)

	keyboard=cv2.waitKey(30)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()



MAIN PROJECT PROGRAM

import cv2
body_classifier=cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_fullbody.xml")
cap=cv2.VideoCapture("cctv.mp4")
while cap.isOpened():
	ret, frame = cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	bodies=body_classifier.detectMultiScale(gray,1.1,3)
	for(x,y,w,h) in bodies:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.imshow("body detection",frame)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break	
cap.release()
cv2.destroyAllWindows()

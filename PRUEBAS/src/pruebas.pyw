import cv2
import numpy as np
import time
import VideoStream
import time
################################################################################
IM_WIDTH = 640
IM_HEIGHT = 480 
numel = IM_WIDTH * IM_HEIGHT;
FRAME_RATE = 30
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up
cam_quit = 0
font = cv2.FONT_HERSHEY_SIMPLEX
ncnt = 0;
while cam_quit==0:
    millisI = int(round(time.time() * 1000))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.imwrite( 'w.png', image )
        cam_quit = 1
    image = videostream.read()
    if( image == None ):
        continue;
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray,(5,5))
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
    sobel2 = sobelx * sobelx + sobely * sobely
    sobel = cv2.compare( sobel2, (8*np.sum(sobel2)/(numel)), cv2.CMP_GT )
    sobel_3_channel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    numpy_horizontal_concat = np.concatenate((image, sobel_3_channel), axis=1)
    millisE = (round(time.time() * 1000)) - millisI
    str = "%d millisegundos" % ( round(millisE) )
    cv2.putText(numpy_horizontal_concat,
                            str,(10,40), font, 0.7,(200,255-ncnt,ncnt),2,cv2.LINE_AA)
    cv2.imshow( "ColorOr", numpy_horizontal_concat )
    cv2.imwrite( 'w.png', numpy_horizontal_concat )
    ncnt = ncnt + 0.01
    
###########################################
videostream.stop()
cv2.destroyAllWindows()

################################################################################
import cv2;#import numpy as np
import os;import time;
from imagemat import*;
# import pruebas
from detect_and_say import*;
###################################################
# pruebas;os._exit(0);
###################################################
IM_WIDTH = 1280
IM_HEIGHT = 720 
#IM_WIDTH = 640;IM_HEIGHT = 480
#IM_WIDTH = 640;IM_HEIGHT = 360
#IM_WIDTH = 320;IM_HEIGHT = 240
FRAME_RATE = 30
########################################################################

videostream = None;
while videostream==None:
    videostream = findcamera( IM_WIDTH, IM_HEIGHT, FRAME_RATE );
    if videostream==None:
        print( "No se detecto ninguna camara" )
        play_audio( "audios\\sincamara.wav", 1 )
        time.sleep(5)
    
print( "Camara detectada..." )
time.sleep(1)
cam_quit = 0
img = videostream.read()
cv2.imshow( "ColorOr", img )
topelem = 4000;
while cam_quit==0:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        cv2.imwrite( 'w.png', img )
        cam_quit = 1
    img = videostream.read()
    
    #img[:,:,0] = cv2.multiply(img[:,:,0], 1.2 );
    #img[:,:,1] = cv2.multiply(img[:,:,1], 1.2 );
    #img[:,:,2] = cv2.multiply(img[:,:,2], 1.2 );
    
    if( img.all() == None ):continue;
    meth = 0;
    NCartas, millisE0  = detect_and_say_func( img, 0, False, ChrtFinded, xPosFinded, topelem );
    if NCartas<=0:
        print( '                                               Metodo 1' )
        meth = 1;
        NCartas, millisE0  = detect_and_say_func( img, 1, False, ChrtFinded, xPosFinded, topelem );
        if NCartas<0:
            play_audio( "audios\\entornoli.wav", 0.1 )
            cv2.putText(img, "Ajuste la iluminacion del entrono.",(20,30), font, fontsize,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow( "ColorOr", img )
        else:
            cv2.putText(img, "%d milisegundos" % round(millisE0),(20,30), font, fontsize,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow( "ColorOr", img )
            
    print( millisE0, 'milisegundos' )
    if ( NCartas>0 ):
        print( NCartas, 'Cartas encontradas. con el metodo ', meth )
        say_charts( img, NCartas, ChrtFinded, Snums, Schtype );
    
videostream.stop()
cv2.destroyAllWindows()
time.sleep(1)
os._exit(0);
########################################################################

import cv2;import numpy as np;import os;import time
import VideoStream;import imagemat as imat
import time;from imagemat import*;  from detect_and_say import*;
################################################################################
millisI = int(round(time.time() * 1000))
img = cv2.imread( 'ima_s\\000MARCO_04.jpg' ) # 000MARCO_02 (3 5 6 8 9 10)
#img = cv2.imread( 'ima_s\\46_2.jpg' )
################################################################################
NCartas, millisE0  = detect_and_say_func( img, 0, False, ChrtLast, ChrtFinded, xPosFinded );
millisE = (round(time.time() * 1000)) - millisI
print millisE, 'milisegundos TOTAL'

if ( NCartas>0 ):
    print millisE0, 'milisegundos METH (0)'
    say_charts( img, NCartas, ChrtFinded, Snums, Schtype );

print NCartas, 'Cartas encontradas.'
#cv2.imshow( "ColorOr", img );cv2.waitKey()
os._exit(0);

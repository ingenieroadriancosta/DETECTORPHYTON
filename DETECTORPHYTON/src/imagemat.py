import cv2
import numpy as np
#import wave
import pygame
import time
import VideoStream
#########################################################################
def imreadopt( fname, n ):
    img = cv2.imread( fname, n )
    return img
#########################################################################
def imread( fname ):
    img = cv2.imread( fname )
    return img
#########################################################################
def rgb2gray( img ):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray
#########################################################################
def im2bw( img, th ):
    if len(img.shape)>2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        retval, img_bin = cv2.threshold( img_gray, th, 255,cv2.THRESH_BINARY );
    else:
        retval, img_bin = cv2.threshold( img, th, 255,cv2.THRESH_BINARY );
    return img_bin
#########################################################################
def im2bw_not( img, th ):
    if len(img.shape)>2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        retval, img_bin = cv2.threshold( img_gray, th, 255,cv2.THRESH_BINARY_INV );
    else:
        retval, img_bin = cv2.threshold( img, th, 255,cv2.THRESH_BINARY_INV );
    return img_bin
#########################################################################
def imhist( img_gray ):
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256]);
    return hist
#########################################################################
def graythresh( img_gray ):
    if len(img_gray.shape)>2:
        return 0
    level = cv2.threshold( img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )[0];
    return (float(level/255.0))
#########################################################################
def im2bw_otsu( img_gray ):
    if len(img_gray.shape)>2:
        return 0
    imgbw = cv2.threshold( img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )[1]
    return imgbw
#########################################################################
def im2bw_otsu_not( img_gray ):
    if len(img_gray.shape)>2:
        return 0
    imgbw = cv2.threshold( img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )[1]
    return imgbw
#########################################################################
def promth( img_gray ):
    if len(img_gray.shape)>2:
        return 0
    sm = np.sum(img_gray)
    level = sm/( img_gray.shape[0] * img_gray.shape[1] );
    return (float((level/255.0)))
#########################################################################
def imerode( img, KType, N ):
    kernelNxN = np.ones( (N,N), np.uint8 )
    erosion = cv2.erode( img, kernelNxN, iterations = 1 )
    return erosion
#########################################################################
def imdilate( img, KType, N ):
    kernelNxN = np.ones( (N,N), np.uint8 )
    erosion = cv2.dilate( img, kernelNxN, iterations = 1 )
    return erosion
#########################################################################
def imopen( img, KType, N ):
    kernelNxN = np.ones( (N,N), np.uint8 )
    erosion = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelNxN)
    return erosion
#########################################################################
def imclose( img, KType, N ):
    kernelNxN = np.ones( (N,N), np.uint8 )
    erosion = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelNxN)
    return erosion
#########################################################################
#########################################################################
#########################################################################
def imfill( img, N ):
    h, w = img.shape
    imfilled = np.array(img, copy=True)
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(imfilled, mask, (0,0), 255);
    fill_image = cv2.bitwise_or(img, ~imfilled)
    return fill_image
#########################################################################    
def bwlabel( img, n4u8 ):
    if len(img.shape)>2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, img_bin = cv2.threshold( img_gray, 128, 255,cv2.THRESH_BINARY_INV );
        retval, labels	= cv2.connectedComponents( thresh, 8, cv2.CV_32S )
    else:
        retval, labels	= cv2.connectedComponents( img, 8, cv2.CV_32S )
    nobj = cv2.minMaxLoc(labels)[1]
    return labels, nobj
#########################################################################    
def loadtemplete( imgname ):
    tmp2ret = cv2.imread( imgname )
    tmp2ret = rgb2gray(tmp2ret)
    tmp2ret = im2bw(tmp2ret, 255*graythresh(tmp2ret) )
    result = cv2.matchTemplate( tmp2ret, tmp2ret, cv2.TM_CCOEFF );
    (_, score, _, _) = cv2.minMaxLoc(result)
    return tmp2ret, score
#########################################################################    
def edgesobel( im_in, NordS, delcorner ):
    IM_WIDTH  = im_in.shape[1]
    IM_HEIGHT = im_in.shape[0]
    numel = IM_WIDTH * IM_HEIGHT
    sobelx = cv2.Sobel(im_in,cv2.CV_64F,1,0,ksize=NordS)
    sobely = cv2.Sobel(im_in,cv2.CV_64F,0,1,ksize=NordS)
    sobel2 = sobelx * sobelx + sobely * sobely
    sobel = cv2.compare( sobel2, (8*np.sum(sobel2)/(numel)), cv2.CMP_GT )
    if delcorner==True:
        sobel[0:IM_HEIGHT, 0:NordS] = 0
        sobel[0:IM_HEIGHT, IM_WIDTH-NordS:IM_WIDTH] = 0
        sobel[0:NordS,0:IM_WIDTH] = 0
        sobel[IM_HEIGHT-NordS:IM_HEIGHT,0:IM_WIDTH] = 0
    return sobel
    
#########################################################################    
def setprint( anyforp, bset ):
    if bset:
        print( anyforp )
    return bset
#########################################################################    
def compare_templete( imgb, tmplt, scrt, perc ):
    resized_image = cv2.resize(imgb, (tmplt.shape[1], tmplt.shape[0]))
    #resized_image = imdilate( resized_image, 'square', 3 )
    result = cv2.matchTemplate( resized_image, tmplt, cv2.TM_CCOEFF );
    (_, scoreh, _, _) = cv2.minMaxLoc(result)
    #print ((scoreh)/scrt)
    if (float((scoreh)/scrt))>=perc:
        return True;
    return False
#########################################################################   
def getROI( img, Nblur, Nsobel, Bsepare ):
    if len(img.shape)>2:
        img_gray = rgb2gray( img, );
    else:
        img_gray = img;
    IM_WIDTH  = img_gray.shape[1]
    IM_HEIGHT = img_gray.shape[0]
    ########################################################################
    numel = IM_WIDTH * IM_HEIGHT;
    NordB = Nblur;
    NordS = Nsobel;
    if NordB>0:
        img_gray = cv2.blur(img_gray,(NordB,NordB))
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=NordS)
    sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=NordS)
    sobel2 = sobelx * sobelx + sobely * sobely
    sobel = cv2.compare( sobel2, int((8*np.sum(sobel2)/(numel))), cv2.CMP_GT )
    sobel[0:IM_HEIGHT, 0:NordS] = 0
    sobel[0:IM_HEIGHT, IM_WIDTH-NordS:IM_WIDTH] = 0
    sobel[0:NordS,0:IM_WIDTH] = 0
    sobel[IM_HEIGHT-NordS:IM_HEIGHT,0:IM_WIDTH] = 0
    sobelf = cv2.bitwise_and( imfill( sobel, 8 ), cv2.bitwise_not(sobel) );
    #
    sobelf = imclose( sobelf, 'square', NordS )
    #
    if Bsepare:
        mxv = int(round( np.max(cv2.distanceTransform( sobelf, cv2.DIST_L2, 3 ))/2 ))
        sobelf = imopen( sobelf, 'square', int(mxv) );
    else:
        sobelf = imopen( sobelf, 'square', int(15) )
    #
    # cv2.imshow( "Color", (sobel) );cv2.waitKey()
    #
    # sobelf = imfill( sobelf, 8 );
    return sobelf
#########################################################################   
def getnum( img, tmplA, tmpl2, tmpl3, tmpl4, tmpl5, tmpl6, tmpl7, tmpl8, tmpl9, 
                    tmpl10, tmplJ, tmplQ, tmplK,
                scrA, scr2, scr3, scr4, scr5, scr6, scr7, scr8, scr9, scr10,
                scrJ, scrQ, scrK, 
                perA, per2, per3, per4, per5, per6, per7, per8, per9, per10, 
                perJ, perQ, perK
                            ):
    if compare_templete(img, tmplA, scrA, perA ):
        return 1;
    if compare_templete(img, tmpl2, scr2, per2 ):
        return 2;
    if compare_templete(img, tmpl3, scr3, per3 ):
        return 3;
    if compare_templete(img, tmpl4, scr4, per4 ):
        return 4;
    if compare_templete(img, tmpl5, scr5, per5 ):
        return 5;
    if compare_templete(img, tmpl6, scr6, per6 ):
        return 6;
    if compare_templete(img, tmpl7, scr7, per7 ):
        return 7;
    if compare_templete(img, tmpl8, scr8, per8 ):
        return 8;
    if compare_templete(img, tmpl9, scr9, per9 ):
        return 9;
    if compare_templete(img, tmpl10, scr10, per10 ):
        return 10;
    if compare_templete(img, tmplJ, scrJ, perJ ):
        return 11;
    if compare_templete(img, tmplQ, scrQ, perQ ):
        return 12;
    if compare_templete(img, tmplK, scrK, perK ):
        return 13;
    return 0;
    
#########################################################################   
def getcharttype( img, 
                    tmplH, tmplD, tmplT, tmplP,
                    scrH, scrD, scrT, scrP,
                    perH, perD, perT, perP ):
    if compare_templete(img, tmplH, scrH, perH ):
        return 1;
    if compare_templete(img, tmplD, scrD, perD ):
        return 2;
    if compare_templete(img, tmplT, scrT, perT ):
        return 3;
    if compare_templete(img, tmplP, scrP, perP ):
        return 4;
    return 0;
#########################################################################   
def play_audio( auname, timw ):
    pygame.mixer.init()
    pygame.mixer.music.load(auname)
    pygame.mixer.music.play()
    time.sleep(timw)
    while pygame.mixer.music.get_busy() == True:
        cv2.waitKey(1)
        time.sleep(timw)
        continue;
    pygame.mixer.music.stop()
    time.sleep(timw)
#########################################################################   
def repro_audio( numc, typc, timw ):
    pygame.mixer.init()
    
    
    nm = "audios\\%s.wav" %numc;
    ty = "audios\\%s.wav" %typc;
    
    pygame.mixer.music.load(nm)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        cv2.waitKey(1)
        time.sleep(timw)
        continue;
    pygame.mixer.music.stop()
    time.sleep(timw)
    
    pygame.mixer.music.load('audios\\de.wav')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        cv2.waitKey(1)
        time.sleep(timw)
        continue;
    pygame.mixer.music.stop()
    time.sleep(timw)
    
    pygame.mixer.music.load(ty)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        cv2.waitKey(1)
        time.sleep(timw)
        continue;
    pygame.mixer.music.stop()
    time.sleep(timw)
#########################################################################    
def sort_charts( Chfind, xposfnd, NChr ):
    for i0 in range(0, NChr):
        for i1 in range(i0, NChr):
            if (xposfnd[i1]<xposfnd[i0]):
                i0t_00 = Chfind[i0,0]
                i0t_01 = Chfind[i0,1]
                x0t = xposfnd[i0]
                Chfind[i0, 0] = Chfind[i1, 0];
                Chfind[i0, 1] = Chfind[i1, 1];
                xposfnd[i0] = xposfnd[i1];
                #
                Chfind[i1, 0] = i0t_00;
                Chfind[i1, 1] = i0t_01;
                xposfnd[i1]   = x0t;
                
                
                
    return 0;
#########################################################################    
def say_charts( img, NCartas, ChrtFinded, Snums, Schtype ):
        cv2.imshow( "ColorOr", img )
        print( 'Nuevas cartas diferentes encontradas: ', NCartas )
        for ind in range(0,NCartas):
            str = "%s de %s" % (Snums[ChrtFinded[ind,0]], Schtype[ChrtFinded[ind,1]])
            print( str )
            try:
                repro_audio( Snums[ChrtFinded[ind,0]], Schtype[ChrtFinded[ind,1]], 0.1 );
            except:
                continue;
    
#########################################################################


def findcamera( IM_WIDTHP, IM_HEIGHTP, FRAME_RATEP):
    print( "Buscando camara..." )
    for PiOrUSB_P in range(1,3):
        for cmr_P in range(0,16):
            try:
                videostream = VideoStream.VideoStream((IM_WIDTHP,IM_HEIGHTP),FRAME_RATEP,PiOrUSB_P,cmr_P)
                videostream.start()
                break;
            except  Exception as e:
                print( e )
                videostream = None;
        if videostream==None:
            continue;
        break;
    if videostream==None:
        return videostream;
    time.sleep(1) # Give the camera time to warm up
    img = videostream.read()
    if img.all() == None:
        videostream.stop();
        return videostream;
    sap = np.sum(img)
    if sap == 0:
        videostream.stop();
        videostream = None;
    return videostream;
    




perA = 0.64;
per2 = 0.64;
per3 = 0.7;
per4 = 0.64;
per5 = 0.65;
per6 = 0.64;
per7 = 0.64;
per8 = 0.64;
per9 = 0.67;
per10 = 0.7;
perJ = 0.64;
perQ = 0.64;
perK = 0.64;

perctemp_t   = 0.8;
perctemp_pic = 0.8;
perctemp_h   = 0.8;
perctemp_d   = 0.8;
tmplA, scrA = loadtemplete( 'pat\\A.jpg' )
tmpl2, scr2 = loadtemplete( 'pat\\2.jpg' )
tmpl3, scr3 = loadtemplete( 'pat\\3.jpg' )
tmpl4, scr4 = loadtemplete( 'pat\\4.jpg' )
tmpl5, scr5 = loadtemplete( 'pat\\5.jpg' )
tmpl6, scr6 = loadtemplete( 'pat\\6.jpg' )
tmpl7, scr7 = loadtemplete( 'pat\\7.jpg' )
tmpl8, scr8 = loadtemplete( 'pat\\8.jpg' )
tmpl9, scr9 = loadtemplete( 'pat\\9.jpg' )
tmpl10, scr10 = loadtemplete( 'pat\\10.jpg' )
tmplJ, scrJ = loadtemplete( 'pat\\J.jpg' )
tmplQ, scrQ = loadtemplete( 'pat\\Q.jpg' )
tmplK, scrK = loadtemplete( 'pat\\K.jpg' )

templeteh, scrh = loadtemplete( 'pat\\heart.jpg' )
templeted, scrd = loadtemplete( 'pat\\diamond.jpg' )
templetet, scrt = loadtemplete( 'pat\\trebol.jpg' )
templetep, scrp = loadtemplete( 'pat\\Picas.jpg' )
################################################################################
Snums = [ ' ', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K' ]
Schtype = [ " ", "Corazones", "Diamantes", "Treboles", "Picas" ]
################################################################################
MaxNChart = 10
ChrtFinded = np.zeros( (MaxNChart, 2), 'int' );
xPosFinded = np.zeros( (MaxNChart, 1), 'int' );
################################################################################
bshowbin  = False
bshowpass = False
font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 0.6;
######################
NordB = 3;
NordS = 3;
################################################################################

import cv2;import numpy as np;import os;import time
import time;from imagemat import*; 
#########################################################################    
def detect_and_say_func( img, method, saychart, ChrtFinded, xPosFinded, topelem ):
    millisI = int(round(time.time() * 1000))
    img_gray = rgb2gray(img);
    IM_WIDTH  = img_gray.shape[1]
    IM_HEIGHT = img_gray.shape[0]
    if method==0:
        sobelf = getROI( img_gray, NordB, NordS, True );
        imf = cv2.bitwise_and( img_gray, sobelf );
    else:
        imf = img_gray;
    #level    = 90.0/255;#graythresh(imf)
    level    = graythresh(imf);#graythresh(imf)
    #  level = (np.sum( imf )/(1.0*cv2.countNonZero(imf)))/255.0;
    img_bin = im2bw_not(imf, round(255*level) )
    # img_bin = cv2.bitwise_or( img_bin, edgesobel( imf, 3, True ) );
    #cv2.imshow( "ColorOr", img_bin );cv2.waitKey()
    ################################################################################
    cnts, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = False) # get largest five contour area
    rects = []
    ########################################################################
    NCartas = 0;
    ChrtFinded[0:MaxNChart,0] = 0
    ChrtFinded[0:MaxNChart,1] = 0
    #for ind in range(0,MaxNChart):
        #ChrtFinded[ind,0] = 0
        #ChrtFinded[ind,1] = 0
    ########################################################################
    xymaxpi = 20
    W_MAXDEL = 0.9*IM_WIDTH
    H_MAXDEL = 0.9*IM_HEIGHT
    if len(cnts)>topelem:
        return -1, 0;
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP( c, 0.02 * peri, True )
        x, y, w, h = cv2.boundingRect(approx)
        if ((w<xymaxpi or h<xymaxpi) or (w>=(W_MAXDEL) or h>=(H_MAXDEL)) ):
            continue;
        rect = (x, y, w, h)
        rects.append(rect)
        img_t = img_gray[y:y+h, x:x+w];
        imt = im2bw_otsu_not(img_t);
        xptext = int(x+w/2-w/6)
        yptext = int(y+h/2)
        ############################################################################
        numoc = getnum( imt, tmplA, tmpl2, tmpl3, tmpl4, tmpl5, tmpl6, tmpl7, tmpl8, tmpl9, 
                        tmpl10, tmplJ, tmplQ, tmplK,
                        scrA, scr2, scr3, scr4, scr5, scr6, scr7, scr8, scr9, scr10,
                        scrJ, scrQ, scrK, 
                        perA, per2, per3, per4, per5, per6, per7, per8, per9, per10, 
                        perJ, perQ, perK
                                );
        if numoc>0:
            ###
            XADV = int(round( x - w/2.0 ));
            if XADV<0:
                XADV = 0;
            WADV = int(round( 2.5*w ));
            if WADV>IM_WIDTH:
                WADV = IM_WIDTH;
            ###
            HADV = int(round(3*h));
            if HADV>IM_HEIGHT:
                HADV = IM_HEIGHT
            ###
            img_typeg = img_gray[y:y+HADV, XADV:XADV+WADV];
            img_typeb = im2bw_otsu_not(img_typeg);
            typfinded = False;
            cntsT, hierarchy = cv2.findContours(img_typeb.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cntsT = sorted(cntsT, key = cv2.contourArea, reverse = False) # get largest five contour area
            for cT in cntsT:
                peri = cv2.arcLength(cT, True)
                approx = cv2.approxPolyDP( cT, 0.02 * peri, True )
                xt, yt, wt, ht = cv2.boundingRect(approx)
                img_t = img_typeb[yt:yt+ht, xt:xt+wt];
                ictyp = getcharttype( img_t, 
                            templeteh, templeted, templetet, templetep,
                            scrh, scrd, scrt, scrp,
                            perctemp_h, perctemp_d, perctemp_t, perctemp_pic );
                if ictyp>0:
                    typfinded = True;
                    break;
            if typfinded:
                W_MAXDEL = 3*w
                H_MAXDEL = 3*h
                xymaxpi = int( min( w/2.0, h/2.0 ) )
                ##
                xPosFinded[NCartas] = x;
                ChrtFinded[NCartas,0] = numoc;
                ChrtFinded[NCartas,1] = ictyp;
                NCartas = NCartas + 1;
                ##
                str = "%s de %s" % (Snums[numoc],Schtype[ictyp])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2);
                cv2.putText(img, str,(xptext,yptext), font, fontsize,(255,255,255),2,cv2.LINE_AA)
                continue;
    ####################################################################################
    millisE = (round(time.time() * 1000)) - millisI
    sort_charts( ChrtFinded, xPosFinded, NCartas );
    bsetcharts = NCartas>0;
    ##
    if bsetcharts and saychart:
        cv2.imshow( "ColorOr", img )
        print( 'Nuevas cartas diferentes encontradas: ', NCartas )
        for ind in range(0,NCartas):
            str = "%s de %s" % (Snums[ChrtFinded[ind,0]], Schtype[ChrtFinded[ind,1]])
            print( str )
            try:
                repro_audio( Snums[ChrtFinded[ind,0]], Schtype[ChrtFinded[ind,1]], 0.1 );
            except:
                continue;
    ################################################################################
    return NCartas, millisE;

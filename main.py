import cv2 as cv
import numpy as np
import glob

from numpy.lib.type_check import imag
path = "input\*.*"
def color_picker(image):
    def empty(a):
        pass
    cv.namedWindow("Range HSV")
    cv.resizeWindow("Range HSV", 500, 350)
    cv.createTrackbar("HUE Min", "Range HSV", 0,180,empty)
    cv.createTrackbar("HUE Max", "Range HSV", 180,180,empty)
    cv.createTrackbar("SAT Min", "Range HSV", 0,255,empty)
    cv.createTrackbar("SAT Max", "Range HSV", 255,255,empty)
    cv.createTrackbar("VALUE Min", "Range HSV", 0,255,empty)
    cv.createTrackbar("VALUE Max", "Range HSV", 255,255,empty)
    while True:
        # get value from trackbar
        h_min = cv.getTrackbarPos("HUE Min", "Range HSV")
        h_max = cv.getTrackbarPos("HUE Max", "Range HSV")
        s_min = cv.getTrackbarPos("SAT Min", "Range HSV")
        s_max = cv.getTrackbarPos("SAT Max", "Range HSV")
        v_min = cv.getTrackbarPos("VALUE Min", "Range HSV")
        v_max = cv.getTrackbarPos("VALUE Max", "Range HSV")
        lower_range = np.array([h_min,s_min,v_min])
        upper_range = np.array([h_max, s_max, v_max])
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        thresh = cv.inRange(hsv, lower_range, upper_range)
        bitwise = cv.bitwise_and(image, image, mask=thresh)
        cv.imshow("Original Image", image)
        cv.imshow("Thresholded", thresh)
        cv.imshow("Bitwise", bitwise)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
                mode = not mode
        elif k == 27:
                break
    cv.destroyAllWindows()
dict_colours={
    3:[[0,55,0],[10,255,255]],#red
    2:[[43,60,47],[103,255,255]],#green
    1:[[82,51,0],[170,255,255]],#blue
    4:[[8,58,0],[17,255,255]],#orange
    5:[[17,80,0],[60,255,255]],#yellow
    6:[[13,0,63],[56,22,221]]#white
}
for file in glob.glob(path):
    image=cv.imread(file)
    #cv.imshow("Input",cv.resize(image,(480,480)))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)[1]
    # cv.imshow("Threshold",thresh)
    # cv.waitKey()
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    cnt_max=[]
    mx_area=-1
    #print("Shape of image",image.shape)
    for i in cnts:
        #print("Contour of area ",cv.contourArea(i),"Found/n % 'of' area covered = ",cv.contourArea(i)/(image.shape[0]*image.shape[1]))
        if mx_area<=cv.contourArea(i):
            mx_area=cv.contourArea(i)
            cnt_max=i
    #print("MAX AREA CONTOUR ",cv.contourArea(cnt_max))

    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv.minAreaRect(cnt_max)
    box = np.int0(cv.boxPoints(rect))
    cv.drawContours(image, [box], 0, (36, 255, 12), 3)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(
        image, M, (width, height))  # will be used further
    warped = cv.resize(warped, (480, 480))
    warped[:,0:8,:]=0
    warped[:,warped.shape[0]-9:,:]=0
    warped[0:8,:,:]=0
    warped[warped.shape[1]-8:,:,:]=0
    #cv.imshow("Warped Img small",warped)
    #print("Rotation = ",rect[2])
    if rect[2]>=45:
        M = cv.getRotationMatrix2D((240, 240), -90, 1.0)
        rotated = cv.warpAffine(warped, M, (480, 480))
        #cv.imshow("Rotated by -90 Degrees", rotated)
    elif rect[2]<=-45:
        M = cv.getRotationMatrix2D((240, 240), -90, 1.0)
        rotated = cv.warpAffine(warped, M, (480, 480))
        #cv.imshow("Rotated by 90 Degrees", rotated)
    else :
        rotated=warped.copy()
    ##################Ref Image Making##########################
    ref_image=rotated.copy()
    font=cv.FONT_HERSHEY_COMPLEX
    ref_image=cv.putText(ref_image,"1",(80-20,80-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"2",(240-20,80-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"3",(400-20,80-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"4",(80-20,240-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"5",(240-20,240-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"6",(400-20,240-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"7",(80-20,400-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"8",(240-20,400-20),font,1.5,(0,255,0))
    ref_image=cv.putText(ref_image,"9",(400-20,400-20),font,1.5,(0,255,0))
    #cv.imshow("Reference Image",ref_image)
    ################################################################
    dict_results={}
    final_img=rotated.copy()
    for clr in dict_colours:
        hsv = cv.cvtColor(rotated, cv.COLOR_BGR2HSV)
        hsv_low = np.array(dict_colours[clr][0], np.int)
        hsv_high = np.array(dict_colours[clr][1], np.int)
        mask = cv.inRange(hsv, hsv_low, hsv_high)
        res = cv.bitwise_and(rotated, rotated, mask=mask)
        res=cv.cvtColor(res,cv.COLOR_BGR2GRAY)
        ret, thrash = cv.threshold(res, 40 , 255, cv.THRESH_BINARY)
        #cv.imshow("Thesg massk "+clr,res)
        #cv.waitKey()
        contours , hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cntrRect = []
        for i in contours:
            epsilon = 0.05*cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,epsilon,True)
            #print("For colour ",clr," Found contour of edge ",len(approx),"Area is ",cv.contourArea(i))
            if len(approx) == 4 and cv.contourArea(i)>=15000 and cv.contourArea(i)<=23000:
                #print("For colour ",clr,"Found number of contours ",len(contours))
                cv.drawContours(final_img,[approx],-1,(0,255,0),2)
                M = cv.moments(i)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #print("Founded Center ",cX," ",cY)
                final_img=cv.putText(final_img,str(cX)+','+str(cY),(cX-20,cY-20),font,0.4,(0,255,0))
                if cY<=95:
                    if cX<=95:
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['1']=clr
                    elif cX>95 and cX<=255:
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['2']=clr
                    elif cX>255 and cX<=415 :
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['3']=clr 
                elif cY>95 and cY<=255:
                    if cX<=95:
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['4']=clr
                    elif cX>95 and cX<=255:
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['5']=clr
                    elif cX>255 and cX<=415 :
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['6']=clr
                elif cY>255 and cY<=415:
                    if cX<=95:
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['7']=clr
                    elif cX>95 and cX<=255:
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['8']=clr
                    elif cX>255 and cX<=415 :
                        final_img=cv.circle(final_img,(cX,cY),5,(0,0,0),-1)
                        dict_results['9']=clr 
                cntrRect.append(approx)
        #print("Intermediate dictionery ",dict_results)
    s='Output'+file[5:file.index('.')]+'.txt'
    cnt=0
    lst=[]
    tm_lst=[]
    print("Result ",dict_results)
    cnt=0
    for key in sorted(dict_results):
        cnt+=1
        tm_lst.append(dict_results[key])
        if cnt%3==0:
            lst.append(tm_lst)
            tm_lst=[]
    textfile = open(s, "w")
    textfile.write(str(np.array(lst)))
    textfile.close()
    #cv.imshow("Final Image",final_img)
    #cv.waitKey()
# im=cv.imread("test/cube_rand.jpg")
# color_picker(im)

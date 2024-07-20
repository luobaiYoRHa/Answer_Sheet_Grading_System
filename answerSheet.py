import numpy as np
import imutils
import cv2

image_path = 'images/test_01.png'
#Correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

	# top-left, top-right, bottom-right, bottom-left in order
	# tl and br
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

	#tr and bl
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# get distance 
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	#coordinates after transforming
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0], #Avoid exceeding maximum width and height
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	#Computer transform matrix
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

image = cv2.imread(image_path)
contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv_show('blurred', blurred)
edged = cv2.Canny(blurred, 75, 200)
#cv_show('edged', edged)

#Contour Detection
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(contours_img,cnts,-1,(0,0,255),3) 
#cv_show('contours_img',contours_img)
docCnt = None

if len(cnts) > 0:
	# sort by area of coutours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		#Check if the contours have four sides
        if len(approx) == 4:        
            docCnt = approx
            break
        
# Implement perspective transforming
warped = four_point_transform(gray, docCnt.reshape(4, 2))
#cv_show('warped',warped)

# Otsu's threshhold 
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
#cv_show('thresh',thresh)
thresh_Contours = thresh.copy()

# Circle Contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(thresh_Contours,cnts,-1,(0,0,255),3) 
#cv_show('thresh_Contours',thresh_Contours)

questionCnts = []

#Detect answer circle from all contours
for c in cnts:
	# Compute ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    #This is cutomized for different shapes of choices
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
        
# Order from top to bot, 5 choices form a batch
questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
# 5 choices per row
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	# order 5 choices per row
    cnts = sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):
		# mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        #cv_show('mask',mask)
		# Count non-zero pixcel to get correct answer
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        #cv_show('mask',mask)
        total = cv2.countNonZero(mask)

		# Check by threshold
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

	# Compare correct answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    
    #correct choice
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(warped, [cnts[k]], -1, color, 3)
    
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)


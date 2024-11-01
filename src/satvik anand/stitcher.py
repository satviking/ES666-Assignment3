import pdb
import glob
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
class PanaromaStitcher():
    def __init__(self):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm =0, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        self.imageSet = [cv2.imread(each) for each in all_images]
        self.images = [cv2.resize(each,(480,320)) for each in self.imageSet ]
        #self.images = [cv2.resize(each,(int(0.5*each.shape[0]),int(0.5*each.shape[1])),interpolation = cv2.INTER_AREA) for each in self.imageSet ]
        self.pimages = [self.cylinderproj(each) for each in self.images]
        self.count = len(self.images)
        self.left_list, self.right_list,self.center_im = [],[],None
        self.arrange_imgs()

        self.say_hi()

        # Collect all homographies calculated for pair of images and return
        
        # Return Final panaroma
        stitched_image,Homogr = self.pan_creator()
        #####
        homography_matrix_list =Homogr
        
        return stitched_image, homography_matrix_list

    def say_hi(self):
        print('Hi From satvik')
    
    def match(self,i1,i2,direction = None):
        imageSet1 = self.Sift_out(i1)
        imageSet2 = self.Sift_out(i2)
        matches = self.flann.knnMatch(
            imageSet2['des'],
            imageSet1['des'],
            k =2)
        matched_features =[]
        for i,(m,n) in enumerate(matches):
            if(m.distance) < (0.5*n.distance):
                matched_features.append((m.trainIdx,m.queryIdx))
        if len(matched_features) >=10:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32(
                [pointsCurrent[i].pt for (_,i) in matched_features]
                )
            matchedPointsPrev = np.float32(
                [pointsPrevious[i].pt for (i,_) in matched_features]
                )
            H,inliers_curr,inliers_prev = self.ransac(matchedPointsCurrent,matchedPointsPrev,4)
            H = self.hom_calc(inliers_curr,inliers_prev)
            return H
        return None

    def hom_calc(self,current,previous):
        a_vals = []
        for i in range(len(current)):
            p1 = np.matrix([current[i][0],current[i][1],1])
            p2 = np.matrix([previous[i][0],previous[i][1], 1])

            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                  p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                  p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            a_vals.append(a1)
            a_vals.append(a2)

        matrixA = np.matrix(a_vals)

        #svd composition
        u, s, v = np.linalg.svd(matrixA)

        #reshape the min singular value into a 3 by 3 matrix
        H = np.reshape(v[8], (3, 3))

        #normalize and now we have h
        H = (1/H.item(8)) * H
        return H

    def ransac(self,current,previous, thresh):
        maxInliers_curr, maxInliers_prev =[],[]
        finalH = None
        random.seed(2)
        for i in range(1000):
            currFour = np.empty((0, 2))
            preFour = np.empty((0,2))
            for j in range(4):
                random_pt = random.randrange(0, len(current))
                curr = current[random_pt]
                pre = previous[random_pt]
                currFour = np.vstack((currFour,curr))
                preFour = np.vstack((preFour,pre))


            #call the homography function on those points
            h = self.hom_calc(currFour,preFour)
            inliers_curr = []
            inliers_prev =[]
            for i in range(len(current)):
                d = self.Distance(current[i],previous[i], h)
                if d < 10:
                    inliers_curr.append([current[i][0],current[i][1]])
                    inliers_prev.append([previous[i][0],previous[i][1]])

            if len(inliers_curr) > len(maxInliers_curr):
                maxInliers_curr = inliers_curr
                maxInliers_prev = inliers_prev
                finalH = h

            if len(maxInliers_curr) > (len(current)*thresh):
                break

        return finalH, maxInliers_curr,maxInliers_prev


    def Distance(self,current,previous, h):

        p1 = np.transpose(np.matrix([current[0], current[1], 1]))
        estimatep2 = np.dot(h, p1)
        estimatep2 = (1/estimatep2.item(2))*estimatep2

        p2 = np.transpose(np.matrix([previous[0], previous[1], 1]))
        error = p2 - estimatep2
        return np.linalg.norm(error)

    def Sift_out(self,im):
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        kp,des = self.sift.detectAndCompute(gray,None)
        return {'kp':kp,'des':des}
    def remove_extra_pix(self, image):
        # Ensure the image is uint8 format
        image = image.astype(np.uint8)

        # Convert BGR to RGB (if needed)
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale to find non-zero regions
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find the bounding box of non-zero pixels (non-black areas)
        coords = cv2.findNonZero(gray)  # Returns all non-zero points
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box from points

        # Crop the image using the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Save the cropped image
        cv2.imwrite("Cropped_Image.jpg", cropped_image)

        return cropped_image

    def arrange_imgs(self):
        self.centerIdx = int(self.count/2)
        self.center_im = self.images[self.centerIdx]
        for i in range(self.count):
            if i <= self.centerIdx :
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    def homogeneous_coordinate(self,coordinate):
        x = coordinate[0]/coordinate[2]
        y = coordinate[1]/coordinate[2]
        return x, y


    def wrap(self,a,b):
        H = self.match(a,b,"left")
        if H is None:
            return a , H
        h1, w1 = b.shape[:2]
        h2, w2 = a.shape[:2]

        row_number, column_number = int(b.shape[0]), int(b.shape[1])
        homography = H
        up_left_cor = self.homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
        up_right_cor = self.homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
        low_left_cor = self.homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
        low_right_cor = self.homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))
        corners2 =np.float32([up_left_cor,low_left_cor,low_right_cor,up_right_cor]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        Hnew = Ht.dot(H)
        homography = Hnew

        offset_x = math.floor(xmin)
        offset_y = math.floor(ymin)

        max_x = math.ceil(xmax)
        max_y = math.ceil(ymax)

        size_x = max_x - offset_x
        size_y = max_y - offset_y

        dsize = [size_x,size_y]
        homography_inverse = np.linalg.inv(homography)

        tmp = np.zeros((dsize[1], dsize[0], 3))
        tmp1= np.zeros((dsize[1],dsize[0],3))

        for x in range(size_x):
            for y in range(size_y):
                point_xy = self.homogeneous_coordinate(np.dot(homography_inverse, [[x], [y], [1]]))
                point_x = int(point_xy[0])
                point_y = int(point_xy[1])

                if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number):
                    tmp[y, x, :] = b[point_y, point_x, :]

        tmp1[t[1]:h2+t[1], t[0]:w2+t[0]] = a
        tmp = np.where(np.all(tmp == 0, axis=-1, keepdims=True), tmp1, tmp)
        tmp1 = np.where(np.all(tmp1 == 0, axis=-1, keepdims=True), tmp, tmp1)
        alpha = 0.5
        img_final = cv2.addWeighted(tmp1, alpha, tmp, 1 - alpha, 0)
        img_final = self.remove_extra_pix(img_final)
        img_final = img_final.astype(a.dtype)
        return img_final , H

    def pan_creator(self):
        a = self.pimages[self.centerIdx]
        hom = []
        for b in self.pimages[self.centerIdx+1:]:
            a , homo = self.wrap(a,b)
            if homo is not None:
                hom = np.append(hom,homo)
        group1 = a
        a = self.pimages[self.centerIdx]
        for b in self.pimages[0:self.centerIdx][::-1]:
            a, homo= self.wrap(a,b)
            if homo is not None:
                hom = np.append(hom,homo)
        group2 = a
        a = group1
        b = group2
        result,homo = self.wrap(a,b)
        if homo is not None:
            hom = np.append(hom,homo)
        self.leftImage = result
        self.homog = hom
        hom = np.array(hom)
        hom = hom.reshape((-1,3,3))
        return result,hom

    def Convert_xy(self,x, y):
        global center, f

        xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
        yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]

        return xt, yt


    def cylinderproj(self,GivenImages):
        global w, h, center, f
        h, w = GivenImages.shape[:2]
        center = [w // 2, h // 2]
        f = 600     # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

        modified_image = np.zeros(GivenImages.shape, dtype=np.uint8)

        AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
        ti_m = AllCoordinates_of_ti[:, 0]
        ti_n = AllCoordinates_of_ti[:, 1]

        ii_m, ii_n = self.Convert_xy(ti_m, ti_n)

        ii_tl_m = ii_m.astype(int)
        ii_tl_n = ii_n.astype(int)

        GoodIndices = (ii_tl_m >= 0) * (ii_tl_m <= (w-2)) * \
                      (ii_tl_n >= 0) * (ii_tl_n <= (h-2))

        ti_m = ti_m[GoodIndices]
        ti_n = ti_n[GoodIndices]

        ii_m = ii_m[GoodIndices]
        ii_n = ii_n[GoodIndices]

        ii_tl_m = ii_tl_m[GoodIndices]
        ii_tl_n = ii_tl_n[GoodIndices]

        dx = ii_m - ii_tl_m
        dy = ii_n - ii_tl_n

        weight_tl = (1.0 - dx) * (1.0 - dy)
        weight_tr = (dx)       * (1.0 - dy)
        weight_bl = (1.0 - dx) * (dy)
        weight_br = (dx)       * (dy)

        modified_image[ti_n, ti_m, :] = ( weight_tl[:, None] * GivenImages[ii_tl_n,     ii_tl_m,     :] ) + \
                                          ( weight_tr[:, None] * GivenImages[ii_tl_n,     ii_tl_m + 1, :] ) + \
                                          ( weight_bl[:, None] * GivenImages[ii_tl_n + 1, ii_tl_m,     :] ) + \
                                          ( weight_br[:, None] * GivenImages[ii_tl_n + 1, ii_tl_m + 1, :] )


        # Getting x coorinate to remove black region from right and left in the transformed image
        min_m = min(ti_m)

        # Cropping out the black region from both sides (using symmetricity)
        modified_image = modified_image[:, min_m : -min_m, :]

    #     return modified_image, ti_m-min_m, ti_n
        return modified_image
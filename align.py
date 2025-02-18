from __future__ import print_function
import cv2
import numpy as np
from time import time


MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.3


def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1Gray = cv2.medianBlur(im1Gray, 11)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    im1Gray = cv2.filter2D(im1Gray, -1, kernel=kernel)
    im1Gray = cv2.equalizeHist(im1Gray)

    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.medianBlur(im2Gray, 11)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    im2Gray = cv2.filter2D(im2Gray, -1, kernel=kernel)
    im2Gray = cv2.equalizeHist(im2Gray)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':

    # Read reference image
    refFilename = "frames/frame_4.jpg"
    print("Reading reference image : ", refFilename)
    # imReference = cv2.imread(refFilename)[332: ,718:, :]
    imReference = cv2.imread(refFilename)[:, 0:718, :]

    # Read image to be aligned
    imFilename = "frames/frame_480.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename)[:, 0:718, :]
    # im = cv2.imread(imFilename)[332: ,718:, :]

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    cv2.imwrite('aligned/frame2.jpg', im)
    cv2.imwrite('aligned/orb_frame1.jpg', imReference)
    cv2.imwrite('aligned/orb_frame2.jpg', imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)

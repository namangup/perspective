import cv2
import numpy as np
import sys
from pdf2image import convert_from_path
from argparse import ArgumentParser
import PIL
import os

def click(event, x, y, flags, param):
    global coords, img0
    if event ==  cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img0, (x, y), 2, (0, 0, 255), thickness=-1)
        coords.append([x, y])
        # print(x, y)

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts
    # print(pts)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

if __name__ == "__main__":
    global coords, img0
    parser = ArgumentParser('perspective')
    parser.add_argument('--pdf', default='', help='Work with a PDF')
    parser.add_argument('--dir', default='', help='Directory Path')
    parser.add_argument('--img', default='', help='Image Path')
    parser.add_argument('--out', default='', help='Output Directory')
    args = parser.parse_args()

    if args.dir != '':
        path = [os.path.join(args.dir, i) for i in os.listdir(args.dir)]
        imgs = []
        for i in path:
            imgs.append(cv2.imread(i))
        if args.out == '':
            args.out = 'processed/'

    elif args.img != '':
        path = args.img
        imgs = [cv2.imread(path)]
        if args.out == '':
            args.out = path.split('.')[0]+'/'

    elif args.pdf != '':
        imgs = convert_from_path(args.pdf)
        if args.out == '':
            args.out = args.pdf.split('.')[0]+'/'
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            imgs[i] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if not os.path.exists(os.path.join(os.getcwd(), args.out)):
        os.mkdir(os.path.join(os.getcwd(), args.out))

    idx = 0
    for img in imgs:
        coords = []
        h, w = img.shape[:2]
        img0 = cv2.resize(img, (960, 960))
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', click)
        while True:
            cv2.imshow('Image', img0)
            k = cv2.waitKey(delay=100)
            if k == 27:
                cv2.destroyAllWindows()
                break

        if len(coords) % 4 != 0:
            print('Number of clicks not a multiple of 4, Exiting')
            sys.exit(1)
        elif len(coords) == 0:
            continue
        else:
            pts = np.array(coords, dtype='float32')
            pts[:, 0] *= (w/960)
            pts[:, 1] *= (h/960)

            for i in range(len(pts)//4):
                k = 4*i
                pt = pts[k:k+4]
                transformed = four_point_transform(img, np.round(pt))
                # cv2.imshow('img', transformed)
                # cv2.waitKey(0)
                cv2.imwrite(f'{args.out}{idx}.jpg', transformed)
                idx += 1
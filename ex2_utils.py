import math
import numpy as np
import cv2

def myID() -> np.int:
    return 3022989963

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    k_size = k_size[::-1]
    #doing the convultion with multiplying
    return np.array([ np.dot(in_signal[max(0,i):min(i+len(k_size),len(in_signal))],
           k_size[max(-i,0):len(in_signal)-i*(len(in_signal)-len(k_size)<i)],)
       for i in range(1-len(k_size),len(in_signal)) ])



def conv2D(image, kernel):
    k=kernel
    #fliping the kernel
    kernel = np.flipud(np.fliplr(kernel))
    padding=k.shape[0] // 2
    # add zero paddings to the input image
    temp = np.pad(image, (k.shape[0] // 2, k.shape[1] // 2), 'edge').astype('float32')
    new_ = np.zeros((int(((image.shape[0] - kernel.shape[0] + 2 * padding) ) + 1), 
                       int(((image.shape[1] - kernel.shape[1] + 2 * padding) ) + 1)))
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image
    # Loop over every pixel of the image
    for y in range(image.shape[1]):
            for x in range(image.shape[0]):
                # element-wise multiplication of the kernel and the image

                        new_[x, y] = (temp[x:x + k.shape[0], y:y + k.shape[1]] * k).sum()
    return new_



def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    #the kernel for the derivative
    kernel = np.array([[0, 1, 0],[0, 0, 0], [0, -1, 0]])
    magnitude   = np.sqrt(np.abs(conv2D(in_image, kernel.transpose()))**2 + np.abs(conv2D(in_image, kernel))**2)
    #doing over all the way's
    div = np.arctan2(conv2D(in_image, kernel), conv2D(in_image, kernel.transpose()))
    return div, magnitude 


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    #create the kernel
    kernel = np.zeros((k_size, k_size))
    sigma = 1
    #building the Gaussian kernel to blur 
    for i in range(k_size):
        for j in range(k_size):
            chan = np.sqrt((i - (int)(k_size / 2)) ** 2 + (j - (int)(k_size / 2)) ** 2)
            kernel[i, j] = np.exp(-(chan ** 2) / (2 * sigma ** 2))
    #doing convolution btw original to the kernel we built
    return conv2D(in_image,kernel / np.sum(kernel))

def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    #sigma by the cv libary(recommended)
    gaussian_kernel = cv2.getGaussianKernel(k_size, 0.3*((k_size-1)*0.5 - 1) + 0.8 )
    kernel_2D = gaussian_kernel @ gaussian_kernel.transpose()
    return cv2.filter2D(in_image, -1, kernel_2D,borderType=cv2.BORDER_REPLICATE)



def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray, np.ndarray):
    if img.max() > 1:
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #using laplacian kernel
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    #bluring the image before starting
    blur = blurImage2(in_image=img, k_size=5)
    #building laplacian
    laplacian = conv2D(blur, lap)
    img_ = np.zeros(img.shape)
    l_rows, l_cols = lap.shape
    img_rows, img_cols = laplacian.shape
    for i in range(img_rows - (l_rows - 1)):
        for j in range(img_cols - (l_cols - 1)):
            if laplacian[i][j] == 0:
                # check neighbours
                if (laplacian[i][j - 1] < 0 and laplacian[i][j + 1] > 0) or \
                        (laplacian[i][j - 1] < 0 and laplacian[i][j + 1] < 0) or \
                        (laplacian[i - 1][j] < 0 and laplacian[i + 1][j] > 0) or \
                        (laplacian[i - 1][j] > 0 and laplacian[i + 1][j] < 0):
                    img_[i][j] = 255
            if laplacian[i][j] < 0:
                if (laplacian[i][j - 1] > 0) or (laplacian[i][j + 1] > 0) or (laplacian[i - 1][j] > 0) or (
                        laplacian[i + 1][j] > 0):
                    img_[i][j] = 255
    return img_



def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    _list = []
    #thresh choosed to start
    thresh1=0.7
    dire = np.arctan2(cv2.Sobel(img, cv2.CV_64F, 0, 1, thresh1), cv2.Sobel(img, cv2.CV_64F, 1, 0, thresh1)) * 180 / np.pi
    direct=np.radians(dire)
    #building
    circle=np.zeros((len(img),len(img[0]),max_radius+1))
    canny_ = cv2.Canny((img * 255).astype(np.uint8), 50, 100) / 255
    for x in range(0,len(canny_)):
        for y in range(0,len(canny_[0])):
            if canny_[x][y] > 0:
                for r in range(min_radius,max_radius+1):
                    cy1 = int(y+ r * np.sin(direct[x, y]-np.pi/2))
                    cx1 = int(x - r * np.cos(direct[x, y]-np.pi/2))
                    cy2 = int(y - r * np.sin(direct[x, y]-np.pi/2))
                    cx2 = int(x + r * np.cos(direct[x, y]-np.pi/2))
                    if 0 < cx1 < len(circle) and 0 < cy1 < len(circle[0]):
                        circle[cx1,cy1,r]+=1
                    if 0 < cx2 < len(circle) and 0 < cy2 < len(circle[0]):
                        circle[cx2,cy2,r]+=1
     # filter with  thresh
    thresh = 0.50*circle.max()
    b_center,a_center,radius=np.where(circle>=thresh)
    # delete similar circles
    eps = 10
    for j in range(0, len(a_center)):
        if a_center[j] == 0 and b_center[j] == 0 and radius[j] == 0:
            continue
        temp = (b_center[j], a_center[j], radius[j])
        index_a = np.where((temp[0]-eps <= b_center) & (b_center <= temp[0]+eps)
                           & (temp[1]-eps <= a_center) & (a_center <= temp[1]+eps)
                           & (temp[2]-eps <= radius) & (radius <= temp[2]+eps))[0]
        for i in range(1, len(index_a)):
            b_center[index_a[i]] = 0
            a_center[index_a[i]] = 0
            radius[index_a[i]] = 0
    # building the last list
    for i in range(0, len(a_center)):
        if a_center[i] == 0 and b_center[i] == 0 and radius[i] == 0:
            continue
        _list.append((a_center[i], b_center[i], radius[i]))
    print("Thresh: ",thresh)
    return _list



 

 
 
def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    img_m = np.zeros(in_image.shape)
    # Creates a gaussian kernel of given dimension.
    kernel = np.zeros((k_size, k_size))
    for i in range(0, k_size):
        for j in range(0, k_size):
            kernel[i, j] = math.sqrt(
                abs(i - k_size // 2) ** 2 + abs(j - k_size // 2) ** 2 )
    # For applying gaussian function for each element in matrix.
    sigma = math.sqrt(sigma_color)
    cons = 1 / (sigma * math.sqrt(2 * math.pi))
    gauss_kernel=cons * np.exp(-((kernel / sigma) ** 2) * 0.5)
    #passing over the image
    for i in range(k_size // 2, in_image.shape[0] - k_size // 2):
        for j in range(k_size // 2, in_image.shape[1] - k_size // 2):
            half = k_size // 2
            imgS= in_image[i - half : i + half + 1, j - half : j + half + 1]
            imgI = imgS - imgS[k_size // 2, k_size // 2]
            sigma = math.sqrt(sigma_color)
            cons = 1 / (sigma * math.sqrt(2 * math.pi))
            imgIG = cons * np.exp(-((imgI / sigma) ** 2) * 0.5)
            weights = np.multiply(gauss_kernel, imgIG)
            vals = np.multiply(imgS, weights)
            val = np.sum(vals) / np.sum(weights)
            img_m[i, j] = val
    #cv bilateral
    img_cv=cv2.bilateralFilter(in_image,k_size,sigma_color, sigma_space)
    return img_cv,img_m

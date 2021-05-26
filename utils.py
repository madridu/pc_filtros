import numpy as np
from numba import jit
from scipy import ndimage
from skimage.color import rgb2gray


def add_salt_pepper(image, prob):
    """
    Add salt and pepper noise to a given image.
    
    Parameters
    ----------
    image : numpy.array
        Image to add noise to.
    prob: float
        Probability of the noise.
    
    References
    ----------
    Taken from:
    https://gist.github.com/lucaswiman/1e877a164a69f78694f845eab45c381a
    """
    output = image.copy()
    
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    
    probs = np.random.random(image.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    
    return output


def add_noise(img, mean=0, std=1, cast=True):
    """Add gaussian noise to a given image to create a noisy
    image.
    
    Parameters
    ----------
    img : numpy.array
        Image to add noise to.
    mean : int or float
        Mean of the gaussian noise.
    std : int or float
        Standard deviation of the gaussian noise.
    
    References
    ----------
    .. [1] StackOverflow - Why does adding Gaussian noise to image give white screen?
       https://stackoverflow.com/questions/50641860/why-does-adding-gaussian-noise-to-image-give-white-screen
    """
    noisy_img = np.clip(img + np.random.normal(mean, std, img.shape), 0, 255)
    
    if cast:
        noisy_img = noisy_img.astype('uint8') # cast to unsigned integer. See references
    
    return noisy_img


def get_sketch(img, contour):
    """
    Create a sketch-like image given an image and its edges (contours)
    
    Parameters
    ----------
    img : numpy.array
        Image to get sketch from.
    contour : numpy.array
        Contour (edges) of the image.
        
    Returns
    -------
    sketch : numpy.array
        Sketch-like image.
    """
    sketch = img * contour
    return np.clip(sketch, 0, 1)


@jit(nopython=False)
def adaptive_median_filter(img, kernel_size=3):
    """
    Apply an adaptive median filter.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply the filter to.
    kernel_size : int
        Kernel size (sliding window)
        
    Returns
    -------
    out : numpy.array
        Filtered image.
    """
    h,w,c = img.shape
    
    # Zero padding
    pad = kernel_size // 2
    out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=np.float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
    
    # Zona adaptativa
    A_size = 7
    
    # Proceso de filtrado
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                max_tresh = np.mean(tmp[y:y+A_size,x:x+A_size,ci]) + np.std(tmp[y:y+A_size,x:x+A_size,ci])
                min_tresh = np.mean(tmp[y:y+A_size,x:x+A_size,ci]) - np.std(tmp[y:y+A_size,x:x+A_size,ci])
                if out[pad+y,pad+x,ci] > max_tresh or out[pad+y,pad+x,ci] < min_tresh:
                    out[pad+y,pad+x,ci] = np.median(tmp[y:y+kernel_size,x:x+kernel_size,ci])                 
    
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
    
    return out


@jit(nopython=False)
def median_filter(img, kernel_size=3):
    """
    Apply a median filter.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply the filter to.
    kernel_size : int
        Kernel size (sliding window)
        
    Returns
    -------
    out : numpy.array
        Filtered image.
    """
    h,w,c = img.shape
    
    # zero padding (handle borders)
    pad = kernel_size // 2
    out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=np.float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
    
    # filtering process
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci] = np.median(tmp[y:y+kernel_size,x:x+kernel_size,ci])
    
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
    return out


@jit(nopython=False)
def mean_filter(img, kernel_size=3):
    """
    Apply a median filter.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply the filter to.
    kernel_size : int
        Kernel size (sliding window)
        
    Returns
    -------
    out : numpy.array
        Filtered image.
    """
    h,w,c = img.shape
    
    # zero padding (handle borders)
    pad = kernel_size // 2
    out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=np.float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
    
    # filtering process
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci] = np.mean(tmp[y:y+kernel_size,x:x+kernel_size,ci])
    
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
    return out


def get_rgb_channels(img):
    """
    Gets the RGB channels of a given RGB image.
    
    Parameters
    ----------
    img : numpy.array
        RGB image
        
    Returns
    -------
    red : numpy.array
        Red channel.
    green : numpy.array
        Green channel.
    blue : numpy.array
        Blue channel.
    """
    msg = 'Input must be RGB image'
    assert len(img.shape) == 3, msg
    
    red = img.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0
    
    green = img.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    
    blue = img.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0
    
    return red, green, blue


def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.
        
    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton
    
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n)
    
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    
    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    
    # inverse fourier transform
    clean_data = np.fft.ifft(fft)
    
    if to_real:
        clean_data = clean_data.real
    
    return clean_data


def fourier_transform_denoising(img, n_components=1000):
    """
    Filters and image using the Fourier Transform
    
    Parameters
    ----------
    img : numpy.array
        RGB image.
    n_components : int or float
        Number of components above which the frequencies
        will be filtered.
    
    References
    ----------
    .. [1] Apeer Micro - Tutorial 40 - What is Fourier transform 
       and how is it relevant for image processing?
       https://www.youtube.com/watch?v=RVE-CSZijAI&ab_channel=Apeer_micro
    
    .. [2] Apeer Micro - Tutorial 41 - mage filtering using Fourier
       transform in python
       https://www.youtube.com/watch?v=9mLeVn8xzMw&ab_channel=Apeer_micro
    """
    shape = img.shape
    red, green, blue = get_rgb_channels(img=img)
    
    channel_list = [red, green, blue]
    filtered_list = []
    
    # denoise, cast and clip
    for channel in channel_list:
        res = fft_denoiser(x=channel.ravel(), n_components=n_components, to_real=True)
        res = res.reshape(shape).astype('uint')
        res = np.clip(res, 0, 255)
        
        filtered_list.append(res)
    
    return sum(filtered_list)


# ----------- Morphologic operations
def thresholding(img, umbral):
    """
    Umbralize (thresholding) operation.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply operation to.
    umbral : scalar
        Threshold.
        
    Returns
    -------
    np.array
        Umbralized image.
    """
    img_grey = rgb2gray(img)
    
    above = np.where(img_grey > umbral, 255, img_grey)
    below_equal = np.where(img_grey <= umbral, 0, above)
    
    return below_equal


def dilate(img, kernel_size=3):
    """
    Dilate operation.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply operation to.
    kernel_size : int
        Window (kernel) size.
    
    Returns
    -------
    out : numpy.array
        Resulting image.
    """
    h,w = img.shape
    
    # Zero padding
    pad = kernel_size // 2
    out = np.zeros((h + 2*pad, w + 2*pad), dtype=float)
    out[pad:pad+h, pad:pad+w] = img.copy().astype(float)
    
    se = np.array([[255,255,255],[255,255,255],[255,255,255]])
    
    for y in range(h):
        for x in range (w):
            mask = out[y:y+kernel_size, x:x+kernel_size] == se
            
            try:
                cond = mask.any()
            except:
                cond = mask
            
            if cond:
                out[y,x] = 255
    
    out = out[pad:pad+h, pad:pad+w].astype(np.uint8)
                
    return out


def erosion(img, kernel_size=3):
    """
    Erosion operation.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply operation to.
    kernel_size : int
        Window (kernel) size.
    
    Returns
    -------
    out : numpy.array
        Resulting image.
    """
    h,w = img.shape
    
    # Zero padding
    pad = kernel_size//2
    out = np.zeros((h + 2*pad,w + 2*pad),dtype=float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(float)
    
    se = np.array([[255,255,255],[255,255,255],[255,255,255]])
    
    for y in range(h):
        for x in range (w):
            comparition = out[y:y+kernel_size, x:x+kernel_size] == se
            if comparition.all():
                out[y,x] = 255
            else:
                out[y,x] = 0
    
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
                
    return out


def opening(img):
    """
    Opening operation.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply operation to.
    
    Returns
    -------
    np.array
        Resulting image.
    """
    eroded = erosion(img)
    dilated = dilate(eroded)
    
    return dilated


def closing(img):
    """
    Closing operation.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply operation to.
    
    Returns
    -------
    np.array
        Resulting image.
    """
    dilated = dilate(img)
    eroded = erosion(dilated)
    
    return eroded


def gradient(img):
    """
    Gradient operation.
    
    Parameters
    ----------
    img : numpy.array
        Image to apply operation to.
    
    Returns
    -------
    np.array
        Resulting image.
    """
    eroded = erosion(img)
    dilated = dilate(img)
    
    return eroded - dilated
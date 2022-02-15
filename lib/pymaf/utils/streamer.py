import cv2
import torch
import numpy as np
import imageio

def aug_matrix(w1, h1, w2, h2):
    dx = (w2 - w1) / 2.0
    dy = (h2 - h1) / 2.0
    
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])


    scale = np.min([float(w2)/w1, float(h2)/h1])

    M = get_affine_matrix(
        center = (w2 / 2.0, h2 / 2.0), 
        translate = (0, 0), 
        scale = scale)
    
    M = np.array(M + [0., 0., 1.]).reshape(3, 3)
    M = M.dot(matrix_trans)
    
    return M


def get_affine_matrix(center, translate, scale):
    cx, cy = center
    tx, ty = translate

    M = [1, 0, 0,
         0, 1, 0]
    M = [x * scale for x in M]

    # Apply translation and of center translation: RSS * C^-1
    M[2] += M[0] * (-cx) + M[1] * (-cy)
    M[5] += M[3] * (-cx) + M[4] * (-cy)

    # Apply center translation: T * C * RSS * C^-1
    M[2] += cx + tx
    M[5] += cy + ty
    return M


class BaseStreamer():
    """This streamer will return images at 512x512 size.
    """
    def __init__(self, 
                 width=512, height=512, pad=True, 
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 **kwargs):
        self.width = width
        self.height = height
        self.pad = pad
        self.mean = np.array(mean)
        self.std = np.array(std)
        
        self.loader = self.create_loader()

    def create_loader(self):
        raise NotImplementedError
        yield np.zeros((600, 400, 3)) # in RGB (0, 255)

    def __getitem__(self, index):
        image = next(self.loader)
        in_height, in_width, _ = image.shape
        M = aug_matrix(in_width, in_height, self.width, self.height, self.pad)
        image = cv2.warpAffine(
            image, M[0:2, :], (self.width, self.height), flags=cv2.INTER_CUBIC)
                
        input = np.float32(image)
        input = (input / 255.0 - self.mean) / self.std # TO [-1.0, 1.0]
        input = input.transpose(2, 0, 1) # TO [3 x H x W]
        return torch.from_numpy(input).float()
    
    def __len__(self):
        raise NotImplementedError
        

class CaptureStreamer(BaseStreamer):
    """This streamer takes webcam as input.
    """
    def __init__(self, id=0, width=512, height=512, pad=True, **kwargs):
        super().__init__(width, height, pad, **kwargs)
        self.capture = cv2.VideoCapture(id)
    
    def create_loader(self):
        while True:
            _, image = self.capture.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB
            yield image

    def __len__(self):
        return 100_000_000

    def __del__(self):
        self.capture.release()


class VideoListStreamer(BaseStreamer):
    """This streamer takes a list of video files as input.
    """
    def __init__(self, files, width=512, height=512, pad=True, **kwargs):
        super().__init__(width, height, pad, **kwargs)
        self.files = files
        self.captures = [imageio.get_reader(f) for f in files]
        self.nframes = sum([int(cap._meta["fps"] * cap._meta["duration"]) \
                            for cap in self.captures])
    
    def create_loader(self):
        for capture in self.captures:
            for image in capture: # RGB
                yield image

    def __len__(self):
        return self.nframes

    def __del__(self):
        for capture in self.captures:
            capture.close()


class ImageListStreamer(BaseStreamer):
    """This streamer takes a list of image files as input.
    """
    def __init__(self, files, width=512, height=512, pad=True, **kwargs):
        super().__init__(width, height, pad, **kwargs)
        self.files = files
    
    def create_loader(self):
        for f in self.files:
            image = cv2.imread(f, cv2.IMREAD_UNCHANGED)[:, :, 0:3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB
            yield image

    def __len__(self):
        return len(self.files)




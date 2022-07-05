import random
import yaml
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import ndimage

class Overlayer:
    """ Transform module can help you put
    the anomalies from your anomaly database and
    random shapes on the source image
    
    Args:
    
    - masks_path (depricated)
    - preps_path (depricated)
    - anomalies_path <str>: path to your anomaly database with subfolders
    - anml_amount_max <int>: max amount of anomalies from your database 
                             which will be putted on the image (random limit)
    - probability <int>: probability of putting any anomalies from your
                         database on the image. From 0 to 100
    - anomaly_min_pxl_intensity <int>: [heuristics: 2000] used in the
                                        I - heavy_score / anml  
                                        (random limit)
    - anomaly_max_pxl_intensity <int>: [heuristics: 6000-8000] the same
                                        pursose as anomaly_min_pxl_intensity
                                        (random limit)
    - min_points_amount <int>: min amount of points of drawn polygon (random limit)
    - max_points_amount <int>: max amount of points of drawn polygon (random limit)
    - max_polygons_amount <int>: max amount of polygon that will be drawn
                               (random limit)
    """
    def __init__(self,
                 masks_path,
                 preps_path,
                 anomalies_path,
                 anml_amount_max=3,
                 probability=50,
                 anomaly_min_pxl_intensity=2000,
                 anomaly_max_pxl_intensity=5000,
                 min_points_amount=3,
                 max_points_amount=30,
                 max_polygons_amount=3 
        ):
        self.masks_path = masks_path
        self.preps_path = preps_path
        self.anomalies_path = anomalies_path
        
        self.masks_paths = sorted(glob.glob(self.masks_path+"/*"))
        self.preps_paths = sorted(glob.glob(self.preps_path+"/*"))
        self.anomalies_paths = sorted(glob.glob(self.anomalies_path+"/*/*"))
        
        self.anml_amount_max = anml_amount_max
        self.probability = probability
        assert type(self.probability) is int, f"Should be int, not {type(probability)}"
        
        self.min_points_amount = min_points_amount
        self.max_points_amount = max_points_amount
        self.max_polygons_amount = max_polygons_amount
        
        self.anomaly_min_pxl_intensity = anomaly_min_pxl_intensity
        self.anomaly_max_pxl_intensity = anomaly_max_pxl_intensity
        
    def _load(self, idx):
        print(self.masks_paths[idx])
        print(self.preps_paths[idx])
        
        mask = np.array(Image.open(self.masks_paths[idx]))  # load mask
        mask = mask > 0  # load mask
        prep = np.array(Image.open(self.preps_paths[idx]))  # load prep
        
        return prep, mask
    
    @staticmethod
    def _choose_anml_position(img, mask):
        h_i, w_i = img.shape
        h_m, w_m = mask.shape

        max_x_lim = w_i - (w_m + 1) - 150
        max_y_lim = h_i - (h_m + 1) - 200

        # print(f"Image (h,w)=({h_i},{w_i}). Mask (h,w)=({h_m},{w_m}). Xlim: ({150}-{max_x_lim}). Ylim: ({200}-{max_y_lim})")
        
        if max_y_lim <= 200:
            rand_y = random.randint(0, h_i)
        else:
            rand_y = random.randint(200, max_y_lim)  # same with y dim
            
        if max_y_lim <= 200:
            rand_x = random.randint(0, w_i)
        else:
            rand_x = random.randint(150, max_x_lim)  # man in the missle of picture

        return rand_y, rand_x
    
    def _build_appendix(self, prep):
        while True:
            anml_path = random.choice(self.anomalies_paths)
            anml = np.array(Image.open(anml_path))  # load anomaly
            
            if len(anml.shape) == 2:
                break
        
        # randomly rotate the image
        angle = random.randint(0, 180)
        anml = ndimage.rotate(anml, angle, cval=255)
        
        # with 60% probability we gonna randomly upscale the image
        if random.randint(1, 10) <= 6:
            a_height, a_width = anml.shape
            p_height, p_width = prep.shape
            delta_height, delta_width = p_height - a_height - 1, p_width - a_width - 1
            a_height += random.randint(0, delta_height//18)
            a_width  += random.randint(0, delta_width//18)

            if random.randint(1, 10) <= 8:
                anml = cv2.resize(anml, dsize=(a_width, a_height), interpolation=cv2.INTER_CUBIC)
        
        anml_mask = anml < 255  # build anomaly mask
        # cut off the junk dims if exists
        if len(anml.shape) == 3:
            anml = anml[:, :, 0]
        if len(anml_mask.shape) == 3:
            anml_mask = anml_mask[:, :, 0]

        y, x = self._choose_anml_position(prep, anml)

        appendix = prep[y:y+anml_mask.shape[0], x:x+anml_mask.shape[1]]
        
        heavy_score = random.randint(self.anomaly_min_pxl_intensity, self.anomaly_max_pxl_intensity)
        
        appendix = np.where(
            ~anml_mask,
            appendix,
            appendix - heavy_score/(anml+1e-6)
        )
        
        appendix[appendix > 255] = 255
        appendix[appendix < 0] = 0
        
        return appendix, anml_mask, y, x
    
    def _generate_shapes(self, img, mask):
        h, w = mask.shape
        h -= 200  # indeed anomalies can put on the platform
        w -= 150  # man in the missle of picture

        y_c, x_c = random.randint(200, h), random.randint(150, w)

        amount = random.randint(self.min_points_amount, self.max_points_amount)

        points = [
            # сначала x, потом y!
            (x_c + random.randint(-50, 50), y_c + random.randint(-50, 50))
            for _ in range(0, amount)
        ]
        
        pil_mask = mask.copy()
        pil_mask = (pil_mask*255).astype(np.uint8)
        pil_mask = Image.fromarray(pil_mask)
                
        ImageDraw.Draw(pil_mask).polygon(points, outline=1, fill=255)
        
        pil_mask = np.array(pil_mask) > 254
        
        black_matrix = np.abs(np.random.normal(loc=1, scale=15.0, size=pil_mask.shape))
        black_matrix[black_matrix>255] = 255
        black_matrix[black_matrix<0] = 0
        
        heavy_score = random.randint(self.anomaly_min_pxl_intensity, self.anomaly_max_pxl_intensity)
        
        img = np.where(
            ~pil_mask,
            img,
            img - heavy_score/(255-black_matrix)
        )
        
        img[img>255] = 255
        img[img<0] = 0
        
        return img, np.logical_or(pil_mask, mask)

    def overlay(self, sample):
        prep, mask, name = sample['image'], sample['label'], sample['name']
        
        res_img = prep.copy()
        new_mask = mask.copy()
            
        if random.randint(1, 10) <= self.probability/10:
            # Choose amount of anomalies randomly
            amount = random.randint(0, self.anml_amount_max)

            for i in range(amount):
                appendix, anml_mask, y, x = self._build_appendix(prep)

                res_img[y:y+anml_mask.shape[0], x:x+anml_mask.shape[1]] = np.where(
                    ~anml_mask,
                    prep[y:y+anml_mask.shape[0], x:x+anml_mask.shape[1]],
                    appendix
                )

                new_mask[y:y+anml_mask.shape[0], x:x+anml_mask.shape[1]] = np.where(
                    ~anml_mask,
                    new_mask[y:y+anml_mask.shape[0], x:x+anml_mask.shape[1]],
                    np.full(new_mask[y:y+anml_mask.shape[0], x:x+anml_mask.shape[1]].shape, True)
                )
        
        polygons_amount = random.randint(0, self.max_polygons_amount)
        for _ in range(polygons_amount):
            res_img, new_mask = self._generate_shapes(res_img, new_mask)

        response = {
            "image": res_img,
            "mask": new_mask,
            "name": name
        }

        return response
    
    def __call__(self, payload):
        return self.overlay(payload)
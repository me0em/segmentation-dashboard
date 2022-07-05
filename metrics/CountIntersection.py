import cv2 
from functools import partial
import numpy as np 
import torch


class CountIntersection(torch.nn.Module):
    def __init__(self, treshold=0):
        super(CountIntersection, self).__init__()
        assert treshold >= 0 and treshold <= 1, "treshold need between 0 and 1"

        self.treshold = treshold
        
        
    def contourIntersect(self, original_image: np.array, contour1:np.array, contour2:np.array) -> list:
        """
            Находит пересечение масок
        """
        # объеденяем контуры в один массив (для применения их в методе cv2.drawContours)
        contours = [contour1, contour2]

        # матрица на которую будем накладывать объекты 
        blank = np.zeros(original_image.shape[0:2])

        # выделяем объекты по координатам границ и закрашиваем внутренности
        image1 = cv2.drawContours(blank.copy(), contours, 0, color = 1, thickness=-1)
        image2 = cv2.drawContours(blank.copy(), contours, 1, color = 1, thickness=-1)

        # находим пересечение
        intersection = np.logical_and(image1, image2)
        return intersection.any() * intersection.sum() / image1.astype(bool).sum()

    def forward(self, mask: np.array, target: np.array) -> float:
        mask = mask.cpu().numpy()
        target = target.cpu().numpy()
        
        bl_img = cv2.medianBlur(target, 5)
        bl_mask = cv2.medianBlur(mask, 5)
        # нахождение границ
        canny_img = cv2.Canny(bl_img, 10, 250)
        canny_mask = cv2.Canny(bl_mask, 10, 250)
        # ядро с элементами 1, влияет на объединение близких облъектов в одну границу
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # создание полноценных областей вокруг объектов на основе границ
        closed_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
        closed_mask = cv2.morphologyEx(canny_mask, cv2.MORPH_CLOSE, kernel)

        # поиск замкнутых контуров
        contours_img = cv2.findContours(closed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_mask = cv2.findContours(closed_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        res = []
        for one_mask in contours_img: 
            # создаю функцию, с фиксированием изображения и настоящей маски
            chek = partial(self.contourIntersect, target, one_mask)
            # применяю сразу ко всем выделенным моделью объектам и записываю результат
            out = map(chek, contours_mask)

            # поиск объектоы которые покарывают оригинальные маски больше чем на treshold процентов
            res.append(bool(list(filter(lambda x: x > self.treshold, out))))

        return sum(res) / len(contours_img)
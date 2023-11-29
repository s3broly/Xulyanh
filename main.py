from PIL import Image
import numpy as np
import cv2
import os

class ImageProcess:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img_raw = self.read_image()

    def read_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not read image from path: {self.image_path}")
        return img

    def negative_transform(self):
        # Chuyển đổi ảnh thành ảnh âm bản.
        img_array = 255 - self.img_raw
        return Image.fromarray(img_array)

    def threshold_transform(self, threshold_value):

        # Chuyển đổi ảnh thành ảnh nhị phân với ngưỡng.
        _, thresholded_img_array = cv2.threshold(self.img_raw, threshold_value, 255, cv2.THRESH_BINARY)
        return Image.fromarray(thresholded_img_array)

    def logarithmic_transform(self):
        
        # Áp dụng biến đổi logarit cho ảnh.
        
        img_array = np.log1p(self.img_raw)
        return Image.fromarray(img_array)

    def median_transform(self, kernel_size):
        
        # Lọc trung vị ảnh với kích thước kernel được cung cấp.
        
        median_filtered_img_array = cv2.medianBlur(self.img_raw, kernel_size)
        return Image.fromarray(median_filtered_img_array)

    def roberts_operator(self):
        
        # Áp dụng toán tử Roberts lên ảnh.
        
        img_array = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2GRAY)
        roberts_x = cv2.filter2D(img_array, cv2.CV_64F, np.array([[-1, 0], [0, 1]]))
        roberts_y = cv2.filter2D(img_array, cv2.CV_64F, np.array([[0, -1], [1, 0]]))
        roberts_img_array = np.sqrt(np.square(roberts_x) + np.square(roberts_y))
        return Image.fromarray(roberts_img_array.astype('uint8'))

    def sobel_operator(self):
        
        # Áp dụng toán tử Sobel lên ảnh.
        
        img_array = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        sobel_img_array = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        return Image.fromarray(sobel_img_array.astype('uint8'))

    def prewitt_operator(self):
        
        # Áp dụng toán tử Prewitt lên ảnh.
        
        img_array = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2GRAY)
        prewitt_x = cv2.filter2D(img_array, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitt_y = cv2.filter2D(img_array, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        prewitt_img_array = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
        return Image.fromarray(prewitt_img_array.astype('uint8'))

    def canny_edge_detection(self):
        
        # Phát hiện cạnh bằng phương pháp Canny.
        
        img_array = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_array, 100, 200)
        return Image.fromarray(edges)

    def otsu_threshold(self):
        
        # Áp dụng ngưỡng Otsu lên ảnh.
        
        _, otsu_img_array = cv2.threshold(self.img_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(otsu_img_array)

    def histogram_equalizing(self):
        
        # Cân bằng lược đồ màu của ảnh.
        
        img_to_yuv = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2YUV)
        img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        return Image.fromarray(hist_equalization_result)

def main():
    image_path = input("Nhập đường dẫn của ảnh: ")

    try:
        image_processor = ImageProcess(image_path)

        negative_image = image_processor.negative_transform()
        negative_image.show()

        thresholded_image = image_processor.threshold_transform(128)
        thresholded_image.show()

        logarithmic_image = image_processor.logarithmic_transform()
        logarithmic_image.show()

        median_filtered_image = image_processor.median_transform(3)
        median_filtered_image.show()

        roberts_image = image_processor.roberts_operator()
        roberts_image.show()

        sobel_image = image_processor.sobel_operator()
        sobel_image.show()

        prewitt_image = image_processor.prewitt_operator()
        prewitt_image.show()

        edges_image = image_processor.canny_edge_detection()
        edges_image.show()

        otsu_image = image_processor.otsu_threshold()
        otsu_image.show()

        hist_equalized_image = image_processor.histogram_equalizing()
        hist_equalized_image.show()

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

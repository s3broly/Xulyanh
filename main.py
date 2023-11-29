from PIL import Image
import numpy as np
import cv2

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
        max_pixel_value = np.max(self.img_raw)
        img_array = np.log1p(self.img_raw / max_pixel_value)
        return Image.fromarray((img_array * 255).astype('uint8'))

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
        if len(self.img_raw.shape) == 3:
            img_array = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_array = self.img_raw
        _, otsu_img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(otsu_img_array)

    def histogram_equalizing(self):
        
        # Cân bằng lược đồ màu của ảnh.
        
        img_to_yuv = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2YUV)
        img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        return Image.fromarray(hist_equalization_result)

def main():
    image_path = input("Nhập đường dẫn của ảnh (ví dụ: image\8.jpg): ")
    print("\n")

    try:
        image_processor = ImageProcess(image_path)

        print("Chọn phương thức:\n")
        print("1. Negative Transform")
        print("2. Threshold Transform")
        print("3. Logarithmic Transform")
        print("4. Median Transform")
        print("5. Histogram Equalizing")
        print("6. Roberts Operator")
        print("7. Sobel Operator")
        print("8. Prewitt Operator")
        print("9. Canny Edge Detection")
        print("10. Otsu Threshold")
        
    
        choice = int(input("\nNhập số tương ứng với phương thức bạn muốn sử dụng: "))

        if choice == 1:
            result_image = image_processor.negative_transform()
        elif choice == 2:
            result_image = image_processor.threshold_transform(128)
        elif choice == 3:
            result_image = image_processor.logarithmic_transform()
        elif choice == 4:
            result_image = image_processor.median_transform(3)
        elif choice == 5:
            result_image = image_processor.histogram_equalizing()
        elif choice == 6:
            result_image = image_processor.roberts_operator()
        elif choice == 7:
            result_image = image_processor.sobel_operator()
        elif choice == 8:
            result_image = image_processor.prewitt_operator()
        elif choice == 9:
            result_image = image_processor.canny_edge_detection()
        elif choice == 10:
            result_image = image_processor.otsu_threshold()
        result_image.show()

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

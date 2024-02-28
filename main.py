import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
import time

image_folder = "/Users/dylangrech/Desktop/InitialImages"
output_folder = "/Users/dylangrech/Desktop/ProcessedImages"

class QRCodeProcessor:
    def __init__(self):
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.processed_qr_codes = set()

    def decode_qr_code(self, frame):
        try:
            qr_code_data_list = decode(frame)
            return qr_code_data_list
        except Exception as e:
            print(f"Error decoding QR code: {str(e)}")
            return []

    def preprocess_image(self, frame):
        # Scale down the image
        scale_factor = 2.0  # You can adjust this value as needed 2.0(355)
        scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        #equalized_frame = cv2.equalizeHist(gray_frame)
        #edges = cv2.Canny(gray_frame, 10, 10)

        # Increase the contrast of the frame
        alpha = 1.0 #1.0 yields a higher success rate(387)
        beta = 0.1
        contrast_frame = cv2.convertScaleAbs(gray_frame, alpha=alpha, beta=beta)
        smoothed_frame = cv2.bilateralFilter(contrast_frame, -30, 15, 15)#(-30,15,15) best outcome
        #threshold_frame = cv2.threshold(smoothed_frame, 0, 512, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #sharpening_filter = np.array([[0, -1, 0],
        #                      [-1,  5, -1],
        #                      [0, -1, 0]])
        #sharpened_frame = cv2.filter2D(contrast_frame, -1, kernel=sharpening_filter)


        blurred_frame = cv2.GaussianBlur(smoothed_frame, (1, 1), 1)


        return blurred_frame


    def run(self):
        try:
            start_time = time.time()  # Record the start time
            total_qr_codes_read = 0  # Initialize a counter for total QR codes read
            print(f"Total images to process: {len(self.image_files)}\n")
            for image_file in self.image_files:
                print(f"Processing image: {image_file}\n")
                image_path = os.path.join(image_folder, image_file)
                frame = cv2.imread(image_path)

                # Preprocess the image
                processed_frame = self.preprocess_image(frame)

                qr_code_data_list = self.decode_qr_code(processed_frame)

                # Save the processed image with a different filename
                output_path = os.path.join(output_folder, f"processed_{image_file}")
                cv2.imwrite(output_path, processed_frame)

                print(f"Number of QR codes detected: {len(qr_code_data_list)}\n")

                if not qr_code_data_list:
                    print(f"No QR code found for '{image_file}'.\n")
                    continue

                total_qr_codes_read += len(qr_code_data_list)  # Update the counter

                for idx, qr_code_data in enumerate(qr_code_data_list):
                    qr_code_str = qr_code_data.data.decode('utf-8')
                    print(f"Decoded String {idx + 1}: {qr_code_str}")
                    print(f"QR code found for '{image_file}'.\n")
                    self.processed_qr_codes.add(qr_code_str)

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            print(f"Total QR codes read: {total_qr_codes_read} out of {len(self.image_files)} images.")
            print(f"Total runtime: {elapsed_time:.2f} seconds")

        except KeyboardInterrupt:
            print("Program terminated by user.")

if __name__ == "__main__":
    qr_processor = QRCodeProcessor()
    qr_processor.run()

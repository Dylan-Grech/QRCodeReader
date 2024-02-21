import cv2
from pyzbar.pyzbar import decode
import os
import numpy as np

image_folder = "/Users/dylangrech/Desktop/InitialImages"
output_folder = "/Users/dylangrech/Desktop/ProcessedImages"

class QRCodeProcessor:

    def __init__(self):
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.processed_qr_codes = set()

    def decode_qr_code(self, frame):
        try:
            if len(frame.shape) == 2:
                qr_code_data_list = decode(frame)
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                qr_code_data_list = decode(gray_frame)
            return qr_code_data_list
        except Exception as e:
            print(f"Error decoding QR code: {str(e)}")
            return []

    def publish_qr_code(self, qr_code_data):
        url = qr_code_data.data.decode('utf-8')
        return url

    def run(self):
        try:
            for image_file in self.image_files:
                image_path = os.path.join(image_folder, image_file)
                frame = cv2.imread(image_path)

                # Apply image processing steps
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                alpha = 1.5
                beta = 20
                contrast_frame = cv2.convertScaleAbs(gray_frame, alpha=alpha, beta=beta)

                qr_code_data_list = self.decode_qr_code(contrast_frame)

                # Save the processed image with a different filename
                output_path = os.path.join(output_folder, f"processed_{image_file}")
                cv2.imwrite(output_path, contrast_frame)

                if not qr_code_data_list:
                    print(f"No QR code found for '{image_file}'.\n")
                    continue

                for qr_code_data in qr_code_data_list:
                    if qr_code_data.data not in self.processed_qr_codes:
                        decoded_string = self.publish_qr_code(qr_code_data)
                        print(f"Decoded String: {decoded_string}")
                        print(f"QR code found for '{image_file}'.\n")
                        self.processed_qr_codes.add(qr_code_data.data)

        except KeyboardInterrupt:
            print("Program terminated by user.")

if __name__ == "__main__":
    qr_processor = QRCodeProcessor()
    qr_processor.run()

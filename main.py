import sys
import cv2
from pyzbar.pyzbar import decode
import os
import time

image_folder = "/Users/dylangrech/Desktop/InitialImages"
output_folder = "/Users/dylangrech/Desktop/ProcessedImages"

class QRCodeProcessor:

    # Instance of the class is created
    def __init__(self):
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.processed_qr_codes = set()

    # Decoding the qr code through this function
    def decode_qr_code(self, frame):
        try:
            qr_code_data_list = decode(frame)
            return qr_code_data_list
        except Exception as e:
            print(f"Error decoding QR code: {str(e)}")
            return []

    # Function to process and adjust the image
    def preprocess_image(self, frame, brightness_factor=1.0):
        # Scale down the image
        scale_factor = 2.0
        scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

        # Increase the contrast of the frame
        alpha = 1.0
        beta = 0.1
        contrast_frame = cv2.convertScaleAbs(gray_frame, alpha=alpha, beta=beta)

        # Adjust brightness
        brightened_frame = cv2.convertScaleAbs(contrast_frame, alpha=brightness_factor, beta=30)
        smoothed_frame = cv2.bilateralFilter(brightened_frame, -30, 15, 15)

        # Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(60, 60))
        contrast_enhanced_frame = clahe.apply(smoothed_frame)

        # Apply Gaussian Blur
        blurred_frame = cv2.GaussianBlur(contrast_enhanced_frame, (1, 1), 1)

        return blurred_frame

    # Function that creates a small loading effect
    def loading_effect(self):
        spinner_chars = ['-', '\\', '|', '/']
        iterations = 5

        sys.stdout.write("Processing images: ")
        sys.stdout.flush()

        for _ in range(iterations):
            for char in spinner_chars:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write('\b')

    # Runs the program
    def run(self):
        try:
            # Record the start time
            start_time = time.time()  
            # Initialize a counter for total QR codes read
            total_qr_codes_read = 0  
            print(f"Total images to process: {len(self.image_files)}")
            self.loading_effect()  # Add loading effect

            for image_file in self.image_files:

                image_path = os.path.join(image_folder, image_file)
                frame = cv2.imread(image_path)

                # Preprocess the image
                processed_frame = self.preprocess_image(frame)
                # Decode the qr code
                qr_code_data_list = self.decode_qr_code(processed_frame)
                # Save the processed image with a different filename
                output_path = os.path.join(output_folder, f"processed_{image_file}")
                cv2.imwrite(output_path, processed_frame)

                my_file = open("/Users/dylangrech/Desktop/QrCodes.txt", "a")

                if not qr_code_data_list:
                    my_file.write(f"No QR code found for '{image_file}'.\n")
                    continue
                # Update the counter
                total_qr_codes_read += len(qr_code_data_list)  
                for idx, qr_code_data in enumerate(qr_code_data_list):
                    qr_code_str = qr_code_data.data.decode('utf-8')
                    my_file.write(f"Decoded String {idx + 1}: {qr_code_str}")
                    my_file.write(f"QR code found for '{image_file}'.\n")
                    self.processed_qr_codes.add(qr_code_str)
                my_file.close()

            # Record the end time
            end_time = time.time()  
            elapsed_time = end_time - start_time
            # Output information about the program runtime and success rate
            print(f"\nTotal QR codes read: {total_qr_codes_read} out of {len(self.image_files)} images.")
            successRate = 100 * (total_qr_codes_read / len(self.image_files))
            print(f"Success rate is {successRate:.2f}%")
            print(f"Total runtime: {elapsed_time:.2f} seconds")

        except KeyboardInterrupt:
            print("Program terminated by user.")

if __name__ == "__main__":
    qr_processor = QRCodeProcessor()
    qr_processor.run()

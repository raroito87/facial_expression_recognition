import sys
#print(sys.path)
sys.path.append("/Users/raroito/PycharmProjects/facial_expression_recognition/src/")


from image_utils import ImageCapture, ImageExporter


if not __name__ == '__main_':

    cap_img = ImageCapture()
    frame = cap_img.capture_image()

    if frame is not None:
        img_exp = ImageExporter()
        img_exp.save_capture(frame)

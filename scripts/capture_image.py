from image_utils import ImageCapture, ImageExporter


if not __name__ == '__main_':

    cap_img = ImageCapture()
    frame = cap_img.capture_image()

    img_exp = ImageExporter()
    img_exp.save_capture(frame)

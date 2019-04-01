from image_utils import ImageImporter
import matplotlib.pyplot as plt
import cv2


if not __name__ == '__main_':

    img_imp = ImageImporter(name = 'fer2013')

    idx = 1000
    amount = 48
    plt.figure(1, figsize=(20, 20))
    for i in range(amount):
        img, emotion = img_imp.load_data_as_img(index = idx + i)
        # Call signature: subplot(nrows, ncols, index, **kwargs)
        ax = plt.subplot(6, 8, 1 + i)
        ax.set_title(emotion, pad = 1.0)
        ax.axis('off')
        plt.imshow(img, cmap='gray', interpolation='none')

    plt.show()


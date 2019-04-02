import sys
#print(sys.path)
sys.path.append("/Users/raroito/PycharmProjects/facial_expression_recognition/src/")

from image_utils import ImageImporter, ImageExporter
import matplotlib.pyplot as plt

if not __name__ == '__main_':

    img_imp = ImageImporter(name = 'fer2013')
    img_exp = ImageExporter()

    path = img_imp.directory + '/images/'
    idx = 1000
    amount = 48
    plt.figure(1, figsize=(20, 20))
    for i in range(amount):
        img, emotion = img_imp.load_data_as_img(index = idx + i)
        filename = f'{emotion}_{i}'
        img_exp._save_original(frame = img, path = path, name = filename)
        # Call signature: subplot(nrows, ncols, index, **kwargs)
        ax = plt.subplot(6, 8, 1 + i)
        ax.set_title(emotion, pad = 1.0)
        ax.axis('off')
        plt.imshow(img, cmap='gray', interpolation='none')

    plt.show()


import numpy as np
from keras import models
import pickle
from sklearn.preprocessing import Normalizer
import cv2
import tensorflow as tf
from mtcnn.mtcnn import MTCNN


class Classifier_model:

    class_name = ['An', 'Hien', 'Hoai', 'Linh', 'Thuy', 'Tra', 'Dat', 'Nhi', 'Thu', 'Tuan']
    model = None
    embedding_model = None
    graph = None
    detector = None

    def __init__(self, model_path):

        self.model = pickle.load(open(model_path, 'rb'))
        self.embedding_model = models.load_model(r'C:\Users\Luxury\PycharmProjects\webdemo\Data_pretrain\facenet_keras.h5')
        self.embedding_model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.detector = MTCNN()

    def detect(self, image):
        images = []
        faces = self.detector.detect_faces(image)
        if faces is None:
            print('Không tìm thấy mặt người trong ảnh')
            return None
        else:
            for face in faces:
                confidence = face['confidence']
                if confidence > 0.95:
                    bounding_box = face['box']
                    face_image = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                                       bounding_box[0]:bounding_box[0] + bounding_box[2]]
                    face_image = cv2.resize(face_image, (160, 160))
                    images.append(face_image)
                    cv2.rectangle(image,
                                  (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                  (255, 0, 0),
                                  thickness=2)
        return images, image

    def predict(self, X_test):
        y_nums = []
        y_names = []
        for x_test in X_test:
            x_test = x_test.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = x_test.mean(), x_test.std()
            x_test = (x_test - mean) / std
            # transform face into one sample
            samples = np.expand_dims(x_test, axis=0)
            with self.graph.as_default():
                x_test_embbeding = self.embedding_model.predict(samples)

            x_test_embbeding = x_test_embbeding.reshape((len(x_test_embbeding), 128))
            in_encoder = Normalizer(norm='l2')
            x_test_embbeding = in_encoder.transform(x_test_embbeding)
            y_predict = self.model.predict(x_test_embbeding)[0]
            y_nums.append(y_predict)
            y_names.append(self.class_name[y_predict])
        return y_nums, y_names





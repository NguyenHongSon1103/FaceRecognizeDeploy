import json

from flask import Flask, request, render_template, jsonify, Response
import time 
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import cv2 
#from flask_cors import CORS
from flask_restplus import Api, Resource
from parsers import file_upload
from Keras_Classifier.model import Classifier_model

classify_model = Classifier_model(r'Data_pretrain/model.pkl')
app = Flask(__name__)
#CORS(app)
api = Api(app, title='Deep learning Classifier APIs', description='APIs of Face Classification')
name_space = api.namespace('main', description='Main APIs')


def read_image(img_bytes):
    # if (img_bytes is np.array):
    #     return img_bytes
    return cv2.imdecode(np.asarray(bytearray(img_bytes.read()), dtype="uint8"), cv2.IMREAD_COLOR)


@name_space.route('/showViewClassifyFace/classifyFace')
@name_space.expect(file_upload)
class Keras_Classifier(Resource):
    def post(self):
        start_time = time.time()

        data = json.dumps({"success": False})
        if request.method == "POST":
            if 'image' not in request.files:
                return jsonify({'error': 'no file'}), 400

            image = request.files['image']  # image được lấy dưới dạng file-storage
            if image is not None:
                try:
                    # Read image by Opencv
                    image = read_image(image.read())  # chuyển từ dạng file-storage sang dạng ảnh
                    # classify the input image
                    (class_index, class_name) = classify_model.predict(image)
                    data["class_index"] = class_index
                    data["class_name"] = class_name
                    # indicate that the request was a success
                    data["success"] = True
                except Exception as ex:
                    print("Exception")
                    # data['error'] = ex
                    print(str(ex))
            else:
                fs_img = request.form['image']
                # image = Image.open(fs_img)
                image = np.fromfile(fs_img, np.uint8)
                image = cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                if image is not None:
                    try:
                        # # Read Image
                        # image = image.split("base64,")[1]
                        # image = BytesIO(base64.b64decode(image))
                        # image = Image.open(image)
                        # image = Image.composite(image, Image.new('RGB', image.size, 'white'), image)
                        # image = image.convert('L')
                        # image = image.resize((160, 160), Image.ANTIALIAS)

                        # classify the input image
                        class_index, class_name = classify_model.predict(image)

                        data["class_index"] = class_index
                        data["class_name"] = class_name
                        # indicate that the request was a success
                        data["success"] = True
                    except Exception as ex:
                        data['error'] = ex
                        print(str(ex))

        data['run_time'] = "%.2f" % (time.time() - start_time)
        # return the data dictionary as a JSON response
        return Response(data, status=201)

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3000)





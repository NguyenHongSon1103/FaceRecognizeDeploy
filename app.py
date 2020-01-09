import os

from flask import Flask, Response
from flask import render_template
from flask import request, jsonify
import cv2
import numpy as np
import json


from Keras_Classifier.model import Classifier_model
from io import BytesIO
import base64
from PIL import Image
import time
SERVER_NAME = 'http://localhost:5000'

app = Flask(__name__)
classify_model = Classifier_model(r'Data_pretrain/model.pkl')


def encode_np_array(image):
    ''' images: list of RGB images '''
    success, encoded_image = cv2.imencode('.jpg', image[:,:,::-1])
    byte_image = encoded_image.tobytes()
    b64_string_image = base64.b64encode(byte_image).decode()
    return b64_string_image

#Tranfer bytes to image:
def read_image(img_bytes):
    return cv2.imdecode(np.asarray(bytearray(img_bytes), dtype="uint8"), cv2.IMREAD_COLOR)


@app.route("/classifyFace", methods=['POST'])
def classifyFace():
    start_time = time.time()

    data = {"success": 'False'}
    if request.method == "POST":
        if 'image' not in request.files:
            data['error'] = 'image not found'
            return Response(json.dumps(data), status=400)

        image = request.files['image']  # image được lấy dưới dạng file-storage
        if str.lower(image.content_type) not in ('image/jpeg', 'image/png'):
            data['error'] = 'input is not image'
            return Response(json.dumps(data), status=400)
        if image is not None:
            #try:
                # Read image by Opencv
                image = read_image(image.read())  # chuyển từ dạng file-storage sang dạng ảnh
                # classify the input image
                faces, img_and_bounding = classify_model.detect(image)
                #cv2.imwrite(r'\webdemo\BoundingImage\boundingImage.jpg', img_and_bounding)

                img_and_bounding = cv2.cvtColor(img_and_bounding, cv2.COLOR_BGR2RGB)
                b64_image_string = encode_np_array(img_and_bounding)

                (class_indexs, class_names) = classify_model.predict(faces)

                for i in range(len(class_indexs)):
                    data['face'+str(i)] = {'class_index':str(class_indexs[i]),'class_name': str(class_names[i])}
                    # indicate that the request was a success
                data['base64_image'] = str(b64_image_string)
            # except Exception as ex:
            #     print("Exception")
            #     data['error'] = str(ex)
            #     print(str(ex))
        # else:
        #     fs_img = request.form['image']
        #     # image = Image.open(fs_img)
        #     image = np.fromfile(fs_img, np.uint8)
        #     image = cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        #     if image is not None:
        #         try:
        #             # # Read Image
        #             # image = image.split("base64,")[1]
        #             # image = BytesIO(base64.b64decode(image))
        #             # image = Image.open(image)
        #             # image = Image.composite(image, Image.new('RGB', image.size, 'white'), image)
        #             # image = image.convert('L')
        #             # image = image.resize((160, 160), Image.ANTIALIAS)
        #
        #             # classify the input image
        #             class_index, class_name = classify_model.predict(image)
        #
        #             data["class_index"] = class_index
        #             data["class_name"] = class_name
        #             # indicate that the request was a success
        #             data["success"] = True
        #         except Exception as ex:
        #             data['error'] = ex
        #             print(str(ex))
                data['num_of_faces'] = str(len(class_indexs))
    data['run_time'] = "%.2f" % (time.time() - start_time)
    data['success'] = 'True'
    # return the data dictionary as a JSON response
    return Response(json.dumps(data), status=201)


@app.route("/",methods=["GET", "POST"])
def main():
    return render_template("classifyFace.html")


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=3000)

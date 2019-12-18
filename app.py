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


app = Flask(__name__)
classify_model = Classifier_model(r'Data_pretrain/model.pkl')


#Tranfer bytes to image:
def read_image(img_bytes):
    return cv2.imdecode(np.asarray(bytearray(img_bytes), dtype="uint8"), cv2.IMREAD_COLOR)


@app.route("/classifyFace", methods=['POST'])
def classifyFace():
    start_time = time.time()

    data = {"success": 'False'}
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({'error': 'no file'}), 400

        image = request.files['image']  # image được lấy dưới dạng file-storage
        if image is not None:
            #try:
                # Read image by Opencv
                image = read_image(image.read())  # chuyển từ dạng file-storage sang dạng ảnh
                print("Đã đọc được ảnh")

                # classify the input image
                (class_index, class_name) = classify_model.predict(image)
                print("Đã predict được kết quả")
                data["class_index"] = str(class_index)
                data["class_name"] = str(class_name)
                # indicate that the request was a success
                data["success"] = 'True'
            # except Exception as ex:
            #     print("Exception")
            #     data['error'] = str(ex)
            #     print(str(ex))
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
    return Response(json.dumps(data), status=201)


@app.route("/",methods=["GET", "POST"])
def main():
    return render_template("classifyFace.html")


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=3000)

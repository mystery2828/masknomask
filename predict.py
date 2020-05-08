import operator
import cv2
from keras.models import model_from_json
from mtcnn import MTCNN

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")


def do():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        detector = MTCNN()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)

        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        if results:
            for i in range(len(results)):
                bounding_box = results[i]['box']

                cv2.rectangle(image,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (255, 0, 0),
                              thickness=4)
                roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                      bounding_box[0]:bounding_box[0] + bounding_box[2]]
                roi = cv2.resize(roi, (64, 64))

                result = loaded_model.predict(roi.reshape(1, 64, 64, 3))
                print(result)
                prediction = {'mask': result[0][0],
                              'no-mask': result[0][1],
                              }
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

                cv2.putText(image, prediction[0][0], (bounding_box[0]-30, bounding_box[1]-30), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (0, 0, 255), 2)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imshow("Frame", image)
                cv2.imshow("Frameoriginal", frame)

        if cv2.waitKey(40) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    do()

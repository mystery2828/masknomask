import cv2
from mtcnn import MTCNN
from os import listdir

count = 0
for file in listdir('D:\\maskornomask\\data\\record\\mask'):
    filename = 'D:\\maskornomask\\data\\record\\mask\\' + file
    detector = MTCNN()

    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    print('Started {}'.format(count))
    count+=1
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    if result:
        for i in range(len(result)):
            bounding_box = result[i]['box']

            cv2.rectangle(image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255, 0, 0),
                          thickness=4)
            roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                  bounding_box[0]:bounding_box[0] + bounding_box[2]]
            roi = cv2.resize(roi, (64, 64))
            print(roi.shape)
            # roi = np.expand_dims(roi, axis=0)
            cv2.imwrite('D:\\maskornomask\\data\\train\\mask\\' + file, roi)
            print('Ended {}'.format(count))
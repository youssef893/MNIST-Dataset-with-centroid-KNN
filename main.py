import cv2
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def readAndReshapeData(imagesPath, labelsPath):
    images, images_labels = loadlocal_mnist(
        images_path=imagesPath,
        labels_path=labelsPath)

    images = images.reshape(len(images), 28, 28)
    images = np.array(images)
    return images, images_labels


def preprocessingImage(input_images):
    input_images = input_images.reshape(len(input_images), 784)
    outputImages_list = []

    for image in input_images:
        bgrImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # change image to BGR
        grayImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)  # change image gray scale
        ret, binaryImage = cv2.threshold(grayImage, 127, 255, 0)  # change image to binaryImage
        moment = cv2.moments(binaryImage)  # calculate moment to get centroid
        cX = int(moment["m10"] / moment["m00"])  # calculate x
        cY = int(moment["m01"] / moment["m00"])  # calculate y
        output_Image = cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)  # add circle at centroid point
        outputImages_list.append(output_Image)  # add centroid to feature vector

    outputImages_list = np.array(outputImages_list)
    return outputImages_list


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def centerOfMass(image):
    Sumx, Sumy, num = 0, 0, 0
    cx, cy = 0, 0

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if image[i][j] != 0:  # if pixel has a value
                Sumx += i  # add index of x-axis
                Sumy += j  # add index of y-axis
                num += 1  # get their number

    if num != 0:
        cx = Sumx / num  # get point of centroid on x-axis
        cy = Sumy / num  # get point of centroid on y-axis
    return cx, cy


def extractFeatures(imagesList):
    inputFeatures = []

    for image in imagesList:
        featureVector = []
        # split matrix into 16 sub matrices
        for matrix in split(image, 7, 7):
            cx1, cy1 = centerOfMass(matrix)
            featureVector.append((cx1, cy1))  # collect centroid of sub matrices into one list
        inputFeatures.append(featureVector)  # get feature vector of all images

    print(inputFeatures)
    return np.array(inputFeatures)


def applyKNN(trainFeatures, trainLabels, testFeatures):
    print('Wait for fitting knn classifier and testing it...')
    knn = KNeighborsClassifier(5, metric='euclidean')  # Apply KNN classifier with k = 5
    knn.fit(trainFeatures, trainLabels)  # fit train data
    prediction = knn.predict(testFeatures)  # test data
    return prediction


def openCv_model(trainFeatures, testFeatures, trainLabels, testLabels):
    train_inputs_openCV = preprocessingImage(trainFeatures)
    test_inputs_openCV = preprocessingImage(testFeatures)

    prediction = applyKNN(train_inputs_openCV, trainLabels, test_inputs_openCV)
    return accuracy_score(testLabels, prediction)


def second_model(trainFeatures, testFeatures, trainLabels, testLabels):
    train_inputs = extractFeatures(trainFeatures)
    test_inputs = extractFeatures(testFeatures)

    train_inputs = train_inputs.reshape(60000, 32)
    test_inputs = test_inputs.reshape(10000, 32)

    prediction = applyKNN(train_inputs, trainLabels, test_inputs)
    return accuracy_score(testLabels, prediction)


def main():
    train_features, train_labels = readAndReshapeData("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    test_features, test_labels = readAndReshapeData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    accuracy_score_for_openCv = openCv_model(train_features, test_features, train_labels, test_labels)
    accuracyScore = second_model(train_features, test_features, train_labels, test_labels)

    print("Accuracy Score for openCv =", accuracy_score_for_openCv * 100, "%")
    print("Accuracy Score =", accuracyScore * 100, "%")


if __name__ == '__main__':
    main()

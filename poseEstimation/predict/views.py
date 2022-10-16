from django.shortcuts import render,redirect
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import ImageOps
import PIL.Image
from .forms import ImageForm
from .models import ImageModel


interpreter = tf.lite.Interpreter(model_path='C:\Anirudh\IBM Hack Challenge\poseEstimation\static\lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite')
interpreter.allocate_tensors()

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 2, (255,0,0), -1)
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def predictHome(request):
    ImageModel.objects.all().delete()
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('predictionResult')
    else:
        form = ImageForm()
    return render(request, 'predict/home.html', {'form': form})


def predictionResult(request):
    # Fetching the image from database
    image_url = ImageModel.objects.all()[0].image_entry.url
    print('C:/Anirudh/IBM Hack Challenge/poseEstimation/{}'.format(image_url))
    # Making detections

    # Resizing
    img_original = PIL.Image.open('C:/Anirudh/IBM Hack Challenge/poseEstimation{}'.format(image_url))


    image_result = resize_with_padding(img_original,(256,256)) # Converting the image to 256x256 with padding
    image_result = np.array(image_result.getdata()).reshape(image_result.size[1], image_result.size[0], 3) # Converting the image to ndarray
    image_result = np.ascontiguousarray(image_result, dtype=np.uint8) # Making the array contiguous

    img = image_result.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img,axis=0),256,256)
    input_image =tf.cast(img,dtype=tf.uint8)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    draw_connections(image_result, keypoints_with_scores, EDGES, 0.3)
    draw_keypoints(image_result, keypoints_with_scores, 0.3)

    image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
    image_result = cv2.resize(image_result,(384,384))
    filename = 'C:/Anirudh/IBM Hack Challenge/poseEstimation/static/result_images/1.jpg'
    cv2.imwrite(filename,image_result)
    print('File saved successfully')
    cf_dict={
        'Nose': "{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[0]),
        'Left Eye': "{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[1]),
        'Right Eye':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[2] ),
        'Left Ear':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[3] ),
        'Right Ear':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[4] ),
        'Left Shoulder': "{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[5]),
        'Right Shoulder':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[6] ),
        'Left Elbow':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[7]),
        'Right Elbow':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[8]),
        'Left Wrist':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[9]),
        'Right Wrist':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[10]),
        'left hip':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[11]),
        'Right hip':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[12]),
        'Left Knee':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[13]),
        'Right Knee':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[14]),
        'Left Ankle':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[15]),
        'Right Ankle':"{:.2f}".format((keypoints_with_scores[0][0][:,-1]*100)[16]),

    }

    # Detection of pose..

    # Create a function to import an image and resize it to be able to be used with our model
    def load_and_prep_image(filename, img_shape=224):

        # Read in target file (an image)
        img = tf.io.read_file(filename)

        # Decode the read file into a tensor & ensure 3 colour channels 
        # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
        img = tf.image.decode_image(img, channels=3)

        # Resize the image (to the same size our model was trained on)
        img = tf.image.resize(img, size = [img_shape, img_shape])

        # Rescale the image (get all values between 0 and 1)
        img = img/255.
        return img
    def pred_and_plot(model, filename, class_names):
        """
        Imports an image located at filename, makes a prediction on it with
        a trained model and plots the image with the predicted class as the title.
        """
        # Import the target image and preprocess it
        img = load_and_prep_image(filename)

        # Make a prediction
        pred = model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        pred_class = class_names[int(tf.round(pred)[0][0])]
        return pred_class

    class_names = ['bowling',
        'catch',
        'dive',
        'drive',
        'pull',
        'reverseSweep',
        'sweep',
        'throw',
        'wicketKeeper']
    model_0 = tf.keras.models.load_model('C:/Anirudh/IBM Hack Challenge/trained_model_0.h5')
    result = pred_and_plot(model_0, 'C:/Anirudh/IBM Hack Challenge/poseEstimation/{}'.format(image_url) , class_names)




    return render(request, 'predict/result.html',{'cf_dict':cf_dict, 'result':result})
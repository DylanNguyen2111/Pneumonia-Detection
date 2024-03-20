import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import mlflow



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):

        model = load_model(os.path.join("artifacts","training", "model.h5"))
        # model = load_model(os.path.join("model", "model.h5"))
 
        # #load model from mlflow uri
        # model = mlflow.pyfunc.load_model('models:/VGG16Model/11')

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Apply rescaling
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Pneumonia'
            return [{ "image" : prediction}]
    

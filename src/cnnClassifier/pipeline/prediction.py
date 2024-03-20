import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mlflow.pyfunc
import os



class PredictionPipeline:
    def __init__(self, filename) -> None:
        self.filename=filename

    
    def predict(self):
        #load model from training local folder
        # model = load_model(os.path.join("artifacts", "training", "model.h5"))
        #load model from mlflow uri
        model = mlflow.pyfunc.load_model('models:/VGG16Model/1')

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(f"This is result class: {result[0]}")

        if result[0] == 1:
            prediction = "Pneumonia"
            return [{"image" : prediction}]
        elif result[0] == 0:
            prediction = "Normal"
            return [{"image" : prediction}]
    

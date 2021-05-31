from tensorflow import keras

model = keras.models.load_model('trained_model.h5')

#inputprediction = [[Water, Cement, Fine aggregate, FM, Silica fume, superplasticizer]]
inputprediction = [[111.67, 252.9, 420.41, 2.5, 27.49, 7]]
#inputprediction = [[]]

def make_prediction():
    print(model.summary())
    
    prediction = model.predict(inputprediction)
    print(prediction)
    return prediction
make_prediction()
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class MultiLayerNetKeras:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        self.model = Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='valid',activation= 'relu', input_shape=[28,28,1]),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dropout(0.25),                        
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dropout(0.10),                         
            keras.layers.Dense(units=10, activation='softmax')
                         
            ])

    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def summary(self):
        self.model.summary()

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        return self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
    
    def save_model(self, file_name="model.h5"):
        self.model.save(file_name)
        print("Model saved successfully.")

    def load_model(self, file_name="model.h5"):
        self.model = keras.models.load_model(file_name)
        print("Model loaded successfully.")

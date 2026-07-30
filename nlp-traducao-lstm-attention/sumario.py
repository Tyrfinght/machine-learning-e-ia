from tensorflow import keras

load_model = keras.models.load_model

pretrained_model = load_model('model.h5')

for layer in pretrained_model.layers:
    print(f"Layer Name: {layer.name}")
    print(f"Layer Type: {type(layer)}")
    print(f"Layer Config: {layer.get_config()}")
    print("\n")
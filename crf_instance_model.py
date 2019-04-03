from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, ZeroPadding2D, Dropout, Conv2DTranspose, Cropping2D, Add,Dense
from crfasrnn_layer_instance import CrfRnnLayerForInstance
import object_detection_class

def get_instance_model(d):
    

    channels, height, weight = 21, 500, 500

    # Input
    input_shape = (height, weight, d)
    img_input = Input(shape=input_shape)
    box_term = Input(shape = (height, weight, d))
    global_term = Input(shape = (height, weight, d))

#     b = Dense(32)(img_input)
#     g = Dense(32)(img_input)
    
#     box_layer = Lambda(lambda x: boxterm)(b)
    
#     global_layer = Lambda(lambda x: globalterm)(g)
    
    output = CrfRnnLayerForInstance(image_dims=(height, weight), num_detections = d,
                         num_classes=d,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([img_input, box_term, global_term])
    
    

    # Build the model
    model = Model(inputs = [img_input,box_term, global_term] ,outputs = output, name='crfrnn_net')

    return model
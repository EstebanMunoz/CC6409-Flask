from app import model, imagenet_class_index, feature_extractor
import io
import torchvision.transforms as transforms
from PIL import Image


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


# out será un diccionario{'bloque': tensor}
# podríamos sacar features de más de un bloque.
# la llave será el nombre del bloque según nodes (variable en app.py)
def get_features(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    out = feature_extractor(tensor)
    return list(out.values())[0]

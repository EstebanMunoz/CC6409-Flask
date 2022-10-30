import json
from torchvision import models
from flask import Flask
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()

# Agregaremos un extractor a la red para sacar deep features.
# La variable nodes guarda todos los posibles lugares desde donde podríamos extraer.
nodes, _ = get_graph_node_names(model)
#print(nodes)

# En return_nodes debemos indicar a qué componentes les queremos agregar un extractor.
feature_extractor = create_feature_extractor(
    model, return_nodes=['adaptive_avg_pool2d'])
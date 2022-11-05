import os
import json

from flask import Flask
from dotenv import load_dotenv
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


load_dotenv()
IMAGNET_CLASS_PATH = os.getenv('IMAGNET_CLASS_PATH')

app = Flask(__name__)
imagenet_class_index = json.load(open(f'{IMAGNET_CLASS_PATH}'))
model = densenet121(weights=DenseNet121_Weights.DEFAULT)
model.eval()

# Agregaremos un extractor a la red para sacar deep features.
# La variable nodes guarda todos los posibles lugares desde donde podríamos extraer.
nodes, _ = get_graph_node_names(model)
#print(nodes)

# En return_nodes debemos indicar a qué componentes les queremos agregar un extractor.
feature_extractor = create_feature_extractor(
    model, return_nodes=['adaptive_avg_pool2d'])
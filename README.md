# CC6409-Flask

Ejemplo de deploy ML en Flask para el curso CC6409

El código para la API de Flask (carpeta `redcnn`) es una modificación de una parte de un tutorial propiedad de Pytorch [Link al tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)

El repositorio se compone de dos proyectos en Flask:

- `buscador`, que contiene una aplicación simple la cual recibe una
  imagen y devuelve las 3 imágenes más parecidas mediante el llamado a una API.
- `redcnn`, que corresponde a la API Rest. Dicho proyecto instancia un modelo DenseNet121
  preentrenado y lo utiliza para obtener el vector de características de la imagen que recibe por POST,
  devolviendo un archivo con el vector calculado.

## Instrucciones

- Instalar librerías necesarias.
  
  ```bash
   conda install flask
   conda install python-dotenv -c conda-forge
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
  ```

- Nótese que el comando para instalar pytorch podría variar según su sistema operativo. 
  Se recomienda visitar [Pytorch - Install](https://pytorch.org) y copiar el comando que corresponda según su configuración.

- En un primer terminal, levantar la API.
  
  ```bash
   cd redcnn
   python main.py
  ```

- En un segundo terminal, levantar la app de frontend:
  
  ```bash
   cd buscador
   python main.py
  ```

- Por defecto, la API quedará en el puerto 5001 y el front en el puerto 5000.

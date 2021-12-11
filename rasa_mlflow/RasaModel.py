from rasa.model_training import train_nlu
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter

class RasaTestModel(mlflow.pyfunc.PythonModel):
  """
  The python_function model flavor serves as a default model interface for MLflow Python models. Any MLflow Python model is expected to be loadable as a python_function model.
  Rasa model is saved in ".tar.gz" format in models directory after each training. To log the model in mlflow for Inferencing we will use RasaTestModel class.
  model logging command -> mlflow.pyfunc.log_model("RasaTestModel", python_model=obj)
  """
  def __init__(self):
    self.my_model_path = f"./models/20211001-142423.tar.gz"
	
  """
  def call_latest_model(self, model_dir):
	# Instead of hardcoding model filepath, this function can be called to automatically log last trained model.
	
	model_files = [f'{model_dir}/{f}' for f in os.listdir(model_dir)]
	return max(model_files, key=os.path.getctime)
  """

  def load_context(self, context):
    import numpy as np
    import pandas as pd
    import mlflow
    import rasa
    from rasa.cli.utils import get_validated_path
    from rasa.model import get_model, get_model_subdirectories
    from rasa.nlu.model import Interpreter
    return
  
  def load_interpreter(self):
    """
    This loads the Rasa NLU interpreter. It is able to apply all NLU pipeline steps to a text that you provide it.
    """
    model = get_validated_path(self.my_model_path, "model")
    self.my_model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(self.my_model_path)
    return Interpreter.load(nlu_model)
  
  def predict(self, context, model_input):
    """
    After logging the model in mlflow, at the time of prediction, predict function will be used to take in model input i.e any word/para for intent and entity classification and make predictions.
    """
    model = get_validated_path(self.my_model_path, "model")
    self.my_model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(self.my_model_path)
    nlu_interpreter = Interpreter.load(nlu_model)
    res = nlu_interpreter.parse(model_input)
    # test_nlu(model = self.latest_model_path, nlu_data = model_input, output_directory = None, additional_arguments = None)
    return res
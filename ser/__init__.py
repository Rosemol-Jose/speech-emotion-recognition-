#emodb dataset
import sys

class Model(object):
   

    def __init__(self, save_path=None, name='Not Specified', **params):
        
        
        self.model = None
        self.save_path = save_path
        self.name = name
        self.trained = False

    def train(self, x_train, y_train, x_val=None, y_val=None):
        
        self.model.fit(x_train, y_train)
        self.trained = True
        if self.save_path:
            self.save_model()

    def predict(self, data):
      
    
        if not self.trained:
            sys.stderr.write("Train the model before doing predict\n")
            sys.exit(-1)
        return self.model.predict(data)

    def restore_model(self, load_path=None):
      
        to_load = load_path or self.save_path
        if to_load is None:
            sys.stderr.write("Provide a path to load from or save_path of the model\n")
            sys.exit(-1)
        self.load_model(to_load)
        self.trained = True

    def load_model(self, to_load):
        
        
        raise NotImplementedError()

    def save_model(self):
               raise NotImplementedError()

    def evaluate(self, x_test, y_test):
       
        # This will be specific to model so should be implemented by child classes
        raise NotImplementedError()

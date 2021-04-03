from datetime import datetime
from sklearn import preprocessing

class DatasetMirror():

    def __init__(self, target=None, ignore_columns = []):
        self.ACTIONS_DICT = {"drop_columns":self.drop_columns,
                             "apply_fcn":self.apply_fcn}
                             
        self._actions = []
        self.ignore_columns=ignore_columns
        self._fcn={}
        self.target = target
        
    def set_ignore_columns(self, ignore_columns):
        self.ignore_columns = set(list(self.ignore_columns)  + ignore_columns)
        
    def register_function(self, name, fcn):
        if name in self._fcn:
            raise ValueError("Duplicated function: {}".format(name))
        self._fcn[name] = fcn
        
    def _ignore_columns(self, columns):
        columns = set(columns)
        # Get new set with elements that are only in a but not in b
        return columns.difference(self.ignore_columns)
  

    def drop_columns(self,data, columns, is_training=True, ignore_column_enabled=True):
        columns = self.__common_pre_actions("drop_columns",columns, is_training, ignore_column_enabled)
        return data.drop(columns, axis=1)
    

    def __common_pre_actions(self, fcn_name, columns, is_training, ignore_column_enabled):
        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append((fcn_name, columns))
        return columns

    def list_transform(self):
        for action in self._actions:
            print ("{}:\n\tparams: {}\n".format(action[0],action[1:]))
            
    def transform(self, data, is_training=False):
        for action in self._actions:
            data=self.ACTIONS_DICT[action[0]](data,*action[1:], is_training=is_training)
        return data

    def apply_fcn(self, fcn_name, data, columns=[], params={}, is_training=True, ignore_column_enabled=True):
        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append(("apply_fcn",fcn_name, columns, params))
        data=self._fcn[fcn_name](data, *columns, **params)
        return data

    
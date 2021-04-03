from datetime import datetime
from sklearn import preprocessing


def compare(key, value ):
    if type(value) is float:
        return 0
    if key in value:
        return 1
    return 0

class DatasetMirror():

    def __init__(self, target=None, ignore_columns = []):
        self.ACTIONS_DICT = {"filter":self.filter_by_column,
                             "drop":self.drop_columns,
                             "drop_na":self.drop_na_columns,
                             "fill_na_mode":self.fill_na_mode,
                             "mean_encoder":self.mean_encoder,
                             "year_encoder":self.year_encoder,
                             "apply_fcn":self.apply_fcn,
                             "data_scaler": self.data_scaler,
                             "data_scaler_continous":self.data_scaler_continous,
                             "list_2_categorical":self.list_2_categorical,
                             "reorder_columns":self.reorder_columns,
                             "filter_threshold":self.filter_threshold}
                             
        self.data_scaler_dict ={'default':self.data_scaler,
                                'continous':self.data_scaler_continous,
                                'skip':self.data_scaler_no_apply}

        self.target = target
        self.scaler = preprocessing.StandardScaler()
        self.continuos_scaler = preprocessing.MinMaxScaler()
        self.target_scaler = preprocessing.MinMaxScaler()
        self._actions = []
        self.config_dict={}
        self.ignore_columns=ignore_columns
        self._fcn={}
        
    def set_ignore_columns(self, ignore_columns):
        self.ignore_columns = set(list(self.ignore_columns)  + ignore_columns)
        
    def register_function(self, name, fcn):
        if name in self._fcn:
            raise ValueError("Duplicated function: {}".format(name))
        self._fcn[name] = fcn
        
    def _ignore_columns(self, columns):
        print ("Ignore columns: input", columns)
        # Create sets of a,b
        columns = set(columns)
        # Get new set with elements that are only in a but not in b
        return columns.difference(self.ignore_columns)
  
    def filter_threshold(self, data, column, threshold, is_training=True):
        if is_training:
            self._actions.append(("filter_threshold", column, threshold))
        
        total= len(data[column])
        outliers = len(data[data[column]>=threshold])
        print ("{}/{} -- Remove {}%".format(outliers,total, float(outliers)/total))            
        
        return data[data[column]<threshold]
    
    def filter_by_column(self, data, column, values, is_training=True):
        if is_training:
            self._actions.append(("filter", column, values))
        return data[data[column].isin(values)]

    def drop_columns(self,data, columns, is_training=True, ignore_column_enabled=True):
        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append(("drop", columns))
        return data.drop(columns, axis=1)
    
    def drop_na_columns(self,data, columns, is_training=True, ignore_column_enabled=True):
        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append(("drop_na", columns))
        
        for column in columns:
            na = data[column].isna().sum()    
            print ("{}: {}/{} --- Remove {}%".format(column,na, len(data),float(na)/len(data)))
        return data.dropna(subset=columns)
    
    def apply_fcn(self, data, fcn_name, columns, params={}, is_training=True):
#                   ignore_column_enabled=True):
#         if ignore_column_enabled:
#             columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append(("apply_fcn",fcn_name, columns, params))
        data=self._fcn[fcn_name](data, columns, **params)
        return data
    
    def fill_na_mode(self, data, columns=None, is_training=True, ignore_column_enabled=True):
        if columns is None:
            columns=data.columns

        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
            
        if is_training:
            self._actions.append(("fill_na_mode", columns))
       
        for column in columns:
            data[column].fillna(data[column].mode()[0],inplace=True)
        return data

    def list_2_categorical(self, data, column, list_keys, drop_column = True, is_training = True):
        """
        Convert a column with list of values to categorical
        """
        
        if is_training:
            self._actions.append(("list_2_categorical", column,list_keys))
    
        for key in list_keys:
            data[column+ " " + key]=data[column].apply(lambda x: compare(key, x))

        if drop_column:
            data=data.drop(column, axis=1)
        return data
        
    def mean_encoder(self, data, columns, drop_column = True, is_training=True , ignore_column_enabled=True):
        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append(("mean_encoder", columns, drop_column))
            
        for column in columns:
            if is_training:
                self.config_dict[column]={"method":"mean", "mean":data.groupby(column)[self.target].mean()}

            data.loc[:,column+" Mean"] = data[column].map(self.config_dict[column]["mean"])

            if drop_column:
                data=data.drop(column, axis=1)
        
        return data

    def year_encoder(self, data, columns, drop_column = True, is_training=True, ignore_column_enabled=True):
        if ignore_column_enabled:
            columns = self._ignore_columns(columns)
        if is_training:
            self._actions.append(("year_encoder",columns,drop_column))

        for column in columns:
            data[column] = data[column].apply(lambda x: datetime.strptime(x,'%Y-%m-%d') if type(x)==str else x)
            data[column + ' years'] = data[column].apply(lambda x: 2020 - x.year)

            if drop_column:
                data=data.drop(column, axis=1)
        return data
    def list_transform(self):
        for action in self._actions:
            print ("{}:\n\tparams: {}\n".format(action[0],action[1:]))
            
    def transform(self, data, is_training=False):
        for action in self._actions:
            data=self.ACTIONS_DICT[action[0]](data,*action[1:], is_training=is_training)
        return data
    
    def transform_split(self, data):
        data = self.transform(data)
        return self.split_data(data)
    
    def split_data(self, data, scale_data='default', scale_target=False, is_training= True):
        scaler_fcn=  self.data_scaler_dict[scale_data]

        cols = data.columns.tolist()
        cols.insert(0, cols.pop(cols.index(self.target)))
        data = data[cols]
        
#        values = scaler_fcn(data.values, is_training)
        values = data.values
        y = self.data_scaler_target(values[:,0:1], 
                                    is_training=is_training,
                                    scale_target=scale_target)
        
        X = scaler_fcn(values[:,1:], is_training)
        return X, y, cols[1:]
    
    def data_scaler(self, data, is_training=True):
        if is_training:
            self.scaler.fit(data)
        return self.scaler.transform(data)
    
    def data_scaler_continous(self, data, is_training=True):
        if is_training:
            self.continuos_scaler.fit(data)
        
        return self.continuos_scaler.transform(data)
    
    def data_scaler_no_apply(self, data, is_training=True):
        return data
        
            
    def data_scaler_target(self, data, is_training=True, scale_target=True):
        if not scale_target:
            return data
        
        if is_training:
            self.target_scaler.fit(data)
            
        return self.target_scaler.transform(data)
    
    def reorder_columns(self, data, columns, is_training=True):
        if is_training:
            self._actions.append(("reorder_columns",columns))
        return data[columns]

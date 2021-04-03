import unittest
from threading import Event
from  dtools.datamirror import DatasetMirror
import pandas as pd

class TestDataMirror(unittest.TestCase):

    def setUp(self):
        self.datamirror = DatasetMirror()
        
    def test_class_empty(self):
        """Test class no params"""
        datamirror = DatasetMirror()
        self.assertEqual(datamirror.target, None)
        self.assertEqual(datamirror.ignore_columns, [])
        
    def test_class_parameters(self):
        """Test class with params"""
        datamirror = DatasetMirror("param1", ["param2"])
        self.assertEqual(datamirror.target, "param1")
        self.assertEqual(datamirror.ignore_columns, ["param2"])
        
    def test_set_ignore_columns(self):
        """Test set_ignore_columns"""
        self.datamirror.set_ignore_columns(['param1','param1'])
        self.assertEqual(self.datamirror.ignore_columns,set(["param1"]))

    def test_register_function(self):
        """Test register_function"""
        self.datamirror.register_function("register_event", "test")
        self.assertTrue("register_event" in self.datamirror._fcn.keys())
        self.assertEqual(self.datamirror._fcn["register_event"],"test")

    

    def test_apply_fcn(self):
        """Test apply_fcn"""
        test_event = Event()

        def register_event(data, columns, **params):
            test_event.set()
            return data, columns, params

        self.datamirror.register_function("register_event", register_event)
        data = self.datamirror.apply_fcn("register_event", "test", ['param1'],{'test':'param2'})
        self.assertTrue(test_event.is_set())
        self.assertEqual(data, ("test", 'param1',{'test':'param2'}))

    def test_drop_columns(self):
        """Test drop_columns"""
        
        data = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        
        data = self.datamirror.drop_columns(data, ['col2'])

        self.assertEqual(data.columns, ['col1'])

    def test_transform(self):
        """Test transform"""
        
        data = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        data2 = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        
        data = self.datamirror.drop_columns(data, ['col2'])

        data2 = self.datamirror.transform(data2)
        self.assertTrue(data.equals(data2))


    


if __name__ == '__main__':
    unittest.main(verbosity=2)

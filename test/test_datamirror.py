import unittest
from  dtools.datamirror import DatasetMirror
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
        self.datamirror = set_ignore_columns(['param1','param1'])
        self.assertEqual(self.datamirror.ignore_columns,set(["param1"]))

if __name__ == '__main__':
    unittest.main(verbosity=2)

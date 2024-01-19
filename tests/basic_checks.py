import unittest
import os
import importlib.util

class TestSentimentFinData(unittest.TestCase):

    def test_library_torch_installed(self):
        """ Test if torch library is installed """
        torch_installed = importlib.util.find_spec("torch") is not None
        self.assertTrue(torch_installed, "torch library is not installed")

    def test_library_transformers_installed(self):
        """ Test if transformers library is installed """
        transformers_installed = importlib.util.find_spec("transformers") is not None
        self.assertTrue(transformers_installed, "transformers library is not installed")
        
    def test_library_numpy_installed(self):
        """ Test if numpy library is installed """
        numpy_installed = importlib.util.find_spec("numpy") is not None
        self.assertTrue(numpy_installed, "numpy library is not installed")
        
    def test_library_pandas_installed(self):
        """ Test if OpenCV library is installed """
        pandas_installed = importlib.util.find_spec("pandas") is not None
        self.assertTrue(pandas_installed, "pandas library is not installed")
        
    def test_library_datasets_installed(self):
        """ Test if datasets library is installed """
        datasets_installed = importlib.util.find_spec("datasets") is not None
        self.assertTrue(datasets_installed, "datasets library is not installed")
        
    def test_library_sklearn_installed(self):
        """ Test if sklearn library is installed """
        sklearn_installed = importlib.util.find_spec("sklearn") is not None
        self.assertTrue(sklearn_installed, "sklearn library is not installed")

    def test_data_file_exists(self):
        """ Test if the data file exists """
        data_file_path = '/mnt/code/all-data.csv'
        self.assertTrue(os.path.isfile(data_file_path), "Data file does not exist")

if __name__ == '__main__':
    unittest.main()

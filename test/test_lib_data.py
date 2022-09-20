from lib_data import create_dataframe

class TestCreateDataframe:
    def test_num_classes(self):
        ds_df, num_classes = create_dataframe("./test/mnist_mini/")
        assert num_classes == 10

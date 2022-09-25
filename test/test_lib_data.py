from lib_data import create_dataframe

class TestCreateDataframe:
    def test_num_classes(self):
        ds_df, num_classes = create_dataframe("./test/mnist_mini/")
        assert num_classes == 10
    def test_df_shape(self):
        ds_df, num_classes = create_dataframe("./test/mnist_mini/")
        assert ds_df.shape == (1000,2)
    def test_df_columns(self):
        ds_df, num_classes = create_dataframe("./test/mnist_mini/")
        assert ds_df.columns[0] == 'File'
        assert ds_df.columns[1] == 'Label'
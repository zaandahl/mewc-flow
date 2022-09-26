from lib_data import create_dataframe, sample_dataframe

# mnist_mini dataset contains numbers 0-9 with 100 images in each class
class TestCreateDataframe:
    def test_num_classes(self):
        ds_df, num_classes,class_labels = create_dataframe("./test/mnist_mini/")
        assert num_classes == 10
    def test_df_shape(self):
        ds_df, num_classes,class_labels = create_dataframe("./test/mnist_mini/")
        assert ds_df.shape == (1000,2)
    def test_df_columns(self):
        ds_df,num_classes,class_labels = create_dataframe("./test/mnist_mini/")
        assert ds_df.columns[0] == 'File'
        assert ds_df.columns[1] == 'Label'

class TestSampleDataframe:
    def test_sample_replace(self):
        ds_df,num_classes,class_labels = create_dataframe("./test/mnist_mini/")
        samp_df = sample_dataframe(ds_df,num_classes,class_labels,n=[20])
        assert samp_df.shape == (200,2)
        samp_df = sample_dataframe(ds_df,num_classes,class_labels,n=[50,40,20,10,10,5,5,3,3,3])
        assert samp_df.shape == (149,2)
        samp_df = sample_dataframe(ds_df,num_classes,class_labels,n=[200],rep=False)
        assert samp_df.shape == (2000,2)

from lib_data import create_dataframe,sample_dataframe,split_dataframe,create_dataset

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

class TestSplitDataframe:
    def test_split_dataframe(self):
        ds_df,num_classes1,class_labels = create_dataframe("./test/mnist_mini/")
        train_df,val_df,test_df,num_classes2 = split_dataframe(ds_df, seed=42, val_split=0.2, test_split=0)
        assert num_classes1 == num_classes2
        assert train_df.shape == (800,2)
        assert val_df.shape == (200,2)
        assert test_df == None

class TestCreateDataset:
    def test_create_dataset(self):
        ds_df,num_classes,class_labels = create_dataframe("./test/mnist_mini/")
        tensor_ds = create_dataset(ds_df, img_size=28, batch_size=10, magnitude=0, seed=42, augment=False)
        assert tensor_ds.cardinality().numpy() == 100
        tensor_ds = create_dataset(ds_df, img_size=28, batch_size=10, magnitude=0, seed=42, augment=True)
        assert tensor_ds.cardinality().numpy() == 100

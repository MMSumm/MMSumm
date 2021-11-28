# Generate lineidx files
# Creates lookup tables for data search within files

from oscar.utils.tsv_file import TSVFile

prefix = '/scratch/summ_data_imgs_test/'

labels = TSVFile(prefix + 'labels.tsv', generate_lineidx=True)
vgg_features = TSVFile(prefix + 'vgg_features.tsv', generate_lineidx=True)
features = TSVFile(prefix + 'features.tsv', generate_lineidx=True)
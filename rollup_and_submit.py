__author__ = 'sheraz'

from os import listdir
from os.path import isfile, join
import datetime
import csv


# Script to take all of the item level predictions and roll them up into a master submission
def rollup():

    itempath = '../wmt_data/submissions/by_item/'
    storepath = '../wmt_data/submissions/by_store/'

    all_item_files, all_store_files = get_files(itempath)
    now = datetime.datetime.now()
    cur_time = str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)

    with open('../wmt_data/submissions/submission{}.csv'.format(cur_time), 'w') as f_submission:
        f_submission.write('id,units\n')

        # Loop through item level prediction files
        for cur_file in all_item_files:
            print("writing for {}".format(cur_file))
            with open(itempath+cur_file, 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    f_submission.write('%s,%s\n' % (row[0], row[1]))

            f.close()

        # Loop through store level prediction files
        for cur_file in all_store_files:
            print("writing for {}".format(cur_file))
            with open(storepath+cur_file, 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    f_submission.write('%s,%s\n' % (row[0], row[1]))

            f.close()

        f_submission.close()


def get_files(item_path):

    itemfiles = [f for f in listdir(item_path) if isfile(join(item_path, f))]
    storefiles = []
    items_to_add_by_store = ['8', '9', '98', '95', '87', '81', '78', '67', '63','13', '17', '24', '28',
                             '35', '38', '41', '48', '58', '68', '70', '74']
    file_template = 'item_'
    files_to_remove = []

    for num in items_to_add_by_store:
        this_file = file_template+str(num)+'.csv'
        files_to_remove.append(this_file)

    for remove_file in files_to_remove:
        itemfiles.remove(remove_file)

    file_template = 'itemstore_'

    for item in items_to_add_by_store:
        for store in range(1,46):
            if store != 35:
                this_file = file_template+str(item)+'_'+str(store)+'.csv'
                storefiles.append(this_file)

    return itemfiles, storefiles

if __name__ == "__main__":
    rollup()
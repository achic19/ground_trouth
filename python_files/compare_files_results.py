import pandas as pd
import os


def compare_results(self,new_data):
    data = pd.read_csv(os.path.join(self.workspace_csv_progress, 'avarage_spd' + str(self.time_for_avg) + '.csv'))
    if self.user == 'ped':
        new_data = data.drop(data[data['speed'] > 1.5].index)
        size = new_data.shape[0]
        stat_dic = {
            'number of records': size,
            'jack': (new_data.drop(new_data[new_data['ROWSTATUS'] != 'Out of range'].index)).shape[0] / size * 100,
            'ours': (new_data.drop(new_data[new_data['avarage_spd'] < 1.5].index)).shape[0] / size * 100}
    else:
        size = new_data.shape[0]
        stat_dic = {
            'number of records': size,
            'jack': (new_data.drop(new_data[new_data['ROWSTATUS'] != 'Valid'].index)).shape[0] / size * 100,
            'ours':
                (new_data.drop(new_data[new_data['speed'] < 1.5 and new_data['avarage_spd'] > 1.5].index)).shape[
                    0] / size * 100}
    print(stat_dic)
    save_dic = os.path.join(self.workspace_csv_progress, 'stat_file.txt')
    with open(save_dic, 'w') as f:
        print(stat_dic, file=f)

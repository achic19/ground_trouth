import math
import os

import arcpy
import numpy as np
import pandas as pd


# Python 3.6 (arcgispro-py3) C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe


class SortByGroundTruth:

    def __init__(self, data_date, workspace_main_progress, workspace, bt_file, gps_file, def_to_run='all'):
        """

        :param data_date:
        :param workspace_main_progress:
        :param workspace:
        :param bt_file:
        :param gps_file:
        :param def_to_run:
        """
        self.gps_file = gps_file
        meta_data = str.split(os.path.basename(self.gps_file), '_')
        if meta_data[len(meta_data) - 1] == 'car.csv':
            self.user = 'car'
        else:
            self.user = 'ped'
        self.surveyor = meta_data[0]
        self.mac = []
        self.mac.append(meta_data[1])
        self.short_name = '_'.join([self.surveyor, self.mac[0]])
        self.date = data_date.replace('.csv', '')
        self.obj_name = '_'.join([self.short_name, self.date])
        self.workspace_path = workspace
        self.workspace_gis_progress = workspace_main_progress
        self.network_links_gis = os.path.join(self.workspace_path, r'general.gdb/links_network')
        self.workspace_csv_progress = os.path.join(workspace_main_progress, 'progress_files', self.short_name)
        self.bt_file_path = bt_file
        self.time_for_avg = 300

        print('The program run on {} ,{} on {} {}'.format(self.surveyor, self.mac[0], self.date, self.user))
        self.name_gis = str.replace(self.obj_name, '.', '_')
        if def_to_run != 'no':
            if os.path.isdir(self.workspace_csv_progress):
                print('data are processed')
            else:
                os.makedirs(self.workspace_csv_progress)
            self.create_new_gdb()
            print('finish to create_new_gdb')

            arcpy.env.workspace = os.path.join(self.workspace_gis_progress, self.name_gis + '.gdb')

            self.add_itm_cor_to_csv_file()
            print('finish to add_itm_cor_to_csv_file')

            self.calc_speed()
            print('finish to calc_speed for each gps points')

            self.remove_points_near_intersection()
            print('finish to remove_points_near_intersection')

            self.spatial_join()
            print('finish to execute_points_near_intersection')

            self.join_data()
            print('finish to execute_join_data')

            # self.calc_neto_length()
            # print('finish to execute_calc_neto_length')
            #
            # self.calc_stats(self.workspace_csv_progress, self.obj_name)
            # print('finish to execute_calc_stats')
            #
            # self.use_same_length_for_each_dir(self.workspace_csv_progress)
            # print('finish to execute_use_same_length_for_each_dir')
            # join_df = pd.read_csv(os.path.join(self.workspace_csv_progress, 'join' + self.date + '.csv'))
            #
            # self.calculate_speed(join_df)
            # print('finish to execute_calculate_speed')
            #
            # self.calc_avg_speed(self.workspace_csv_progress)
            # print('finish to execute_calculate_speed')

    def create_new_gdb(self):
        # Execute CreateFileGDB
        arcpy.CreateFileGDB_management(self.workspace_gis_progress, self.name_gis)

    def add_itm_cor_to_csv_file(self):
        fc_file = self.name_gis
        arcpy.management.XYTableToPoint(self.gps_file, fc_file,
                                        "lon", "lat", coordinate_system=arcpy.SpatialReference(4326))

        obj_name_pro = fc_file + '_pro'

        # Process: Project
        arcpy.Project_management(in_dataset=fc_file, out_dataset=obj_name_pro,
                                 out_coor_system=arcpy.SpatialReference(2039),
                                 transform_method="WGS_1984_To_Israel_CoordFrame",
                                 in_coor_system=arcpy.SpatialReference(4326),
                                 preserve_shape="NO_PRESERVE_SHAPE", max_deviation="", vertical="NO_VERTICAL")

        # Process: Add Geometry Attributes
        arcpy.AddGeometryAttributes_management(Input_Features=obj_name_pro,
                                               Geometry_Properties="POINT_X_Y_Z_M", Length_Unit="METERS", Area_Unit="",
                                               Coordinate_System="")
        # Process: Table To Table
        arcpy.TableToTable_conversion(obj_name_pro, self.workspace_csv_progress, fc_file + 'itm.csv')

    def calc_speed(self):
        csv_file_to_add_time_stamp = os.path.join(self.workspace_csv_progress, self.name_gis + 'itm.csv')
        meas_data = pd.read_csv(csv_file_to_add_time_stamp)
        meas_data['new_time'] = meas_data['time'].astype(str).map(lambda x: x.replace('T', ' '))
        meas_data['new_time'] = meas_data['new_time'].astype(str).map(lambda x: x.replace('Z', ''))
        meas_data['new_time'] = meas_data['new_time'].map(lambda x: pd.to_datetime(x).timestamp())
        meas_data.sort_values(by=['new_time'])

        meas_data['speed_1'] = ''
        # Calculate speed
        for index, record in meas_data.iterrows():
            if index == 0:
                continue
            else:
                lst_index = index - 1
                dx = record['POINT_X'] - meas_data.at[lst_index, 'POINT_X']
                dy = record['POINT_Y'] - meas_data.at[lst_index, 'POINT_Y']
                tr_time = record['new_time'] - meas_data.at[lst_index, 'new_time']
                length = math.sqrt(dx * dx + dy * dy)
                meas_data.at[index, 'speed_1'] = length / tr_time

        meas_data.to_csv(os.path.join(self.workspace_csv_progress, r'stamp_time.csv'))

    def remove_points_near_intersection(self):
        itm_fc_name = 'all_gps_pnts'
        arcpy.Intersect_analysis([self.network_links_gis], 'links_intersections', output_type='POINT')
        arcpy.Buffer_analysis('links_intersections', 'buffer_intersection', 30)

        arcpy.management.XYTableToPoint(os.path.join(self.workspace_csv_progress, 'stamp_time.csv'), itm_fc_name,
                                        "POINT_X", "POINT_Y", coordinate_system=arcpy.SpatialReference(2039))

        arcpy.Intersect_analysis(['buffer_intersection', itm_fc_name], 'point_in_buffer', output_type='POINT')
        arcpy.SymDiff_analysis(itm_fc_name, 'point_in_buffer', "symdiff")

    def spatial_join(self):
        # Delete unnecessary fields
        field_mapping = 'time "time" true true false 8000 Text 0 0,First,#,symdiff,time,0,8000;' \
                        'POINT_X "POINT_X" true true false 8 Double 0 0,First,#,symdiff,POINT_X,-1,-1;' \
                        'POINT_Y "POINT_Y" true true false 8 Double 0 0,First,#,symdiff,POINT_Y,-1,-1;' \
                        'new_time "new_time" true true false 8 Double 0 0,First,#,symdiff,new_time,-1,-1;' \
                        'speed_1 "speed_1" true true false 8 Double 0 0,First,#,symdiff,speed_1,-1,-1'

        arcpy.FeatureClassToFeatureClass_conversion(in_features="symdiff", out_path=arcpy.env.workspace,
                                                    out_name="gps_pnt_to_join",
                                                    field_mapping=field_mapping)

        arcpy.SpatialJoin_analysis("gps_pnt_to_join", self.network_links_gis, 'gps_point_joined',
                                   join_type='KEEP_COMMON',
                                   match_option='CLOSEST', search_radius=70, distance_field_name='distance')
        arcpy.TableToTable_conversion('gps_point_joined', self.workspace_csv_progress, 'pnts_to_calc.csv')

    def join_data(self):

        # path to csv files and feature class

        # from feature class to dataframe format
        via_to = list()
        shape_length = list()
        cursor = arcpy.da.SearchCursor(self.network_links_gis, ['via_to', 'Shape_Length'])
        for row in cursor:
            via_to.append(row[0])
            shape_length.append(row[1])
        df_network = pd.DataFrame({'via_to': [], 'Shape_Length': []})
        df_network['via_to'] = via_to
        df_network['Shape_Length'] = shape_length
        # print(df_network)

        # Merge feature class and csv file
        all_bt_file = pd.read_csv(self.bt_file_path)
        all_bt_file['via_to'] = all_bt_file['VIAUNITC'] + all_bt_file['TOUNITC']
        df = pd.merge(all_bt_file, df_network, on=['via_to'], how='inner')
        df.to_csv(os.path.join(self.workspace_csv_progress, 'join' + self.date + '.csv'))

    def calc_neto_length(self):
        """
        The method calculates length between enter to exit link by using travel time as given in the TB file and speed as
        given in gps file. The relation between these file is based on link name and time
        :return:
        """

        gps_pnts = pd.read_csv(os.path.join(self.workspace_csv_progress, 'pnts_to_calc.csv'))

        # Filter speed more than 2 m/s (7 km/h)
        if self.user == 'ped':
            gps_pnt_less_2 = gps_pnts.loc[gps_pnts['speed_1'] < 2]
        else:
            gps_pnt_less_2 = gps_pnts

        bt_links = pd.read_csv(os.path.join(self.workspace_csv_progress, 'join' + self.date + '.csv'))
        bt_links = bt_links.loc[(bt_links['MAC'].isin(self.mac))]

        # Add two new columns: sum_spd and n with zero values
        new_array = np.zeros(bt_links.shape[0])
        bt_links['sum_spd'] = new_array
        bt_links['n'] = new_array
        # for each point find the candidates   link by one of 2 condition: via= via AND to= to OR  via= to and  to =via
        for index, row in gps_pnt_less_2.iterrows():

            candidates_1 = bt_links.loc[(row['VIA'] == bt_links['VIAUNITC']) & (row['TO'] == bt_links['TOUNITC'])]
            candidates_2 = bt_links.loc[(row['VIA'] == bt_links['TOUNITC']) & (row['TO'] == bt_links['VIAUNITC'])]
            candidates = candidates_1.append(candidates_2)
            # if  more than one record found
            if len(candidates.index) > 1:
                # find the record with the min time diff base on comparison to   entrance/exit from a link
                candidates_1 = candidates.iloc[(candidates['LASTDISCOTS'] - row['new_time']).abs().argsort()[:1]]
                candidates_2 = candidates.iloc[(candidates['CLOSETS'] - row['new_time']).abs().argsort()[:1]]
                # if candidates_1 and candidates_2 are the same
                if candidates_1.iloc[0]['PK_UID'] == candidates_2.iloc[0]['PK_UID']:
                    bt_index = candidates_1.iloc[0]['PK_UID']
                else:
                    # if candidates_1 and candidates_2 are not the  same , select the min one between them
                    if abs(candidates_1.iloc[0]['LASTDISCOTS'] - row['new_time']) < abs(
                            candidates_2.iloc[0]['CLOSETS'] - row[
                                'new_time']):
                        bt_index = candidates_1.iloc[0]['PK_UID']
                    else:
                        bt_index = candidates_2.iloc[0]['PK_UID']

            elif len(candidates.index) > 0:
                bt_index = candidates.iloc[0]['PK_UID']
            else:
                # print("NO link {} is found ".format(row['via_to']))
                continue

            # Add to 'sum_spd' in the proper record the new speed and continue to count the number of gps points
            # that will participate in the average speed later
            pk_ud_ind = (bt_links.index[bt_links['PK_UID'] == bt_index])[0]
            # if there is more than 5 minutes difference don't do anything
            if abs(bt_links.at[pk_ud_ind, 'CLOSETS'] - row['new_time']) > 300:
                continue
            bt_links.at[pk_ud_ind, 'sum_spd'] = bt_links.at[pk_ud_ind, 'sum_spd'] + row['speed_1']
            bt_links.at[pk_ud_ind, 'n'] = bt_links.at[pk_ud_ind, 'n'] + 1

        # Calculate the  average speed
        bt_links['avg_spd'] = bt_links['sum_spd'] / bt_links['n']

        # Calculate neto length
        bt_links['neto_length'] = bt_links['avg_spd'] * (bt_links['CLOSETS'] - bt_links['LASTDISCOTS'])

        # Make it one directional by change the “via_to” field (if necessary) to smaller first in TB file
        for index, row in bt_links.iterrows():
            if row['VIAUNITC'] > row['TOUNITC']:
                bt_links.at[index, 'via_to'] = row['TOUNITC'] + row['VIAUNITC']
        # Delete links without calculated average speed
        bt_links = bt_links.drop(bt_links[bt_links['n'] == 0].index)
        bt_links.to_csv(os.path.join(self.workspace_csv_progress, 'neto_length_' + self.obj_name + '.csv'))

        return bt_links

    @staticmethod
    def papulate_frames(frame, folder, file_not_papulte):
        for date_data in os.listdir(folder):
            if date_data == 'general' or date_data == 'raw_data' or date_data == 'missing_bt' \
                    or date_data == 'test':
                continue
            if date_data in file_not_papulte:
                continue
            else:
                users = os.path.join(folder, date_data + '/progress_files')
                for user in os.listdir(users):
                    bt_file_path = os.path.join(users, user, 'neto_length_' + user + '_' + date_data + '.csv')
                    frame.append(pd.read_csv(bt_file_path))

    @staticmethod
    def join_preprocess_to_one_file(frames, path):
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(path)

    @staticmethod
    def calc_stats(workspace_csv_progress, obj_name, all=False):
        '''
        :param df: database with net length for each record
        for each link calc averge and std base on neto length of each record
        :return:
        '''
        print('run stat on net length')
        if all == False:
            df = pd.read_csv(os.path.join(workspace_csv_progress, 'neto_length_' + obj_name + '.csv'))
        else:
            df = pd.read_csv(os.path.join(workspace_csv_progress, 'neto_length.csv'))
        # Drop file with time span less than 1
        df = df.drop(df[(df['CLOSETS'] - df['LASTDISCOTS']) <= 0].index)
        df = df[df['n'] > 5]
        # df = df[df['Shape_Length'] >df['neto_length']]
        # df = df[df['neto_length'] < df['Shape_Length']]
        # Delete links with radius larger than 100 meters
        if all == '100_meters_restriction':
            df = df.drop(df[df['Shape_Length'] - df['neto_length'] > 200].index)
        # to evaluate the results all the neto_length by pk will be store at the stat file
        # df['pk_neto_length'] = df.loc[:, ['PK_UID', 'neto_length']].apply(list, axis=1)
        groups_via_to = df.groupby(['via_to'])
        if all == 'median':
            gk7 = groups_via_to['neto_length'].median()
        else:
            gk7 = groups_via_to['neto_length'].mean()
        # statistic calculations
        series = groups_via_to['neto_length'].apply(lambda x: SortByGroundTruth.calculte_avg_with_error_gross(x))
        temp_df = pd.DataFrame([[a, b] for a, b in series.values], columns=['mean', 'std'], index=series.index)
        gk2 = groups_via_to['neto_length'].count()
        gk0 = temp_df['mean']
        gk1 = temp_df['std']
        # gk0 = groups_via_to['neto_length'].mean()
        # gk1 = groups_via_to['neto_length'].std()
        gk3 = gk1 / gk0
        gk4 = groups_via_to['neto_length'].min()
        gk5 = groups_via_to['neto_length'].max()
        gk6 = groups_via_to['neto_length'].apply(list)

        # With records fields all the length are stored
        # for group_name in df['via_to'].unique():
        #     group = str(df.get_group(group_name)['time'].values.tolist())
        gk = pd.concat([gk0, gk1, gk2, gk3, gk4, gk5, gk6, gk7], axis=1, )
        gk.columns = ['mean', 'std', 'count', 'ratio', 'min', 'max', 'neto_length', 'old_mean']
        gk.sort_values(by=['ratio'], inplace=True, ascending=False)
        gk.to_csv(os.path.join(workspace_csv_progress, 'mean_std.csv'))

    @staticmethod
    def use_same_length_for_each_dir(workspace_csv_progress):
        """
        df file include data about average neto length for links only as one direction, so the method copy the length and
        sdt to the opposite direction
        """
        df = pd.read_csv(os.path.join(workspace_csv_progress, 'mean_std.csv'))
        df = df[pd.notna(df['mean'])]
        temp_list = []
        for index, row in df.iterrows():
            split_row = row['via_to'].split('T')
            temp_row = ['T' + split_row[2] + 'T' + split_row[1], row['mean'], row['std'], row['count']]
            temp_list.append(temp_row)
        reverse_df = pd.DataFrame(temp_list, columns=['via_to', 'mean', 'std', 'count'])
        new_df = df.append(reverse_df).reset_index()
        new_df.to_csv(os.path.join(workspace_csv_progress, 'mean_std_bidirectional.csv'))

    @staticmethod
    def calculte_avg_with_error_gross(df):

        mean_0 = df.mean()
        n = df.shape[0]
        sus_meas_index = abs(df - mean_0).idxmax()
        df_temp = df.drop(index=sus_meas_index)
        mean_1 = df_temp.mean()
        error = abs(mean_1 - df[sus_meas_index])
        error_tol = df_temp.std()
        if error > error_tol:
            return mean_1, error_tol
        else:
            return mean_0, df.std()

    def calculate_speed(self, join_df, all=False, new_path=''):
        print("calculate speed")
        if all:
            mean_std_bidirectional_df = pd.read_csv(os.path.join(new_path, 'mean_std_bidirectional.csv'))

        else:
            # Store only records with matching links as appear  in mean_std_bidirectional_df
            mean_std_bidirectional_df = pd.read_csv(
                os.path.join(self.workspace_csv_progress, 'mean_std_bidirectional.csv'))
        rel_links = join_df.loc[(join_df['via_to'].isin(mean_std_bidirectional_df['via_to']))]
        rel_links.to_csv(os.path.join(self.workspace_csv_progress, 'rel_links.csv'))

        # Build dictionary of links and length
        my_dict = {row[2]: row[3] for row in mean_std_bidirectional_df.values}
        my_dict_std = {row[2]: row[4] for row in mean_std_bidirectional_df.values}
        temp_len = len(rel_links.columns)
        # assign new columns to our database
        rel_links.insert(loc=temp_len, column='net_link', value='')
        rel_links.insert(loc=temp_len, column='std_net_link', value='')
        rel_links.insert(loc=temp_len, column='speed', value='')
        rel_links.insert(loc=temp_len, column='std_speed', value='')
        for index, row in rel_links.iterrows():
            if row['CLOSETS'] - row['LASTDISCOTS'] > 0:
                rel_links.at[index, 'net_link'] = my_dict[row['via_to']]
                rel_links.at[index, 'std_net_link'] = my_dict_std[row['via_to']]
                rel_links.at[index, 'speed'] = my_dict[row['via_to']] / (row['CLOSETS'] - row['LASTDISCOTS'])
                rel_links.at[index, 'std_speed'] = my_dict_std[row['via_to']] / (row['CLOSETS'] - row['LASTDISCOTS'])
            else:
                rel_links.at[index, 'speed'] = -1000
        # drop records without speed and change 'std_speed' with null to zero
        rel_links['std_speed'].fillna(0, inplace=True)
        rel_links = rel_links.drop(rel_links[rel_links['speed'] == -1000].index)
        rel_links.to_csv(os.path.join(self.workspace_csv_progress, 'speed.csv'))

    def calc_avg_speed(self, workspace_csv_progress, all=False):
        """

        :param workspace_csv_progress:
        :param filtered_db:
        :param all:

        :return:
        """
        print("avg_speed")

        filtered_db = pd.read_csv(os.path.join(self.workspace_csv_progress, 'speed.csv'))
        # Calc our trip time, speed and average speed
        filtered_db.set_index('PK_UID', inplace=True, drop=False)
        filtered_db['avarage_spd'] = ''
        filtered_db['avarage_spd_10800'] = ''

        filtered_db['std_spd_avg'] = ''
        filtered_db['num_of_recs'] = ''

        # group by link name regardless direction
        gk = filtered_db.groupby('via_to')

        # For each record in each group find all the other records that was in the same link in the same time ( -30 sec) and
        #  calculate avarage and standard deviation
        for group_name in filtered_db.via_to.unique():
            group = gk.get_group(group_name)
            mac_records = group.loc[(group.MAC.isin(self.mac))]

            for index, record in mac_records.iterrows():
                pk_id = record['PK_UID']
                # In the next rows the code count the number of records
                # in the current moment ('CLOSETS') on the link ( the worst case) for a specific record
                record_LASTDISCOTS = record['LASTDISCOTS']
                result = group.loc[(record_LASTDISCOTS - group['LASTDISCOTS'] < self.time_for_avg) & (
                        record_LASTDISCOTS - group['LASTDISCOTS'] >= 0)]
                result_10800 = group.loc[(record_LASTDISCOTS - group['LASTDISCOTS'] < 10800) & (
                        record_LASTDISCOTS - group['LASTDISCOTS'] >= 0)]
                if result_10800.shape[0] > 1:
                    filtered_db.at[pk_id, 'avarage_spd'] = result.speed.mean()
                    filtered_db.at[pk_id, 'avarage_spd_10800'] = result_10800.speed.mean()
                    # filtered_db['std_spd_avg'].loc[filtered_db['PK_UID'] == pk_id] = math.sqrt(
                    #     (result_10800['speed'] ** 2).sum()) / result_10800.shape[0]
                    filtered_db.at[pk_id, 'std_spd_avg'] = result_10800.speed.std()
                    filtered_db.at[pk_id, 'num_of_recs'] = result_10800.shape[0]
                else:
                    filtered_db.at[pk_id, 'avarage_spd_10800'] = result_10800.speed.mean()
                    filtered_db.at[pk_id, 'avarage_spd'] = result_10800.speed.mean()
                    filtered_db.at[pk_id, 'std_spd_avg'] = 0
                    filtered_db.at[pk_id, 'num_of_recs'] = 1
        filtered_db.reset_index(inplace=True, drop=True)
        filtered_db = filtered_db.loc[filtered_db['avarage_spd'] != '']
        filtered_db.to_csv(
            os.path.join(workspace_csv_progress, 'avarage_spd_' + self.obj_name + '_' + self.user + '.csv'))
        if self.user == 'ped':
            if all:
                gps_file = pd.read_csv(os.path.join(workspace_csv_progress, 'neto_length.csv'))
            else:
                gps_file = pd.read_csv(os.path.join(workspace_csv_progress, 'neto_length_' + self.obj_name + '.csv'))
            gps_file = gps_file.rename(columns={'avg_spd': 'gps_spd'})
            filtered_db = pd.merge(filtered_db, gps_file[['PK_UID', 'gps_spd']], on=['PK_UID'], how='inner')
            filtered_db = filtered_db.drop(filtered_db[filtered_db['gps_spd'] > 1.5].index)
        else:
            if all:
                gps_file = pd.read_csv(os.path.join(workspace_csv_progress, 'neto_length.csv'))
            else:
                gps_file = pd.read_csv(os.path.join(workspace_csv_progress, 'neto_length_' + self.obj_name + '.csv'))
            gps_file = gps_file.rename(columns={'avg_spd': 'gps_spd'})
            filtered_db = pd.merge(filtered_db, gps_file[['PK_UID', 'gps_spd']], on=['PK_UID'], how='inner')
            filtered_db = filtered_db.drop(filtered_db[filtered_db['gps_spd'] < 1.5].index)
        filtered_db.to_csv(
            os.path.join(workspace_csv_progress, 'avarage_spd_' + self.obj_name + '_' + self.user + '.csv'))
        return filtered_db


def compare_results(data, user_type, workspace_csv_progress, confusion_matrix, k_s=0.9):
    """

    :param data_path:
    :param user_type:
    :param workspace_csv_progress:
    :param confusion_matrix by the classified results
    :return:
    """

    size = data.shape[0]
    k_a_s = 1
    data["std_speed"].fillna(0, inplace=True)  # replacing nan values in std_speed with zero
    # data = pd.read_csv(os.path.join(self.workspace_csv_progress, 'avarage_spd' + str(self.time_for_avg) + '.csv'))
    if user_type == 'ped':
        # If we know for sure no traffic jam is exist don't look up on the average

        our_result = data.loc[data['speed'] - k_s * data['std_speed'] < 1.5]
        our_result = our_result.loc[data['avarage_spd'] > data['avarage_spd_10800'] - k_a_s * data['std_spd_avg']]
        tp = our_result.shape[0]
        stat_dic = {
            'number of records_peds': size,
            'jack': (data.drop(data[data['ROWSTATUS'] != 'Out of range'].index)).shape[0] / size * 100,
            'ours': tp / size * 100}
        confusion_matrix['TP'] = tp
        confusion_matrix['FP'] = size - tp

    else:
        our_result = data.loc[
            (data['speed'] - k_s * data['std_speed'] > 1.5) | (
                    data['avarage_spd'] < data['avarage_spd_10800'] - k_a_s * data['std_spd_avg'])]
        tn = our_result.shape[0]
        stat_dic = {
            'number of records_cars': size,
            'jack': (data.drop(data[data['ROWSTATUS'] != 'Valid'].index)).shape[0] / size * 100,
            'ours': tn / size * 100}
        confusion_matrix['FN'] = size - tn
        confusion_matrix['TN'] = tn
    print(stat_dic)
    save_dic = os.path.join(workspace_csv_progress, user_type + '_stat_file.txt')
    with open(save_dic, 'w') as f:
        print(stat_dic, file=f)
    our_result.to_csv(os.path.join(workspace_csv_progress, user_type + '_right_records.csv'))


def final_sorting(path, k, name):
    '''

    :param path: path to the folder where _car.csv and _ped_csv are stored
    :param k: list for number of std , for speed and for avarage speed
    :return:
    '''
    # the next functions take all our records and sort them again to positive (1) or negative in
    # new field called 'right' and then group by via_to to get statistic

    # sort the files and append them
    ped_file = prepration('ped', k)
    car_file = prepration('car', k)
    final_sorting_file = ped_file.append(car_file)
    final_sorting_file.to_csv(os.path.join(path, 'ver_2/final_sorting_' + name + '.csv'))

    # groupby and for each group calculate the number of positive in respect to all its records
    final_sorting_stat = final_sorting_file.groupby(['via_to'])
    gk0 = final_sorting_stat['right'].count()
    gk1 = final_sorting_stat['right'].sum()
    gk2 = gk1 / gk0
    gk = pd.concat([gk0, gk1, gk2], axis=1, )
    gk.columns = ['count', 'right', 'ratio']
    gk.sort_values(by=['ratio'], inplace=True, ascending=False)
    gk.to_csv(os.path.join(path, 'ver_2/stat_final_' + name + '.csv'))


def prepration(type, k):
    file = pd.read_csv(os.path.join(start_to_run['final_sorting'], '_' + type + '.csv'))
    file['type'] = type
    file['via_to'] = file.apply(lambda x: ab_direct(x), axis=1)
    file = file[~file['via_to'].isin(filter_list)]
    file['right'] = 0
    if type == 'car':
        file['right'] = file.apply(lambda x: sorting_car(x, k[0], k[1]), axis=1)
    else:
        file['right'] = file.apply(lambda x: sorting_ped(x, k[0], k[1]), axis=1)
    return file


def sorting_car(data, k_s, k_a_s):
    # check if car is car by our condition. YES- retrun a No - return 0
    if (data['speed'] - k_s * data['std_speed'] > 1.5) | (
            data['avarage_spd'] < data['avarage_spd_10800'] - k_a_s * data['std_spd_avg']):
        return 1
    else:
        return 0


def sorting_ped(data, k_s, k_a_s):
    # check if car is pedestrian  by our condition. YES- retrun a No - return 0
    if (data['speed'] - k_s * data['std_speed'] < 1.5) & (
            data['avarage_spd'] > data['avarage_spd_10800'] - k_a_s * data['std_spd_avg']):
        return 1
    else:
        return 0


def ab_direct(row):
    # make the links undirected
    if row['VIAUNITC'] > row['TOUNITC']:
        return row['TOUNITC'] + row['VIAUNITC']
    else:
        return row['via_to']


if __name__ == '__main__':
    # Parameters to run
    # Control whether start the process from the start
    #  Control whether start the process from the start and on which file/day to perform on in case of separate files
    # ran_all_def =['no'], ['all'], ['date_1',' date_2' ]
    name_eva = 'loop_2'
    ran_all_def = ['no']
    proceess_gps = ['no']
    # don't run o n
    filter_list = [
        'TA37TA38',
        'TA2TA257',
        'TA55TA76',
        'TA262TA78',
        'TA122TA251',
        'TA105TA83',
        'TA23TA26',
        'TA183TA81',
        'TA18TA8'
    ]
    # filter_list = [
    #     'TA2TA257',
    #     'TA262TA78',
    #     'TA146TA184',
    #     'TA122TA251',
    #     'TA105TA83',
    #     'TA18TA8',
    #     'TA55TA76',
    #     'TA23TA26',
    #     'TA183TA81'
    # ]
    # filter_list = [
    #     'fgdads'
    # ]
    # Control whether net length should be calculated or not: the options are :
    # 'papulate_frame'- True means aggregate relevant files from the disk for neto length calculation except 'papulate_frame'[1]
    # 'preprocessing' - calculation for specific mac in specific date
    start_to_run = {'papulate_frame': (False, ['0.0.0']), 'preprocessing': False, 'net length': False,
                    'calculate speed': False,
                    'calculate_avg speed': False,
                    'compare results': [True],  # 'filter_list'
                    'final_sorting': False,
                    'avg_test':True}
    # Control whether to model car records or not
    # 'final_sorting': r'D:\Users\Technion\Sagi Dalyot - AchituvCohen\Jacques data\new\ground_trouth\csv_files\general\neto_length_all_file'}

    car = True
    # penetration rate parameter
    pen_rate = 4.545
    options = ['as_separete_files', 'neto_length_all_file', 'median', '100_meters_restriction']
    type_analysis = options[1]
    print(type_analysis)
    # To Analysis ground_truth data two file are required : BT file of specific date ( from Jack system ) and GPS
    # trajectories on the same date by specific
    workspace_path = os.path.split(os.path.split(__file__)[0])[0]
    dates_data = os.path.join(workspace_path, r'csv_files')
    # This lists  stores the final files of the each measure files
    frame = list()
    frame_car = list()
    frame_ped = list()

    # calculate net length
    # Sort each file separately

    if start_to_run['preprocessing']:
        for date_data in os.listdir(dates_data):
            # To run the code for specific data ( in case or new  coming data)
            if ran_all_def[0] != 'no' and ran_all_def[0] != 'all':
                if date_data not in ran_all_def:
                    continue
            elif date_data == 'general' or date_data == 'raw_data' or date_data == 'missing_bt' or date_data == 'test':
                continue
            date_data_path = os.path.join(dates_data, date_data)
            bt_file_path = os.path.join(date_data_path, date_data + '.csv')
            gps_csv_files = os.path.join(date_data_path, 'gps')

            for gps_csv_file in os.listdir(gps_csv_files):
                gps_file_path = os.path.join(gps_csv_files, gps_csv_file)
                if type_analysis in options:
                    my_object = SortByGroundTruth(date_data, date_data_path, workspace_path, bt_file_path,
                                                  gps_file_path,
                                                  proceess_gps[0])
                if type_analysis != 'as_separete_files':
                    frame.append(my_object.calc_neto_length())
                else:
                    if my_object.user == 'ped':
                        frame_ped.append(
                            my_object.calc_avg_speed(my_object.workspace_csv_progress))
                    else:
                        frame_car.append(
                            my_object.calc_avg_speed(my_object.workspace_csv_progress))

        # Aggregate the data for  cars and peds and Calc statistic

        # Aggregate the data to calculate the average neto length
    # in case when much data is already ready
    if start_to_run['papulate_frame'][0]:
        SortByGroundTruth.papulate_frames(frame=frame, folder=dates_data,
                                          file_not_papulte=start_to_run['papulate_frame'][1])
    if type_analysis in options:
        path_to_save = os.path.join(workspace_path, r'csv_files/general', type_analysis)
        if start_to_run['net length']:
            SortByGroundTruth.join_preprocess_to_one_file(frame, os.path.join(path_to_save, 'neto_length.csv'))
            SortByGroundTruth.calc_stats(path_to_save, 'general', type_analysis)
            SortByGroundTruth.use_same_length_for_each_dir(path_to_save)
        if start_to_run['calculate speed']:
            for date_data in os.listdir(dates_data):
                if date_data == 'general' or date_data == 'raw_data' or date_data == 'missing_bt' or date_data == 'test':
                    continue
                date_data_path = os.path.join(dates_data, date_data)
                bt_file_path = os.path.join(date_data_path, date_data + '.csv')
                gps_csv_files = os.path.join(date_data_path, 'gps')

                for gps_csv_file in os.listdir(gps_csv_files):
                    gps_file_path = os.path.join(gps_csv_files, gps_csv_file)
                    my_object = SortByGroundTruth(date_data, date_data_path, workspace_path, bt_file_path,
                                                  gps_file_path,
                                                  'no')
                    join_df = pd.read_csv(
                        os.path.join(my_object.workspace_csv_progress, 'join' + my_object.date + '.csv'))
                    if start_to_run['calculate speed']:
                        my_object.calculate_speed(join_df, True, path_to_save)
                    if my_object.user == 'ped':
                        frame_ped.append(
                            my_object.calc_avg_speed(path_to_save, True))
                    else:
                        frame_car.append(
                            my_object.calc_avg_speed(path_to_save, True))
            # Aggregate the data for  cars and peds and Calc statistic
            path_ped = os.path.join(path_to_save, '_ped.csv')
            SortByGroundTruth.join_preprocess_to_one_file(frame_ped, path_ped)
            path_car = os.path.join(path_to_save, '_car.csv')
            SortByGroundTruth.join_preprocess_to_one_file(frame_car, path_car)

    # compare result ( calculate metrics for different k_s )
    if start_to_run['compare results'][0]:
        path_to_save = os.path.join(workspace_path, r'csv_files/general', type_analysis)
        path_ped = os.path.join(path_to_save, '_ped.csv')
        path_car = os.path.join(path_to_save, '_car.csv')
        # calculate metrics for different k_s
        list_matrices = []
        # Cross validation

        # Devide to training and test

        data_ped = pd.read_csv(path_ped)
        data_car = pd.read_csv(path_car)
        print(data_ped.shape[0])
        print(data_car.shape[0])
        # sort all file in filter list
        data_ped['via_to_2'] = data_ped.apply(lambda x: ab_direct(x), axis=1)
        data_car['via_to_2'] = data_car.apply(lambda x: ab_direct(x), axis=1)
        data_ped = data_ped[~data_ped['via_to_2'].isin(filter_list)]
        data_car = data_car[~data_car['via_to_2'].isin(filter_list)]
        print(data_ped.shape[0])
        print(data_car.shape[0])
        date_ped_copy = data_ped.copy()
        train_set_ped = date_ped_copy.sample(frac=0.7)
        test_set_ped = date_ped_copy.drop(train_set_ped.index)
        # Devide to training and test

        data_car_copy = data_car.copy()
        train_set_car = data_car_copy.sample(frac=0.7)
        test_set_car = data_car_copy.drop(train_set_car.index)
        # data = pd.read_csv(path_ped)
        for k_s in range(0, 21):
            k_s = k_s / 10

            confusion_matrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
            compare_results(train_set_ped, 'ped', path_to_save, confusion_matrix, k_s)
            compare_results(train_set_car, 'car', path_to_save, confusion_matrix, k_s)

            accuracy = (confusion_matrix['TP'] + confusion_matrix['TN']) / (
                    confusion_matrix['TP'] + confusion_matrix['TN'] + confusion_matrix['FN'] + confusion_matrix['FP'])
            precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
            recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
            f1 = (2 * precision * recall) / (precision + recall)
            temp_list = [k_s, accuracy, precision, recall, f1, confusion_matrix['TP'],
                         confusion_matrix['FP'], confusion_matrix['FN'], confusion_matrix['TN']]
            print(temp_list)
            list_matrices.append(temp_list)
        df = pd.DataFrame(list_matrices,
                          columns=['K for speed', 'accuracy', 'precision', 'recall', 'f1', 'TP', 'FP', 'FN', 'TN'])
        df.to_csv('matrices_' + name_eva +'.csv')

        # run on test
        k_s = float(input("select k : "))

        confusion_matrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        compare_results(test_set_ped, 'ped', path_to_save, confusion_matrix, k_s)
        compare_results(test_set_car, 'car', path_to_save, confusion_matrix, k_s)

        accuracy = (confusion_matrix['TP'] + confusion_matrix['TN']) / (
                confusion_matrix['TP'] + confusion_matrix['TN'] + confusion_matrix['FN'] + confusion_matrix['FP'])
        precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
        recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
        f1 = (2 * precision * recall) / (precision + recall)
        temp_list = [k_s, accuracy, precision, recall, f1, confusion_matrix['TP'],
                     confusion_matrix['FP'], confusion_matrix['FN'], confusion_matrix['TN']]
        print(temp_list)
    if start_to_run['final_sorting'] != False:
        final_sorting(start_to_run['final_sorting'], [k_s, 1], name_eva)
        final_sorting(start_to_run['final_sorting'], [k_s, 1], name_eva)

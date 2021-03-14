import pandas as pd
import os
import numpy as np
import arcpy
import pandas as pd
import math
import shutil


class NetoLength():
    def __init__(self, data_date, workspace_main_progress, workspace, bt_file, gps_file):
        self.gps_file = gps_file
        meta_data = str.split(os.path.basename(self.gps_file), '_')
        if meta_data[len(meta_data) - 1] == 'car':
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
        self.time_for_avg = 900

        print('The program run on {} ,{} on {}'.format(self.surveyor, self.mac[0], self.date))
        self.name_gis = str.replace(self.obj_name, '.', '_')
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

        self.calc_neto_length()
        print('finish to execute_calc_neto_length')

        self.calc_stats()
        print('finish to execute_calc_stats')

        self.use_same_length_for_each_dir()
        print('finish to execute_use_same_length_for_each_dir')
        join_df = pd.read_csv(os.path.join(self.workspace_csv_progress, 'join' + self.date + '.csv'))

        self.calculate_speed(join_df)
        print('finish to execute_calculate_speed')

        self.calc_avg_speed()
        print('finish to execute_calculate_speed')

        self.compare_results()
        print('finish to execute_compare_results')


    def create_new_gdb(self):
        # Execute CreateFileGDB
        arcpy.CreateFileGDB_management(self.workspace_gis_progress, self.obj_name)


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
                                   match_option='CLOSEST', search_radius=30, distance_field_name='distance')
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
        gps_pnt_less_2 = gps_pnts.loc[gps_pnts['speed_1'] < 2]

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


class SortByGroundTruth:
    def __init__(self, analysis_name):

        self.workspace_ground_truth = os.path.split(os.path.split(__file__)[0])[0]
        self.workspace = os.path.join(self.workspace_ground_truth, 'csv_files/analysis_folder', analysis_name)
        self.bt_file_path = ''
        self.gps_file_path = ''

    def delete_folder(self):
        # Delete folder to run same analysis again
        shutil.rmtree(self.workspace)

    def separate_files(self):
        # Sort each file separately
        os.makedirs(os.path.join(self.workspace, 'separate_files'))
        dates_data = os.path.join(self.workspace_ground_truth, r'csv_files')
        for date_data in os.listdir(dates_data):
            if date_data == 'general' or date_data == 'raw_data' or date_data == 'missing_bt' or date_data == 'test':
                continue
            date_data_path = os.path.join(dates_data, date_data)
            self.bt_file_path = os.path.join(date_data_path, date_data + '.csv')
            gps_csv_files = os.path.join(date_data_path, 'gps')
            NetoLength()
            for gps_csv_file in os.listdir(gps_csv_files):
                self.gps_file_path = os.path.join(gps_csv_files, gps_csv_file)

        os.makedirs(os.path.join(self.workspace, 'separate_files'))
        print('finish to create_new_gdb')
        self.create_new_gdb()
        print('finish to create_new_gdb')

    # def __init__(self, data_date, workspace_main_progress, workspace, bt_file, gps_file):
    #     self.gps_file = gps_file
    #     meta_data = str.split(os.path.basename(self.gps_file), '_')
    #     if meta_data[len(meta_data) - 1] == 'car':
    #         self.user = 'car'
    #     else:
    #         self.user = 'ped'
    #     self.surveyor = meta_data[0]
    #     self.mac = []
    #     self.mac.append(meta_data[1])
    #     self.short_name = '_'.join([self.surveyor, self.mac[0]])
    #     self.date = data_date.replace('.csv', '')
    #     self.obj_name = '_'.join([self.short_name, self.date])
    #     self.workspace_path = workspace
    #     self.workspace_gis_progress = workspace_main_progress
    #     self.network_links_gis = os.path.join(self.workspace_path, r'general.gdb/links_network')
    #     self.workspace_csv_progress = os.path.join(workspace_main_progress, 'progress_files', self.short_name)
    #     self.bt_file_path = bt_file
    #     self.time_for_avg = 900
    #
    #     print('The program run on {} ,{} on {}'.format(self.surveyor, self.mac[0], self.date))
    #     self.name_gis = str.replace(self.obj_name, '.', '_')

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

    self.calc_neto_length()
    print('finish to execute_calc_neto_length')

    self.calc_stats()
    print('finish to execute_calc_stats')

    self.use_same_length_for_each_dir()
    print('finish to execute_use_same_length_for_each_dir')
    join_df = pd.read_csv(os.path.join(self.workspace_csv_progress, 'join' + self.date + '.csv'))

    self.calculate_speed(join_df)
    print('finish to execute_calculate_speed')

    self.calc_avg_speed()
    print('finish to execute_calculate_speed')

    self.compare_results()
    print('finish to execute_compare_results')

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
                                   match_option='CLOSEST', search_radius=30, distance_field_name='distance')
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
        gps_pnt_less_2 = gps_pnts.loc[gps_pnts['speed_1'] < 2]

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

    @staticmethod
    def join_preprocess_to_one_file(frames, path):
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(os.path.join(path, 'neto_length.csv'))

    def calc_stats(self, all=False):
        '''
        :param df: database with net length for each record
        for each link calc averge and std base on neto length of each record
        :return:
        '''
        if all:
            df = pd.read_csv(os.path.join(self.workspace_csv_progress, 'neto_length.csv'))
        else:
            df = pd.read_csv(os.path.join(self.workspace_csv_progress, 'neto_length_' + self.obj_name + '.csv'))
        gk0 = df.groupby(['via_to'])['neto_length'].mean()
        gk1 = df.groupby(['via_to'])['neto_length'].std()
        gk2 = df.groupby(['via_to'])['neto_length'].count()
        gk = pd.concat([gk0, gk1, gk2], axis=1)
        gk.to_csv(os.path.join(self.workspace_csv_progress, 'mean_std.csv'), header=['mean', 'std', 'count'])

    def use_same_length_for_each_dir(self):
        """
        df file include data about average neto length for links only as one direction, so the method copy the length and
        sdt to the opposite direction
        """
        df = pd.read_csv(os.path.join(self.workspace_csv_progress, 'mean_std.csv'))
        df = df[pd.notna(df['mean'])]
        temp_list = []
        for index, row in df.iterrows():
            split_row = row['via_to'].split('T')
            temp_row = ['T' + split_row[2] + 'T' + split_row[1], row['mean'], row['std'], row['count']]
            temp_list.append(temp_row)
        reverse_df = pd.DataFrame(temp_list, columns=['via_to', 'mean', 'std', 'count'])
        new_df = df.append(reverse_df).reset_index()
        new_df.to_csv(os.path.join(self.workspace_csv_progress, 'mean_std_bidirectional.csv'))

    def calculate_speed(self, join_df, all=False):
        if all:
            mean_std_bidirectional_df = pd.read_csv(os.path.join(self.workspace_path,
                                                                 r'csv_files\general\mean_std_bidirectional.csv'))
        else:
            # Store only records with matching links as appear  in mean_std_bidirectional_df
            mean_std_bidirectional_df = pd.read_csv(
                os.path.join(self.workspace_csv_progress, 'mean_std_bidirectional.csv'))
        rel_links = join_df.loc[(join_df['via_to'].isin(mean_std_bidirectional_df['via_to']))]
        rel_links.to_csv(os.path.join(self.workspace_csv_progress, 'rel_links.csv'))

        # Build dictionary of links and length
        my_dict = {row[2]: row[3] for row in mean_std_bidirectional_df.values}
        for index, row in rel_links.iterrows():
            if row['CLOSETS'] - row['LASTDISCOTS'] > 0:
                rel_links.at[index, 'speed'] = my_dict[row['via_to']] / (row['CLOSETS'] - row['LASTDISCOTS'])
            else:
                rel_links.at[index, 'speed'] = -1000
        # drop records without speed
        rel_links = rel_links.drop(rel_links[rel_links['speed'] == -1000].index)
        rel_links.to_csv(os.path.join(self.workspace_csv_progress, 'speed.csv'))
        return rel_links

    def calc_avg_speed(self):
        # Calc our trip time, speed and average speed
        filtered_db = pd.read_csv(os.path.join(self.workspace_csv_progress, 'speed.csv'))
        filtered_db['avarage_spd'] = ''
        filtered_db['std_spd'] = ''
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
                record_LASTDISCOTS = record['LASTDISCOTS']
                result = group.loc[(record_LASTDISCOTS - group['LASTDISCOTS'] < self.time_for_avg) & (
                        record_LASTDISCOTS - group['LASTDISCOTS'] >= 0)]
                if result.shape[0] > 1:
                    filtered_db['avarage_spd'].loc[filtered_db['PK_UID'] == pk_id] = result.speed.mean()
                    filtered_db['std_spd'].loc[filtered_db['PK_UID'] == pk_id] = result.speed.std()
                    filtered_db['num_of_recs'].loc[filtered_db['PK_UID'] == pk_id] = result.shape[0]
                else:
                    filtered_db['avarage_spd'].loc[filtered_db['PK_UID'] == pk_id] = record['speed']

        filtered_db = filtered_db.loc[filtered_db['avarage_spd'] != '']
        filtered_db.to_csv(os.path.join(self.workspace_csv_progress, 'avarage_spd' + str(self.time_for_avg) + '.csv'))

    # def compare_results(self):
    #     data = pd.read_csv(os.path.join(self.workspace_csv_progress, 'avarage_spd' + str(self.time_for_avg) + '.csv'))
    #     if self.user == 'ped':
    #         new_data = data.drop(data[data['speed'] > 1.5].index)
    #         size = new_data.shape[0]
    #         stat_dic = {
    #             'number of records': size,
    #             'jack': (new_data.drop(new_data[new_data['ROWSTATUS'] != 'Out of range'].index)).shape[0] / size * 100,
    #             'ours': (new_data.drop(new_data[new_data['avarage_spd'] < 1.5].index)).shape[0] / size * 100}
    #     else:
    #         size = new_data.shape[0]
    #         stat_dic = {
    #             'number of records': size,
    #             'jack': (new_data.drop(new_data[new_data['ROWSTATUS'] != 'Valid'].index)).shape[0] / size * 100,
    #             'ours':
    #                 (new_data.drop(new_data[new_data['speed'] < 1.5 and new_data['avarage_spd'] > 1.5].index)).shape[
    #                     0] / size * 100}
    #     print(stat_dic)
    #     save_dic = os.path.join(self.workspace_csv_progress, 'stat_file.txt')
    #     with open(save_dic, 'w') as f:
    #         print(stat_dic, file=f)

# if __name__ == '__main__':
#     my_object = SortByGroundTruth()
#     my_
#
#     # To Analysis ground_truth data two file are required : BT file of specific date ( from Jack system ) and GPS
#     # trajectories on the same date by specific
#     workspace_path = os.path.split(os.path.split(__file__)[0])[0]
#
#     # Sort each file separately
#     dates_data = os.path.join(workspace_path, r'csv_files')
#     for date_data in os.listdir(dates_data):
#         if date_data == 'general' or date_data == 'raw_data' or date_data == 'missing_bt' or date_data == 'test':
#             continue
#         date_data_path = os.path.join(dates_data, date_data)
#         bt_file_path = os.path.join(date_data_path, date_data + '.csv')
#         gps_csv_files = os.path.join(date_data_path, 'gps')
#
#         for gps_csv_file in os.listdir(gps_csv_files):
#             gps_file_path = os.path.join(gps_csv_files, gps_csv_file)
#             my_object = SortByGroundTruth(date_data, date_data_path, workspace_path, bt_file_path, gps_file_path)
#             join_df = pd.read_csv(os.path.join(my_object.workspace_csv_progress, 'join' + my_object.date + '.csv'))
#             my_object.calculate_speed(join_df, True)
#             my_object.calc_avg_speed()
#             my_object.compare_results()
#
#     # dates_data = os.path.join(workspace_path, r'csv_files/general/inputs')
#     # path_to_save = os.path.join(workspace_path, r'csv_files/general')
#     # frame = list()
#     # for date_data in os.listdir(dates_data):
#     #     date_data_path = os.path.join(dates_data, date_data)
#     #     df = pd.read_csv(date_data_path)
#     #     frame.append(df)
#     # SortByGroundTruth.join_preprocess_to_one_file(frame, path_to_save)
#     # my_object.workspace_csv_progress = path_to_save
#     # my_object.calc_stats(True)
#     # my_object.use_same_length_for_each_dir()
#
#     #
#     # my_object.workspace_csv_progress = os.path.join(workspace_path, r'csv_files/general/progress_files')
#     # my_object.mac = ['D938AA04677', '5DDA77510E26']
#     # my_object.calc_stats()
#     # my_object.use_same_length_for_each_dir()
#     # dates_data = os.path.join(workspace_path, r'csv_files/general/join')
#     # speed_all_day = list()
#     # for date_data in os.listdir(dates_data):
#     #     date_data_path = os.path.join(dates_data, date_data)
#     #     df = pd.read_csv(date_data_path)
#     #     speed_all_day.append(my_object.calculate_speed(df))
#     # result = pd.concat(speed_all_day, ignore_index=True)
#     # result.to_csv(os.path.join(my_object.workspace_csv_progress, 'speed.csv'))
#     # my_object.calc_avg_speed()
#     # my_object.compare_results()
#
# # # Calculate speed of GPS file
# # workspace_path = os.path.split(os.path.split(__file__)[0])[0]
# # gps_file = os.path.join(workspace_path, r'GPS_data/av_D938AA04677C_21_11_19.csv')
# # workspace_gis = os.path.join(workspace_path, r'gis_files.gdb')
# # arcpy.env.workspace = workspace_gis
# # # add_itm_cor_to_csv_file(workspace_gis, 'av_D938AA04677C_21_11_19')
# # # calc_time_stamp(os.path.join(workspace_path, r'GPS_data/av_D938AA04677C_21_11_19_itm.csv'))
# # # remove_points_near_intersection('links_network', os.path.join(workspace_path, r'GPS_data/stamp_time.csv'),
# # #                                 'all_gps_pnts')
# # # spatial_join()
# # # join_data(workspace_path)
# # calc_neto_length('D938AA04677C')
# # # for_test_only = r'C:\Users\achic\Technion\Sagi Dalyot - AchituvCohen\Jacques ' \
# # #                 r'data\new\python_files\eval_files\neto_length - Copy.csv'
# # # join_preprocess_to_one_file(r'eval_files/neto_length.csv', for_test_only)
# #
# # df = pd.read_csv('eval_files/neto_length.csv')
# # calc_stats(df)
# #
# # df = pd.read_csv('eval_files/mean_std.csv')
# # use_same_length_for_each_dir(df)
# #
# # join = pd.read_csv('join.csv')
# # mean_std_bidirectional = pd.read_csv('eval_files/mean_std_bidirectional.csv')
# # calculate_speed(join, mean_std_bidirectional)
# # calc_avg_speed(['D938AA04677C'], pd.read_csv('eval_files/speed.csv'), 300)
#
# #     # Stat data
# #     name = list()
# #     after_join = list()
# #     filter_not_valid = list()
# #     filter_not_valid_percentage = list()
# #     filter_high_speed = list()
# #     filter_high_speed_percentage = list()
# #     ped = list()
# #     ped_percentage = list()
# #     not_sure = list()
# #     not_sure_percentage = list()
# #
# #     # join = pd.read_csv('eval_files/join.csv')
# #     # mean_std_bidirectional = pd.read_csv('eval_files/mean_std_bidirectional.csv')
# #     # calculate_speed(join, mean_std_bidirectional)
# #     for time in [240,300]:
# #         calc_avg_speed(['5DDA77510E26'], pd.read_csv('eval_files/speed.csv'),time)
# #     # data_with_avg = pd.read_csv('eval_files/avarage_spd.csv')
# #     # my_data = data_with_avg.loc[(data_with_avg['MAC'].isin(['5DDA77510E26']))]
# #     # source_size = my_data.shape[0]
# #     # no_high_spd = filter_high_speed_def(my_data)
# #     #
# #     # select_peds_not_sure(no_high_spd)
# #
# #     # join_data(workspace_path)
# #     # df = pd.read_csv('eval_files/neto_length.csv')
# #     # calc_stats(df)
# #     # df = pd.read_csv('eval_files/mean_std.csv')
# #     # use_same_length_for_each_dir(df)
# #
# # #
# # # stat = pd.DataFrame({'name': name,
# # #                      'after_join': after_join,
# # #                      'filter_not_valid': 0,
# # #                      'filter_not_valid_percentage': 0,
# # #                      'filter_high_speed': filter_high_speed,
# # #                      'filter_high_speed_percentage': filter_high_speed_percentage,
# # #                      'ped': ped,
# # #                      'ped_percentage': ped_percentage,
# # #                      'not_sure': not_sure,
# # #                      'not_sure_percentage': not_sure_percentage})
# # #
# # # path = os.path.join(data_table_path0, 'stat.csv')
# # # stat.to_csv(path)

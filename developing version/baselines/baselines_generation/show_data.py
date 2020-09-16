import sys
import configparser
import pandas as pd
import glob
import utils.plot_utils as plot_utils
import numpy as np

if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
        
    # source_folder = './data/gen_data/baseline2_1days_folder/'
    source_folder = './data/gen_data/5times/'
    # source_folder = './data/raw_data/'
    
    if '-flow' in sys.argv or '-all' in sys.argv:
        print('reach show raw_data')

        all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)

        x_data = [source_folder]
        y_data = [[]]
        for t_hour in range(24):   
            y_data_i = []
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data[0].append(len(chunk.index))
        
        plot_utils.temporal_lineplot(x_data, y_data, x_label="hour", y_label="#flow", title="#flow distribution over 100 users")

    if '-bytperflow' in sys.argv or '-all' in sys.argv:
        print('reach show raw_data')
        # ll = []
        # for i in range(1,6):
        #     ll += [pd.read_csv(f) for f in glob.glob(source_folder+'5times%d_baseline1_1days_folder/*.csv'%i)]
        # print("lenlen", len(ll))
        # all_record = pd.concat(ll, ignore_index = True)
        
        all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'5times%d_baseline1_1days_folder/*.csv'%5)], ignore_index = True)
        # all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)

        x_data = [source_folder]
        y_data = [[]]
        for t_hour in range(24):   
            y_data_i = []
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data[0].append(sum(np.log(chunk['byt'].values.tolist()))/len(chunk.index))
        
        plot_utils.temporal_lineplot(x_data, y_data, x_label="hour", y_label="#byt", title="average #byt per flow distribution over 100 users")#, lim=8.0)

    if '-byttotal' in sys.argv or '-all' in sys.argv:
        print('reach show raw_data')

        all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)

        x_data = [source_folder]
        y_data = [[]]
        for t_hour in range(24):   
            y_data_i = []
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data[0].append(sum(np.log(chunk['byt'].values.tolist())))
        
        plot_utils.temporal_lineplot(x_data, y_data, x_label="hour", y_label="#byt", title="#byt sum distribution over 100 users")

    if '-bytdist' in sys.argv or '-all' in sys.argv:
        all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)

        x_data = []
        y_data = []

        for t_hour in range(24):   
            x_data.append(str(t_hour))
            y_data_i = []
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data_i = np.log(chunk['byt'].values.tolist())
            # y_data_i = chunk['byt'].values.tolist()
            print(t_hour, len(y_data_i), y_data_i)
            y_data.append(y_data_i)
        
        # plot_utils.temporal_lineplot(x_data, y_data)
        plot_utils.boxplot(x_data, y_data, x_label='hour', y_label='byt (log)', title='hour distribution of all '+source_folder)

    

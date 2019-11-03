import pandas as pd
import logging
import os

def main():


    logging.basicConfig(level=logging.DEBUG)

    logging.info("Loading data....\n")
    repo_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = ("data")
    data_dir_path = os.path.join(repo_path, data_dir)
    
    csv_list = []

    
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            logging.debug(os.path.join(data_dir_path, file))
            csv_list.append(os.path.join(data_dir_path, file))


    data = pd.DataFrame()
    logging.info("Creating Dateframe from CSV files ...\n")
    for csv_file in csv_list:
        
        if  csv_file[-5:]=="1.csv":
            temp_data = pd.read_csv(csv_file, delimiter=r"\s+",low_memory=False)
        else:
            temp_data = pd.read_csv(csv_file, sep=",",low_memory=False)
        
        logging.info("\nCSV file:{}\nSize = {}\nShape ={}\nShape[0] x Shape[1] = {}".format(csv_file,temp_data.size, temp_data.shape, temp_data.shape[0]*temp_data.shape[1]))
        data = data.append(temp_data, ignore_index = True, sort=True)
    
        


    # data = data.append(pd.read_csv(csv_list[1], sep=",",low_memory=False), ignore_index = True, sort=True) 
    logging.info("\n\n-------Final dataframe-------:\nSize = {}\nShape ={}\nShape[0] x Shape[1] = {}".format(csv_file,data.size, data.shape, data.shape[0]*data.shape[1]))        
    print(data.head())
    
    # data = pd.read_csv(csv_list[0],sep=' ')
    # data = pd.read_csv(csv_list[0], delimiter=r"\s+")    
    # Preview the first 5 lines of the loaded data 

    # print(data.head())

if __name__ == '__main__':
    main()

##### 使用するライブラリ・モジュールのインポート

import os
import csv
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from sklearn.metrics import r2_score
import pandas as pd
import math
import datetime

# 実行時間取得
updated_date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M')
print("updated_date: " + updated_date)
print(type(updated_date))


##### データフォルダ・データシート作成（データフォルダが存在しない場合のみ実行）

current_directory = os.getcwd().replace(os.sep, "/")
print("current_directory: " + current_directory)

data_path = current_directory + "/Data"
print("data_path: " + data_path)

datasheet_all_path = data_path + "/datasheet_all" + ".csv"
print("datasheet_all_path: " + datasheet_all_path)

cols_for_all = ["Data No", "Gonio", "Echo", "Date", "Subject", "Pattern", "Trial Num", "Depth", "RMSE", "R2", "Corrcoef", "Updated Date"]

# データフォルダが存在しなければ作成
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print("Data folder was successfully created.")
else:
    print("Data folder already exists.")

# datasheet_allが存在しなければ作成
if not os.path.exists(datasheet_all_path):
    df_for_all = pd.DataFrame(columns=cols_for_all)
    df_for_all.set_index("Data No", inplace=True)
    # datasheet_allへの新規書き込み
    df_for_all.to_csv(datasheet_all_path, encoding="shift_jis")
    print("datasheet_all was successfully created.")
    
else:
    print("Datasheet already exists.")


##### datasheet_all.csv 読み込み

# インデックスを"Data No"としたデータフレームとして読み込み
datasheet_all_df = pd.read_csv(datasheet_all_path, header=0, index_col=["Data No"], encoding='shift_jis')
datasheet_all_df = datasheet_all_df.dropna(how='all', axis=0)
print("-------------  datasheet_all_df  ---------------------")
print(datasheet_all_df)
print("------------------------------------------------------")

data_No_list = datasheet_all_df.index.tolist()
if datasheet_all_df.empty:
    data_last_No = 0
else:
    data_last_No = data_No_list[-1]
print("data_No_list: " + str(data_No_list))
print("data_last_No: " + str(data_last_No))


##### 書き込み
#### DATA_INFO(date, subject, pattern) を入力
####  Data_INFO から該当するデータを読み込み， data_No, Data_PATH(Gonio, Echo), trial_num を決定し Data_INFOとともに datasheet_all に書き込み

print("書き込みモードです．")
print("DATA_INFOを入力してdatasheet_all.csvに書き込むデータを指定してください．")

## DATA_INFO
# 実験日
date = input("Date: ")

# 被験者
subject = input("Subject: ")

# 実験パターン
pattern = input("Pattern: ")

# 深度
depth = input("Depth: ")


info_str = date + "_" + subject + "_pattern" + pattern
data_per_info_path = data_path + "/" + info_str
print("DATA_INFO: " + info_str)

# data_per_info フォルダが存在しなければ作成
if not os.path.exists(data_per_info_path):
    make_new = input("該当するフォルダがありません．新規作成しますか？(yes/no)")
    if make_new == "yes":
        os.makedirs(data_per_info_path)
        print("\nフォルダを新規作成しました．\n\n")
        print("このフォルダに実験データを入れてからもう一度実行してください．")
        print("※それぞれのフォルダ名は'GonioData'，'EchoData'とすること．")
        raise Exception("Data doesn't exist.")
    else:
        raise Exception("Try again.")


## data_per_info フォルダが存在していれば，以下の処理をする

# データがなければエラーを表示
gonio_data_path = data_per_info_path + "/GonioData"
echo_data_path = data_per_info_path + "/EchoData"
if not os.path.exists(gonio_data_path):
    print("ゴニオデータがありません")
    raise Exception("GonioData doesn't exist.")
if not os.path.exists(echo_data_path):
    print("エコーデータがありません")
    raise Exception("EchoData doesn't exist.")

# ゴニオデータファイルをリストで取得
gonio_files = list()
for file in os.listdir(gonio_data_path):
    if file.endswith(".csv"):
        gonio_files.append(file)

# エコーデータファイルをリストで取得
echo_files = list()
for file in os.listdir(echo_data_path):
    if file.endswith(".mp4"):
        echo_files.append(file)

print("gonio_files: " + str(gonio_files))
print("echo_files: " + str(echo_files))

# 試行数
num_of_trial = len(gonio_files)
print("num_of_trial: " + str(num_of_trial))

# 書き込み用のデータフレームを用意
new_df_for_write = pd.DataFrame(columns=cols_for_all)
new_df_for_write.set_index("Data No", inplace=True)

# 各試行ごとのデータ取得
for trial in range(num_of_trial):
    # 試行番号
    trial_num = trial + 1
    
    # データ番号
    data_No = data_last_No + trial_num
    
    ## DATA_PATH
    # ゴニオファイル名
    gonio_file = gonio_files[trial]

    # エコーファイル名
    echo_file = echo_files[trial]
    
    # print("trial_num: " + str(trial_num))
    # print("data_No: " + str(data_No))
    # print("gonio_file: " + str(gonio_file))
    # print("echo_file: " + str(echo_file))
    
    # 各試行ごとのデータフレームを作成
    new_data = [(data_No, gonio_file, echo_file, date, subject, pattern, trial_num, depth, "", "", "", updated_date)]
    new_df_element = pd.DataFrame(data=new_data, columns=cols_for_all)
    new_df_element.set_index("Data No", inplace=True)
    # print("-------------  new_df_element  ---------------------")
    # print(new_df_element)
    # print("----------------------------------------------------")
    
    # 書き込み用データフレームに追加
    new_df_for_write = new_df_for_write.append(new_df_element)

print("-------------  new_df_for_write  ---------------------")
print(new_df_for_write)
print("------------------------------------------------------")

# datasheet_all.csvに書き込み
datasheet_all_df = datasheet_all_df.append(new_df_for_write)
datasheet_all_df.to_csv(datasheet_all_path, encoding="shift_jis")

print("datasheet_all.csvに新しくデータを書き込みました．")
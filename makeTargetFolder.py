import os
import csv

##########################   Parameter   ###########################
# 被験者
subject = "Katagi"
# 実験日
date = "20221024"
# 補足
additional_info = "modified"


# カレントディレクトリ
current_directory = "C:/Users/katagi/Desktop/Research/UltrasoundImaging"
#####################################################################

path = os.getcwd().replace(os.sep, "/")
# pathが正しくカレントディレクトリになっているかを確認
if (path != current_directory):
  exit()

data_path = path + "/Data"
# DataフォルダがなければDataフォルダとデータシート作成（最初の1回のみ）
if not os.path.exists(data_path):
  os.makedirs(data_path)
  with open(data_path + "/datasheet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["INFO", "", "", "", "", "", "RESULTS"])
    writer.writerow(["Date", "Subject", "Info", "Patern", "Trial_Num", "", "RMSE", "R2", "Corrcoef"])


new_experiment = data_path + "/" + subject + "_" + date  + "_" + additional_info
os.makedirs(new_experiment)
gonio_path = new_experiment + "/Goniometer"
echo_path = new_experiment + "/Echo"
results_path = new_experiment + "/Results"
os.makedirs(gonio_path)
os.makedirs(echo_path)
os.makedirs(results_path)
echo_original = echo_path + "/Original"
echo_preprocessed = echo_path + "/Preprocessed"
echo_OpticalFlow = echo_path + "/OpticalFlow"
os.makedirs(echo_original)
os.makedirs(echo_preprocessed)
os.makedirs(echo_OpticalFlow)

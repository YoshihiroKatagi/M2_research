import os
import numpy as np
import matplotlib.pyplot as plt

##########################   Parameter   ###########################
# 被験者
subject = "Katagi"
# 実験日
date = "20221024"
# 補足
additional_info = "modified"
# 実験パターン
patern = 1

# カレントディレクトリ
current_directry = "C:/Users/katagi/Desktop/Research/UltrasoundImaging"
#####################################################################

##########################   Folder & File   ########################
path = os.getcwd().replace(os.sep, "/")
# pathが正しくカレントディレクトリになっているかを確認
if (path != current_directry):
  exit()

data_path = path + "/Data/" + subject + "_" + date + "_" + additional_info
# 特徴点データ取得
feature_points_data_path = data_path + "/Echo/FeaturePointsData.npy"
feature_points_data = np.load(feature_points_data_path)
# ゴニオデータ取得
gonio_data_path = data_path + "/Goniometer/Processed/GonioData.npy"
gonio_data = np.load(gonio_data_path)

# データ形状取得
data_shape = feature_points_data.shape

# 結果保存用パス
results_path = data_path + "/Results/AngleVSFeaturePoints"
if not os.path.exists(results_path):
  os.makedirs(results_path)

#####################################################################

###########################  Visualize  #############################
def Visualize(x, y, save_path, point_num):
  
  fig = plt.figure()
  # plt.title("Wrist angle VS. Feature point", fontsize=18)
  plt.xlabel("Pixel [px]", fontsize=20)
  plt.ylabel("Wrist angle [deg]", fontsize=20)
  plt.scatter(x, y, color="cornflowerblue", linewidth=2, label="Estimated angle")
  plt.grid(True)
  plt.tight_layout()
  fig.savefig(save_path + "/point" + str(point_num) + ".png")

  # plt.show()
  plt.close()
#####################################################################

#############################  Main  ################################
for i in range(data_shape[0]):
  trial_num = i + 1
  print("--試行" + str(trial_num) + "--")

  each_trial_path = results_path + "/trial" + str(trial_num)
  if not os.path.exists(each_trial_path):
    os.makedirs(each_trial_path)

  Feature_points, Theta = feature_points_data[i], (gonio_data[i]) # (3212, 100), (3212, 1)
  theta = np.squeeze(Theta) # (3212,)

  for j in range(Feature_points.shape[1]):
    # 特徴点10個分のみ
    if j >= 20:
      break

    point_num = j + 1
    feature_point = Feature_points[:, j] # (3212,)

    # グラフ作成
    Visualize(feature_point, theta, each_trial_path, point_num)

print("DONE")
#####################################################################
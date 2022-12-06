import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import r2_score

##########################   Parameter   ###########################
# 被験者
subject = "Katagi"
# 実験日
date = "20221024"
# date = datetime.now().strftime("%Y%m%d")
# 補足
additional_info = "modified"
# 実験パターン
patern = 1

# フレームレート
frame_rate = 73 # Depth 20
# frame_rate = 62 # Depth 30
# frame_rate = 54 # Depth 40
# frame_rate = 48 # Depth 50
# frame_rate = 43 # Depth 60

# グラフ
min_height = -80
max_height = 30

# 正則化パラメータ
# lam = 0.1
lam = 0.5

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
feature_points_data = np.load(feature_points_data_path) # (5, 1548, 100)
# ゴニオデータ取得
gonio_data_path = data_path + "/Goniometer/Processed/GonioData.npy"
gonio_data = np.load(gonio_data_path) # (5, 1548, 1)

# # 正規化で用いた平均値と標準偏差取得
# with open(data_path + "/Goniometer/Processed/ThetasMean.csv") as f:
#   reader = csv.reader(f)
#   thetas_mean = next(reader)
#   f.close()
# with open(data_path + "/Goniometer/Processed/ThetasStd.csv") as f:
#   reader = csv.reader(f)
#   thetas_std = next(reader)
#   f.close()

# データ形状取得
data_shape = feature_points_data.shape # (5, 1548, 100)

# 訓練データとテストデータの区切り位置
devide_num = data_shape[1] * 4 // 5

# 結果保存用パス
results_path = data_path + "/Results"

# 結果用のcsvファイル作成
with open(results_path + "/RMSEandR2.csv", "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["RMSE", "R2", "coeff"])

# データシート
datasheet = path + "/Data/datasheet.csv"
f = open(datasheet, "r")
datasheet_all = csv.reader(f)
datasheet_list = [e for e in datasheet_all]
#####################################################################

###################  Divide into Train and Test  ####################
def DivideIntoTrainAndTest(x, theta):# x:(1548, 100), theta: (1548, 1)
  x_train, theta_train = x[:devide_num], theta[:devide_num]
  x_test, theta_test = x, theta

  return x_train, theta_train, x_test, theta_test
#####################################################################

############################  Analysis  #############################
def Analysis(x, theta): #x: (1238, 100), theta: (1238, 1)
  x_T = x.T
  x_T_x = np.dot(x_T, x) # (100, 100)
  I = np.eye(x_T_x.shape[0])
  inv = np.linalg.pinv(x_T_x)
  # inv = np.linalg.pinv(x_T_x + lam * I)
  W =  np.dot(np.dot(inv, x_T), theta) # (100, 1)

  return W
#####################################################################

###########################  Visualize  #############################
def Visualize(y, y_pred, num):
  x = np.arange(y.shape[0]) / frame_rate # T:0 ~ 1543 → 0 ~ 36[s] に修正
  y = y.reshape(-1) # (1548,)
  y_pred = y_pred.reshape(-1) # (1548,)

  fig1 = plt.figure()
  # plt.title("A result of estimating wrist joint angle", fontsize=18)
  plt.xlabel("Time [s]", fontsize=20)
  plt.ylabel("Wrist angle [deg]", fontsize=20)
  plt.plot(x, y_pred, color="cornflowerblue", linewidth=2, label="Estimated angle")
  plt.plot(x, y, color="tomato", linewidth=2, label="Measured angle")
  line_position = devide_num / frame_rate
  plt.vlines(line_position, min_height, max_height, "gray", linestyles="dashed")
  plt.ylim(min_height, max_height) #extensor
  plt.xlim(0, x.shape[0] / frame_rate)
  plt.legend(loc="upper left", fontsize=16)
  plt.grid(True)
  plt.tight_layout()
  fig1.savefig(results_path + "/plot" + str(num) + ".png")

  fig2 = plt.figure()
  # plt.title("Scatter plots of estimated and measured angle", fontsize=16)
  plt.xlabel("Measured angle [deg]", fontsize=20)
  plt.ylabel("Estimated angle [deg]", fontsize=20)
  plt.scatter(y[devide_num:], y_pred[devide_num:])
  plt.tight_layout()
  fig2.savefig(results_path + "/scatter" + str(num) + ".png")

  # plt.show()
  plt.close()
#####################################################################

########################  Calc_RMSEandR2  ###########################
def Calc_RMSEandR2(theta, theta_pred, T, num):
  theta = theta.reshape(-1)[devide_num:] # (310, )
  theta_pred = theta_pred.reshape(-1)[devide_num:] # (310, )

  def _coef(theta, theta_pred):
    corrcoef = np.corrcoef(theta, theta_pred)
    corrcoef = corrcoef[0][1]
    return corrcoef
  
  def _RMSE(theta, theta_pred, T):
    L = np.sum((theta - theta_pred)**2)
    RMSE = np.sqrt(L/T)
    return RMSE

  RMSE = _RMSE(theta, theta_pred, T)
  R2 = r2_score(theta, theta_pred)
  coef = _coef(theta, theta_pred)
  print("RMSE = " + str(RMSE))
  print("決定係数 R2 = " + str(R2))
  print("相関係数: " + str(coef) + "\n")

  return RMSE, R2, coef
#####################################################################

#######################   Write Results   ###########################
def Write_Results(RMSE, R2, coef, num):
  # RMSEandR2.csvに結果を書き込み
  with open(results_path + "/RMSEandR2" + ".csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([RMSE, R2, coef])
    f.close()
  
  # データシートに結果を書き込み
  flag = 1
  for row in datasheet_list:
    if row[0] == date and row[1] == subject and row[2] == additional_info and row[3] == str(patern) and row[4] == str(num):
      row[6], row[7], row[8] = RMSE, R2, coef
      flag = 0
  if flag:
    datasheet_list.append([date, subject, additional_info, patern, num, "", RMSE, R2, coef])

  with open(datasheet, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(datasheet_list)
#####################################################################

#############################  Main  ################################
for i in range(data_shape[0]):
  trial_num = i + 1
  print("--試行" + str(trial_num) + "--")

  X, Theta = feature_points_data[i], gonio_data[i] # (1548, 100), (1548, 1)

  X_train, Theta_train, X_test, Theta_test = DivideIntoTrainAndTest(X, Theta) # (1238, 100), (1238, 1), (1548, 100), (1548, 1)
  W = Analysis(X_train, Theta_train) # (100, 1)

  Theta_pred = np.dot(X_test, W) # (1548, 1)

  # #正規化の逆変換
  # theta_mean = float(thetas_mean[i])
  # theta_std = float(thetas_std[i])
  # Theta_test = -1 * (Theta_test * theta_mean + theta_std)
  # Theta_pred = -1 * (Theta_pred * theta_mean + theta_std)

  # グラフ作成
  Visualize(Theta_test, Theta_pred, trial_num)

  # RMSE, R2を導出
  T = X_test.shape[0] - X_train.shape[0]
  RMSE, R2, coef = Calc_RMSEandR2(Theta_test, Theta_pred, T, trial_num)

  # csvファイルに結果を書き込み
  Write_Results(RMSE, R2, coef, trial_num)
#####################################################################
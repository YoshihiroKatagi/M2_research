# LinearRegressionForEachData Ver2
# difference:
# Ver1: Wが間違っている(179, 100) 時系列は関係ないはず
# Ver2: Wを修正(1, 100) そのためにAnalysis()を修正
# Ver2: 訓練データを含む全データでテストを行うX_test = X
# Ver2: 12パターンでループc
# Ver2: 特徴点の動作をチェックする関数作成 CheckFeature()
# Ver2: 特徴点にハイパスフィルタをかける HighpassFilter()
# Ver2: ハイパスフィルタチェック用のグラフ XYgraph()
# Ver2: 相関係数を決定係数に変更


import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import cv2
from scipy import signal
from sklearn.metrics import r2_score

#####################  Path and Parameter  ###########################
# 使用するUSデータの形状
default_US_shape = (898, 100)

# 該当フォルダの日付（計測日と異なる場合は手入力）
target_date = "2022-05-30"
# target_date = datetime.now().strftime("%Y-%m-%d")

target_path = "C:/Users/katagi/Desktop/Research/UltrasoundImaging/dataset/" + target_date + "/forMachineLearning/"

# 試行回数
trial_num = 10

# 検証するパターン数（12パターン × 10試行 = 120個）
patern_num = 12
#####################################################################

#######################  Make Results Path  #########################
# 結果保存用パス
def MakeResultsPath(patern):
  result_folder = "dataset/" + target_date + "/results"
  result_path = result_folder + "/patern" + str(patern)

  if not os.path.exists(result_path):
    os.makedirs(result_path)

  with open(result_path + "/RMSEandR2_P" + str(patern) + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["RMSE", "R2", "coeff"])
  
  return result_path
#####################################################################

###########################  Read Data  #############################
def ReadData(): # 10試行分（どのパターンかは上のパラメータで指定する）
  feature_points_data_path = target_path + "FeaturePointsData.npy"
  feature_points_data = np.load(feature_points_data_path)
  feature_points_data = feature_points_data[10*(patern - 1): 10*patern, :, :100]
  # print(feature_points_data.shape)

  gonio_data_path = target_path + "gonioData.npy"
  gonio_data = np.load(gonio_data_path)
  gonio_data = gonio_data[10*(patern - 1) : 10*patern]

  return feature_points_data, gonio_data
#####################################################################

###################  Divide into Train and Test  ####################
def DivideIntoTrainAndTest(x, theta):# Train:(718, 100), (718, 1)  Test:(898, 100), (898, 1)
  x_train, theta_train = x[:718], theta[:718]
  x_test, theta_test = x, theta

  return x_train, theta_train, x_test, theta_test
#####################################################################

############################  Analysis  #############################
def Analysis(x, theta): #x: (718, 101), theta: (718, 1)
  x_T = x.T
  x_T_x_inv = np.linalg.pinv(np.dot(x_T, x))
  W =  np.dot(np.dot(x_T_x_inv, x_T), theta) # (101, 1)

  return W
#####################################################################

###########################  Visualize  #############################
def Visualize(y, y_pred, n):
  x = np.arange(y.shape[0])/30 # T:0 ~ 30に変更
  y = y.reshape(-1) # (898,)
  y_pred = y_pred.reshape(-1) # (898,)

  fig1 = plt.figure()
  # plt.title("A result of estimating wrist joint angle", fontsize=18)
  plt.xlabel("Time [s]", fontsize=20)
  plt.ylabel("Wrist angle [deg]", fontsize=20)
  plt.plot(x, y_pred, color="cornflowerblue", linewidth=2, label="Estimated angle")
  plt.plot(x, y, color="tomato", linewidth=2, label="Measured angle")
  plt.vlines(718/30, -80, 30, "gray", linestyles="dashed")
  plt.ylim(-80, 30) #extensor
  plt.xlim(0, y.shape[0]/30)
  plt.legend(loc="upper left", fontsize=16)
  plt.grid(True)
  plt.tight_layout()
  fig1.savefig(result_path + "/plot" + str(n) + ".png")

  fig2 = plt.figure()
  # plt.title("Scatter plots of estimated and measured angle", fontsize=16)
  plt.xlabel("Measured angle [deg]", fontsize=20)
  plt.ylabel("Estimated angle [deg]", fontsize=20)
  plt.scatter(y[718:], y_pred[718:])
  plt.tight_layout()
  fig2.savefig(result_path + "/scatter" + str(n) + ".png")

  # plt.show()
  plt.close()
#####################################################################

########################  Calc_RMSEandR2  ###########################
def Calc_RMSEandR2(theta, theta_pred, T):
  theta = theta.reshape(-1)[718:] # (180, )
  theta_pred = theta_pred.reshape(-1)[718:] # (180, )

  def _coeff(theta, theta_pred):
    corrcoef = np.corrcoef(theta, theta_pred)
    corrcoef = corrcoef[0][1]
    return corrcoef
  
  def _RMSE(theta, theta_pred, T):
    L = np.sum((theta - theta_pred)**2)
    RMSE = np.sqrt(L/T)
    return RMSE

  RMSE = _RMSE(theta, theta_pred, T)
  R2 = r2_score(theta, theta_pred)
  coeff = _coeff(theta, theta_pred)
  print("RMSE = " + str(RMSE))
  print("決定係数 R2 = " + str(R2))
  print("相関係数: " + str(coeff) + "\n")

  with open(result_path + "/RMSEandR2_P" + str(patern) + ".csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([RMSE, R2, coeff])
#####################################################################


#############################  Main  ################################
for p in range(patern_num):
  patern = p + 1
  result_path = MakeResultsPath(patern)
  print("-----Patern" + str(patern) + "-----\n")

  Xs, Thetas = ReadData() # (10, 898, 100), (10, 898, 1)
  # b = np.ones([Xs.shape[1], 1]) # (898, 1)
  for i in range(trial_num):
    test_num = i + 1
    print("--test" + str(test_num) + "--")

    X, Theta = Xs[i], Thetas[i] # (898, 100), (898, 1)
    # # バイアスを追加
    # X = np.append(X, b, axis=1) # (898, 101)

    X_train, Theta_train, X_test, Theta_test = DivideIntoTrainAndTest(X, Theta) # (718, 101), (718, 1), (898, 101), (898, 1)
    W = Analysis(X_train, Theta_train) # (101, 1)

    Theta_pred = np.dot(X_test, W) # (898, 1)

    # グラフ作成
    Visualize(Theta_test, Theta_pred, test_num)

    # RMSE, R2を導出
    T = X_test.shape[0] - X_train.shape[0]
    Calc_RMSEandR2(Theta_test, Theta_pred, T)
    # exit()
  exit()
#####################################################################
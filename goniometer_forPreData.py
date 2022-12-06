# ※ 20220909 のデータ用

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

##########################   Parameter   ###########################
# 被験者
subject = "Katagi"
# 実験日
date = "20220909"
# date = datetime.now().strftime("%Y%m%d")
# 
additional_info = ""

# データの基本情報
gonio_frame_rate = 1000
echo_frame_rate = 43
total_time = 34

numOfData = echo_frame_rate * total_time # 1462

interpolate_rate = gonio_frame_rate * echo_frame_rate # 43000

# The number of gonio data per person
# numOfTrial * numOfDataPerTrial
#      5     * (   1000 * 36   )
#      5     *       36000

# csvファイル内の目的データの開始位置
cut_time = 2 # 最初の2秒はカットする
start_row = gonio_frame_rate * cut_time + 6 # 2006
end_row = start_row + gonio_frame_rate * total_time # 36006
target_column = 2

fp = 1.5 # 通過域端周波数[Hz] #フーリエ変換にあわせて調整
fs = 7.5 # 阻止域端周波数[Hz] # fp*5くらい
gpass = 3 # 通過域端最大損失[dB]
gstop = 40 # 阻止域端最小損失[dB]

# time = np.linspace(0, total_time, gonio_frame_rate * total_time) # 36000
time = np.linspace(0, total_time, numOfData) # (3285,)

# カレントディレクトリ
current_directry = "C:/Users/katagi/Desktop/Research/UltrasoundImaging"
#####################################################################

##########################   Folder & File   ##########################
path = os.getcwd().replace(os.sep, "/")
# pathが正しくカレントディレクトリになっているかを確認
if (path != current_directry):
  exit()

gonio_path = path + "/Data/" + subject + "_" + date + "_" + additional_info + "/Goniometer"
gonio_save_path = gonio_path + "/Processed"
if not os.path.exists(gonio_save_path):
  os.makedirs(gonio_save_path)
fig_save_path = gonio_path + "/fig"
if not os.path.exists(fig_save_path):
  os.makedirs(fig_save_path)


gonio_data = list()
for file in os.listdir(gonio_path):
  if file.endswith(".csv"):
    gonio_data.append(file)
#######################################################################

##########################   データ読み込み   ##########################
# 同被験者（，同条件）でのデータを読み込む（5試行分）
def read_csv():
  thetas = list()
  for i in range(len(gonio_data)):
    each_gonio_path = gonio_path + "/" + gonio_data[i]

    with open(each_gonio_path, encoding="utf-8") as f:
      reader = csv.reader(f)
      theta = list()
      for i, row in enumerate(reader):
        if i < start_row:
          continue
        if i >= end_row:
          break

        theta.append(row[target_column])
      thetas.append(theta)
      # print(len(theta))
      f.close()
    
  # print(len(thetas))
  # exit()

  return np.array(thetas).astype(float)[:, :] # (5, 4500)
#######################################################################

# #########################   フレームレート調整   ########################
# # gonioとechoのフレームレートを合わせる
# # 計測時にgonioは大きくとっておき，echoに合わせる
# def adjust_frame_rate(thetas):
#   new_thetas = list()
#   adjust_num = gonio_frame_rate // echo_frame_rate
#   for theta in thetas:
#     new_theta = list()
#     for i in range(len(theta)):
#       if i % adjust_num == 0:
#         new_theta.append(theta[i])
#         if len(new_theta) == numOfData:
#           break
#     new_thetas.append(new_theta)  # 5 * 1548

#   new_thetas = np.array(new_thetas).astype(float)[:, :, np.newaxis]

#   return new_thetas # (5, 1548, 1)
# #######################################################################

#########################   フレームレート調整   ########################
# gonioとechoのフレームレートを合わせる
# gonioデータを補完(interpolate)しechoデータにあわせてリサンプリングする
#https://watlab-blog.com/2019/09/19/resampling/

def adjust_frame_rate(thetas):

  t0 = 0    # 初期時間[s]
  dt = 1/gonio_frame_rate  # 時間刻み[s] 1/100
  t = np.arange(t0, total_time, dt) # (4500,)

  interpolate_num = interpolate_rate * total_time # 43000 * 34 = 1462000
  t_interpolate = np.linspace(t0, total_time - dt, interpolate_num)

  resampled_thetas = list()

  # 補間
  for theta in thetas:
    theta = np.squeeze(theta)
    f = interpolate.interp1d(t, theta, kind="cubic")
    interpolated_theta = f(t_interpolate) # (328500,)

    # リサンプリング
    new_theta = list()
    for i in range(len(interpolated_theta)):
      if i % gonio_frame_rate == 0: # 1462000 / 1000 = 1462(=43*34)コ
        new_theta.append(interpolated_theta[i])
      if (len(new_theta)) >= numOfData:
        break
    resampled_thetas.append(new_theta) # 5 * 3285
  
  resampled_thetas = np.array(resampled_thetas).astype(float)[:, :, np.newaxis] # (5, 3285, 1)
  return resampled_thetas
#######################################################################

##############################    正規化    ############################
#NumPyで配列の正規化(normalize)、標準化する方法
#https://deepage.net/features/numpy-normalize.html

def zscore(thetas, axis = None):
  thetas_zscore = list()
  thetas_mean = list()
  thetas_std = list()
  for theta in thetas:
    theta_mean = theta.mean(axis=axis, keepdims=True)
    theta_std = np.std(theta, axis=axis, keepdims=True)

    theta_zscore = (theta - theta_mean) / theta_std

    thetas_zscore.append(theta_zscore)
    thetas_mean.append(float(theta_mean))
    thetas_std.append(float(theta_std))
  with open(gonio_save_path + "/ThetasMean" + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(thetas_mean)
    f.close()
  with open(gonio_save_path + "/ThetasStd" + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(thetas_std)
    f.close()
  return thetas_zscore
#######################################################################

#########################   ローパスフィルター   ########################
# 角度情報の加工
#Pythonによるデータ処理4 ～ フィルタ処理
#https://atatat.hatenablog.com/entry/data_proc_python4
#PythonのSciPyでローパスフィルタをかける！
#https://watlab-blog.com/2019/04/30/scipy-lowpass/

def lowpass(thetas, samplerate, fp, fs, gpass, gstop):
  thetas_low = list()
  for theta in thetas:
    theta = np.squeeze(theta)
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    theta_low = signal.filtfilt(b, a, theta)                  #信号に対してフィルタをかける
    print(theta, theta_low)
    
    thetas_low.append(theta_low)
    # # 図で確認
    # plt.scatter(time, theta, label='raw')
    # plt.scatter(time, theta_low, label='filtered')
    # plt.show()
  thetas_low = np.array(thetas_low).astype(float)[:, :, np.newaxis] # (5, 1462, 1)
  return thetas_low
#######################################################################

#########################  Show Figure  ###############################
# データの図を出力して確認
def showFigure(original_thetas, processed_thetas):
  x = time
  for i in range(thetas.shape[0]):
    y1 = np.squeeze(original_thetas[i]) # (3285,)
    y2 = np.squeeze(processed_thetas[i]) # (3285,)

    fig = plt.figure()
    plt.title("Preprocess of Gonio Data")
    plt.scatter(x, y1, label='original')
    plt.scatter(x, y2, label='processed')
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Wrist angle [deg]", fontsize=20)
    plt.legend(loc="upper left", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(fig_save_path + "/fig" + str(i+1) + ".png")
    
    # plt.show()

#######################################################################

#############################  Main  ##################################
# 生データ読み込み
thetas = read_csv() # (5, 34000)
print(thetas.shape)

# # 正規化
# thetas = zscore(thetas)

# フレームレート調整
thetas = adjust_frame_rate(thetas) # (5, 1462, 1)
print("Shape of Resampled Thetas: " + str(thetas.shape))
original = thetas

# # ローパスフィルター
# thetas = lowpass(thetas, gonio_frame_rate, fp, fs, gpass, gstop) # (5, 1462, 1)

# 加工前・加工後のデータの保存・出力
showFigure(original, thetas)

# save data
np.save(gonio_save_path + "/GonioData", thetas)
print("SAVE COMPLETED")
#######################################################################


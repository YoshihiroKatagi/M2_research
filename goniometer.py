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
date = "20221024"
# 補足
additional_info = "modified"

# データの基本情報
gonio_frame_rate = 100

echo_frame_rate = 73 # Depth 20
# echo_frame_rate = 62 # Depth 30
# echo_frame_rate = 54 # Depth 40
# echo_frame_rate = 48 # Depth 50
# echo_frame_rate = 43 # Depth 60

total_time = 44

numOfData = echo_frame_rate * total_time # 3212

interpolate_rate = gonio_frame_rate * echo_frame_rate # 7300

# csvファイル内の目的データの開始位置
cut_time = 12 # 最初の12秒はカットする
start_row = gonio_frame_rate * cut_time # 1200
end_row = start_row + gonio_frame_rate * total_time # 5600
## start_row <= x < end_row (row:1201 ~ 5600, index: 1200 ~ 5599)
target_column = 1

# ローパスフィルタ パラメータ
fp = 1.5 # 通過域端周波数[Hz] #フーリエ変換にあわせて調整
fs = 7.5 # 阻止域端周波数[Hz] # fp*5くらい
gpass = 3 # 通過域端最大損失[dB]
gstop = 40 # 阻止域端最小損失[dB]

time = np.linspace(0, total_time, numOfData) # (3212,)

# カレントディレクトリ
current_directry = "C:/Users/katagi/Desktop/Research/UltrasoundImaging"
#####################################################################

##########################   Folder & File   ##########################
path = os.getcwd().replace(os.sep, "/")
# pathが正しくカレントディレクトリになっているかを確認
if (path != current_directry):
  exit()

target_path = path + "/Data/" + subject + "_" + date + "_" + additional_info
if not os.path.exists(target_path):
  print("target_path doesn't exist")
  exit()

gonio_path = target_path + "/Goniometer"
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

  return np.array(thetas).astype(float)[:, :] # (5, 4400)
#######################################################################

#########################   フレームレート調整   ########################
# gonioとechoのフレームレートを合わせる
# gonioデータを補完(interpolate)しechoデータにあわせてリサンプリングする
#https://watlab-blog.com/2019/09/19/resampling/

def adjust_frame_rate(thetas):

  t0 = 0    # 初期時間[s]
  dt = 1/gonio_frame_rate  # 時間刻み[s] 1/100
  t = np.arange(t0, total_time, dt) # (4400,)

  interpolate_num = interpolate_rate * total_time # 7300 * 44 = 321200
  t_interpolate = np.linspace(t0, total_time - dt, interpolate_num)

  resampled_thetas = list()

  # 補間
  for theta in thetas:
    theta = np.squeeze(theta)
    f = interpolate.interp1d(t, theta, kind="cubic")
    interpolated_theta = f(t_interpolate) # (321200,)

    # リサンプリング
    new_theta = list()
    for i in range(len(interpolated_theta)):
      if i % gonio_frame_rate == 0: # 231200 / 100 = 3212(=73*44)コ
        new_theta.append(interpolated_theta[i])
      if (len(new_theta)) >= numOfData:
        break
    resampled_thetas.append(new_theta) # 5 * 3212
  
  resampled_thetas = np.array(resampled_thetas).astype(float)[:, :, np.newaxis] # (5, 3212, 1)
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
    
    thetas_low.append(theta_low)
    # # 図で確認
    # plt.scatter(time, theta, label='raw')
    # plt.scatter(time, theta_low, label='filtered')
    # plt.show()
  thetas_low = np.array(thetas_low).astype(float)[:, :, np.newaxis] # (5, 3212, 1)
  return thetas_low
#######################################################################

#########################  Show Figure  ###############################
# データの図を出力して確認
def showFigure(original_thetas, processed_thetas):
  x = time
  for i in range(thetas.shape[0]):
    y1 = np.squeeze(original_thetas[i]) # (3212,)
    y2 = np.squeeze(processed_thetas[i]) # (3212,)

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
print("--- TARGET DATA: " + subject + "_" + date + "_" + additional_info + " ---")
print("GONIO FRAME RATE: " + str(gonio_frame_rate))
print("ECHO FRAME RATE: " + str(echo_frame_rate))
print("TOTAL TIME: " + str(total_time))
print("NUM OF DATA: " + str(numOfData))

# 生データ読み込み
thetas = read_csv() # (5, 4400)

# # 正規化
# thetas = zscore(thetas)

# フレームレート調整
thetas = adjust_frame_rate(thetas) # (5, 3212, 1)
print("Shape of Resampled Thetas: " + str(thetas.shape))
original = thetas

# ローパスフィルター
thetas = lowpass(thetas, gonio_frame_rate, fp, fs, gpass, gstop)

# 加工前・加工後のデータの保存・出力
showFigure(original, thetas)

# save data
np.save(gonio_save_path + "/GonioData", thetas)
print("SAVE COMPLETED")
#######################################################################
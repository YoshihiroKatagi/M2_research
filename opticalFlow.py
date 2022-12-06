import os
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

##########################   Parameter   ###########################
# 被験者
subject = "Katagi"
# 実験日
date = "20221024"
# 補足
additional_info = "modified"

# 計測時間(s)
start_time = 12
end_time = 56
total_time = end_time - start_time # 44

# トリミング（元動画の超音波画像の部分のみに切り取る）
# 元動画サイズ：(1172, 608)
top, bottom = 40, 600
left, right = 70, 1100 # Depth: 20
# left, right = 85, 1085 # Depth: 30
# left, right = 200, 970 # Depth: 40
# left, right = 280, 890 # Depth: 50
# left, right = 325, 845 # Depth: 60
# # トリミング後サイズ：(right - left, 560)



# 見切れ対策（最初の特徴点をトリミングした範囲内から抽出）(w:510, h:560)
trim_w = 80
trim_h = 60

# 特徴点数
feature_num = 30

####     オプティカルフローパラメータ     ####
maxCorners=500       # 特徴点の最大数
qualityLevel=0.2     # 特徴点を選択するしきい値で、高いほど特徴点は厳選されて減る。
minDistance=25       # 特徴点間の最小距離
blockSize=15         # 特徴点の計算に使うブロック（周辺領域）サイズ

# フレーム内へ角度情報を挿入する位置
wrist_angle_position = (40, 530)

# カレントディレクトリ
current_directry = "C:/Users/katagi/Desktop/Research/UltrasoundImaging"
#####################################################################

########################   Folder & File   ##########################
path = os.getcwd().replace(os.sep, "/")
# pathが正しくカレントディレクトリになっているかを確認
if (path != current_directry):
  exit()

data_path = path + "/Data/" + subject + "_" + date + "_" + additional_info

# エコーのパス（動画取得・保存用）
echo_path = data_path + "/Echo"
echo_original_path = echo_path + "/Original"
echo_preprocessed_path = echo_path + "/Preprocessed"
echo_opticalflow_path = echo_path + "/OpticalFlow"
# エコー動画をリストで取得
echo_original = os.listdir(echo_original_path)

# # ゴニオメータのデータを取得（goniometer.pyを実行して.npyファイルにまとめておく）
gonio_data_path = data_path + "/Goniometer/Processed/GonioData.npy"
gonio_data = np.load(gonio_data_path) # (5, 3212, 1)
#####################################################################

######################   Preprocess of movie   ######################
# 1. 動画時間を計測時間に合わせる（必要なフレーム数だけを保存）
# 2. 画像内の不要な部分を削る(スクリーン中のエコー部分のみ)
def Preprocess(echo_file_name):
  target_path = echo_original_path + "/" + echo_file_name
  save_path = echo_preprocessed_path + "/" + echo_file_name

  video = cv2.VideoCapture(target_path)

  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print("WIDTH: " + str(width) + ", HEIGHT: " + str(height))

  frame_rate = int(video.get(cv2.CAP_PROP_FPS))   # 73, 62, 54, 48, or 43
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  # フレームレート確認
  print("FRAME RATE: " + str(frame_rate) + ",\nFRAME COUNT: " + str(frame_count) + "\n")
  # exit()

  # 必要なフレームのみ取り出す
  start_frame = start_time * frame_rate # 876
  end_frame = end_time * frame_rate # 4088

  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  new_size = (right - left, bottom - top)
  save = cv2.VideoWriter(save_path, fmt, frame_rate, new_size)

  for i in range(end_frame):    # 1.の処理
    ret, frame = video.read()
    if ret == False:
      break
    if i < start_frame:
      continue

    frame = frame[top:bottom, left:right]   # 2.の処理
    frame = cv2.resize(frame, new_size)
    cv2.imshow("check frame", frame)
    key = cv2.waitKey(10)
    if key == 27:
      break
    save.write(frame)

  save.release()
  video.release()
  cv2.destroyAllWindows()
#####################################################################

###########################  Optical Flow  ##########################
def OpticalFlow(echo_file_name):
  target_path = echo_preprocessed_path + "/" + echo_file_name

  video = cv2.VideoCapture(target_path)

  # Shi-Tomasi法のパラメータ（コーナー：物体の角を特徴点として検出）
  ft_params = dict(maxCorners=maxCorners,       # 特徴点の最大数
                  qualityLevel=qualityLevel,    # 特徴点を選択するしきい値で、高いほど特徴点は厳選されて減る。
                  minDistance=minDistance,       # 特徴点間の最小距離
                  blockSize=blockSize)         # 特徴点の計算に使うブロック（周辺領域）サイズ

  # Lucal-Kanade法のパラメータ（追跡用）
  lk_params = dict(winSize=(80,80),     # オプティカルフローの推定の計算に使う周辺領域サイズ
                  maxLevel=4,          # ピラミッド数
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))       # 探索アルゴリズムの終了条件

  # #properties
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_rate = int(video.get(cv2.CAP_PROP_FPS)) # 73, 62, 54, 48, or 43

  size = (width, height)  # (right - left, 560)
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 3212
  print("FRAME COUNT: " + str(frame_count) + ",\nWIDTH: " + str(width) + ", HEIGHT: " + str(height) + "\n")
  # exit()

  # 最初のフレームを取得してグレースケール変換
  ret, frame = video.read()
  frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # 最初のフレームのみさらにトリミング
  frame_pre_first = frame_pre[trim_h : height - trim_h, trim_w : width - trim_w]

  # Shi-Tomasi法で特徴点の検出
  feature_pre = cv2.goodFeaturesToTrack(frame_pre_first, mask=None, **ft_params)

  # 座標をトリミング前のものに修正
  for v in feature_pre:
    v[0][0] += trim_w
    v[0][1] += trim_h

  # mask用の配列を生成
  mask = np.zeros_like(frame)

  frame_num = 0
  # 動画終了まで繰り返し
  while(video.isOpened() and frame_num < frame_count):
    
    # 次のフレームを取得し、グレースケールに変換
    ret, frame = video.read()
    if ret == False:
      break

    frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade法でフレーム間の特徴点のオプティカルフローを計算
    feature_now, status, err = cv2.calcOpticalFlowPyrLK(frame_pre, frame_now, feature_pre, None, **lk_params)

    # オプティカルフローを検出した特徴点を取得
    good1 = feature_pre[status == 1] # 1フレーム目
    good2 = feature_now[status == 1] # 2フレーム目

    # 座標を保存する配列を初期化、初期位置を保存
    if frame_num == 0:
      feature_points_of_all = np.empty([0, good1.shape[0], 2])
      feature_points_of_t = good1.reshape([1, good1.shape[0], 2])
      feature_points_of_all = np.append(feature_points_of_all, feature_points_of_t, axis=0)

      first_num = good1.shape[0]
      print("Num of first feature point: " + str(first_num))

    # statusが0となるインデックスを取得
    vanish = np.where(status == 0)[0]
    # # 確認用
    # if len(vanish) != 0:
    #   print("frame_num: " + str(frame_num))
    #   print(vanish)

    # position_allからstatus=0の要素を削除
    for i, v in enumerate(vanish):
      # 最初のフレーム間で特徴点が消えている場合は何もしない
      if frame_num == 0:
        break
      # print("i, v: " + str(i) + ", " + str(v))
      feature_points_of_all = np.delete(feature_points_of_all, v - i, 1)
    
    # 各時刻における座標を保存
    feature_points_of_t = good2.reshape([1, good2.shape[0], 2])
    feature_points_of_all = np.append(feature_points_of_all, feature_points_of_t, axis=0)

    # # 特徴点とオプティカルフローをフレーム・マスクに描画
    # for i, (pt1, pt2) in enumerate(zip(good1, good2)):
    #   x1, y1 = pt1.ravel() # 1フレーム目の特徴点座標
    #   x2, y2 = pt2.ravel() # 2フレーム目の特徴点座標

    #   # 軌跡を描画（過去の軌跡も残すためにmaskに描く）
    #   mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), [128, 128, 128], 1)

    #   # 現フレームにオプティカルフローを描画
    #   frame = cv2.circle(frame, (int(x2), int(y2)), 5, [0, 0, 200], -1)
    
    # # フレームとマスクの論理積（合成）
    # img = cv2.add(frame, mask)

    # # ウィンドウに表示
    # cv2.imshow('mask', img)

    # 次のフレーム、ポイントの準備
    frame_pre = frame_now.copy() # 次のフレームを最初のフレームに設定
    feature_pre = good2.reshape(-1, 1, 2) # 次の点を最初の点に設定

    # qキーが押されたら途中終了
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break
    
    frame_num += 1

  last_num = good2.shape[0]
  print("Num of last feature point: " + str(last_num) + "\n\n")

  feature_points_of_all = np.delete(feature_points_of_all, np.s_[feature_num:], 1) # (3212, 50, 2)
  feature_points_of_all = feature_points_of_all.reshape([1, feature_points_of_all.shape[0], feature_points_of_all.shape[1] * 2]) # (1, 3212, 100)

  # 終了処理
  # cv2.destroyAllWindows()
  video.release()

  return feature_points_of_all
#####################################################################

######################  Save Proccessed Image  ######################
def SaveProccessedImage(echo_file_name, feature_points, theta):
  target_path = echo_preprocessed_path + "/" + echo_file_name
  save_path = echo_opticalflow_path + "/" + echo_file_name

  feature_points = feature_points[:, :, :100].reshape([feature_points.shape[1], feature_num, 2]) # (3212, 50, 2)
  video = cv2.VideoCapture(target_path)

  # #properties
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  size = (width, height)  # (right - left, 560)
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  #3212
  frame_rate = int(video.get(cv2.CAP_PROP_FPS)) # 43

  # for save
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  save = cv2.VideoWriter(save_path, fmt, frame_rate, size)

  # 最初のフレームを取得
  ret, frame = video.read()
  # frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # mask用の配列を生成
  mask = np.zeros_like(frame)

  # 最初の特徴点の座標を取得
  points_pre = feature_points[0] # (50, 2)
  # 動画終了まで繰り返し
  for t in range(feature_points.shape[0]):
    if t+1 == feature_points.shape[0]:
      break
    
    # 現在のフレームを取得
    ret, frame = video.read()
    if ret == False:
      break
    # frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 現在の特徴点の座標を取得
    points_now = feature_points[t+1]

    # 現在の関節角度を取得
    theta_now = round(theta[t, 0], 2)

    # オプティカルフローと現在の特徴点をmask, frameに描画
    for p_pre, p_now in zip(points_pre, points_now):
      x1, y1 = p_pre[0], p_pre[1]
      x2, y2 = p_now[0], p_now[1]

      mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), [128, 128, 128], 1)
      frame = cv2.circle(frame, (int(x2), int(y2)), 5, [0, 0, 200], -1)

      # 関節角度情報を描画
      angle = "Wrist Angle: " + str(theta_now)
      # org = (20, 460) # 挿入する座標
      cv2.putText(frame, angle, wrist_angle_position, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255))

    # frameとmaskの合成
    img = cv2.add(frame, mask)

    # ウィンドウに表示
    cv2.imshow("mask", img)

    # フレームごとに保存
    save.write(img)

    # pointsの更新
    points_pre = points_now

    # qキーが押されたら途中終了
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break

  # 終了処理
  cv2.destroyAllWindows()
  video.release()
  save.release()
#####################################################################

#######################  Check Echo Feature  ########################
# 動画情報の確認
def CheckEchoFeature(echo_file_name):
  target_path = echo_original_path + "/" + echo_file_name
  video = cv2.VideoCapture(target_path)
  
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_rate = int(video.get(cv2.CAP_PROP_FPS))   # 73
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

  print("WIDTH: " + str(width) + ", HEIGHT: " + str(height))
  print("FRAME RATE: " + str(frame_rate) + ",\nFRAME COUNT: " + str(frame_count) + "\n")
#####################################################################

##############################  Main  ###############################
print("--- TARGET DATA: " + subject + "_" + date + "_" + additional_info + " ---")
# # 各エコーのフレームレート等を確認
# for i, each_echo in enumerate(echo_original):
#   print("\n----Target: " + "No." + str(i + 1) + ", " + str(each_echo) + "----")
#   CheckEchoFeature(each_echo)
# exit()

# 各動画ごとにオプティカルフローを実行
for i, each_echo in enumerate(echo_original):
  print("---Target Trial: " + "No." + str(i + 1) + ", " + str(each_echo) + "---")

  # 動画の前処理
  Preprocess(each_echo)

  # オプティカルフローを実行し，特徴点抽出
  feature_points_data = OpticalFlow(each_echo)

  # 50個の特徴点とその時の関節角度をUS画像に描画
  Theta = gonio_data[i]
  SaveProccessedImage(each_echo, feature_points_data, Theta)

  if i == 0:
    feature_points_data_of_all = np.empty([0, feature_points_data.shape[1], feature_points_data.shape[2]])
  feature_points_data_of_all = np.append(feature_points_data_of_all, feature_points_data, axis=0)

print("Feature points data of all: " + str(feature_points_data_of_all.shape) + "\n") # (5, 3212, 100)

# 全画像における特徴点の座標を保存
feature_points_save_path = echo_path + "/FeaturePointsData"
np.save(feature_points_save_path, feature_points_data_of_all)
#####################################################################


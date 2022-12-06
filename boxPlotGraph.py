import matplotlib.pyplot as plt

import numpy as np

# # Gain
# x1 = [21.28862641, 23.30387596, 19.10971598, 25.15257667, 25.69376452]
# x2 = [39.78861084, 25.15861155, 20.98695767, 43.85255669, 47.66865305]
# x3 = [30.50155391, 17.21865637, 27.14565798, 14.32638521, 63.99181007]
# x4 = [31.31731511, 34.68201866, 28.69627809, 47.8967141, 30.16978029]
# x5 = [20.687336, 42.56246061, 19.46655275, 38.41162404, 29.00738416]
# x6 = [32.09068124, 25.90539144, 22.45437425, 45.1350952, 25.88737919]

# ave:
# 22.9, 35.5, 30.6, 34.6, 30.0, 30.2

# Depth and Focus
# x1 = np.array([26.05411447, 46.0055156, 71.65635163, 24.25826828, 27.32773648])
# x2 = np.array([29.06428817, 21.22984094, 19.60105286, 29.41734475, 29.53214159])
# x3 = np.array([48.52722293, 26.6394216, 104.2327717, 46.86252947, 33.194838])
# x4 = np.array([106.4480476, 18.8129559, 21.91640042, 46.13297203, 43.91902805])
# x5 = np.array([71.11917489, 20.43810026, 24.47285047, 21.05603945, 50.90598437])

# print(np.mean(x1), np.mean(x2),  np.mean(x3),  np.mean(x4),  np.mean(x5), )

# ave:
# , 25.8, , , 







# Frequency and Dynamic Range
x1 = np.array([39.13549284, 18.63456565, 103.300427, 13.2510279, 132.1799928])
x2 = np.array([40.97334674, 21.79751174, 56.41573263, 52.9379834, 42.12401725])
x3 = np.array([44.1498156, 86.07911008, 31.05901888, 67.04151877, 17.20729081])
x4 = np.array([27.50689271, 31.19998622, 30.02340133, 22.22981303, 26.64595943])
x5 = np.array([60.63693788, 29.67039072, 78.28587587, 9.439399584, 18.80474887])

print(np.mean(x1), np.mean(x2),  np.mean(x3),  np.mean(x4),  np.mean(x5), )





# fig1, ax = plt.subplots()
# ax.boxplot((x1, x2, x3, x4))
# ax.set_xticklabels(["transverse", "diagonal-1", "longitudinal", "diagonal-2"], fontsize=16)
# plt.xlabel('Probe Position', fontsize=20)
# plt.ylabel('RMSE', fontsize=20)
# # plt.show()
# # fig1.savefig("boxplot1" + ".png")

# fig2, ax = plt.subplots()
# ax.boxplot((x5, x6, x7, x8))
# ax.set_xticklabels(["transverse", "diagonal-1", "longitudinal", "diagonal-2"], fontsize=16)
# plt.xlabel('Probe Position', fontsize=20)
# plt.ylabel('RMSE', fontsize=20)
# # plt.show()
# # fig2.savefig("boxplot2" + ".png")




# # Gain
# fig, ax = plt.subplots()
# ax.boxplot((x1, x2, x3, x4, x5, x6))
# ax.set_xticklabels(["20", "30", "40", "60", "70", "80"], fontsize=12)
# plt.xlabel('Gain[%]', fontsize=16)
# plt.ylabel('RMSE', fontsize=18)
# plt.tight_layout()
# # plt.show()
# fig.savefig("boxplot_Gain" + ".png")


# # Depth and Focus
# fig, ax = plt.subplots()
# ax.boxplot((x1, x2, x3, x4, x5))
# ax.set_xticklabels(["De_20", "De_30", "De_40", "Fo_7-17", "Fo_17-30"], fontsize=12)
# plt.xlabel('Depth[mm] and Focus[mm]', fontsize=16)
# plt.ylabel('RMSE', fontsize=18)
# plt.tight_layout()
# # plt.show()
# fig.savefig("boxplot_De_Fo" + ".png")

# # Frequency and Dynamic Range
# fig, ax = plt.subplots()
# ax.boxplot((x1, x2, x3, x4, x5))
# ax.set_xticklabels(["Fr_5", "Fr_9", "F_11", "Dy_42", "Dy_78"], fontsize=12)
# plt.xlabel('Frequency[MHz] and Dynamic Range[dB]', fontsize=16)
# plt.ylabel('RMSE', fontsize=18)
# plt.tight_layout()
# # plt.show()
# fig.savefig("boxplot_Fr_Dy" + ".png")
import matplotlib.pyplot as plt

## Stats for mid map

r1_mid_data_with_zero_noise = {
    "points": [2000, 10000, 20000],
    "noise": 0,
    "train_time": [3.73, 12.45, 1515.66],
    "test_time": [1.295, 2.99, 7.29],
    "roc_auc": [0.7936, 0.8270, 0.8367],
    "nll": [0.4890, 0.4384, 0.4324],
    "unknown_percentage": [59.63, 9.73, 8.22]
}

r2_mid_data_with_zero_noise = {
    "points": [2000, 10000, 20000],
    "noise": 0,
    "train_time": [7.063, 14.884, 1684],
    "test_time": [1.02, 2.844, 7.1044],
    "roc_auc": [0.8239, 0.8677, 0.8725],
    "nll": [0.4814, 0.4188, 0.4106],
    "unknown_percentage": [48.67, 10.12, 8.16]
}

merged_mid_data_with_zero_noise = {
    "points": [2000, 10000, 20000],
    "noise": 0,
    # "train_time": [7.063, 14.884, 1684],
    # "test_time": [1.02, 2.844, 7.1044],
    "roc_auc": [0.8979, 0.9293, 0.9333],
    "nll": [0.4155, 0.3528, 0.3490],
    "unknown_percentage": [18.54, 10.73, 9.37]
}


## Noise 0.05
r1_mid_data_with_05_noise = {
    "points": 10000,
    "noise": 0.05,
    "train_time": 9.571,
    "test_time": 2.585,
    "roc_auc": 0.8279,
    "nll": 0.4147,
    "unknown_percentage": 9.77
}

r2_mid_data_with_05_noise = {
    "points": 10000,
    "noise": 0.05,
    "train_time": 10.168,
    "test_time": 2.533,
    "roc_auc": 0.8550,
    "nll": 0.3957,
    "unknown_percentage": 9.8
}

merged_mid_data_with_05_noise = {
    "points": 10000,
    "noise": 0.05,
    # "train_time": [7.063, 14.884, 1684],
    # "test_time": [1.02, 2.844, 7.1044],
    "roc_auc": 0.9189,
    "nll": 0.3245,
    "unknown_percentage": 11.57
}


## Noise 0.1
r1_mid_data_with_1_noise = {
    "points": 10000,
    "noise": 0.1,
    "train_time": 8.746,
    "test_time": 2.61,
    "roc_auc": 0.8113,
    "nll": 0.4574,
    "unknown_percentage": 13.64
}

r2_mid_data_with_1_noise = {
    "points": 10000,
    "noise": 0.1,
    "train_time": 8.884,
    "test_time": 2.566,
    "roc_auc": 0.8517,
    "nll": 0.4490,
    "unknown_percentage": 14.70
}

merged_mid_data_with_1_noise = {
    "points": 10000,
    "noise": 0.1,
    # "train_time": [7.063, 14.884, 1684],
    # "test_time": [1.02, 2.844, 7.1044],
    "roc_auc": 0.9112,
    "nll": 0.3851,
    "unknown_percentage": 14.35
}


## Plot ROC vs number of points

plt.figure()
plt.plot(r1_mid_data_with_zero_noise['points'], r1_mid_data_with_zero_noise['roc_auc'], label='Robot 1', marker='o')
plt.plot(r2_mid_data_with_zero_noise['points'], r2_mid_data_with_zero_noise['roc_auc'], label='Robot 2', marker='o')
plt.plot(merged_mid_data_with_zero_noise['points'], merged_mid_data_with_zero_noise['roc_auc'], label='Merged', marker='o')
plt.xlabel('Number of Points')
plt.ylabel('ROC AUC')
plt.legend()
# plt.show()

# Plot NLL vs number of points
plt.figure()
plt.plot(r1_mid_data_with_zero_noise['points'], r1_mid_data_with_zero_noise['nll'], label='Robot 1', marker='o')
plt.plot(r2_mid_data_with_zero_noise['points'], r2_mid_data_with_zero_noise['nll'], label='Robot 2', marker='o')
plt.plot(merged_mid_data_with_zero_noise['points'], merged_mid_data_with_zero_noise['nll'], label='Merged', marker='o')
plt.xlabel('Number of Points')
plt.ylabel('NLL')
plt.legend()
# plt.show()

# Plot train time vs number of points
plt.figure()
plt.plot(r1_mid_data_with_zero_noise['points'], r1_mid_data_with_zero_noise['train_time'], label='Robot 1', marker='o')
plt.plot(r2_mid_data_with_zero_noise['points'], r2_mid_data_with_zero_noise['train_time'], label='Robot 2', marker='o')
# plt.plot(merged_mid_data_with_zero_noise['points'], merged_mid_data_with_zero_noise['train_time'], label='Merged', marker='o')
plt.xlabel('Number of Points')
plt.ylabel('Train Time')
plt.legend()
# plt.show()

# Plot test time vs number of points
plt.figure()
plt.plot(r1_mid_data_with_zero_noise['points'], r1_mid_data_with_zero_noise['test_time'], label='Robot 1', marker='o')
plt.plot(r2_mid_data_with_zero_noise['points'], r2_mid_data_with_zero_noise['test_time'], label='Robot 2', marker='o')
# plt.plot(merged_mid_data_with_zero_noise['points'], merged_mid_data_with_zero_noise['test_time'], label='Merged', marker='o')
plt.xlabel('Number of Points')
plt.ylabel('Test Time')
plt.legend()
# plt.show()

# Plot unknown percentage vs number of points
plt.figure()
plt.plot(r1_mid_data_with_zero_noise['points'], r1_mid_data_with_zero_noise['unknown_percentage'], label='Robot 1', marker='o')
plt.plot(r2_mid_data_with_zero_noise['points'], r2_mid_data_with_zero_noise['unknown_percentage'], label='Robot 2', marker='o')
plt.plot(merged_mid_data_with_zero_noise['points'], merged_mid_data_with_zero_noise['unknown_percentage'], label='Merged', marker='o')
plt.xlabel('Number of Points')
plt.ylabel('Unknown Percentage')
plt.legend()
# plt.show()

# Plot ROC vs noise for 10000 points
plt.figure()
plt.plot([r1_mid_data_with_zero_noise['noise'], r1_mid_data_with_05_noise['noise'], r1_mid_data_with_1_noise['noise']], [r1_mid_data_with_zero_noise['roc_auc'][1], r1_mid_data_with_05_noise['roc_auc'], r1_mid_data_with_1_noise['roc_auc']], label='Robot 1', marker='o')
plt.plot([r2_mid_data_with_zero_noise['noise'], r2_mid_data_with_05_noise['noise'], r2_mid_data_with_1_noise['noise']], [r2_mid_data_with_zero_noise['roc_auc'][1], r2_mid_data_with_05_noise['roc_auc'], r2_mid_data_with_1_noise['roc_auc']], label='Robot 2', marker='o')
plt.plot([merged_mid_data_with_zero_noise['noise'], merged_mid_data_with_05_noise['noise'], merged_mid_data_with_1_noise['noise']], [merged_mid_data_with_zero_noise['roc_auc'][1], merged_mid_data_with_05_noise['roc_auc'], merged_mid_data_with_1_noise['roc_auc']], label='Merged', marker='o')
plt.xlabel('Noise')
plt.ylabel('ROC AUC')
plt.legend()
# plt.show()

# Plot NLL vs noise for 10000 points
plt.figure()
plt.plot([r1_mid_data_with_zero_noise['noise'], r1_mid_data_with_05_noise['noise'], r1_mid_data_with_1_noise['noise']], [r1_mid_data_with_zero_noise['nll'][1], r1_mid_data_with_05_noise['nll'], r1_mid_data_with_1_noise['nll']], label='Robot 1', marker='o')
plt.plot([r2_mid_data_with_zero_noise['noise'], r2_mid_data_with_05_noise['noise'], r2_mid_data_with_1_noise['noise']], [r2_mid_data_with_zero_noise['nll'][1], r2_mid_data_with_05_noise['nll'], r2_mid_data_with_1_noise['nll']], label='Robot 2', marker='o')
plt.plot([merged_mid_data_with_zero_noise['noise'], merged_mid_data_with_05_noise['noise'], merged_mid_data_with_1_noise['noise']], [merged_mid_data_with_zero_noise['nll'][1], merged_mid_data_with_05_noise['nll'], merged_mid_data_with_1_noise['nll']], label='Merged', marker='o')
plt.xlabel('Noise')
plt.ylabel('NLL')
plt.legend()
# plt.show()

# Plot unknown percentage vs noise for 10000 points
plt.figure()
plt.plot([r1_mid_data_with_zero_noise['noise'], r1_mid_data_with_05_noise['noise'], r1_mid_data_with_1_noise['noise']], [r1_mid_data_with_zero_noise['unknown_percentage'][1], r1_mid_data_with_05_noise['unknown_percentage'], r1_mid_data_with_1_noise['unknown_percentage']], label='Robot 1', marker='o')
plt.plot([r2_mid_data_with_zero_noise['noise'], r2_mid_data_with_05_noise['noise'], r2_mid_data_with_1_noise['noise']], [r2_mid_data_with_zero_noise['unknown_percentage'][1], r2_mid_data_with_05_noise['unknown_percentage'], r2_mid_data_with_1_noise['unknown_percentage']], label='Robot 2', marker='o')
plt.plot([merged_mid_data_with_zero_noise['noise'], merged_mid_data_with_05_noise['noise'], merged_mid_data_with_1_noise['noise']], [merged_mid_data_with_zero_noise['unknown_percentage'][1], merged_mid_data_with_05_noise['unknown_percentage'], merged_mid_data_with_1_noise['unknown_percentage']], label='Merged', marker='o')
plt.xlabel('Noise')
plt.ylabel('Unknown Percentage')
plt.legend()
plt.show()



## Large Map
r1_large_data_with_0_noise = {
    "points": [2000, 10000],
    "noise": 0.0,
    "train_time": [8.555, 21.806],
    "test_time": [35.1, 139.79],
    "roc_auc": [0.5134, 0.5070],
    "nll": [0.6387, 0.5914],
    "unknown_percentage": [88.99, 92.75]
}

r2_large_data_with_0_noise = {
    "points": [2000, 10000],
    "noise": 0.0,
    "train_time": [4.639, 26.809],
    "test_time": [34.717, 139.87],
    "roc_auc": [0.5420, 0.4810],
    "nll": [0.3099, 1.1693],
    "unknown_percentage": [0.35, 0.77]
}

merged_large_data_with_0_noise = {
    "points": [2000, 10000],
    "noise": 0.0,
    # "train_time": 10.168,
    # "test_time": 34.802,
    "roc_auc": [0.5372, 0.4830],
    "nll": [0.4144, 1.1087],
    "unknown_percentage": [1.37, 1.06]
}


r1_large_data_with_05_noise = {
    "points": 2000,
    "noise": 0.05,
    "train_time": 8.16,
    "test_time": 34.729,
    "roc_auc": 0.5089,
    "nll": 0.1814,
    "unknown_percentage": 0.16
}

r2_large_data_with_05_noise = {
    "points": 2000,
    "noise": 0.05,
    "train_time": 4.639,
    "test_time": 34.802,
    "roc_auc": 0.5474,
    "nll": 0.2804,
    "unknown_percentage": 0.28
}

merged_large_data_with_05_noise = {
    "points": 2000,
    "noise": 0.05,
    # "train_time": 10.168,
    # "test_time": 34.802,
    "roc_auc": 0.5291,
    "nll": 0.1384,
    "unknown_percentage": 0.17
}
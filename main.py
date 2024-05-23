# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import random

import numpy as np


def sort(input, descend=False):
    """
    插入排序 （insertion_sort)
    :param input: 数据
    :param descend: 降序标志
    :return: 排列后的数据
    """
    if isinstance(input, list):
        for i in range(1, len(input)):
            temp = input[i]
            j = i - 1
            if descend is False:
                while j >= 0 and input[j] > temp:
                    input[j + 1] = input[j]
                    j = j - 1
            else:
                while j >= 0 and input[j] < temp:
                    input[j + 1] = input[j]
                    j = j - 1
            input[j + 1] = temp
    return input


def seek(squeeze_data, key):
    """
    线性查找
    :param squeeze_data:
    :param key:
    :return:
    """
    if key in squeeze_data:
        mark = []
        for i in range(0, len(squeeze_data)):
            if key == squeeze_data[i]:
                mark.append(i)
    else:
        mark = 'NIL'
    return mark


def choose_sort(in_put):
    """
    选择排序
    :param in_put: 
    :return:
    """
    for k in range(len(in_put)):
        mark = in_put[k]
        j = k
        # 获得最小值
        for i in range(k, len(in_put)):
            if in_put[i] < mark:
                mark = in_put[i]
                j = i
        in_put[j] = in_put[k]
        in_put[k] = mark
    return in_put


def binary_addition(data1, data2):
    """
    二进制加法
    :param data1:
    :param data2:
    :return:
    """
    len1 = len(data1)
    len2 = len(data2)
    if len1 > len2:
        length = len1
    else:
        length = len2
    data3 = []
    C_ = 0
    for i in range(0, length + 1):
        data3.append(0)
    for i in range(0, length):
        j1 = len1 - i - 1
        j2 = len2 - i - 1
        if j1 >= 0 and j2 >= 0:
            temp1 = C_ + data1[j1] + data2[j2]
        elif j1 < 0:
            temp1 = C_ + data2[j2]
        elif j2 < 0:
            temp1 = C_ + data1[j1]

        if temp1 == 3:
            C_ = 1
            temp1 = 1
        elif temp1 == 2:
            C_ = 1
            temp1 = 0
        elif temp1 == 1:
            C_ = 0
            temp1 = 1
        elif temp1 == 0:
            C_ = 0
            temp1 = 0
        data3[length - i] = temp1
    data3[0] = C_
    return data3


def merge_sort(data1, descend=False):
    """
    归并排序（分治法），不断分成两个进行排序，大数据是用时更短
    :param data1:
    :param descend:
    :return:
    """
    len_data = len(data1)
    if len_data % 2 == 0:
        len_temp = int(len_data / 2)
    else:
        len_temp = int((len_data + 1) / 2)
    data11 = data1[0:len_temp]
    data22 = data1[len_temp:]
    if len(data11) != 1:
        merge_sort(data11)
    if len(data22) != 1:
        merge_sort(data22)
    data11.append(100000)
    data22.append(100000)
    key1 = 0
    key2 = 0
    if descend is False:
        for i in range(len_data):
            if data11[key1] <= data22[key2]:
                data1[i] = data11[key1]
                key1 += 1
            else:
                data1[i] = data22[key2]
                key2 += 1
            if data11[key1] == 100000 and data22[key2] == 100000:
                break
    else:
        for i in range(len_data):
            if data11[key1] <= data22[key2]:
                data1[i] = data22[key2]
                key2 += 1
            else:
                data1[i] = data11[key1]
                key1 += 1
            if data11[key1] == 100000 and data22[key2] == 100000:
                break
    return data1


# 按间距中的绿色按钮以运行脚本。
'''
if __name__ == '__main__':
    data = []
    while len(data) < 10:
        temp = random.randint(0, 15)
        if temp not in data:
            data.append(temp)
    print(f"merge_sort:{merge_sort(data)}")
    print(sort(data))
    print(choose_sort(data))
    print(sort(data, True))
    print(seek(data, random.randint(0, 15)))
    one = []
    two = []
    mode = random.randint(0, 2)
    if mode == 0:
        random.seed(23)
        while len(one) < 10:
            temp1 = random.randint(0, 1)
            temp2 = random.randint(0, 1)
            one.append(temp1)
            two.append(temp2)
    elif mode == 1:
        random.seed(23)
        while len(one) < 10:
            temp1 = random.randint(0, 1)
            one.append(temp1)
        random.seed(450)
        while len(two) < 12:
            temp2 = random.randint(0, 1)
            two.append(temp2)
    elif mode == 2:
        random.seed(23)
        while len(one) < 12:
            temp1 = random.randint(0, 1)
            one.append(temp1)
        random.seed(450)
        while len(two) < 10:
            temp2 = random.randint(0, 1)
            two.append(temp2)
    print(binary_addition(one, two))

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
'''
a = 32.07633637908162 + 32.18736362444816 + 31.416398298870945 + 31.59770847093465 + 28.018431815171514 + 27.031696542177865 + 26.344514669410714 + 26.369404218750176 + 28.499085756551327 + 26.84742634067655 + 29.800484389494112
a = a / 11
print(a)
# 225
PSNR = [31.958439783464748, 29.62574085314737, 30.670021795266223, 34.094292187682925, 37.68113090128216,
        34.401722991115285, 37.152024256615384, 33.711093058191636, 35.20539750827044, 33.38147858361989,
        33.947953450256335, 32.333667112844, 36.12635777161543, 37.1097703825644, 33.94524120061854,
        30.670873458105483, 34.15617381597953, 32.35349457870668, 35.69953119879863, 34.442191028977234]
len = len(PSNR)
print(np.mean(PSNR))
SSIM = [0.8687791228294373, 0.8765277862548828, 0.947134256362915, 0.8987076878547668, 0.9534338116645813,
        0.8964259624481201, 0.946872889995575, 0.8870436549186707, 0.9331608414649963, 0.8977872133255005,
        0.8886672854423523, 0.9434426426887512, 0.9249361753463745, 0.9480819702148438, 0.8982111215591431,
        0.885063886642456, 0.9011298418045044, 0.8981084823608398, 0.9374396204948425, 0.9037889242172241]
print(np.mean(SSIM))
# 230
PSNR1 = [31.442606151709334, 30.062992208529877, 32.25337162139115, 34.05411452706684, 37.495350180371055,
         34.11352887569245, 36.99911899297051, 33.168730221657334, 34.662759234762106, 33.221433507630714,
         33.32416575142163, 32.25320474905598, 35.7245203281876, 36.07622158511578, 33.87654016441272,
         30.78627693278345, 33.88427012181219, 32.27405933347808, 35.51982614315469, 34.39794280541302]
print(np.mean(PSNR1))
SSIM1 = [0.8570912480354309, 0.8639309406280518, 0.9516983032226562, 0.8962507843971252, 0.9524533748626709,
         0.8944130539894104, 0.9459761381149292, 0.8824405670166016, 0.9332714080810547, 0.8964347243309021,
         0.885223388671875, 0.9450889229774475, 0.9238075613975525, 0.9449077248573303, 0.8977339267730713,
         0.8874422907829285, 0.9010509252548218, 0.899117648601532, 0.9372555613517761, 0.9048942923545837]
print(np.mean(SSIM1))

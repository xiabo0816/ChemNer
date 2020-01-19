# -*- coding: UTF-8 -*-

import numpy as np
import os
import argparse
import math
import random
import jieba
import codecs


def get_args_parser():
    parser = argparse.ArgumentParser(description='统计化学术语识别的HMM模型')
    parser.add_argument('-d', '--dictionary', type=str,
                        default='chemwords.dat', help='chemwords.dat')
    parser.add_argument('-m', '--morpheme', type=str,
                        default='chemmorpheme.dat', help='chemmorpheme.dat')
    parser.add_argument('-t', '--taggedterm', type=str,
                        default='chemtagged.dat', help='chemtagged.dat')
    parser.add_argument('-t1', '--testset', type=str,
                        default='chemtagged.dat', help='DEFAULT: chemtagged.dat.')
    parser.add_argument('-t2', '--testsent', type=str,
                        default='testsent.txt', help='DEFAULT: testsent.txt')
    parser.add_argument('-s', '--sentence', default='',
                        type=str, help='单句测试')
    return parser.parse_args()


def readmorpheme(path):
    result = {}
    with open(path, 'r', encoding='utf-8', errors="ignore") as fin:
        lines = fin.readlines()
        for line in lines:
            part = line.strip().split("\t")
            result[part[0]] = part[1]
    return result


def mmseg(term, morpheme, max_chars):
    # 定义一个空列表来存储分词结果
    seged = []
    n = 0
    while n < len(term):
        matched = 0
        for i in range(max_chars, 0, -1):
            s = term[n:n+i]
            # 判断所截取字符串是否在分词词典和停用词典内
            if s in morpheme:
                seged.append(s)
                matched = 1
                n = n+i
                break

        if not matched:
            seged.append(term[n])
            n = n+1

    return seged


def parse(term, morpheme, max_chars):
    seged = mmseg(term, morpheme, max_chars)
    cat = []
    for mor in seged:
        if mor in morpheme:
            cat.append(morpheme[mor])
        else:
            return False, cat, seged
    return True, cat, seged


def readdict(morpheme, path, max_chars):
    result = {}
    with open(path, 'r', encoding='utf-8', errors="ignore") as fin:
        lines = fin.readlines()
        for line in lines:
            flag, lab, mor = parse(line.strip(), morpheme, max_chars)
            if flag:
                result[line.strip()] = {'label': lab, 'morpheme': mor}
    return result


def CountPI(dictionary):
    total = 0
    wulala = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    result = np.zeros(6, dtype=float)
    for ci in dictionary:
        for value in dictionary[ci]['label']:
            result[wulala[value[0]]] += 1
            total += 1
    for i in range(0, 6):
        result[i] = result[i]/total
    return result


def CountA(dictionary):
    wulala = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    result = np.zeros((6, 6), dtype=float)
    count = np.zeros(6, dtype=float)
    for ci in dictionary:
        value = dictionary[ci]['label']
        for v in range(0, len(value) - 1):
            i = value[v]
            j = value[v+1]
            result[wulala[i]][wulala[j]] += 1
            count[wulala[i]] += 1

    for i in range(0, 6):
        for j in range(0, 6):
            result[i][j] = result[i][j]/count[i]
    return result


def CountB(morpheme, morpheme_idx, dictionary):
    wulala = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    count = np.zeros(6, dtype=float)
    result = np.ones((6, len(morpheme)), dtype=float)

    for ci in dictionary:
        for mor in dictionary[ci]['morpheme']:
            result[wulala[morpheme[mor]]][morpheme_idx[mor]] += 1
            count[wulala[morpheme[mor]]] += 1

    for i in range(0, 6):
        for j in range(0, len(morpheme_idx)):
            result[i][j] = result[i][j]/count[i]
    return result


def calc_alpha(pi, A, B, Q, alpha, wula):
    """
    计算前向概率α的值
    pi：初始的随机概率值
    A：状态转移矩阵
    B: 状态和观测值之间的转移矩阵
    Q: 观测值列表
    alpha：前向概率alpha矩阵
    """

    # 1. 初始一个状态类别的顺序
    n = len(A)
    n_range = range(n)

    # 2. 更新初值(t=1)
    for i in n_range:
        if Q[0] not in wula:
            return False
        alpha[0][i] = pi[i] * B[i][wula[Q[0]]]

    # 3. 迭代更新其它时刻
    T = len(Q)
    tmp = [0 for i in n_range]
    for t in range(1, T):
        for i in n_range:
            # 1. 计算上一个时刻t-1累积过来的概率值
            for j in n_range:
                tmp[j] = alpha[t - 1][j] * A[j][i]

            # 2. 更新alpha的值
            # alpha[t][i] = np.sum(tmp) * B[i][wula[Q[i]]]
            if Q[t] not in wula:
                return False
            alpha[t][i] = np.sum(tmp) * B[i][wula[Q[t]]]

    return True

def calc_alpha_log(pi, A, B, Q, alpha, wula):
    """
    计算前向概率α的log值
    pi：初始的随机概率值
    A：状态转移矩阵
    B: 状态和观测值之间的转移矩阵
    Q: 观测值列表
    alpha：前向概率alpha矩阵
    """
    # 1. 初始一个状态类别的顺序
    n = len(A)
    n_range = range(n)

    # 2. 更新初值(t=1)
    for i in n_range:
        if Q[0] not in wula:
            return False
        alpha[0][i] = pi[i] * B[i][wula[Q[0]]]

    # 3. 迭代更新其它时刻
    T = len(Q)
    tmp = [0 for i in n_range]
    for t in range(1, T):
        for i in n_range:
            # 1. 计算上一个时刻t-1累积过来的概率值
            for j in n_range:
                tmp[j] = alpha[t - 1][j] * A[j][i]

            # 2. 更新alpha的值
            # alpha[t][i] = np.sum(tmp) * B[i][wula[Q[i]]]
            if Q[t] not in wula:
                return False
            alpha[t][i] = np.sum(tmp) * B[i][wula[Q[t]]]

    return True

def prob(Q, PI, A, B, morpheme_idx):
    if len(Q) == 1:
        return 99999
    alpha = np.zeros((len(Q), len(A)))
    # 开始计算
    f = calc_alpha(PI, A, B, Q, alpha, morpheme_idx)
    if not f:
        return 99999
    # 输出最终结果
    # print(alpha)

    # 计算最终概率值：
    p = 0
    for i in alpha[-1]:
        p += i

    if p == 0:
        return 99999

    # p = math.log(p, math.pow(len(Q), 10))
    p = math.log(p, 1/len(Q))
    return p


def getratio(threshold, lines, PI, A, B, morpheme, morpheme_idx, morpheme_maxchars, morphemeCountRange):
    correct = 0
    total = 0

    for line in lines:
        part = line.strip().split("\t")
        seged = mmseg(part[0], morpheme, morpheme_maxchars)
        if len(seged) in morphemeCountRange:
            t = prob(seged, PI, A, B, morpheme_idx)
            a = part[1]
            p = 1 if t < threshold else 0
            if int(a) == int(p):
                correct += 1
            total += 1

    # print(threshold,'\t',correct,'\t',total,'\t',correct/total)
    return correct/total


def bestthreshold(start, stop, step, lines, PI, A, B, morpheme_idx, max_chars, morphemeCountRange):
    threshold = 0
    ratio = 0
    for i in np.arange(start, stop, step):
        t = getratio(i, lines, PI, A, B, morpheme, morpheme_idx, max_chars, morphemeCountRange)
        print(t, i)
        # input()
        if(t > ratio):
            threshold = i
            ratio = t

    return threshold, ratio

def getthreshold(seged, thresholds):
    if len(seged) == 0:
        return 0
    if len(seged) == 1:
        return 0
    if len(seged) > len(thresholds):
        return thresholds[1]
    else:
        return thresholds[len(seged)]

def testset(path, PI, A, B, morpheme_maxchars, thresholds, morphemeCountRange):
    correct = 0
    total = 0
    recall = 0

    with open(path, 'r', encoding='utf-8', errors="ignore") as fin:
        lines = fin.readlines()
        for line in lines[0:1000]:
            part = line.strip().split("\t")
            seged = mmseg(part[0], morpheme, morpheme_maxchars)

            if len(seged) in morphemeCountRange:
                t = prob(seged, PI, A, B, morpheme_idx)
                a = part[1]
                p = 1 if t < getthreshold(seged, thresholds) else 0

                if int(a) == int(p):
                    correct += 1
                total += 1

    # print(correct,'\t',total,'\t',correct/total)
    return correct/total

def findAllEntity(sent, PI, A, B, morpheme, morpheme_maxchars, thresholds, morphemeCountRange):
    seged = mmseg(sent, morpheme, morpheme_maxchars)
    result = []
    t = ''
    CF = '白板棒薄变标菜草程橙除大蛋的动度段发法分粉干工光果海黑红花黄灰接空拉蓝冷力粒料流绿马牛排片平气青清缺肉色石属树数水丝体无细香小心学叶用有长纸珠转子紫自'
    CC = '合化的和成-'
    CEND = '的并离制精联中盐混金1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CBEG = '物性盐'
    # print(seged)
    for step in [2,3,4]:
        for mor in range(0, len(seged) - step):
            w = ''
            for s in range(step):
                w += seged[mor + s]
            test = getratio(thresholds[step], [w+'\t1'], PI, A, B, morpheme, morpheme_idx, morpheme_maxchars, morphemeCountRange)
            if int(test) == 1:
                if t != '':
                    t += seged[mor + step - 1]
                else:
                    t += w
            else:
                if t != '':
                    k = t.strip(CF+CC)

                    k = k.lstrip(CBEG)
                    k = k.rstrip(CEND)
                    # if t != k:
                    #     print(t,k)
                    result.append(k)
                t = ''
    # print(result)
    return result

def testsent(path, PI, A, B, morpheme, morpheme_maxchars, thresholds, morphemeCountRange):
    correct = 0
    total   = 0
    recall  = 0
    wordlen = 0
    sentlen = 0
    with open(path, 'r', encoding='utf-8', errors="ignore") as fin,\
         open('chemner.err', 'w', encoding='utf-8', errors="ignore") as fout:
        lines = fin.readlines()
        for line in lines:
            part = line.strip().split("\t")
            wordlen += len(part[0])
            sentlen += len(part[1])
            if len(part[0]) > 100:
                continue
            entities = findAllEntity(part[1], PI, A, B, morpheme, morpheme_maxchars, thresholds, morphemeCountRange)
            if len(entities) != 0:
                recall += 1
            if part[0] in entities:
                correct += 1
            else:
                fout.write('{0}\n'.format(line.strip()))
                # fout.write('{0}\n{1}\n'.format(line.strip(), ' '.join(entities)))
            total += 1

        print(correct,'\t',recall,'\t',correct/recall)
        print(recall,'\t',total,'\t',recall/total)
        print(wordlen,'\t',len(lines),'\t',wordlen/len(lines))
        print(sentlen,'\t',len(lines),'\t',sentlen/len(lines))
    return correct/recall


if __name__ == '__main__':
    args = get_args_parser()
    # print(args)
    morpheme = {}
    morpheme = readmorpheme(args.morpheme)

    # 语素是关键词，自增整数是值
    morpheme_idx = {}
    count = 0
    for key in morpheme.keys():
        morpheme_idx[key] = count
        count += 1
    # 遍历分词字典，获得最大分词长度
    max_chars = 0
    for key in morpheme:
        if len(key) > max_chars:
            max_chars = len(key)

    dictionary = readdict(morpheme, args.dictionary, max_chars)
    # 从这里导出词典
    # with open("chemner.dict", "w", encoding='UTF-8') as f:
    #     for ci in dictionary:
    #         for i in range(len(dictionary[ci]['label'])):
    #             f.write('{0}/{1} '.format(dictionary[ci]['morpheme'][i], dictionary[ci]['label'][i]))
    #         f.write('\n')

    PI = CountPI(dictionary)
    print(PI)
    A = CountA(dictionary)
    print(A)
    B = CountB(morpheme, morpheme_idx, dictionary)
    print(B)
    # thresholds = [ 0, -1.7, -1.5, -1.6, -1.9, -2.3, -2.6, -2.7, -3]
    # thresholds = [0, 17.5, 14.7, 16.8, 18.7, 23.1, 23.4, 22, 23.4, 23.5]
    thresholds = [0, 18, 15, 15.8, 18.7, 23.1, 23.4, 22, 23.4, 23.5]
    print(thresholds)

    print('磷酸肌醇\t', prob(list('磷酸肌醇'), PI, A, B, morpheme_idx))
    print('一元二次方程\t', prob(list('一元二次方程'), PI, A, B, morpheme_idx))
    
    print(set(filter(None, findAllEntity('本发明公开了一种磷酸法生产饲料级磷酸二氢钾的生产工艺', PI, A, B, morpheme, max_chars, thresholds, range(len(thresholds))))))
    print(set(filter(None, findAllEntity('包括下述工艺步骤：合成钛酸锶前驱体溶液：将无水乙醇、异丙醇、钛酸正丁酯、四丁基氢氧化铵甲醇溶液在圆底烧瓶中温度为75', PI, A, B, morpheme, max_chars, thresholds, range(len(thresholds))))))
    print(set(filter(None, findAllEntity('间苯二甲酸乙二醇酯-5-磺酸钠生产工艺', PI, A, B, morpheme, max_chars, thresholds, range(len(thresholds))))))
    print(set(filter(None, findAllEntity('其中常规离子液体为1-丁基-3-甲基咪唑六氟磷酸盐、1-己基-3-甲基咪唑六氟磷酸盐、1-乙基-3甲基咪唑六氟磷酸盐或1-辛基-3-甲基咪唑六氟磷酸盐', PI, A, B, morpheme, max_chars, thresholds, range(len(thresholds))))))
    # exit()

    # 从这里选取阈值
    # thresholds保存了从2到n个语素的阈值，也就是，不同的语素个数有不同的阈值；语素个数从2开始；
    # with open(args.taggedterm, 'r', encoding='utf-8', errors="ignore") as fin:
    #     lines = fin.readlines()
    #     rangemorphemecount = range(2,10)
    #     thresholds = np.zeros(10, dtype=float)
    #     thresholds[0] = 0
    #     print('rangemorphemecount: ???, thresholds[0]: 0')
    #     thresholds[1], ratio = bestthreshold(10, 30, 0.1,
    #                                         lines, PI, A, B, morpheme_idx, max_chars, rangemorphemecount)
    #     print('rangemorphemecount: all, thresholds[1]: ', thresholds[1])
    #     for i in rangemorphemecount:
    #         threshold, ratio = bestthreshold(10, 30, 0.1,
    #                                         lines, PI, A, B, morpheme_idx, max_chars, [i])
    #         print('rangemorphemecount: ', i, ', thresholds['+str(i-2)+']: ', threshold)
    #         thresholds[i] = threshold
    # print('thresholds保存了从2到n个语素的阈值，也就是:\n1.不同的语素个数有不同的阈值；\n2.语素个数从2开始；\n3.t[0]=0;\n4.t[1]=all/avg')
    # print(thresholds)
    # exit()

    print('术语判别测试: ')
    print('语素数: 2\t', testset(args.testset, PI, A, B, max_chars, thresholds, [2]))
    print('语素数: 3\t', testset(args.testset, PI, A, B, max_chars, thresholds, [3]))
    print('语素数: 4\t', testset(args.testset, PI, A, B, max_chars, thresholds, [4]))
    print('语素数: 5\t', testset(args.testset, PI, A, B, max_chars, thresholds, [5]))
    print('任意语素数: \t', testset(args.testset, PI, A, B, max_chars, thresholds, range(len(thresholds))))

    print('测试句子中的术语识别: ')
    print('二元贪婪识别: \t', testsent(args.testsent, PI, A, B, morpheme, max_chars, thresholds, range(len(thresholds))))
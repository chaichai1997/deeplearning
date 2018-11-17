# -*- coding: utf-8 -*-
import numpy as np

def loadDataSet():
    # 训练集文本
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'him', 'my'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'], ]
    # 文本所属分类
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec


class NBayes(object):
    def __init__(self):
        self.vocabulary = []  # 词典
        self.idf = 0  # idf权值向量
        self.tf = 0  # 训练集的权值矩阵
        self.tdm = 0  # P(x|y)
        self.Pcates = {}  # p(yi)类别字典
        self.labels = []  # 每个文本的分类
        self.doclength = 0  # 训练集文本数
        self.vocablen = 0  # 字典词长
        self.testset = 0  # 测试集

    """
     计算数据集在每个分类中的概率
    """
    def cate_prob(self,classVec):
        self.labels = classVec
        labletemps = set(self.labels)  # 获取全部分类
        for labletemp in labletemps:
            # self.labels.count(labletemp) 列表中重复的分类
            self.Pcates[labletemp] = float(self.labels.count(labletemp))/float(len(self.labels))

    """
      生成普通词频向量
    """
    def calc_wordfreq(self,trainset):
        self.idf = np.zeros([1, self.vocablen])  # 1*词典个数
        self.tf = np.zeros([self.doclength, self.vocablen])  # 训练集文件数*词典数
        for indx in range(self.doclength):
            for word in trainset[indx]:  # 找到文本的词在字典中的位置加一
                self.tf[indx, self.vocabulary.index(word)] += 1
            for singleword in set(trainset[indx]):
                self.idf[0, self.vocabulary.index(singleword)] +=  1

    """
     按分类累计计算向量空间的每维值
    """
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates), self.vocablen])
        sumlist = np.zeros([len(self.Pcates), 1])  # 统计每个分类的总值
        for i in range(self.doclength):  # 将同一类词别空间值加总
            self.tdm[self.labels[i]] += self.tf[i]
            sumlist[self.labels[i]] = np.sum(self.tdm[self.labels[i]])
        self.tdm = self.tdm/sumlist  # 生成P(x|yi)

    """
     将测试集映射到当前字典
    """
    def map2vocab(self, testdata):
        self.testset = np.zeros([1, self.vocablen])
        for word in testdata:
            self.testset[0, self.vocabulary.index(word)] += 1

    """
     导入和训练数据集，生成算法所需数据结构
    """
    def train_set(self, trainset, classVec):
        self.cate_prob(classVec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        self.calc_wordfreq(trainset)
        self.build_tdm()

    """
     预测函数，输出预测结果
    """
    def predict(self, testset):
        if np.shape(testset)[1] != self.vocablen:
            print('输出错误')
            exit(0)
        predvalue = 0
        predclass = ""
        for tdm_vect,keyclass in zip(self.tdm, self.Pcates):
            temp = np.sum(testset*tdm_vect*self.Pcates[keyclass])
            if temp > predvalue:
                predclass = keyclass
                predvalue = temp
            return predclass


if __name__ == '__main__':
    dataSet, listClasses = loadDataSet()
    nb = NBayes()
    nb.train_set(dataSet, listClasses)
    nb.map2vocab(dataSet[0])
    print(nb.predict(nb.testset))

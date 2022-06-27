import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.datasets import load_iris

# iris 꽃을 불러오기
iris = load_iris()

# 해당 변수들에 iris 데이터들을 불러오기
data = iris.data
target = iris.target
target_name = iris.target_names

# knn.py 파일의 KNN 클래스 형 변수 생성
knn = KNN()

# 150개의 iris 꽃을 i % 15 == 0 인 꽃들을 test, 나머지를 train data로 분류
for i in range(0, 150):
    add_data = data[i]
    add_target = target[i]
    if i % 15 != 0:
        knn.train_data = np.append(knn.train_data, np.array([add_data]), axis=0)
        knn.train_target = np.append(knn.train_target, np.array(add_target))
    else:
        knn.test_data = np.append(knn.test_data, np.array([add_data]), axis=0)
        knn.test_target = np.append(knn.test_target, np.array(add_target))

# 출력
print('majority_vote')

# test 데이터들을 각각 neighbors 를 구하여 majority_vote algorithm을 이용하여 꽃 판단하기
for i in range(len(knn.test_data)):
    neighbor = knn.get_neighbor(i)
    print('Test Data Index :', end=' ')
    print(i, end='\t')
    print('Computed class :', end=' ')
    index = int(knn.majority_vote(neighbor))
    print(target_name[index], end='\t')
    print('True class : ', end=' ')
    index = int(knn.test_target[i])
    print(target_name[index])

print()
print('weighted_majority_vote')

# test 데이터들을 각각 neighbors 를 구하여 weighted_majority_vote algorithm을 이용하여 꽃 판단하기
for i in range(len(knn.test_data)):
    neighbor = knn.get_neighbor(i)
    print('Test Data Index :', end=' ')
    print(i, end='\t')
    print('Computed class :', end=' ')
    index = int(knn.weighted_majority_vote(neighbor))
    print(target_name[index], end='\t')
    print('True class : ', end=' ')
    index = int(knn.test_target[i])
    print(target_name[index])

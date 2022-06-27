import numpy as np

# numpy library의 유클리드 거리법을 이용하여 거리를 구하는 함수
def distance(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(a - b)

# KNN 클래스
class KNN:
    def __init__(self):
        # 구하고자 하는 이웃의 갯수
        self.K = 9
        # train_data numpy array 초기화
        self.train_data = np.empty((0, 4))
        # train_target numpy array 초기화
        self.train_target = np.array([])
        # test_data numpy array 초기화
        self.test_data = np.empty((0, 4))
        # test_target numpy array 초기화
        self.test_target = np.array([])

    # neighbors 를 구하는 함수
    def get_neighbor(self, test_index, distance=distance):
        # distance 를 저장할 list 초기화
        distances = []
        # 학습되어있는 데이터들과의 거리를 구하여 {dist, train_target}을 list에 저장
        for i in range(len(self.train_data)):
            dist = distance(self.test_data[test_index], self.train_data[i])
            distances.append((dist, self.train_target[i]))

        # dist 를 기준으로 오름차순 정렬
        distances.sort(key=lambda x: x[0])
        # K 개만큼 인덱싱 하여 neighbors 에 저장 및 반환
        neighbors = distances[:self.K]

        return neighbors

    # majority_vote algorithm function
    def majority_vote(self, neighbors):
        # 꽃의 종류의 갯수를 저장할 list 초기화
        cnt = [0, 0, 0]
        # neighbors 의 종류의 갯수를 계산
        for i in range(len(neighbors)):
            if neighbors[i][1] == 0:
                cnt[0] += 1
            elif neighbors[i][1] == 1:
                cnt[1] += 1
            elif neighbors[i][1] == 2:
                cnt[2] += 1

        # numpy 내장함수를 이용하여 cnt list 중 가장 큰 값을 갖는 인덱스 반환
        return np.argmax(cnt)

    # weighted_majority_vote function
    def weighted_majority_vote(self, neighbors):
        # 꽃의 종류의 갯수와 가중치를 저장할 list 초기화
        cnt = [[0, 0], [0, 0], [0, 0]]
        # neighbors 의 종류의 갯수와 거리들의 합을 계산
        for i in range(len(neighbors)):
            if neighbors[i][1] == 0:
                cnt[0][0] += 1
                cnt[0][1] += neighbors[i][0]
            elif neighbors[i][1] == 1:
                cnt[1][0] += 1
                cnt[1][1] += neighbors[i][0]
            elif neighbors[i][1] == 2:
                cnt[2][0] += 1
                cnt[2][1] += neighbors[i][0]

        # neighbors 의 갯수가 0이면 거리를 INT_MAX 로 초기화하고, 그렇지 않다면 평균값 계산 
        if cnt[0][0] != 0:
            cnt[0][1] /= cnt[0][0]
        else:
            cnt[0][1] = 0x7fffffff

        if cnt[1][0] != 0:
            cnt[1][1] /= cnt[1][0]
        else:
            cnt[1][1] = 0x7fffffff

        if cnt[2][0] != 0:
            cnt[2][1] /= cnt[2][0]
        else:
            cnt[2][1] = 0x7fffffff
        
        # 거리들의 평균값을 저장할 배열 초기화 및 값 입력 후 반환
        weighted_cnt = [cnt[0][1], cnt[1][1], cnt[2][1]]
        
        return np.argmin(weighted_cnt)

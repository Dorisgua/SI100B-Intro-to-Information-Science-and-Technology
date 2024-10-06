from check import CheckWin
from copy import deepcopy
# import matplotlib.pyplot as plt

def solve(filename):
    with open(filename) as f:
        strs = f.readlines()
    nameList = strs[1].strip('\n').split(',')
    dealer = nameList.index(strs[0].strip('\n'),0,4)
    tileList = [[] for i in range(4)]
    cnt = [0 for i in range(4)]
    tileInHand = [[] for i in range(4)]
    # Task 1
    winTime = [0 for i in range(4)]
    # Task 2
    winTileTime = {}
    # Task 3
    scoreList = []
    scoreList.append([50000 for i in range(4)])
    lastScore = [50000 for i in range(4)]

    for x in strs:
        if x == strs[0] or x == strs[1]:
            continue
        s = x.strip('\n').split(',')
        for i in range(4):
            if s[i] != '':
                tileList[i].append(s[i])
    endFlag = False
    gameNum = 0
    wall = 136

    with open("winner.csv","w") as f:
        while not endFlag:
            for i in range(dealer, dealer + 4):
                x = i % 4
                if cnt[x] < len(tileList[x]):
                    # Draw a tile
                    if len(tileInHand[x]) < 14:
                        tileInHand[x].append(tileList[x][cnt[x]])
                        wall -= 1
                        if len(tileInHand[x]) == 14:
                            point = CheckWin(tileInHand[x])
                            if point > 0:
                                # print(tileInHand[x])
                                gameNum += 1
                                # Task 1
                                winTime[x] += 1
                                f.write("{}\n".format(nameList[x]))
                                # Task 2
                                if tileList[x][cnt[x]] in winTileTime:
                                    winTileTime[tileList[x][cnt[x]]] += 1
                                else:
                                    winTileTime[tileList[x][cnt[x]]] = 1
                                # Task 3
                                lastScore[x] += point
                                for i in range(4):
                                    if i != x:
                                        lastScore[i] -= point // 3
                                scoreList.append(deepcopy(lastScore))
                                # Initialize
                                wall = 136
                                dealer += 1
                                for i in range(4):
                                    tileInHand[i] = []
                                cnt[x] += 1
                                break
                            # If he or she does not win, drop a tile
                            cnt[x] += 1
                            tileInHand[x].remove(tileList[x][cnt[x]])
                        cnt[x] += 1
                        # DRAW
                        if wall == 0:
                            gameNum += 1
                            # Task 1
                            f.write("Draw\n")
                            # Task 3
                            scoreList.append(deepcopy(lastScore))
                            # Initialize
                            wall = 136
                            dealer += 1
                            for i in range(4):
                                tileInHand[i] = []
                            break
                else:
                    endFlag = True
                    break
        f.write("\n")
        # Task 1
        for i in range(4):
            f.write("{},{}%\n".format(nameList[i],"%.2f" % (100 * winTime[i] / gameNum)))

    # Task 2
    ans = sorted(winTileTime.items(), key = lambda x: (-x[1], x[0]))
    with open("tile.csv", "w") as f:
        for x, y in ans:
            f.write("{},{}%\n".format(x, "%.2f" % (100 * y / gameNum)))
    # Task 3
    with open("battle.csv", "w") as f:
        gameCnt = -1
        for score in scoreList:
            gameCnt += 1
            f.write("Game {}\n".format(gameCnt))
            for i in range(4):
                f.write("{},{}\n".format(nameList[i], int(score[i])))
            f.write("\n")
    """
    for i in range(4):
        x = [j for j in range(gameCnt + 1)]
        y = [score[i] for score in scoreList]
        plt.plot(x, y)
    plt.xticks(x, x)
    plt.title('Result')
    plt.legend(nameList)
    plt.show()
    return plt
    """
# solve("test.csv")
# Task 1(50 pts):
# 输出每一局赢的人是谁，输出胜率，保留两位小数，百分比。
# Task 2(20 pts):
# 将所有和过的牌的和占比输出成csv形式，降序，相同按字典序。
# Task 3(30 pts):
# 假定每位选手初始分50000分，基本分3600分，有一个刻子得分*1.5，全是顺子得7200分，有一个字牌刻子加1200（先计算所有的加法）。得分将会从其他三家均匀收取。
# 将每一局得分输出成csv形式，用matplotlib画出得分图像。


vis = []
def Dfs(flag, chi, pung, honor, maj):
    # Win
    if(flag == True and chi + pung == 4):
        if chi == 4:
            return 7200
        return (3600 + 1200 * honor) * (1.5 ** pung)

    # If sb. wins, record the point
    point = 0
    for i in range(14): 
        if not vis[i]:
            for j in range(i+1, 14): 
                # They are in the same suit
                if (not vis[j]) and maj[i][1] == maj[j][1]:
                    for k in range(j + 1, 14):
                        if (not vis[k]) and maj[j][1] == maj[k][1]:
                            # Pung First
                            if maj[i][0] == maj[j][0] and maj[j][0] == maj[k][0]:
                                vis[i] = vis[j] = vis[k] = True
                                if maj[i][1] == 'z':
                                    point = Dfs(flag, chi, pung + 1, honor + 1, maj)
                                else:
                                    point = Dfs(flag, chi, pung + 1, honor, maj)
                                if point > 0:
                                    return point
                                vis[i] = vis[j] = vis[k] = False

                            # Chow
                            if maj[i][1] != 'z' and int(maj[i][0]) + 1 == int(maj[j][0]) and int(maj[j][0]) + 1 == int(maj[k][0]):
                                vis[i] = vis[j] = vis[k] = True
                                point = Dfs(flag, chi + 1, pung, honor, maj)
                                if point > 0:
                                    return point
                                vis[i] = vis[j] = vis[k] = False
                    
                    # Pair
                    if (not flag) and maj[i][0] == maj[j][0]:
                        vis[i] = vis[j] = True
                        point = Dfs(True, chi, pung, honor, maj) 
                        if point > 0:
                            return point
                        vis[i] = vis[j] = False
    # Fail to win
    return 0


def CheckWin(maj):
    if len(maj) != 14:
        raise RuntimeError("The number of tiles is NOT equal to 14!")
    global vis
    vis = [False for i in range(14)]
    cnt = {}
    # Sort the series
    for i in range(14):
        for j in range(i + 1, 14):
            if maj[i][1] > maj[j][1]:
                maj[i], maj[j] = maj[j], maj[i]
            elif maj[i][1] == maj[j][1] and maj[i][0] > maj[j][0]:
                maj[i], maj[j] = maj[j], maj[i]
        if maj[i] in cnt:
            cnt[maj[i]] += 1
        else:
            cnt[maj[i]] = 1
        if cnt[maj[i]] > 4:
            raise RuntimeError("There are more than 4 same tiles!")

    return Dfs(False, 0, 0, 0, maj)

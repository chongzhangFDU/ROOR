def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError(f"invalid truth value {val}")


def transitive_closure_dfs(edges):
    """ 求传递闭包
    """
    if edges == []: return []
    n = max(max(i, j) for i, j in edges) + 1
    visited = [False for _ in range(n)]
    graph = {node: [] for node in range(n)}
    reachable = {node: [] for node in range(n)}

    # 构建图的邻接表
    for u, v in edges:
        graph[u].append(v)

    # DFS方式计算node和node的后继的后继集合
    def dfs(node):
        if not visited[node]:
            visited[node] = True
            for neighbour in graph[node]:
                reachable[node].append(neighbour)
                dfs(neighbour)
                for neighbour_reach in reachable[neighbour]:
                    if not neighbour_reach in reachable[node]:
                        reachable[node].append(neighbour_reach)

    # 对于每个节点执行DFS
    for node in range(n):
        dfs(node)

    # 从可达性集合中提取传递闭包的边
    new_edges = []
    for i in range(n):
        for j in reachable[i]:
            new_edges.append((i, j))

    return new_edges

if __name__ == '__main__':
    edges = [0, 16], [1, 17], [2, 9], [3, 20], [4, 27], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 3], [16, 28], [17, 26], [18, 19], [19, 0], [20, 21], [21, 22], [22, 23], [23, 24], [24, 4], [25, 18], [26, 29], [27, 32], [28, 31], [29, 2], [31, 1], [32, 5]
    print(transitive_closure_dfs(edges))

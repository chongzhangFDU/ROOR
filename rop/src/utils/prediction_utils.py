from collections import defaultdict

def decode(n, edges):
    """ 输入形如{(i, j): confidence_score, ...}的dict，节点下标为0,...,n-1
        输出最终的预测结果[[i, j], ...]
    """
    def find_cycle(n, edges):
        """ 返回找到的第一个自环
        """
        graph = defaultdict(list)
        for i, j in edges.keys():
            graph[i].append(j)
        visited = [False] * n
        rec_stack = [False] * n # node in path的索引
        def dfs(vertex, path):
            visited[vertex] = True
            rec_stack[vertex] = True
            path.append(vertex)
            for neighbor in graph[vertex]:
                if not visited[neighbor]: # 从neighbor继续往下找
                    ret = dfs(neighbor, path)
                    if ret is not None: return ret
                elif rec_stack[neighbor]: # neighbor出现在了当前路径path中，则从path找到环的起点，并返回
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]  # 返回发现的环
            path.pop()
            rec_stack[vertex] = False
            return None
        for node in range(n):
            if not visited[node]:
                cycle = dfs(node, [])
                if cycle:
                    return cycle  # 发现第一个环，返回结果
        return None  # 没有环
    
    # 不断寻找环，如果找到环，就删掉环中confidence最低的一条边
    while True:
        cycle = find_cycle(n, edges)
        if cycle is None: break
        min_edge, min_conf = None, 1e8
        for i, j in zip(cycle, cycle[1:]+cycle[:1]):
            if edges[(i, j)] < min_conf:
                min_edge = (i, j)
                min_conf = edges[(i, j)]
        del edges[min_edge]
        
    return [list(p) for p in sorted(edges.keys())]

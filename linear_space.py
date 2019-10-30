def score(a, b, alphabet, matrix):
    i = alphabet.index(a) if a is not None else -1
    j = alphabet.index(b) if b is not None else -1
    return matrix[i][j]


def find_max_score(matrix):
    max_score = 0
    entry = (len(matrix) - 1, len(matrix[0]) - 1)
    for i, row in enumerate(matrix):
        for j, score in enumerate(row):
            if score > max_score:
                max_score = score
                entry = (i, j)
    return entry


# Linear space dynamic programming method


def score_prefix_alignment(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    # Keep track of starting point
    V = [[(0, (0, 0))] * n for _ in range(2)]
    def S(a, b): return score(a, b, alphabet, scores)  # noqa: E731

    for j in range(1, n):
        V[0][j] = max(
            (V[0][j-1][0] + S(None, t[j-1]), V[0][j-1][1]),
            (0, (0, j)),
        )

    for i in range(1, m):
        V[1][0] = max(
            (V[0][0][0] + S(s[i-1], None), V[0][0][1]),
            (0, (i, 0)),
        )
        for j in range(1, n):
            V[1][j] = max(
                (V[0][j-1][0] + S(s[i-1], t[j-1]), V[0][j-1][1]),
                (V[0][j][0] + S(s[i-1], None), V[0][j][1]),
                (V[1][j-1][0] + S(None, t[j-1]), V[1][j-1][1]),
            )
        for i in range(n):
            V[0][i] = V[1][i]
    return V[0]


def nw_align(alphabet, scores, s, t):
    def S(a, b): return score(a, b, alphabet, scores)  # noqa: E731
    b = float('-inf')
    if len(t) == 1 and len(s) > 1:
        W, Z = nw_align(alphabet, scores, t, s)
        return Z, W
    # Try to align ---s--- with t
    Z = []
    W = list(range(len(t)))
    for i in range(len(t)):
        u = S(s, t[i]) + sum(S(None, t[j]) for j in range(len(t)) if j != i)
        if u > b:
            b = u
            Z = [-1] * i + [0] + [-1] * (len(t) - i - 1)
    return Z, W


def i_add(Z, i):
    return [(z if z == -1 else z+i) for z in Z]


def global_align(alphabet, scores, X, Y):
    Z = []
    W = []
    if len(X) == 0:
        Z = [-1] * len(Y)
        W = list(range(len(Y)))
    elif len(Y) == 0:
        Z = list(range(len(X)))
        W = [-1] * len(X)
    elif len(X) == 1 or len(Y) == 1:
        Z, W = nw_align(alphabet, scores, X, Y)
    else:
        xlen = len(X)
        xmid = xlen // 2
        score_l = score_prefix_alignment(alphabet, scores, X[:xmid], Y)
        score_r = score_prefix_alignment(alphabet, scores, X[xmid:][::-1], Y[::-1])[::-1]
        max_score = float('-inf')
        ymid = 0
        for i, (a, b) in enumerate(zip(score_l, score_r)):
            if a + b > max_score:
                ymid = i
                max_score = a + b
        Z1, W1 = global_align(alphabet, scores, X[:xmid], Y[:ymid])
        Z2, W2 = global_align(alphabet, scores, X[xmid:], Y[ymid:])
        Z = Z1 + i_add(Z2, xmid)
        W = W1 + i_add(W2, ymid)
    return Z, W


def find_local_max(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[0] * n for _ in range(2)]
    def S(a, b): return score(a, b, alphabet, scores)  # noqa: E731
    best_score = float('-inf')
    best_entry = None

    for j in range(1, n):
        V[0][j] = max(V[0][j-1] + S(None, t[j-1]), 0)
        if V[0][j] > best_score:
            best_score = V[0][j]
            best_entry = (0, j)

    for i in range(1, m):
        V[1][0] = max(V[0][0] + S(s[i-1], None), 0)
        if V[1][0] > best_score:
            best_score = V[1][0]
            best_entry = (i, 0)

        for j in range(1, n):
            V[1][j] = max(
                V[0][j-1] + S(s[i-1], t[j-1]),
                V[0][j] + S(s[i-1], None),
                V[1][j-1] + S(None, t[j-1]),
                0,
            )
            if V[1][j] > best_score:
                best_score = V[1][j]
                best_entry = (i, j)
        for i in range(n):
            V[0][i] = V[1][i]
    return best_entry


def score_indexes(alphabet, scores, s, t, Z, W):
    total = 0
    for i, j in zip(Z, W):
        a = s[i] if i != -1 else None
        b = t[j] if j != -1 else None
        total += score(a, b, alphabet, scores)
    return total


def local_align(alphabet, scores, s, t):
    ei, ej = find_local_max(alphabet, scores, s, t)
    si, sj = find_local_max(alphabet, scores, s[:ei][::-1], t[:ej][::-1])
    # If one of the substrings is empty, then just
    # give up now.
    if ei - si == ei or ej - sj == ej:
        return 0, [], []
    Z, W = global_align(alphabet, scores, s[ei-si:ei], t[ej-sj:ej])
    Z = i_add(Z, ei-si)
    W = i_add(W, ej-sj)
    score = score_indexes(alphabet, scores, s, t, Z, W)
    return score, Z, W

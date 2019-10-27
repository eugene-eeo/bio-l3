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


def basic_align(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[0] * n for _ in range(m)]
    M = [[' ']*n for _ in range(m)]
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731

    for i in range(1, m):
        V[i][0], M[i][0] = max(
            (V[i-1][0] + S(s[i-1], None), 'U'),
            (0, 'T'),
        )

    for j in range(1, n):
        V[0][j], M[0][j] = max(
            (V[0][j-1] + S(None, t[j-1]), 'L'),
            (0, 'T'),
        )

    for i in range(1, m):
        for j in range(1, n):
            V[i][j], M[i][j] = max(
                (V[i-1][j-1] + S(s[i-1], t[j-1]), 'D'),
                (V[i-1][j] + S(s[i-1], None), 'U'),
                (V[i][j-1] + S(None, t[j-1]), 'L'),
                (0, 'T'),
            )

    # reconstruct
    s_idxs = []
    t_idxs = []
    i, j = find_max_score(V)

    while i != 0 and j != 0:
        c = M[i][j]
        if c == 'D':
            i -= 1
            j -= 1
            s_idxs.append(i)
            t_idxs.append(j)
        elif c == 'U':
            i -= 1
            s_idxs.append(i)
            t_idxs.append(-1)
        elif c == 'L':
            j -= 1
            s_idxs.append(-1)
            t_idxs.append(j)
        else:  # 'T'
            break

    s_idxs.reverse()
    t_idxs.reverse()

    return V[i][j], s_idxs, t_idxs


def score_prefix_alignment(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[0] * n for _ in range(2)]
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731

    for j in range(1, n):
        V[0][j] = V[0][j-1] + S(None, t[j-1])

    for i in range(1, m):
        V[1][0] = V[0][0] + S(s[i-1], None)
        for j in range(1, n):
            V[1][j] = max(
                V[0][j-1] + S(s[i-1], t[j-1]),
                V[0][j] + S(s[i-1], None),
                V[1][j-1] + S(None, t[j-1]),
            )
        for i in range(n):
            V[0][i] = V[1][i]
    return V[1]


def nw_align(alphabet, scores, s, t):
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731
    if len(s) == 1:
        # Try to align ---s--- with t
        Z = ""
        W = t
        b = float('-inf')
        for i in range(len(t)):
            x = ("-" * i) + s + ("-" * (len(t) - i - 1))
            u = S(s, t[i]) + sum(S(None, t[j]) for j in range(len(t)) if j != i)
            if u > b:
                b = u
                Z = x
        return Z, W
    else:
        # Try to align s with --t--
        Z = s
        W = ""
        b = float('-inf')
        for i in range(len(s)):
            x = ("-" * i) + t + ("-" * (len(s) - i - 1))
            u = S(s[i], t) + sum(S(s[j], None) for j in range(len(s)) if j != i)
            if u > b:
                b = u
                W = x
        return Z, W


def global_align(alphabet, scores, X, Y):
    Z = ""
    W = ""
    if len(X) == 0:
        Z = "-" * len(Y)
        W = Y
    elif len(Y) == 0:
        Z = X
        W = "-" * len(X)
    elif len(X) == 1 or len(Y) == 1:
        Z, W = nw_align(alphabet, scores, X, Y)
    else:
        xlen = len(X)
        xmid = xlen // 2
        score_l = score_prefix_alignment(alphabet, scores, X[:xmid], Y)
        score_r = score_prefix_alignment(alphabet, scores, X[xmid:][::-1], Y[::-1])[::-1]
        ymid = max(range(len(Y)), key=lambda i: score_l[i] + score_r[i])
        Z1, W1 = global_align(alphabet, scores, X[:xmid], Y[:ymid])
        Z2, W2 = global_align(alphabet, scores, X[xmid:], Y[ymid:])
        Z = Z1 + Z2
        W = W1 + W2
    return Z, W


def find_local_max(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[0] * n for _ in range(2)]
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731
    best_score = float('-inf')
    best_entry = None

    for j in range(1, n):
        V[0][j] = max(V[0][j-1] + S(None, t[j-1]), 0)
        best_score, best_entry = max(
            (best_score, best_entry),
            (V[0][j], (0, j)),
        )

    for i in range(1, m):
        V[1][0] = max(V[0][0] + S(s[i-1], None), 0)
        best_score, best_entry = max(
            (best_score, best_entry),
            (V[1][0], (i, 0)),
        )

        for j in range(1, n):
            V[1][j] = max(
                V[0][j-1] + S(s[i-1], t[j-1]),
                V[0][j] + S(s[i-1], None),
                V[1][j-1] + S(None, t[j-1]),
                0,
            )
            best_score, best_entry = max(
                (best_score, best_entry),
                (V[1][j], (i, j)),
            )
        for i in range(n):
            V[0][i] = V[1][i]
    return best_entry


def local_align(alphabet, scores, s, t):
    ei, ej = find_local_max(alphabet, scores, s, t)
    si, sj = find_local_max(alphabet, scores, s[:ei][::-1], t[:ej][::-1])
    return global_align(alphabet, scores, s[ei-si:ei], t[ej-sj:ej])

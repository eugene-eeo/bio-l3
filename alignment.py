from collections import defaultdict


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
    bscore = V[i][j]

    while i != 0 or j != 0:
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

    return bscore, s_idxs, t_idxs


# Linear space dynamic programming method


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
    return V[0]


def nw_align(alphabet, scores, s, t):
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731
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
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731
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


# Heuristic method


def compute_index_table(ktup, s):
    index_table = {}
    for i in range(len(s) - ktup + 1):
        sub = s[i:i+ktup]
        if sub not in index_table:
            index_table[sub] = [i]
        else:
            index_table[sub].append(i)
    return index_table


def score_alignment(alphabet, scores, s, t):
    return sum(score(s[i], t[i], alphabet, scores) for i in range(len(s)))


def find_seeds(alphabet, scores, ktup, index_table, t):
    MIN = (float('+inf'), float('+inf'))
    MAX = (float('-inf'), float('-inf'))
    min_table = {}
    max_table = {}
    score_table = defaultdict(int)

    for j in range(len(t) - ktup + 1):
        sub = t[j:j+ktup]
        matches = index_table.get(sub, ())
        for i in matches:
            d = i - j
            score_table[d] += score_alignment(alphabet, scores, sub, sub)
            min_table[d] = min(min_table.get(d, MIN), (i, j))
            max_table[d] = max(max_table.get(d, MAX), (i, j))

    best_diagonals = sorted(
        score_table.keys(),
        key=score_table.__getitem__,
        reverse=True,
    )[:10]
    return [(min_table[d], max_table[d]) for d in best_diagonals]


def banded_dp(alphabet, scores, s, t, w):
    m = len(s)
    n = len(t)
    w = min(w, n)
    V = [[0] * min(n-i+1, w) for i in range(min(m+1, n+1))]
    M = [[' ']*min(n-i+1, w) for i in range(min(m+1, n+1))]
    S = lambda a, b: score(a, b, alphabet, scores)  # noqa: E731

    for j in range(1, w):
        V[0][j], M[0][j] = max(
            (V[0][j-1] + S(None, t[j-1]), 'L'),
            (0, 'T'),
        )

    for i in range(1, min(m+1, n+1)):
        # V[i][k] holds the entry for V'[i][i+k]
        # left case: only can use D and U
        V[i][0], M[i][0] = max(
            (V[i-1][0] + S(s[i-1], t[i-1]), 'D'),
            (V[i-1][1] + S(s[i-1], None) if len(V[i-1]) >= 2 else float('-inf'), 'U'),
            (0, 'T'),
        )
        cols = min(n-i+1, w)
        # middle case
        for j in range(1, cols):
            V[i][j], M[i][j] = max(
                (V[i-1][j-1] + S(s[i-1], t[i+j-1]), 'D'),
                (V[i][j-1] + S(None, t[i+j-1]), 'L'),
                (0, 'T'),
            )
            # right case: we have no access to the top entry
            if cols == w and j < w-1:
                continue
            # otherwise we can use U
            V[i][j], M[i][j] = max(
                (V[i][j], M[i][j]),
                (V[i-1][j] + S(s[i-1], None), 'U'),
            )

    i, j = find_max_score(V)
    bscore = V[i][j]
    s_idxs = []
    t_idxs = []

    # Need to translate i and j into the normal indices
    while i != 0 or j != 0:
        c = M[i][j]
        if c == 'D':
            i -= 1
            s_idxs.append(i)
            t_idxs.append(i+j)
        elif c == 'U':
            i -= 1
            j += 1
            s_idxs.append(i)
            t_idxs.append(-1)
        elif c == 'L':
            j -= 1
            s_idxs.append(-1)
            t_idxs.append(i+j)
        else:  # 'T'
            break

    s_idxs.reverse()
    t_idxs.reverse()
    return bscore, s_idxs, t_idxs


def fasta_alt(alphabet, scores, s, t):
    w = 12
    ktup = 2

    it = compute_index_table(ktup, s)
    best = None
    for (si, sj), (ei, ej) in find_seeds(alphabet, scores, ktup, it, t):
        curr = banded_dp(alphabet, scores, s[si:ei+ktup], t[sj:ej+ktup], w)
        if best is None or curr[0] > best[0]:
            best = (curr[0],
                    i_add(curr[1], si),
                    i_add(curr[2], sj))
    return best

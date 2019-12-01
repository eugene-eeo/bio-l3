from alignment2 import dynprog, dynproglin, heuralign


params = ("ABC", [[1, -1, -2, -1], [-1, 2, -4, -1], [-2, -4, 3, -2], [-1, -1, -2, 0]], "AABBAACA", "CBACCCBA")

for f in [dynprog, dynproglin, heuralign]:
    a = f(*params)
    print("Score:  ", a[0])
    print("Indices:", a[1], a[2])


alphabet, scores = "ABCD", [
    [1, -5, -5, -5, -1],
    [-5, 1, -5, -5, -1],
    [-5, -5, 5, -5, -4],
    [-5, -5, -5, 6, -4],
    [-1, -1, -4, -4, -9],
]

params2 = [
    ("AAAAACCDDCCDDAAAAACC", "CCAAADDAAAACCAAADDCCAAAA"),
    ("AACAAADAAAACAADAADAAA", "CDCDDD"),
    ("DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDCDADCDCDCDCD",
     "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBDCDCDCDCD"),
]

for s, t in params2:
    for f in [dynprog, dynproglin, heuralign]:
        a = f(alphabet, scores, s, t)
        print("Score:  ", a[0])
        print("Indices:", a[1], a[2])

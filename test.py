import random
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
    ("", "AAAAACCDDCCDDAAAAACC"),
    ("AAAAACCDDCCDDAAAAACC", ""),
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


print()
print("RANDOMISED TESTING")
print()


for x in range(5000):
    m = random.randint(0, 50)
    n = random.randint(0, 50)
    s = "".join(random.choice(alphabet) for _ in range(m))
    t = "".join(random.choice(alphabet) for _ in range(n))
    print(x, m, n)

    # Just check if it runs
    score3, _, _ = heuralign(alphabet, scores, s, t)

    score1, _, _ = dynprog(alphabet, scores, s, t)
    score2, _, _ = dynproglin(alphabet, scores, s, t)

    if score1 != score2 or score3 > score1:
        print(s)
        print(t)
        print(score1)
        print(score2)
        print(score3)

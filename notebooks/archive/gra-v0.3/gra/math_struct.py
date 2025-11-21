import re
def struct_pass(text: str) -> bool:
    t=text.lower()
    eq = bool(re.search(r"a\^?\s*2\s*\+\s*b\^?\s*2\s*=\s*c\^?\s*2", t) or
              re.search(r"hypotenuse\s*(?:squared|\^2).*\bsum of the squares\b", t))
    rt = bool(re.search(r"\bright[-\s]?triangle\b", t))
    hy = bool(re.search(r"\bhypotenuse\b", t))
    tri = bool(re.search(r"a\s*=\s*\d+.*b\s*=\s*\d+.*c\s*=\s*\d+", t, flags=re.S) or
               re.search(r"\b(3[, ]*4[, ]*5|5[, ]*12[, ]*13|8[, ]*15[, ]*17)\b", t))
    return eq and rt and hy and tri

def canonical(tri=(3,4,5)):
    a,b,c = tri
    return (f"In a right triangle, the square of the hypotenuse equals the sum of the squares of the legs (a^2 + b^2 = c^2). "
            f"Example: a={a}, b={b}, c={c} (since {a}^2 + {b}^2 = {a*a} + {b*b} = {a*a+b*b} = {c}^2).")

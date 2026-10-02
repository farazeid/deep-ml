import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    # f = g / h
    # f' = (g'h - gh') / h^2
    def fn(coeffs: list, x: float) -> float:
        power = len(coeffs) - 1
        out = 0
        for coeff in coeffs:
            out += coeff * x ** power
            power -= 1
        return out
    
    def deriv_fn(coeffs: list, x: float) -> float:
        power = len(coeffs) - 1
        out = 0
        for coeff in coeffs[:-1]:
            out += power * coeff * x ** (power - 1)
            power -= 1
        return out

    hx = fn(h_coeffs, x)

    if hx == 0:
        return -1

    gx = fn(g_coeffs, x)
    gdx = deriv_fn(g_coeffs, x)
    hdx = deriv_fn(h_coeffs, x)

    return (gdx * hx - gx * hdx) / (hx ** 2)

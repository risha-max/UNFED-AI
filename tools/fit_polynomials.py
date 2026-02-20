#!/usr/bin/env python3
"""
Polynomial Coefficient Fitting for MPC Approximations.

Computes optimal polynomial coefficients for approximating non-linear
functions in the UNFED AI MPC pipeline using the Remez algorithm
(minimax approximation).

Current targets:
  - sigmoid(x): used in SiLU(x) = x * sigmoid(x)

Usage:
    python -m tools.fit_polynomials
    python -m tools.fit_polynomials --degree 5 --range 6
"""

import argparse
import sys

import numpy as np
import torch


def chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    """Generate n Chebyshev nodes on [a, b]."""
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) / (2 * n) * np.pi)
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes


def fit_minimax_sigmoid(degree: int = 5,
                        x_range: float = 6.0,
                        n_iter: int = 20,
                        n_eval: int = 10000) -> dict:
    """
    Fit a minimax polynomial to sigmoid(x) over [-x_range, x_range].

    Uses an iterative Remez-like approach:
    1. Start with Chebyshev interpolation
    2. Iteratively adjust to minimize max error

    For sigmoid, only odd-degree terms contribute (sigmoid is 0.5 + odd fn).

    Returns dict with coefficients and error stats.
    """
    a, b = -x_range, x_range
    x_eval = np.linspace(a, b, n_eval)
    y_true = 1.0 / (1.0 + np.exp(-x_eval))

    # For sigmoid, we fit: sigmoid(x) - 0.5 ≈ c1*x + c3*x^3 + c5*x^5 + ...
    # (only odd terms due to the anti-symmetry of sigmoid(x) - 0.5)
    y_target = y_true - 0.5

    # Determine odd powers to use
    n_odd = (degree + 1) // 2
    powers = [2 * i + 1 for i in range(n_odd)]

    # Use Chebyshev nodes for initial fit
    n_nodes = len(powers) + 1
    x_nodes = chebyshev_nodes(n_nodes * 4, a, b)
    y_nodes = 1.0 / (1.0 + np.exp(-x_nodes)) - 0.5

    # Vandermonde-like matrix with odd powers
    V = np.column_stack([x_nodes ** p for p in powers])
    coeffs, _, _, _ = np.linalg.lstsq(V, y_nodes, rcond=None)

    # Refine: iterative weighted least squares (simple Remez approximation)
    best_max_err = float('inf')
    best_coeffs = coeffs.copy()

    for iteration in range(n_iter):
        # Evaluate current approximation
        V_eval = np.column_stack([x_eval ** p for p in powers])
        y_approx = V_eval @ coeffs
        errors = y_target - y_approx
        max_err = np.max(np.abs(errors))

        if max_err < best_max_err:
            best_max_err = max_err
            best_coeffs = coeffs.copy()

        # Weight the fit points by error — focus on worst points
        weights = 1.0 + np.abs(errors) / (max_err + 1e-10)

        # Re-fit with weighted samples
        W = np.diag(weights)
        V_w = W @ V_eval
        y_w = W @ y_target
        coeffs, _, _, _ = np.linalg.lstsq(V_w, y_w, rcond=None)

    # Final evaluation with best coefficients
    V_eval = np.column_stack([x_eval ** p for p in powers])
    y_final = V_eval @ best_coeffs
    errors = y_target - y_final
    max_err = np.max(np.abs(errors))
    mean_err = np.mean(np.abs(errors))

    # Build coefficient dict
    coeff_dict = {}
    for p, c in zip(powers, best_coeffs):
        coeff_dict[f"c{p}"] = float(c)

    return {
        "function": "sigmoid",
        "degree": degree,
        "range": [-x_range, x_range],
        "constant_term": 0.5,
        "coefficients": coeff_dict,
        "powers": powers,
        "max_error": float(max_err),
        "mean_error": float(mean_err),
    }


def fit_silu_direct(degree: int = 5,
                    x_range: float = 6.0,
                    n_eval: int = 10000) -> dict:
    """
    Fit polynomial to SiLU(x) = x * sigmoid(x) directly.

    This is an alternative approach — instead of approximating sigmoid
    and then multiplying by x (which requires a secure multiply), we
    could approximate SiLU directly with a polynomial.

    However, SiLU is NOT an odd function, so we need all powers.
    This is less efficient for MPC (more multiplications) but could
    give better accuracy.
    """
    x_eval = np.linspace(-x_range, x_range, n_eval)
    y_true = x_eval / (1.0 + np.exp(-x_eval))

    V = np.column_stack([x_eval ** p for p in range(1, degree + 1)])
    coeffs, _, _, _ = np.linalg.lstsq(V, y_true, rcond=None)

    y_approx = V @ coeffs
    errors = y_true - y_approx
    max_err = np.max(np.abs(errors))
    mean_err = np.mean(np.abs(errors))

    coeff_dict = {f"c{p}": float(c) for p, c in zip(range(1, degree + 1), coeffs)}

    return {
        "function": "silu_direct",
        "degree": degree,
        "range": [-x_range, x_range],
        "coefficients": coeff_dict,
        "max_error": float(max_err),
        "mean_error": float(mean_err),
    }


def verify_with_pytorch(coefficients: dict, x_range: float = 6.0):
    """Verify polynomial approximation against PyTorch native."""
    x = torch.linspace(-x_range, x_range, 1000)
    native_silu = torch.nn.functional.silu(x)

    # Reconstruct polynomial sigmoid
    sig_approx = torch.full_like(x, 0.5)
    for key, val in coefficients.items():
        power = int(key[1:])
        sig_approx = sig_approx + val * x.pow(power)

    silu_approx = x * sig_approx

    error = (native_silu - silu_approx).abs()
    print(f"\nPyTorch verification:")
    print(f"  Max error:  {error.max().item():.6f}")
    print(f"  Mean error: {error.mean().item():.6f}")
    print(f"  At x=0:     {error[500].item():.6f}")
    print(f"  At x=3:     {error[750].item():.6f}")
    print(f"  At x=-3:    {error[250].item():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit polynomial coefficients for MPC approximations")
    parser.add_argument("--degree", type=int, default=5,
                        help="Polynomial degree (default: 5)")
    parser.add_argument("--range", type=float, default=6.0,
                        help="Approximation range [-R, R] (default: 6.0)")
    parser.add_argument("--compare-degrees", action="store_true",
                        help="Compare accuracy across degrees 3, 5, 7")
    args = parser.parse_args()

    if args.compare_degrees:
        print("=" * 60)
        print("Comparing sigmoid approximation accuracy across degrees")
        print("=" * 60)
        for deg in [3, 5, 7]:
            result = fit_minimax_sigmoid(degree=deg, x_range=args.range)
            print(f"\n  Degree {deg}:")
            print(f"    Coefficients: {result['coefficients']}")
            print(f"    Max error:    {result['max_error']:.6e}")
            print(f"    Mean error:   {result['mean_error']:.6e}")
            extra_mults = (deg + 1) // 2 - 1
            print(f"    Extra secure multiplies: {extra_mults}")
        return

    print(f"Fitting degree-{args.degree} minimax polynomial for sigmoid "
          f"over [-{args.range}, {args.range}]")
    print()

    result = fit_minimax_sigmoid(degree=args.degree, x_range=args.range)
    print("Result:")
    print(f"  Function:     {result['function']}")
    print(f"  Degree:       {result['degree']}")
    print(f"  Range:        {result['range']}")
    print(f"  Constant:     {result['constant_term']}")
    print(f"  Coefficients:")
    for k, v in result['coefficients'].items():
        print(f"    {k} = {v:.8f}")
    print(f"  Max error:    {result['max_error']:.6e}")
    print(f"  Mean error:   {result['mean_error']:.6e}")

    print("\nFor mpc_protocols.py, use:")
    for k, v in result['coefficients'].items():
        print(f"    {k} = {v}")

    verify_with_pytorch(result['coefficients'], args.range)


if __name__ == "__main__":
    main()

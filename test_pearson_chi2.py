"""
Unit tests and verification for Pearson χ² divergence implementation
"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from critic import conservative_loss, conservative_loss_clipped


@dataclass
class Args:
    gumbel_max_clip: float = 7.0
    min_clip: float = 1.0
    max_clip: float = 50.0


def test_pearson_chi2_basic_properties():
    """Test basic properties of Pearson χ² divergence loss"""
    print("TEST 1: Basic Properties of Pearson χ² Divergence Loss")
    print("=" * 60)
    
    args = Args()
    alpha = 1.0
    
    # Test 1: Loss should be zero at td_error = 0
    diff_zero = jnp.array([0.0])
    loss_zero, norm_zero = conservative_loss(diff_zero, alpha, args)
    print(f"Loss at TD error = 0: {loss_zero[0]:.6f}")
    print(f"Norm at TD error = 0: {norm_zero:.6f}")
    assert loss_zero[0] < 1e-6, "Loss should be approximately zero at TD error = 0"
    
    # Test 2: Loss should increase quadratically with positive TD error
    diff_pos = jnp.array([1.0])
    loss_pos, _ = conservative_loss(diff_pos, alpha, args)
    print(f"Loss at TD error = 1: {loss_pos[0]:.6f}")
    assert loss_pos[0] > loss_zero[0], "Loss should increase with positive TD error"
    assert abs(loss_pos[0] - 0.5) < 0.1, "Loss should be approximately 0.5 * (1/alpha)^2"
    
    # Test 3: Loss should be symmetric (same for positive and negative TD error)
    diff_neg = jnp.array([-1.0])
    loss_neg, _ = conservative_loss(diff_neg, alpha, args)
    print(f"Loss at TD error = -1: {loss_neg[0]:.6f}")
    assert abs(loss_pos[0] - loss_neg[0]) < 1e-6, "Loss should be symmetric"
    
    print("✓ All basic property tests passed!\n")


def test_pearson_chi2_clipping():
    """Test clipping behavior of Pearson χ² divergence"""
    print("TEST 2: Clipping Behavior")
    print("=" * 60)
    
    args = Args(gumbel_max_clip=5.0)
    alpha = 1.0
    
    # Test with large TD error that should be clipped
    diff_large = jnp.array([10.0, -10.0])
    loss_clipped, _ = conservative_loss_clipped(diff_large, alpha, args)
    loss_unclipped, _ = conservative_loss(diff_large, alpha, args)
    
    print(f"Large TD errors: {diff_large}")
    print(f"Loss with clipping: {loss_clipped}")
    print(f"Loss without clipping: {loss_unclipped}")
    
    # Clipped version should have linear extrapolation beyond clip
    assert loss_clipped[0] < loss_unclipped[0], "Clipped loss should be smaller than unclipped"
    
    print("✓ Clipping tests passed!\n")


def test_pearson_chi2_vs_kl():
    """Compare Pearson χ² with KL divergence behavior"""
    print("TEST 3: Pearson χ² vs KL Divergence Comparison")
    print("=" * 60)
    
    args = Args()
    alpha = 1.0
    
    # Range of TD errors
    td_errors = jnp.linspace(-5, 5, 100)
    
    # Pearson χ² losses
    chi2_losses = []
    for td_err in td_errors:
        loss, _ = conservative_loss(jnp.array([td_err]), alpha, args)
        chi2_losses.append(float(loss[0]))
    
    chi2_losses = np.array(chi2_losses)
    
    # KL divergence losses (exponential)
    kl_losses = []
    for td_err in td_errors:
        z = td_err / alpha
        if args.gumbel_max_clip is not None:
            z = jnp.clip(z, -args.gumbel_max_clip, args.gumbel_max_clip)
        kl_loss = jnp.exp(z) - z - 1
        kl_losses.append(float(kl_loss))
    
    kl_losses = np.array(kl_losses)
    
    print(f"TD error range: [{td_errors[0]:.2f}, {td_errors[-1]:.2f}]")
    print(f"Pearson χ² loss range: [{chi2_losses.min():.4f}, {chi2_losses.max():.4f}]")
    print(f"KL loss range: [{kl_losses.min():.4f}, {kl_losses.max():.4f}]")
    
    # Pearson χ² should grow slower than KL for large positive TD errors
    idx_large_pos = np.argmax(td_errors > 3)
    if idx_large_pos > 0:
        print(f"\nAt large positive TD error ({td_errors[idx_large_pos]:.2f}):")
        print(f"  Pearson χ² loss: {chi2_losses[idx_large_pos]:.4f}")
        print(f"  KL loss: {kl_losses[idx_large_pos]:.4f}")
        assert chi2_losses[idx_large_pos] < kl_losses[idx_large_pos], \
            "Pearson χ² should grow slower than KL for large TD errors"
    
    print("✓ Comparison tests passed!\n")


def test_pearson_chi2_numerical_stability():
    """Test numerical stability of Pearson χ² divergence"""
    print("TEST 4: Numerical Stability")
    print("=" * 60)
    
    args = Args(gumbel_max_clip=10.0)
    alpha = 1.0
    
    # Test with very large TD errors
    diff_very_large = jnp.array([100.0, -100.0, 1000.0])
    
    try:
        loss, norm = conservative_loss(diff_very_large, alpha, args)
        print(f"Very large TD errors: {diff_very_large}")
        print(f"Loss values: {loss}")
        print(f"All losses are finite: {jnp.all(jnp.isfinite(loss))}")
        assert jnp.all(jnp.isfinite(loss)), "All losses should be finite"
        print("✓ Numerical stability test passed!\n")
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}\n")
        raise


def visualize_pearson_chi2():
    """Visualize Pearson χ² divergence loss function"""
    print("TEST 5: Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        args = Args()
        alpha = 1.0
        
        # Range of TD errors
        td_errors = np.linspace(-5, 5, 200)
        
        # Pearson χ² losses
        chi2_losses = []
        for td_err in td_errors:
            loss, _ = conservative_loss(jnp.array([td_err]), alpha, args)
            chi2_losses.append(float(loss[0]))
        
        chi2_losses = np.array(chi2_losses)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(td_errors, chi2_losses, label='Pearson χ²', linewidth=2)
        plt.xlabel('TD Error', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title('Pearson χ² Divergence Loss Function', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pearson_chi2_loss.png', dpi=150)
        print("✓ Visualization saved to pearson_chi2_loss.png\n")
    except ImportError:
        print("Matplotlib not available, skipping visualization\n")


if __name__ == "__main__":
    print("PEARSON χ² DIVERGENCE IMPLEMENTATION TESTS")
    print("=" * 60)
    print()
    
    test_pearson_chi2_basic_properties()
    test_pearson_chi2_clipping()
    test_pearson_chi2_vs_kl()
    test_pearson_chi2_numerical_stability()
    visualize_pearson_chi2()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


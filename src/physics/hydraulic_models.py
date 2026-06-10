"""
Hydraulic models for soil water flow

PURPOSE:
========
These classes define the constitutive relationships between pressure head and:
- θ(p): Water content [m³/m³]
- kr(p): Relative permeability [-]
- Cm(p) = ∂θ/∂p: Moisture capacity [1/m]

Two model types are available:
1. **VanGenuchtenModel**: Analytical equations (Van Genuchten-Mualem)
   - Best for: Standard soils with known parameters
   - Advantages: Smooth, differentiable, widely validated
   - Disadvantages: Limited to Van Genuchten form, sometimes inappropriate curvatures
   
2. **CurveBasedHydraulicModel**: Empirical data interpolation
   - Best for: Custom soils with measured retention/permeability curves
   - Advantages: Can represent any measured behavior, allows specific curves


Both implement the same HydraulicModel interface, so they're interchangeable
in the simulation framework.

USAGE:
======
# Analytical model
vg_params = VanGenuchtenParams(theta_r=0.045, theta_s=0.35, alpha=14.5, n=2.68)
sand_model = VanGenuchtenModel(vg_params)

# Curve-based model from library
till_model = CurveBasedHydraulicModel.from_library("till")

# Curve-based model from custom data
custom_model = CurveBasedHydraulicModel.from_data(
    pressure_heads=[-10, -5, -1, 0],
    theta_values=[0.05, 0.15, 0.30, 0.40],
    kr_values=[0.001, 0.01, 0.1, 1.0]
)
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from .curve_tools import CurveData, CurveInterpolator
from scipy.optimize import differential_evolution, minimize
from typing import Literal, Optional, Dict, Tuple
from firedrake import conditional, min_value, max_value

# ==============================================
# ABSTRACT BASE CLASS
# ==============================================

class HydraulicModel(ABC):
    """
    Abstract base class for hydraulic models
    All models must implement these methods
    """
    
    @property
    @abstractmethod
    def theta_r(self) -> float:
        """Residual water content [m³/m³]"""
        pass
    
    @property
    @abstractmethod
    def theta_s(self) -> float:
        """Saturated water content [m³/m³]"""
        pass
    
    @abstractmethod
    def _theta(self, pressure) -> float:
        """Water content at pressure head [m³/m³]"""
        pass
    
    @abstractmethod
    def _k(self, pressure) -> float:
        """Relative permeability at pressure head [-]"""
        pass
    
    @abstractmethod
    def _Se(self, pressure) -> float:
        """Moisture capacity: ∂θ/∂p [1/m]"""
        pass

# ==============================================
# VAN GENUCHTEN ANALYTICAL MODEL
# ==============================================

@dataclass
class VanGenuchtenParams:
    """Van Genuchten model parameters"""
    theta_r: float      # Residual water content [m³/m³]
    theta_s: float      # Saturated water content [m³/m³]
    alpha: float        # van Genuchten alpha [1/m]
    n: float            # van Genuchten n parameter [-]
    l_param: float = 0.5  # Pore connectivity parameter [-]
    m: float = None     # Computed from n if None
    
    def __post_init__(self):
        """Compute m from n if not provided"""
        if self.m is None:
            self.m = 1.0 - 1.0 / self.n


class VanGenuchtenModel(HydraulicModel):
    """
    Van Genuchten analytical model
    With smooth transition near saturation (epsilon smoothing)
    """
    
    def __init__(self, 
                 params: VanGenuchtenParams, 
                 epsilon: float = 0.051,
                 kr_min: float = 1e-8,
                 Ss: float = 1e-4):
        """
        Parameters:
        -----------
        params : VanGenuchtenParams
            Van Genuchten parameters
        epsilon : float
            Smoothing parameter near water table [m]
            Creates smooth transition between saturated/unsaturated
        kr_min : float
            Minimum relative permeability (prevents division by zero)
        Ss : float
            Specific storage coefficient [1/m]
        """
        self.params = params
        self.epsilon = epsilon
        self.kr_min = kr_min
        self.Ss = Ss
    
    @property
    def theta_r(self) -> float:
        return self.params.theta_r
    
    @property
    def theta_s(self) -> float:
        return self.params.theta_s
        
    def _theta(self, pressure) -> float:
        """
        θ(p): Water content via Van Genuchten equation
        θ(p) = θ_r + S_e(p) * (θ_s - θ_r)
        """
        theta_r = self.params.theta_r
        theta_s = self.params.theta_s
        Se = self._Se(pressure)
        return theta_r + Se * (theta_s - theta_r)
    
    def _k(self, pressure, Ks) -> float:
        """
        K(p): Hydraulic conductivity [m/s]
        K(p) = k_r(p) * K_s
        """
        kr = self._kr(pressure)
        #if hasattr(kr, 'ufl_element'):
            #return kr * Constant(10)**Ks
        return kr *  Ks 

    def _Se(self, pressure):
        """
        S_e(p): Effective saturation [-]
        """
        eps = self.epsilon
        alpha = self.params.alpha
        n = self.params.n
        m = self.params.m
        
        if pressure >= eps:
            return 1.0
        elif pressure <= -eps:
            Se = 1.0 / (1.0 + (alpha * abs(pressure))**n)**m
            return max(0.0, min(1.0, Se))
        else:
            Se_unsat = 1.0 / (1.0 + (alpha * eps)**n)**m
            weight = (pressure + eps) / (2.0 * eps)
            Se = Se_unsat + (1.0 - Se_unsat) * weight
            return max(0.0, min(1.0, Se))
    
    def _kr(self, pressure):
        """
        Kr(p): Relative permeability [-] via Mualem model
        Kr(p) = S_e(p)^l * [1 - (1 - S_e(p)^(1/m))^m]^2
        """
        eps = self.epsilon
        l = self.params.l_param
        m = self.params.m

        if pressure >= eps:
            return 1.0
        elif pressure <= -eps:
            Se = self._Se(pressure)
            term = max(0.0, 1.0 - Se**(1.0/m))
            kr_val = (Se**l) * (1.0 - term**m)**2
            kr_min = self.kr_min
            return max(kr_min, min(kr_val, 1.0))
        else:
            Se_unsat = self._Se(-eps)
            term = max(0.0, 1.0 - Se_unsat**(1.0/m))
            kr_unsat = (Se_unsat**l) * (1.0 - term**m)**2
            kr_min = self.kr_min
            kr_unsat = max(kr_min, min(kr_unsat, 1.0))
            weight = (pressure + eps) / (2.0 * eps)
            kr_val = kr_unsat + (1.0 - kr_unsat) * weight
            return max(kr_min, min(kr_val, 1.0))

    def _Cm(self, pressure):
        """
        Cm(p): Moisture capacity [1/m]
        Cm(p) = alpha * m / (1 - m) * (θ_s - θ_r) * S_e^(1/m) * (1 - S_e^(1/m))^m
        """
        alpha = self.params.alpha
        m = self.params.m
        theta_s = self.params.theta_s
        theta_r = self.params.theta_r
        eps = self.epsilon
        Ss = self.Ss

        if pressure >= eps:
            return Ss
        elif pressure <= -eps:
            Se = self._Se(pressure)
            term = max(0.0, 1.0 - Se**(1.0 / m))
            Cm_val = ((alpha * m) / (1.0 - m) *
                        (theta_s - theta_r) *
                        Se**(1.0 / m) *
                        term**m)
            return max(Ss, Cm_val)
        else:
            Se_unsat = self._Se(-eps)
            term = max(0.0, 1.0 - Se_unsat**(1.0 / m))
            Cm_unsat = ((alpha * m) / (1.0 - m) *
                        (theta_s - theta_r) *
                        Se_unsat**(1.0 / m) *
                        term**m)
            Cm_unsat = max(Ss, Cm_unsat)
            weight = (pressure + eps) / (2.0 * eps)
            Cm_val = Cm_unsat + (Ss - Cm_unsat) * weight
            return max(Ss, Cm_val)

# ==============================================
# CURVE-BASED EMPIRICAL MODEL
# ==============================================

class CurveBasedHydraulicModel(HydraulicModel):
    """
    Hydraulic model using empirical curve data
    Uses generic CurveInterpolator for flexibility
    """
    
    def __init__(self, 
                 theta_curve: CurveData,
                 kr_curve: CurveData,
                 theta_r: float = None,
                 theta_s: float = None,
                 epsilon: float = 0.005,
                 kr_min: float = 1e-8,
                 Ss: float = 1e-4,
                 ):
        """
        Parameters:
        -----------
        theta_curve : CurveData
            Pressure → water content curve
        kr_curve : CurveData
            Pressure → relative permeability curve
        theta_r, theta_s : float, optional
            If None, inferred from curve bounds
        epsilon : float
            Pressure threshold for saturation [m]
        Ss : float
            Specific storage coefficient [1/m]
        kr_min : float
            Minimum relative permeability
        """
        # Infer bounds if not provided
        self._theta_r = theta_r if theta_r is not None else theta_curve.y_min
        self._theta_s = theta_s if theta_s is not None else theta_curve.y_max
        
        self.epsilon = epsilon
        self.Ss = Ss
        self.kr_min = kr_min
        
        # Create interpolators with appropriate settings
        self._theta_interp = CurveInterpolator(
            theta_curve,
            method='linear',
            extrapolate_mode='clip',
            bounds=(self._theta_r, self._theta_s),
            fill_value_above=self._theta_s
        )
        
        self._kr_interp = CurveInterpolator(
            kr_curve,
            method='linear',
            extrapolate_mode='clip',
            bounds=(self.kr_min, 1.0),
            fill_value_above=1.0
        )
        
        self.theta_curve = theta_curve
        self.kr_curve = kr_curve
    
    @property
    def theta_r(self) -> float:
        return self._theta_r
    
    @property
    def theta_s(self) -> float:
        return self._theta_s

    def _theta(self, pressure) -> float:
        """Water content via interpolation"""
        if pressure >= self.epsilon:
            return self._theta_s
        return self._theta_interp(pressure)
    
    def _kr(self, pressure) -> float:
        """Relative permeability via interpolation"""
        if pressure >= self.epsilon:
            return 1.0
        return self._kr_interp(pressure)
    
    def _k(self, pressure, Ks) -> float:
        """Hydraulic conductivity"""
        return self._kr(pressure) * Ks

    def _Cm(self, pressure) -> float:
        """Moisture capacity"""
        Cm = self._theta_interp.derivative(pressure)
        return max(self.Ss, abs(Cm))

    def _Se(self, pressure) -> float:
        """Effective saturation via numerical derivative"""
        theta = self._theta(pressure)
        Se = (theta - self._theta_r) / (self._theta_s - self._theta_r)
        return max(0.0, min(1.0, Se))
    
    def fit_van_genuchten(self, 
                            method: Literal['minimize', 'differential_evolution'] = 'minimize',
                            fit_kr: bool = True,
                            weight_theta: float = 1.0,
                            weight_kr: float = 1.0,
                            initial_guess: Optional[Dict[str, float]] = None,
                            verbose: bool = True) -> Tuple[VanGenuchtenParams, Dict]:
        """
        Fit Van Genuchten parameters to match the empirical curves
        
        This is useful for:
        1. Converting curve data to analytical form (faster evaluation)
        2. Checking if Van Genuchten model can represent your data
        3. Getting smooth derivatives (VG is C∞, curves have numerical noise)
        
        Parameters:
        -----------
        method : str
            'minimize': Fast local optimization (needs good initial guess)
            'differential_evolution': Global optimization (slower but more robust)
        fit_kr : bool
            If True, fit both theta(p) and kr(p) curves
            If False, only fit theta(p) (assumes kr follows Mualem model)
        weight_theta : float
            Relative weight for theta curve in objective function
        weight_kr : float
            Relative weight for kr curve in objective function
        initial_guess : dict, optional
            Initial parameter guesses: {'alpha': ..., 'n': ..., 'l_param': ...}
            If None, uses heuristics based on curve data
        verbose : bool
            Print fitting progress and diagnostics
        
        Returns:
        --------
        vg_params : VanGenuchtenParams
            Best-fit Van Genuchten parameters
        diagnostics : dict
            Fitting diagnostics (RMSE, R², residuals, etc.)
        
        Example:
        --------
        >>> till_model = CurveBasedHydraulicModel.from_library("till")
        >>> vg_params, diag = till_model.fit_van_genuchten(verbose=True)
        >>> print(f"Fitted alpha={vg_params.alpha:.2f}, n={vg_params.n:.2f}")
        >>> print(f"RMSE(theta)={diag['rmse_theta']:.4f}, R²={diag['r2_theta']:.3f}")
        >>> 
        >>> # Use fitted parameters
        >>> vg_model = VanGenuchtenModel(vg_params)
        """
        # ============================================
        # STEP 1: Get reference data from curves
        # ============================================
        # Use curve data points
        p_data = self.theta_curve.x_values
        theta_data = self.theta_curve.y_values
        
        if fit_kr:
            # Ensure kr curve has same pressure points (interpolate if needed)
            kr_data = self._kr_interp(p_data)
        else:
            kr_data = None
        
        # Remove saturated points (p >= epsilon) for better fitting
        mask = p_data < -self.epsilon
        p_fit = p_data[mask]
        theta_fit = theta_data[mask]
        if fit_kr:
            kr_fit = kr_data[mask]
        
        if len(p_fit) < 4:
            raise ValueError("Need at least 4 data points in unsaturated zone for fitting")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING VAN GENUCHTEN PARAMETERS TO CURVES")
            print(f"{'='*60}")
            print(f"Data points: {len(p_fit)} (pressure range: [{p_fit.min():.2f}, {p_fit.max():.2f}] m)")
            print(f"Target: theta_r={self._theta_r:.3f}, theta_s={self._theta_s:.3f}")
        
        # ============================================
        # STEP 2: Set up optimization problem
        # ============================================
        
        # Fixed parameters (from curve data)
        theta_r = self._theta_r
        theta_s = self._theta_s
        
        # Parameters to optimize: [alpha, n, l_param] (if fitting kr)
        # Bounds based on physical constraints
        if fit_kr:
            bounds = [
                (0.1, 50.0),    # alpha [1/m]: typically 0.5-15 for most soils
                (1.1, 10.0),    # n [-]: must be > 1, typically 1.2-5
                (0.0, 1.0)      # l_param [-]: pore connectivity, typically 0.5
            ]
            param_names = ['alpha', 'n', 'l_param']
        else:
            bounds = [
                (0.1, 50.0),    # alpha [1/m]
                (1.1, 10.0),    # n [-]
            ]
            param_names = ['alpha', 'n']
        
        # Initial guess (heuristic or user-provided)
        if initial_guess is None:
            # Heuristic initial guess based on pressure at theta = (theta_r + theta_s)/2
            theta_mid = (theta_r + theta_s) / 2
            idx_mid = np.argmin(np.abs(theta_fit - theta_mid))
            p_mid = abs(p_fit[idx_mid])
            
            # From VG equation: at S_e = 0.5, p ≈ 1/(alpha * 2^(1/n))
            # Rough estimate: alpha ≈ 1/(2 * p_mid)
            alpha_guess = 1.0 / (2.0 * p_mid) if p_mid > 0 else 2.0
            alpha_guess = np.clip(alpha_guess, 0.5, 15.0)
            
            # n typically 1.5-3 for most soils
            n_guess = 2.0
            
            if fit_kr:
                x0 = [alpha_guess, n_guess, 0.5]
            else:
                x0 = [alpha_guess, n_guess]
        else:
            x0 = [initial_guess.get(name, bounds[i][0]) 
                  for i, name in enumerate(param_names)]
        
        if verbose:
            print(f"Initial guess: {dict(zip(param_names, x0))}")
            print(f"Method: {method}")
        
        # ============================================
        # STEP 3: Define objective function
        # ============================================
        
        def objective(params):
            """
            Objective function: weighted sum of squared errors
            """
            if fit_kr:
                alpha, n, l_param = params
            else:
                alpha, n = params
                l_param = 0.5  # Default Mualem value
            
            # Create temporary VG model
            m = 1.0 - 1.0 / n
            vg_params_temp = VanGenuchtenParams(
                theta_r=theta_r,
                theta_s=theta_s,
                alpha=alpha,
                n=n,
                l_param=l_param,
                m=m
            )
            vg_model_temp = VanGenuchtenModel(vg_params_temp, epsilon=self.epsilon)
            
            # Compute predictions
            theta_pred = np.array([vg_model_temp._theta(p) for p in p_fit])
            
            # Theta error (normalized by range)
            theta_range = theta_s - theta_r
            theta_error = np.sum(((theta_pred - theta_fit) / theta_range)**2)
            
            # Kr error (if fitting)
            kr_error = 0.0
            if fit_kr:
                kr_pred = np.array([vg_model_temp._kr(p) for p in p_fit])
                kr_error = np.sum((kr_pred - kr_fit)**2)
            
            # Weighted total error
            total_error = weight_theta * theta_error + weight_kr * kr_error
            
            return total_error
        
        # ============================================
        # STEP 4: Optimize
        # ============================================
        
        if method == 'differential_evolution':
            # Global optimization (more robust, slower)
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=1000,
                seed=42,
                atol=1e-6,
                tol=1e-6,
                disp=verbose
            )
        else:  # 'minimize'
            # Local optimization (faster, needs good initial guess)
            result = minimize(
                objective,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'disp': verbose}
            )
        
        # ============================================
        # STEP 5: Extract fitted parameters
        # ============================================
        
        if fit_kr:
            alpha_opt, n_opt, l_param_opt = result.x
        else:
            alpha_opt, n_opt = result.x
            l_param_opt = 0.5
        
        m_opt = 1.0 - 1.0 / n_opt
        
        vg_params = VanGenuchtenParams(
            theta_r=theta_r,
            theta_s=theta_s,
            alpha=alpha_opt,
            n=n_opt,
            l_param=l_param_opt,
            m=m_opt
        )
        
        # ============================================
        # STEP 6: Compute diagnostics
        # ============================================
        
        # Create VG model with fitted parameters
        vg_model = VanGenuchtenModel(vg_params, epsilon=self.epsilon)
        
        # Predictions
        theta_pred = np.array([vg_model._theta(p) for p in p_fit])
        
        # Theta diagnostics
        theta_residuals = theta_pred - theta_fit
        theta_rmse = np.sqrt(np.mean(theta_residuals**2))
        theta_mae = np.mean(np.abs(theta_residuals))
        
        # R² (coefficient of determination)
        ss_res = np.sum(theta_residuals**2)
        ss_tot = np.sum((theta_fit - np.mean(theta_fit))**2)
        theta_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        diagnostics = {
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit if hasattr(result, 'nit') else None,
            'objective_value': result.fun,
            'rmse_theta': theta_rmse,
            'mae_theta': theta_mae,
            'r2_theta': theta_r2,
            'residuals_theta': theta_residuals,
            'p_fit': p_fit,
            'theta_fit': theta_fit,
            'theta_pred': theta_pred,
        }
        
        if fit_kr:
            kr_pred = np.array([vg_model._kr(p) for p in p_fit])
            kr_residuals = kr_pred - kr_fit
            kr_rmse = np.sqrt(np.mean(kr_residuals**2))
            kr_mae = np.mean(np.abs(kr_residuals))
            
            ss_res_kr = np.sum(kr_residuals**2)
            ss_tot_kr = np.sum((kr_fit - np.mean(kr_fit))**2)
            kr_r2 = 1.0 - (ss_res_kr / ss_tot_kr) if ss_tot_kr > 0 else 0.0
            
            diagnostics.update({
                'rmse_kr': kr_rmse,
                'mae_kr': kr_mae,
                'r2_kr': kr_r2,
                'residuals_kr': kr_residuals,
                'kr_fit': kr_fit,
                'kr_pred': kr_pred,
            })
        
        # ============================================
        # STEP 7: Print results
        # ============================================
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING RESULTS")
            print(f"{'='*60}")
            print(f"Converged: {result.success}")
            print(f"\nFitted parameters:")
            print(f"  alpha = {alpha_opt:.4f} [1/m]")
            print(f"  n = {n_opt:.4f} [-]")
            print(f"  m = {m_opt:.4f} [-]")
            if fit_kr:
                print(f"  l = {l_param_opt:.4f} [-]")
            
            print(f"\nGoodness of fit (theta):")
            print(f"  RMSE = {theta_rmse:.5f} [m³/m³]")
            print(f"  MAE  = {theta_mae:.5f} [m³/m³]")
            print(f"  R²   = {theta_r2:.4f}")
            
            if fit_kr:
                print(f"\nGoodness of fit (kr):")
                print(f"  RMSE = {kr_rmse:.5f} [-]")
                print(f"  MAE  = {kr_mae:.5f} [-]")
                print(f"  R²   = {kr_r2:.4f}")
            
            # Interpretation
            print(f"\nInterpretation:")
            if theta_r2 > 0.95:
                print("  ✓ Excellent fit! VG model represents data very well")
            elif theta_r2 > 0.90:
                print("  ✓ Good fit. VG model is suitable")
            elif theta_r2 > 0.80:
                print("  ⚠ Acceptable fit, but some discrepancies")
            else:
                print("  ✗ Poor fit. VG model may not be appropriate for this soil")
            
            print(f"{'='*60}\n")
        
        return vg_params, diagnostics
    
    def compare_with_van_genuchten(self, vg_params: VanGenuchtenParams, 
                                   n_points: int = 50) -> Dict:
        """
        Visual comparison between curve data and Van Genuchten model
        
        Parameters:
        -----------
        vg_params : VanGenuchtenParams
            VG parameters to compare against
        n_points : int
            Number of points for comparison
        
        Returns:
        --------
        comparison : dict
            Contains pressure, theta, and kr arrays for both models
        
        Example:
        --------
        >>> vg_params, _ = till_model.fit_van_genuchten()
        >>> comp = till_model.compare_with_van_genuchten(vg_params)
        >>> 
        >>> # Plot comparison
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(comp['p'], comp['theta_curve'], 'o', label='Curve data')
        >>> plt.plot(comp['p'], comp['theta_vg'], '-', label='Van Genuchten')
        >>> plt.legend()
        >>> plt.show()
        """
        # Pressure range
        p_min = self.theta_curve.x_min
        p_max = min(self.theta_curve.x_max, -self.epsilon)  # Stay in unsaturated zone
        p_range = np.linspace(p_min, p_max, n_points)
        
        # Curve model predictions
        theta_curve = np.array([self._theta(p) for p in p_range])
        kr_curve = np.array([self._kr(p) for p in p_range])
        
        # VG model predictions
        vg_model = VanGenuchtenModel(vg_params, epsilon=self.epsilon)
        theta_vg = np.array([vg_model._theta(p) for p in p_range])
        kr_vg = np.array([vg_model._kr(p) for p in p_range])
        
        return {
            'p': p_range,
            'theta_curve': theta_curve,
            'theta_vg': theta_vg,
            'kr_curve': kr_curve,
            'kr_vg': kr_vg,
            'theta_diff': theta_vg - theta_curve,
            'kr_diff': kr_vg - kr_curve,
        }
from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_vector, grad, dot, SpatialCoordinate, Constant, exp
)

from ..tools.tools import loading_bar

class ThermoRichardsSolver:
    def __init__(self, V, field_map, source_scenario, bc_manager, config, 
                 monitoring, verbose=True):
        
        self.field_map = field_map
        self.domain = self.field_map.domain
        self.mesh = self.domain.mesh
        self.V = V # Function space (assuming same space for P and T for simplicity)
        
        self.source_scenario = source_scenario
        self.bc_manager = bc_manager
        self.config = config
        self.monitoring = monitoring

        self.verbose = verbose
        self.max_ponding_flux = 1e-6
        self.Se_cutoff = 0.95
        
        # Pressure functions
        self.p_n = Function(self.V, name="Pressure_old")
        self.p_new = Function(self.V, name="Pressure")
        
        # Temperature functions
        self.T_n = Function(self.V, name="Temperature_old")
        self.T_new = Function(self.V, name="Temperature")
        
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """Set initial hydrostatic pressure and temperature profiles"""        
        # Pressure IC
        water_table = self.bc_manager.left_wt_0
        coords_ufl = SpatialCoordinate(self.mesh)
        pressure_expr = water_table - coords_ufl[1]
        self.p_n.interpolate(pressure_expr)
        
        # Temperature IC (e.g., 10 degrees Celsius baseline)
        # You can move this to bc_manager for more complex initial gradients
        self.T_n.interpolate(Constant(10.0)) 
    
    def _get_solver_parameters(self):
        return {
            "ksp_type": "gmres",
            "ksp_gmres_restart": 30,
            "ksp_rtol": 1e-4,
            "ksp_atol": 1e-6,
            "ksp_max_it": 100,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_max_levels": 15,
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
            "pc_hypre_boomeramg_truncfactor": 0.3,
        }
    
    def _get_viscosity_ratio(self, T):
        """
        Returns the UFL expression for the kinematic viscosity ratio: mu_ref / mu(T).
        Assumes T is in Celsius. Reference temperature is 20 C.
        Using an exponential approximation for UFL compatibility and numerical stability.
        """
        T_ref = Constant(20.0)
        # Empirical approximation: Viscosity decreases by ~2.4% per degree C around 20C.
        # This prevents complex UFL division issues and remains highly stable.
        return exp(0.024 * (T - T_ref))

    def solve_timestep(self, t: float):
        # ---------------------------------------------------------
        # 1. THERMAL CONDUCTION STEP
        # ---------------------------------------------------------
        T_trial = TrialFunction(self.V)
        T_test = TestFunction(self.V)
        
        # Fetch thermal parameters (fallback to Constants if not in field_map yet)
        rhoc = getattr(self.field_map, 'rhoc', Constant(2.5e6))    # Volumetric heat capacity [J/(m3 K)]
        lambda_t = getattr(self.field_map, 'lambda_t', Constant(1.5)) # Thermal conductivity [W/(m K)]
        
        F_T = (
            rhoc * (T_trial - self.T_n) / self.config.dt * T_test * dx +
            lambda_t * dot(grad(T_trial), grad(T_test)) * dx
        )
        
        # Fetch thermal BCs (you will need to implement this in your bc_manager)
        bcs_T = getattr(self.bc_manager, 'get_thermal_bcs', lambda time: [])(t)
        
        solve(lhs(F_T) == rhs(F_T), self.T_new, bcs=bcs_T,
              solver_parameters=self._get_solver_parameters())
        
        # ---------------------------------------------------------
        # 2. HYDROGEOLOGICAL STEP (With Viscosity Correction)
        # ---------------------------------------------------------
        p = TrialFunction(self.V)
        q = TestFunction(self.V)

        # Baseline hydraulic parameters
        Cm = self.field_map.get_Cm_field(self.p_n)
        K_base = self.field_map.get_K_field(self.p_n)
        
        # Apply Viscosity Correction: K_eff = K_base * (mu_ref / mu(T))
        viscosity_factor = self._get_viscosity_ratio(self.T_new)
        K_eff = K_base * viscosity_factor
        
        bcs_p = self.bc_manager.get_dirichlet_bcs(t)
        rain_flux = -self.source_scenario.get_flux_expression(t, self.mesh)
        gravity = as_vector([0, 1])

        F_P = (
            Cm * (p - self.p_n) / self.config.dt * q * dx +
            K_eff * dot(grad(p), grad(q)) * dx +
            K_eff * dot(gravity, grad(q)) * dx +
            rain_flux * q * ds(4)
        )

        solve(lhs(F_P) == rhs(F_P), self.p_new, bcs=bcs_p,
              solver_parameters=self._get_solver_parameters())
        
        # ---------------------------------------------------------
        # 3. UPDATE PREVIOUS STATES
        # ---------------------------------------------------------
        self.T_n.assign(self.T_new)
        self.p_n.assign(self.p_new)
    
    def run(self):
        if self.verbose:
            print("Starting thermo-hydro simulation...")
            print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        # Record Initial States
        self.monitoring.record_probe(0.0, self.p_n, "water_table")
        self.monitoring.record_probe(0.0, self.T_n, "temperature")
        self.monitoring.check_and_record_snapshot(0.0, self.config.dt, self.field_map.get_Se_field(self.p_n), "saturation")
        self.monitoring.check_and_record_snapshot(0.0, self.config.dt, self.T_n, "temperature")

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            self.solve_timestep(t)
            
            self.monitoring.record_probe(t, self.p_new, "water_table")
            self.monitoring.record_probe(t, self.T_new, "temperature")
            self.monitoring.check_and_record_snapshot(t, self.config.dt, self.field_map.get_Se_field(self.p_new), "saturation")
            self.monitoring.check_and_record_snapshot(t, self.config.dt, self.T_new, "temperature")

            if self.verbose:
                loading_bar(step, t, self.config)
        
        if self.verbose:
            print("\n\nSimulation complete!")
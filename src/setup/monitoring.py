import pandas as pd
import numpy as np
from datetime import timedelta

class SimulationMonitor:
    """
    Unified monitor using Pandas DataFrames for probes and snapshots.
    
    DataFrames:
    -----------
    probes_df:    MultiIndex(time_s, datetime) | Columns: Probe1_head, Global_mass...
    snapshots_df: Index(time_s) | Columns: pressure, saturation (Values: Firedrake Functions)
    """
    
    def __init__(self, config, mesh, probe_positions=None, names=None, snapshot_times=None):
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data_ro
        
        # --- Time-Series Setup ---
        self.probe_positions = probe_positions or [[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]]
        self.names = names or [f"Probe_{i+1}" for i in range(len(self.probe_positions))]
        self._probe_indices = self._find_probe_nodes()
        
        # Build the 'dedoubled' index (Seconds + Datetime)
        time_s = self.config.time_steps
        if self.config.start_datetime:
            dates = [self.config.start_datetime + timedelta(seconds=s) for s in time_s]
            index = pd.MultiIndex.from_arrays([time_s, dates], names=['time_s', 'datetime'])
        else:
            index = pd.Index(time_s, name='time_s')

        # Pre-allocate probes DataFrame (columns added dynamically)
        self.probes_df = pd.DataFrame(index=index)
        
        # --- Snapshots Setup ---
        self.snapshot_targets = sorted(list(snapshot_times)) if snapshot_times else []
        self.snapshots_df = pd.DataFrame() # Rows added only on snapshot events

    def _find_probe_nodes(self):
        """Finds the single closest mesh node index for each probe coordinate."""
        indices = []
        for x_p, y_p in self.probe_positions:
            dist_sq = (self.coords[:, 0] - x_p)**2 + (self.coords[:, 1] - y_p)**2
            indices.append(np.argmin(dist_sq))
        return indices

    def record_probe(self, t: float, field=None, field_name: str = "value", data=None):
        """Records values into the probes_df at the row corresponding to time t."""
        # Case 1: Generic global scalar
        if data is not None or not hasattr(field, 'dat'):
            val = float(data if data is not None else field)
            self.probes_df.loc[t, f"global_{field_name}"] = val
            return

        # Case 2: Firedrake Function probe extraction
        field_data = field.dat.data_ro
        for name, idx, pos in zip(self.names, self._probe_indices, self.probe_positions):
            val = float(field_data[idx])
            
            # Hydrogeology specific: Pressure head -> Total head conversion
            if field_name == "water_table":
                val += pos[1] 
                
            self.probes_df.loc[t, f"{name}_{field_name}"] = val

    def check_and_record_snapshot(self, t: float, dt: float, field, field_name: str = "field"):
        """Saves the full Firedrake Function into the snapshots_df cell."""
        for req_time in self.snapshot_targets[:]:
            if abs(t - req_time) < dt * 0.6:
                self.snapshot_targets.remove(req_time)
                
                # Store the deepcopy directly in the DataFrame cell
                # Each 'cell' holds a full Firedrake Function object
                self.snapshots_df.at[t, field_name] = field.copy(deepcopy=True)
                vmin = field.dat.data.min()
                vmax = field.dat.data.max()
                
                print(f"  Snapshot at t={t/3600:.0f}h, data ranging from {vmin:.3f}; {vmax}.")
                return True
        return False

    def save_probes_to_csv(self, filename: str):
        """Dumps the probes DataFrame to CSV (includes MultiIndex)."""
        self.probes_df.to_csv(filename)
        print(f"✓ Probe data saved to {filename}")

    def get_snapshot(self, t: float, field_name: str = "field"):
        """Helper to retrieve a Function from the DataFrame snapshots."""
        try:
            return self.snapshots_df.at[t, field_name]
        except KeyError:
            return None
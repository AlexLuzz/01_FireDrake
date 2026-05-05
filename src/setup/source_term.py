from firedrake import Constant, SpatialCoordinate, conditional, And
import pandas as pd
import numpy as np
from ..tools.fetch_weather import fetch_weather_data
from ..tools.csv_loader import load_csv

class SourceManager:
    def __init__(self, config, zones: dict):
        """
        config: SimulationConfig instance
        zones: {'zone_name': (x_min, x_max, y_min, y_max)}
        """
        self.config = config
        self.zones = zones
        # Dataframe stores temporal source fluxes at simulation time scale
        self.data = pd.DataFrame(0.0, index=self.config.time_steps, columns=[])

    def add_event(self, start_h, end_h, value, source_type: str = "Rain (mm/h)"):
        """Manual event definition using hours from simulation start."""
        if source_type not in self.data.columns:
            self.data[source_type] = 0.0
            
        mask = (self.data.index >= start_h * 3600) & (self.data.index < end_h * 3600)
        self.data.loc[mask, source_type] = value

    def import_csv(self, filepath: str, datetime_col: str = None):
        df = load_csv(filepath, datetime_col)

        # --- Ensure DatetimeIndex ---
        if datetime_col and datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        else:
            # Try to convert existing index
            df.index = pd.to_datetime(df.index, errors="coerce")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Data must have a DatetimeIndex or a valid datetime column")

        df = df.sort_index()

        # --- Simulation index ---
        sim_index = self.config.start_datetime + pd.to_timedelta(self.config.time_steps, unit="sec")
        dt = pd.to_timedelta(self.config.dt, unit="sec")

        # --- Resample (rain = rate → mean) ---
        new_data = (
            df.resample(dt).mean()
            .reindex(sim_index)
            .interpolate()
            .fillna(0.0)
        )/24

        new_data.index = self.config.time_steps
        self.data = pd.concat([self.data, new_data], axis=1)

    def import_weather_data(self, start_date, end_date):
        """Fetches station data and converts units to SI (m/s)."""
        rain_df, snow_df = fetch_weather_data(start_date, end_date)
        
        for df, col, factor in [(rain_df, 'rain', 1/(1000*86400)), 
                                (snow_df, 'snow', 1/(100*86400))]:
            df['sim_t'] = (df['date'] - self.config.start_datetime).dt.total_seconds()
            series = df.set_index('sim_t')[col]
            self.data[col] = series.reindex(self.config.time_steps).ffill().fillna(0.0).values * factor
    
    def get_flux_expression(self, t: float, mesh, source_type: str = "Rain (mm/h)"):
        """
        Constructs the UFL expression by applying zone weighting to fluxes.
        """
        x, y = SpatialCoordinate(mesh)
        expr = Constant(0.0)
        
        # Get the current flux values for all columns at time t
        idx = (np.abs(self.config.time_steps - t)).argmin()
        base_value = self.data.iloc[idx].iloc[0]
        
        for name, info in self.zones.items():
            # zone-specific scaling
            val = base_value * info.get("factor", 1.0) * 1/3600000

            if val == 0.0:
                continue

            b = info["bounds"]

            # x-condition (always exists)
            cond = And(x >= b[0], x <= b[1])

            # optional y-condition
            if len(b) > 2 and b[2] is not None:
                cond = And(
                    cond,
                    And(y >= b[2], y <= b[3])
                )

            expr = conditional(cond, Constant(val), expr)

        return expr
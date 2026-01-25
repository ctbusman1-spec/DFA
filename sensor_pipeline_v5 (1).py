# csv_analysis_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

class CSVAnalysisPipeline:
    """
    Offline analysis pipeline for sensor log CSV files.
    No Sense HAT required - processes existing data.
    """
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.sampling_rate = None
        
    def load_data(self):
        """Load and preprocess CSV data"""
        print(f"[INFO] Loading {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        
        # Calculate sampling rate if not in data
        if 'dt' in self.df.columns:
            avg_dt = self.df['dt'].mean()
            self.sampling_rate = 1.0 / avg_dt if avg_dt > 0 else 50.0
        else:
            self.sampling_rate = 50.0  # Default assumption
            
        print(f"[INFO] Loaded {len(self.df)} samples at ~{self.sampling_rate:.1f} Hz")
        
        # Convert step_state to categorical for analysis
        if 'step_state' in self.df.columns:
            # Get unique states and create categorical
            unique_states = self.df['step_state'].unique()
            self.df['step_state_cat'] = pd.Categorical(
                self.df['step_state'], 
                categories=unique_states
            )
        
        return self.df
    
    def analyze_movement_patterns(self):
        """Analyze movement patterns from the data"""
        print("\n" + "="*50)
        print("MOVEMENT PATTERN ANALYSIS")
        print("="*50)
        
        if 'step_state' not in self.df.columns:
            print("[WARNING] No step_state column found")
            return
            
        # 1. Activity distribution
        activity_dist = self.df['step_state'].value_counts(normalize=True) * 100
        print(f"\n1. Activity Distribution:")
        for state, percentage in activity_dist.items():
            print(f"   {state}: {percentage:.1f}%")
        
        # 2. Step count analysis
        if 'step_count' in self.df.columns:
            max_steps = self.df['step_count'].max()
            total_duration = len(self.df) / self.sampling_rate
            steps_per_second = max_steps / total_duration if total_duration > 0 else 0
            print(f"\n2. Step Analysis:")
            print(f"   Total steps: {max_steps}")
            print(f"   Steps/sec: {steps_per_second:.2f}")
        
        # 3. Movement transitions
        if 'step_state' in self.df.columns:
            transitions = []
            for i in range(1, len(self.df)):
                if self.df['step_state'].iloc[i] != self.df['step_state'].iloc[i-1]:
                    transitions.append((i, self.df['step_state'].iloc[i-1], self.df['step_state'].iloc[i]))
            
            print(f"\n3. Movement Transitions: {len(transitions)}")
            for i, (idx, from_state, to_state) in enumerate(transitions[:5]):  # Show first 5
                time_at_transition = idx / self.sampling_rate
                print(f"   {time_at_transition:.1f}s: {from_state} → {to_state}")
            if len(transitions) > 5:
                print(f"   ... and {len(transitions) - 5} more transitions")
        
        return activity_dist
    
    def detect_anomalies(self):
        """Detect anomalies in sensor data"""
        print("\n" + "="*50)
        print("ANOMALY DETECTION")
        print("="*50)
        
        anomalies = []
        
        # Check for NaN values
        nan_count = self.df.isnull().sum().sum()
        if nan_count > 0:
            print(f"[WARNING] Found {nan_count} NaN values in data")
            anomalies.append(('NaN values', nan_count))
        
        # Check for constant values (sensor stuck)
        for col in ['accel_raw_x_g', 'accel_raw_y_g', 'accel_raw_z_g']:
            if col in self.df.columns:
                std_dev = self.df[col].std()
                if std_dev < 0.001:  # Very low variance
                    print(f"[WARNING] {col} has very low variance (std={std_dev:.4f})")
                    anomalies.append((f'{col} low variance', std_dev))
        
        # Check for unrealistic values
        for col in ['accel_raw_x_g', 'accel_raw_y_g', 'accel_raw_z_g']:
            if col in self.df.columns:
                max_val = self.df[col].abs().max()
                if max_val > 10:  # More than 10g is unrealistic for walking
                    print(f"[WARNING] {col} has unrealistic value: {max_val:.1f}g")
                    anomalies.append((f'{col} unrealistic', max_val))
        
        # Gyro range check
        if 'gyro_z' in self.df.columns:
            max_gyro = self.df['gyro_z'].abs().max()
            print(f"   Max gyro_z: {max_gyro:.2f} rad/s")
            if max_gyro > 10:  # Very high rotation
                print(f"[WARNING] Unusually high gyro_z value: {max_gyro:.1f} rad/s")
        
        return anomalies
    
    def calculate_metrics(self):
        """Calculate key performance metrics"""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        metrics = {}
        
        # Signal quality metrics
        if 'accel_raw_x_g' in self.df.columns and 'accel_filt_x_mps2' in self.df.columns:
            # Signal to Noise Ratio (SNR) estimate
            raw_signal = self.df['accel_raw_x_g'].values
            filtered_signal = self.df['accel_filt_x_mps2'].values / 9.81  # Convert to g
            
            # Estimate noise as difference
            noise = raw_signal - filtered_signal
            signal_power = np.mean(filtered_signal**2)
            noise_power = np.mean(noise**2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                metrics['SNR_dB'] = snr_db
                print(f"1. Estimated SNR: {snr_db:.1f} dB")
            else:
                print("1. Could not calculate SNR (noise power is zero)")
        
        # Position drift
        if 'pos_x' in self.df.columns and 'pos_y' in self.df.columns:
            start_pos = np.array([self.df['pos_x'].iloc[0], self.df['pos_y'].iloc[0]])
            end_pos = np.array([self.df['pos_x'].iloc[-1], self.df['pos_y'].iloc[-1]])
            drift_distance = np.linalg.norm(end_pos - start_pos)
            metrics['position_drift_m'] = drift_distance
            print(f"2. Position drift: {drift_distance:.2f} m")
        
        # Step detection consistency
        if 'step_state' in self.df.columns:
            step_events = self.df[self.df['step_state'] == 'STEP_DETECTED']
            if len(step_events) > 1:
                step_intervals = np.diff(step_events.index / self.sampling_rate)
                if np.mean(step_intervals) > 0:
                    step_interval_cv = np.std(step_intervals) / np.mean(step_intervals) * 100
                    metrics['step_interval_cv'] = step_interval_cv
                    print(f"3. Step interval CV: {step_interval_cv:.1f}% (lower = more consistent)")
                else:
                    print("3. Step intervals too short to calculate CV")
        
        return metrics
    
    def generate_plots(self, save_dir='plots'):
        """Generate comprehensive analysis plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n[INFO] Generating plots in '{save_dir}' directory")
        
        fig = plt.figure(figsize=(16, 12))
        time_axis = np.arange(len(self.df)) / self.sampling_rate
        
        # 1. Acceleration over time
        ax1 = plt.subplot(3, 2, 1)
        if 'accel_raw_x_g' in self.df.columns:
            ax1.plot(time_axis, self.df['accel_raw_x_g'], 'gray', alpha=0.5, label='Raw X')
            ax1.plot(time_axis, self.df['accel_raw_y_g'], 'gray', alpha=0.3, label='Raw Y')
            
            if 'accel_filt_x_mps2' in self.df.columns:
                ax1.plot(time_axis, self.df['accel_filt_x_mps2']/9.81, 'b', label='Filtered X', linewidth=1.5)
            
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Acceleration (g)')
            ax1.set_title('Acceleration Signals')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No acceleration data', ha='center', va='center')
            ax1.set_title('Acceleration Signals (No Data)')
        
        # 2. Activity states
        ax2 = plt.subplot(3, 2, 2)
        if 'step_state' in self.df.columns:
            # Create a numerical representation for plotting
            state_mapping = {}
            unique_states = self.df['step_state'].unique()
            for i, state in enumerate(unique_states):
                state_mapping[state] = i
            
            numeric_states = self.df['step_state'].map(state_mapping)
            
            ax2.plot(time_axis, numeric_states, 'g-', linewidth=2)
            ax2.set_yticks(list(state_mapping.values()))
            ax2.set_yticklabels(list(state_mapping.keys()))
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Activity States Over Time')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No step_state data', ha='center', va='center')
            ax2.set_title('Activity States (No Data)')
        
        # 3. Position tracking
        ax3 = plt.subplot(3, 2, 3)
        if 'pos_x' in self.df.columns and 'pos_y' in self.df.columns:
            ax3.plot(self.df['pos_x'], self.df['pos_y'], 'b-', linewidth=1.5)
            if len(self.df) > 0:
                ax3.plot(self.df['pos_x'].iloc[0], self.df['pos_y'].iloc[0], 'go', 
                        markersize=10, label='Start')
                ax3.plot(self.df['pos_x'].iloc[-1], self.df['pos_y'].iloc[-1], 'ro', 
                        markersize=10, label='End')
            ax3.set_xlabel('X Position (m)')
            ax3.set_ylabel('Y Position (m)')
            ax3.set_title('Dead Reckoning Trajectory')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No position data', ha='center', va='center')
            ax3.set_title('Position Tracking (No Data)')
        
        # 4. Step detection markers
        ax4 = plt.subplot(3, 2, 4)
        if 'step_state' in self.df.columns and 'accel_filt_y_mps2' in self.df.columns:
            step_indices = self.df[self.df['step_state'] == 'STEP_DETECTED'].index
            ax4.plot(time_axis, self.df['accel_filt_y_mps2'], 'g-', label='Accel Y')
            if len(step_indices) > 0:
                step_times = step_indices / self.sampling_rate
                step_accels = self.df.loc[step_indices, 'accel_filt_y_mps2'].values
                ax4.scatter(step_times, step_accels, color='red', marker='o', 
                          s=50, label='Step Detected', zorder=5)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Acceleration (m/s²)')
            ax4.set_title(f'Step Detection ({len(step_indices)} steps)')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No step detection data', ha='center', va='center')
            ax4.set_title('Step Detection (No Data)')
        
        # 5. Velocity profile
        ax5 = plt.subplot(3, 2, 5)
        if 'velocity_x' in self.df.columns and 'velocity_y' in self.df.columns:
            velocity_magnitude = np.sqrt(self.df['velocity_x']**2 + self.df['velocity_y']**2)
            ax5.plot(time_axis, velocity_magnitude, 'b-', linewidth=1.5)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Velocity (m/s)')
            ax5.set_title('Velocity Magnitude')
            ax5.grid(True, alpha=0.3)
            
            # Add horizontal lines for walking/running thresholds
            ax5.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Walking threshold')
            ax5.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='Running threshold')
            ax5.legend(fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'No velocity data', ha='center', va='center')
            ax5.set_title('Velocity Profile (No Data)')
        
        # 6. Gyro data
        ax6 = plt.subplot(3, 2, 6)
        if 'gyro_z' in self.df.columns:
            ax6.plot(time_axis, self.df['gyro_z'], 'purple', linewidth=1.5)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Angular Velocity (rad/s)')
            ax6.set_title('Gyroscope Z-axis')
            ax6.grid(True, alpha=0.3)
            
            # Add zero line
            ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No gyroscope data', ha='center', va='center')
            ax6.set_title('Gyroscope Data (No Data)')
        
        plt.suptitle(f'Sensor Data Analysis: {os.path.basename(self.csv_file)}', fontsize=16)
        plt.tight_layout()
        output_filename = f'{save_dir}/analysis_{os.path.basename(self.csv_file).replace(".csv", "")}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plot saved to {output_filename}")
        plt.show()
        
        return fig
    
    def generate_report(self, output_file='analysis_report.txt'):
        """Generate a comprehensive text report"""
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SENSOR DATA ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"File: {self.csv_file}\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Sampling rate: {self.sampling_rate:.1f} Hz\n")
            f.write(f"Duration: {len(self.df)/self.sampling_rate:.1f} seconds\n\n")
            
            # Activity summary
            if 'step_state' in self.df.columns:
                f.write("ACTIVITY SUMMARY:\n")
                f.write("-"*40 + "\n")
                for state, count in self.df['step_state'].value_counts().items():
                    percentage = count / len(self.df) * 100
                    f.write(f"{state:15s}: {count:6d} samples ({percentage:5.1f}%)\n")
            
            # Metrics
            f.write("\nKEY METRICS:\n")
            f.write("-"*40 + "\n")
            metrics = {}
            
            # Calculate metrics without printing
            if 'accel_raw_x_g' in self.df.columns and 'accel_filt_x_mps2' in self.df.columns:
                raw_signal = self.df['accel_raw_x_g'].values
                filtered_signal = self.df['accel_filt_x_mps2'].values / 9.81
                noise = raw_signal - filtered_signal
                signal_power = np.mean(filtered_signal**2)
                noise_power = np.mean(noise**2)
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                    metrics['SNR_dB'] = snr_db
                    f.write(f"SNR (dB): {snr_db:.1f}\n")
            
            if 'pos_x' in self.df.columns and 'pos_y' in self.df.columns:
                start_pos = np.array([self.df['pos_x'].iloc[0], self.df['pos_y'].iloc[0]])
                end_pos = np.array([self.df['pos_x'].iloc[-1], self.df['pos_y'].iloc[-1]])
                drift_distance = np.linalg.norm(end_pos - start_pos)
                metrics['position_drift_m'] = drift_distance
                f.write(f"Position drift (m): {drift_distance:.2f}\n")
            
            # Data quality
            f.write("\nDATA QUALITY CHECK:\n")
            f.write("-"*40 + "\n")
            
            # Check for NaN values
            nan_count = self.df.isnull().sum().sum()
            if nan_count > 0:
                f.write(f"NaN values found: {nan_count}\n")
            else:
                f.write("No NaN values found ✓\n")
            
            # Check for unrealistic values
            unrealistic = []
            for col in ['accel_raw_x_g', 'accel_raw_y_g', 'accel_raw_z_g']:
                if col in self.df.columns:
                    max_val = self.df[col].abs().max()
                    if max_val > 10:
                        unrealistic.append((col, max_val))
            
            if unrealistic:
                f.write("Unrealistic values:\n")
                for col, val in unrealistic:
                    f.write(f"  - {col}: {val:.1f}g\n")
            else:
                f.write("All values in reasonable range ✓\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")
        
        print(f"[INFO] Report saved to {output_file}")
        return output_file

def main():
    """Main function to run the analysis pipeline"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python csv_analysis_pipeline.py <csv_file>")
        print("Example: python csv_analysis_pipeline.py fixed_sensor_log_5.csv")
        return
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        print(f"[TIPS] Make sure you're in the right directory")
        print(f"[TIPS] Try using the full path with forward slashes: C:/Users/.../fixed_sensor_log_5.csv")
        print(f"[TIPS] Or put the CSV file in the same directory as this script")
        return
    
    # Create analysis pipeline
    pipeline = CSVAnalysisPipeline(csv_file)
    
    # Load data
    try:
        df = pipeline.load_data()
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return
    
    # Run analyses
    try:
        pipeline.analyze_movement_patterns()
        pipeline.detect_anomalies()
        pipeline.calculate_metrics()
        
        # Generate plots and report
        pipeline.generate_plots()
        pipeline.generate_report()
        
        print(f"\n[SUCCESS] Analysis complete for {csv_file}")
    except Exception as e:
        print(f"[ERROR] During analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
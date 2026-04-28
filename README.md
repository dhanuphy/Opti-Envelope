# Optical Envelope Simulator Application

## Overview

**OpticalSimulatorApp** is a comprehensive GUI-based tool for analyzing optical transmission spectra and extracting material optical properties using the Swanepole (Envelope) method. This application is designed for researchers and engineers working with thin film materials, semiconductors, and optical materials characterization.

## Application Purpose

The tool enables users to:
- Import optical transmittance spectroscopy data (TXT, CSV, Excel formats)
- Manually select or automatically detect interference fringes (upper and lower envelopes)
- Estimate film thickness from spectral interference patterns
- Calculate optical properties including:
  - Refractive index (n)
  - Extinction coefficient (k)
  - Complex dielectric constants (ε₁, ε₂)
  - Absorption coefficient (α)
  - Bandgap energy (direct and indirect)
  - Additional parameters: tan δ, skin depth, optical density, conductivity
- Perform spectral simulations and curve fitting
- Export calculated optical properties to Excel/TXT files

## Key Features

### Data Analysis Methods
1. **Envelope Method** - For spectra with interference fringes
   - Manual envelope point selection
   - Automatic peak/valley detection
   - Smooth envelope interpolation (linear, quadratic, cubic)
   - Thickness estimation from fringe spacing

2. **Simple Analysis** - For spectra without fringes
   - Direct absorption coefficient calculation
   - Band gap extraction
   - Refractive index estimation

### Interactive GUI
- **Left Panel**: Data import, method selection, refractive index input, envelope controls
- **Middle Panel**: Multi-tab display showing 14 different optical parameters
- **Right Panel**: Simulation controls, thickness adjustment, status monitoring
- **Main Spectrum Tab**: Interactive plot with manual point selection and dragging
- **Parameter Tabs**: Individual plots for each calculated property

### Advanced Features
- Zoom-aware cursor snapping for precise point placement
- Real-time thickness optimization across the full spectral range
- Confidence score calculation for thickness estimates
- Fit quality assessment (%)
- Mouse event handlers for interactive point selection and editing
- Preserved zoom state across plot updates

## System Requirements

- **Python**: 3.11 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 2GB (4GB recommended)
- **Display**: 1920x1080 or higher recommended (application optimized for 1600x900)

## Installation

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd Opti-Envelope
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical computations |
| pandas | ≥1.3.0 | Data import/export |
| PyQt5 | ≥5.15.0 | GUI framework |
| matplotlib | ≥3.4.0 | Plot visualization |
| scipy | ≥1.7.0 | Signal processing, interpolation |
| openpyxl | ≥3.6.0 | Excel file support |

## Usage

### Running the Application
```bash
python opti_envelope_app.py
```

### Basic Workflow

1. **Import Data**
   - Click "📥 Import Data (TXT/CSV/Excel)"
   - Select a file containing wavelength and transmittance data
   - Data must have at least 2 columns: wavelength and transmittance

2. **Choose Analysis Method**
   - Select "🌊 Envelope Method (with fringes)" for spectra with interference patterns
   - Or select "📏 Simple Analysis (no fringes)" for featureless spectra

3. **For Envelope Method**:
   - Enter substrate refractive index (default: 1.4585)
   - **Manual Selection**: Toggle "📈 Upper Envelope" and "📉 Lower Envelope" to manually select points by clicking on the spectrum
   - **Auto Selection**: Click "🎯 Auto Select Tmax & Tmin" to automatically detect peaks and valleys
   - Click "🔗 Interpolate Envelopes" after selecting points
   - Adjust thickness in "Thickness (nm)" field
   - Click "🚀 Simulate Spectra" to run simulation

4. **For Simple Analysis**:
   - Enter film thickness
   - Click "🧮 Calculate Parameters"
   - View results in parameter tabs

5. **View Results**
   - Switch between tabs to view different calculated properties
   - Monitor thickness estimation and fit quality in status panel
   - Zoom, pan, and interact with plots

6. **Export Data**
   - Click "📤 Export Data (Excel/TXT)" to save calculated optical properties

### Input Data Format

Supported file formats: CSV, TXT, Excel (.xlsx, .xls)

**Required columns**:
- Column 1: Wavelength (nm)
- Column 2: Transmittance (0-1 or 0-100%)

**Example**:
```
Wavelength(nm)  Transmittance
400             0.85
410             0.82
420             0.88
...
```

## GUI Controls Reference

### Left Panel - Data & Controls
- **📂 DATA CONTROL**: Import/export data, clear all
- **📊 METHOD SELECTION**: Choose analysis method
- **🔬 REFRACTIVE INDEX**: Set substrate refractive index
- **📈 ENVELOPE CONTROLS**: Manual envelope selection buttons
- **🔧 SIMULATION CONTROL**: Auto select, point addition, thickness input, simulation trigger

### Right Panel - Monitoring
- **📋 STATUS**: Real-time status indicators
  - Envelope selection status
  - Manual and auto point counters
  - Estimated thickness with confidence
  - Fit quality percentage
  - Cursor position display

- **📏 SIMPLE ANALYSIS**: Controls for non-fringe analysis

### Center Panel - Multi-Tab Display
14 parameter visualization tabs:
- **Main**: Raw spectrum and envelopes
- **n**: Refractive index
- **k**: Extinction coefficient
- **ε₁, ε₂**: Complex dielectric constants
- **tanδ**: Loss tangent
- **SD**: Skin depth
- **σ**: Electrical conductivity
- **OD**: Optical density
- **α**: Absorption coefficient
- **(αE)²**: Direct bandgap plot
- **(αE)^½**: Indirect bandgap plot
- **dα/dE**: Absorption edge derivative
- **lnα**: Log-scale absorption

## Algorithm Details

### Envelope Method (Swanepole Method)
1. Smooth experimental transmittance curve
2. Detect peaks (maxima) and valleys (minima)
3. Interpolate envelope curves through detected points
4. Calculate refractive index at each wavelength
5. Estimate thickness from fringe spacing
6. Simulate transmission spectrum
7. Refine thickness through curve fitting

### Thickness Estimation
- Uses alternating pairs of maxima and minima
- Calculates multiple thickness estimates
- Applies weighted averaging based on quality metrics
- Returns confidence score based on consistency

### Optical Properties Calculation
- **n(λ)**: From envelope method or simple analysis
- **k(λ)**: From absorption measurements
- **ε₁, ε₂**: From n and k values
- **α(λ)**: Absorption coefficient
- **Eg**: Direct and indirect bandgaps from α vs E plot

## Output Format

### Excel Export
Contains sheets with:
- Wavelength (nm)
- Transmittance (experimental)
- Refractive index (n)
- Extinction coefficient (k)
- Dielectric constants (ε₁, ε₂)
- Loss tangent (tan δ)
- Skin depth (nm)
- Conductivity (S/cm)
- Optical density
- Absorption coefficient
- Band gap information

### Data Export Columns
All calculated parameters are exported with corresponding wavelength or energy values, enabling further analysis in other tools.

## Troubleshooting

### No peaks detected
- Ensure data has sufficient signal-to-noise ratio
- Check that transmittance values are in valid range (0-1)
- Try adjusting prominence settings in code if needed

### Thickness estimation unreliable
- Ensure at least 2 pairs of maxima/minima are selected
- Verify auto-detected points are accurate (use manual selection if needed)
- Check substrate refractive index value

### Simulated spectrum doesn't match experimental
- Verify thickness value is in reasonable range
- Ensure envelopes are correctly interpolated
- Check that both Tmax and Tmin points are properly selected

### Plots not updating
- Try switching between tabs to trigger redraw
- Check that data was successfully imported
- Restart application if UI becomes unresponsive

## Keyboard & Mouse Shortcuts

- **Left Click**: Add/select envelope points (when mode active)
- **Right Click**: Remove nearest point
- **Drag**: Move selected envelope points
- **Scroll**: Zoom in/out plots
- **Pan Tool** (toolbar): Navigate plots

## File Structure
```
Opti-Envelope/
├── opti_envelope_app.py          # Main application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # License information
```

## Performance Notes

- Application optimized for datasets with 100-2000 data points
- Larger datasets may require more processing time for fitting
- Real-time rendering disabled for datasets >5000 points
- Suggested zoom range: 50-2000 nm for typical thin films

## Future Enhancements

Potential features for future versions:
- Multiple material layer support
- Automated material identification
- Temperature-dependent optical properties
- Fitting to Lorentz or Drude models
- Batch processing of multiple files
- Custom calculation plugins
- Advanced wavelet denoising

## References

The Swanepole (Envelope) method for optical property extraction is based on:
- C.J. Swanepoel, "Determination of the thickness and optical constants of amorphous silicon," J. Phys. E, 1983

## License

See LICENSE file for licensing information.

## Support & Contact

For bug reports, feature requests, or technical support, please contact the development team or open an issue in the repository.

---

**Version**: 1.0  
**Last Updated**: 2026  
**Developed for**: Optical Materials Research & Characterization

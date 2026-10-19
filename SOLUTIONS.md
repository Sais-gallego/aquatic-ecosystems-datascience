# Data Science for Monitoring Aquatic Ecosystems (DATA20067)
## Comprehensive Master's Course Solutions & Study Guide

**Course Code:** DATA20067 (DATA200671 MOOC 3 ECTS + DATA200672 Exam 2 ECTS)  
**Institution:** University of Helsinki - Faculty of Science, Department of Computer Science  
**Student:** Sergio Andrés Ipuz Saavedra (`@Sais-gallego`)  
**Repository:** [https://github.com/Sais-gallego/aquatic-ecosystems-datascience](https://github.com/Sais-gallego/aquatic-ecosystems-datascience)

---

## Table of Contents
1. [Course Overview & Multidisciplinary Framework](#1-course-overview--multidisciplinary-framework)
2. [Chapter 1: Domain Foundations & Scientific Context](#2-chapter-1-domain-foundations--scientific-context)
3. [Chapter 2: Technologies for Aquatic Monitoring](#3-chapter-2-technologies-for-aquatic-monitoring)
4. [Chapter 3: Data Quality, QA/QC & Feature Engineering](#4-chapter-3-data-quality-qaqc--feature-engineering)
5. [Chapter 4: Data Analysis and Machine Learning](#5-chapter-4-data-analysis-and-machine-learning)
6. [Chapter 5: Advanced Deep Learning & Remote Sensing](#6-chapter-5-advanced-deep-learning--remote-sensing)
7. [Chapter 6: Physics, Trust, and Operational Deployment](#7-chapter-6-physics-trust-and-operational-deployment)
8. [Comprehensive Quiz Solutions & Practice Q&A](#8-comprehensive-quiz-solutions--practice-qa)

---

## 1. Course Overview & Multidisciplinary Framework

Aquatic ecosystems (lakes, rivers, estuaries, and coastal seas) are under severe pressure from anthropogenic eutrophication, industrial runoff, and climate-induced thermal stratification changes. This course bridges limnology, physical oceanography, embedded edge sensing, and modern machine learning to build continuous real-time monitoring and early warning systems.

---

## 2. Chapter 1: Domain Foundations & Scientific Context

### 2.1 Key Water Quality Indicators
- **Dissolved Oxygen (DO):** The master variable of aquatic health. Photosynthesis generates DO during daylight; nocturnal cellular respiration and bacterial decay consume DO. Concentrations below $2.0\text{ mg/L}$ cause **hypoxia**; concentrations of $0.0\text{ mg/L}$ indicate **anoxia**.
- **Garcia-Gordon DO Saturation Formulation:**
  Solubility of oxygen is non-linear and governed by Henry's law, decreasing as water temperature and salinity increase:
  $$\ln C_o^*(T, S) = A_0 + A_1 T_s + A_2 T_s^2 + A_3 T_s^3 + A_4 T_s^4 + A_5 T_s^5 + S(B_0 + B_1 T_s + B_2 T_s^2 + B_3 T_s^3) + C_0 S^2$$
  where $T_s = \ln\left(\frac{298.15 - T}{273.15 + T}\right)$.
- **Chlorophyll-a (Chl-a):** Proxy for phytoplankton biomass ($\mu\text{g/L}$). Values above $20\ \mu\text{g/L}$ indicate eutrophic algal bloom conditions.
- **Turbidity & Secchi Depth:** Measure of water clarity and suspended particulate matter.

### 2.2 Thermal Stratification & Lake Physics
In dimictic and monomictic lakes, summer solar heating generates vertical thermal layers:
1. **Epilimnion:** Warm, circulating, well-oxygenated surface water layer.
2. **Metalimnion / Thermocline:** Rapid vertical thermal transition zone where $|-\partial T / \partial z| \ge 1.0^\circ\text{C/m}$.
3. **Hypolimnion:** Dense, cold ($4^\circ\text{C}$ in freshwater), isolated bottom layer prone to oxygen depletion.
- **Brunt-Väisälä Frequency ($N^2$):** Quantifies water column stability against turbulent vertical mixing:
  $$N^2 = \frac{g}{\rho_0} \frac{\partial \rho}{\partial z}$$
  $N^2 > 0$: Stable stratification; $N^2 < 0$: Convective instability.

### 2.3 Carlson's Trophic State Index (TSI)
Evaluates biological productivity from oligotrophic to hypereutrophic:
$$\text{TSI}(\text{Chl-a}) = 9.81 \ln(\text{Chl-a}) + 30.6$$
$$\text{TSI}(\text{Secchi}) = 60 - 14.41 \ln(\text{Secchi})$$

---

## 3. Chapter 2: Technologies for Aquatic Monitoring

### 3.1 In-Situ Sensing Platforms
- **Fixed Mooring Buoys:** Continuous high-frequency surface and vertical chain profiling.
- **CTD Sondes (Conductivity, Temperature, Depth):** High-precision inductive conductivity cells and platinum resistance thermometers.
- **Optical Fluorometers:** Stimulate chlorophyll pigments with blue light ($\approx 470\text{ nm}$) and measure red fluorescence emission ($\approx 685\text{ nm}$).
- **Autonomous Surface Vehicles (ASVs) & Gliders:** Spatial path planning for dynamic adaptive sampling.

### 3.2 Biofouling & Sensor Maintenance
- **Biofouling Mechanism:** Microorganisms and algae adhere to optical sensor windows, creating a biofilm that attenuates optical excitation and emission light, causing apparent chlorophyll signal decay.
- **Mitigation:** Mechanical wipers with copper brushes, UV irradiation LEDs, and automated drift compensation algorithms.

### 3.3 Remote Sensing & Satellite Telemetry
- **Copernicus Sentinel-2 & Sentinel-3 OLCI:** Multi-spectral bands capturing water surface reflectance:
  - **Normalized Difference Water Index (NDWI):**
    $$\text{NDWI} = \frac{R_{\text{Green}} - R_{\text{NIR}}}{R_{\text{Green}} + R_{\text{NIR}}}$$
  - **Floating Algae Index (FAI):** Detects surface cyanobacterial scum:
    $$\text{FAI} = R_{\text{NIR}} - \left(R_{\text{Red}} + (R_{\text{SWIR}} - R_{\text{Red}}) \frac{\lambda_{\text{NIR}} - \lambda_{\text{Red}}}{\lambda_{\text{SWIR}} - \lambda_{\text{Red}}}\right)$$

---

## 4. Chapter 3: Data Quality, QA/QC & Feature Engineering

### 4.1 IOOS QARTOD Automated Quality Control
The **Quality Assurance / Quality Control of Real-Time Oceanographic Data (QARTOD)** standard defines real-time flag assignments:
- `Flag 1 (Pass)`: Valid observation.
- `Flag 3 (Suspect / Warning)`: Abnormally high rate of change or climatological outlier.
- `Flag 4 (Fail)`: Defective sensor, gross out-of-range, or flatline.
- `Flag 9 (Missing)`: Sensor dropout or communication loss.

#### Five Core QARTOD Tests:
1. **Gross Range Test:** Physical limit check (e.g., $-2^\circ\text{C} \le T \le 45^\circ\text{C}$).
2. **Climatology Test:** Checks against historic seasonal limits for specific geographic coordinates.
3. **Spike Detection Test:** Flags sudden isolated single-sample excursions:
   $$|v_t - 0.5(v_{t-1} + v_{t+1})| > \tau_{\text{spike}}$$
4. **Rate of Change Test:** Flags $|\Delta v / \Delta t| > \tau_{\text{rate}}$.
5. **Flatline Test:** Detects frozen or stuck sensor outputs over $k$ consecutive steps ($v_{\max} - v_{\min} \le \epsilon$).

### 4.2 Feature Engineering
- **Apparent Oxygen Deficit (AOD):** $\text{AOD} = \text{DO}_{\text{sat}}(T, S) - \text{DO}_{\text{meas}}$.
- **Cyclical Temporal Encodings:** $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$.
- **Rolling Statistical Moments:** Trailing mean, variance, min, and max over 6-hour, 24-hour, and 7-day windows.

---

## 5. Chapter 4: Data Analysis and Machine Learning

### 5.1 Spatial Autocorrelation & Spatiotemporal Leakage
- **Moran's I:** Measures spatial clustering:
  $$I = \frac{N}{W} \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$
- **Data Leakage Hazard:** Random train/test splits in environmental time series or spatial grids produce artificially inflated validation scores due to spatial proximity and temporal autocorrelation.
- **Solution:** **Blocked Spatiotemporal Cross-Validation** (leaving entire river basins or entire seasonal months out of training).

### 5.2 Supervised Modeling of Algal Blooms
- **Chlorophyll-a Forecasting:** Regularized Ridge Regression or Gradient Boosting predicting $t+1$ concentration.
- **Harmful Algal Bloom (HAB) Risk Classifier:** Logistic Regression / Random Forest predicting $P(\text{Chl-a} \ge 20\ \mu\text{g/L})$.

---

## 6. Chapter 5: Advanced Deep Learning & Remote Sensing

### 6.1 Recurrent Neural Networks (LSTM & GRU)
Recurrent architectures model long-term vertical oxygen consumption and stratification memory. Gating equations handle vanishing gradients across multi-week time horizons.

### 6.2 Spatio-Temporal Graph Neural Networks (ST-GNN)
Hydrological river networks are modeled as directed graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$ where edge weights represent upstream-to-downstream water flow velocities. Graph convolutions propagate pollutant transport downstream in agreement with advective transport.

---

## 7. Chapter 6: Physics, Trust, and Operational Deployment

### 7.1 Physics-Informed Neural Networks (PINNs)
Pure data-driven models often violate basic physical principles (e.g., predicting negative water density or energy creation). A PINN incorporates the 1D vertical heat equation into the loss function:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \frac{1}{M}\sum \left| \frac{\partial T}{\partial t} - \kappa_z \frac{\partial^2 T}{\partial z^2} - \frac{H(z, t)}{\rho_w c_p} \right|^2$$

### 7.2 Model Explainability with SHAP
SHapley Additive exPlanations quantify feature attributions, verifying that models rely on scientifically plausible physical drivers (e.g. solar irradiance and water temperature driving chlorophyll growth) rather than spurious sensor artifacts.

---

## 8. Comprehensive Quiz Solutions & Practice Q&A

### Q1: Why does dissolved oxygen concentration decrease during night-time even when there is no surface turbulence?
**Answer:** During daylight hours, aquatic phytoplankton and macrophytes undergo oxygen-producing photosynthesis, often causing daytime oxygen supersaturation. At night, photosynthesis halts due to the absence of sunlight, but aquatic organisms, algae, and benthic bacteria continue cellular respiration and organic decomposition, consuming dissolved oxygen and causing nocturnal hypoxia.

### Q2: What is the primary difference between a QARTOD Rate of Change test and a Spike Detection test?
**Answer:** A Rate of Change test evaluates the difference between two consecutive samples ($|v_t - v_{t-1}| / \Delta t$) to detect unphysical shifts; if the shift is real (such as a sudden flood plume), subsequent samples will stay elevated. A Spike Detection test evaluates whether a single reading deviates drastically from both its preceding and succeeding neighbors ($|v_t - 0.5(v_{t-1} + v_{t+1})|$), identifying transient electronic glitches or debris passing the optical sensor window that immediately return to baseline.

### Q3: How does optical biofouling affect in-situ fluorometer readings over a 4-week deployment, and how can automated algorithms detect it?
**Answer:** Biofouling coats the optical fluorometer window with a biological film. This film absorbs blue excitation light and attenuates red fluorescence emissions, resulting in an artificial monotonic downward drift in measured chlorophyll-a concentrations. Algorithms detect biofouling by combining flatline tests, comparing against diurnal solar expectations, tracking mechanical wiper current, or evaluating discrepancies against unaffected acoustic or temperature sensors.

---

*(c) 2026 Sergio Andrés Ipuz Saavedra - Department of Computer Science, University of Helsinki.*

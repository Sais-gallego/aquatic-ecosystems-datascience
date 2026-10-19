# Data Science for Monitoring Aquatic Ecosystems (DATA20067)

[![CI Pipeline](https://github.com/Sais-gallego/aquatic-ecosystems-datascience/actions/workflows/ci.yml/badge.svg)](https://github.com/Sais-gallego/aquatic-ecosystems-datascience/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready implementation of the environmental data science pipelines, physical oceanographic algorithms, automated QA/QC test suites, and predictive modeling architectures for **Data Science for Monitoring Aquatic Ecosystems (DATA20067)** at the **University of Helsinki**.

---

## 🌊 Overview & System Architecture

This repository bridges environmental physics, in-situ sensor telemetry, and modern data science to monitor lakes, rivers, and coastal marine waters:

```
aquatic-ecosystems-datascience/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Multi-version Python CI pipeline
├── src/
│   ├── __init__.py
│   ├── domain_foundations.py      # Garcia-Gordon DO saturation, density, TSI, thermocline
│   ├── sensor_telemetry.py        # Buoy diurnal simulation, biofouling, LoRaWAN codec
│   ├── qa_qc_pipeline.py          # IOOS QARTOD real-time quality control & imputation
│   ├── feature_engineering.py     # Diurnal/annual cyclical encodings, AOD, rolling stats
│   ├── ml_models.py               # Chlorophyll-a ridge regression & HAB risk classifier
│   └── physics_informed.py        # 1D heat diffusion PDE residual loss (PINN)
├── tests/
│   ├── test_domain_foundations.py # Limnological physics tests
│   ├── test_sensor_telemetry.py   # Buoy simulation & 10-byte binary codec tests
│   ├── test_qa_qc.py              # QARTOD gross range, spike, rate of change, flatline
│   ├── test_feature_engineering.py# Temporal encodings & feature matrix builder tests
│   └── test_ml_and_physics.py     # ML forecasting, classifier, and PDE loss tests
├── SOLUTIONS.md                   # Complete 6-chapter theoretical guide & exam prep
└── README.md
```

---

## 🧪 Scientific & Mathematical Formulations

### 1. Dissolved Oxygen Saturation (Garcia & Gordon, 1992)
$$\ln C_o^*(T, S) = A_0 + A_1 T_s + A_2 T_s^2 + A_3 T_s^3 + A_4 T_s^4 + A_5 T_s^5 + S(B_0 + B_1 T_s + B_2 T_s^2 + B_3 T_s^3) + C_0 S^2$$

### 2. Carlson's Trophic State Index (TSI)
$$\text{TSI}(\text{Chl-a}) = 9.81 \ln(\text{Chl-a}) + 30.6$$
$$\text{TSI}(\text{Secchi}) = 60 - 14.41 \ln(\text{Secchi})$$

### 3. Physics-Informed 1D Thermal Diffusion PDE
$$\frac{\partial T}{\partial t} = \kappa_z \frac{\partial^2 T}{\partial z^2} + \frac{H(z, t)}{\rho_w c_p}$$
$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{phys}} \frac{1}{M}\sum |\mathcal{R}(T, z, t)|^2$$

---

## ⚡ Running Unit Tests

Execute the 21 automated unit tests locally:

```bash
python -m unittest discover tests -v
```

Output:
```
test_brunt_vaisala_frequency (test_domain_foundations.TestDomainFoundations.test_brunt_vaisala_frequency) ... ok
test_carlson_trophic_state_index (test_domain_foundations.TestDomainFoundations.test_carlson_trophic_state_index) ... ok
test_do_saturation_freshwater (test_domain_foundations.TestDomainFoundations.test_do_saturation_freshwater) ... ok
test_thermocline_detection (test_domain_foundations.TestDomainFoundations.test_thermocline_detection) ... ok
test_water_density (test_domain_foundations.TestDomainFoundations.test_water_density) ... ok
test_apparent_oxygen_deficit (test_feature_engineering.TestFeatureEngineering.test_apparent_oxygen_deficit) ... ok
test_build_ml_feature_matrix (test_feature_engineering.TestFeatureEngineering.test_build_ml_feature_matrix) ... ok
test_cyclical_time_features (test_feature_engineering.TestFeatureEngineering.test_cyclical_time_features) ... ok
test_rolling_statistics (test_feature_engineering.TestFeatureEngineering.test_rolling_statistics) ... ok
test_algae_bloom_classifier (test_ml_and_physics.TestMlAndPhysics.test_algae_bloom_classifier) ... ok
test_chlorophyll_forecaster (test_ml_and_physics.TestMlAndPhysics.test_chlorophyll_forecaster) ... ok
test_physics_heat_source (test_ml_and_physics.TestMlAndPhysics.test_physics_heat_source) ... ok
test_physics_pde_loss (test_ml_and_physics.TestMlAndPhysics.test_physics_pde_loss) ... ok
test_flatline_detection (test_qa_qc.TestQartodPipeline.test_flatline_detection) ... ok
test_gross_range (test_qa_qc.TestQartodPipeline.test_gross_range) ... ok
test_linear_imputation (test_qa_qc.TestQartodPipeline.test_linear_imputation) ... ok
test_rate_of_change (test_qa_qc.TestQartodPipeline.test_rate_of_change) ... ok
test_spike_detection (test_qa_qc.TestQartodPipeline.test_spike_detection) ... ok
test_biofouling_accumulation (test_sensor_telemetry.TestSensorTelemetry.test_biofouling_accumulation) ... ok
test_buoy_simulation_step (test_sensor_telemetry.TestSensorTelemetry.test_buoy_simulation_step) ... ok
test_lorawan_binary_codec (test_sensor_telemetry.TestSensorTelemetry.test_lorawan_binary_codec) ... ok

----------------------------------------------------------------------
Ran 21 tests in 0.008s

OK
```

---

## 👤 Author & Academic Details

- **Student:** Sergio Andrés Ipuz Saavedra
- **GitHub:** [@Sais-gallego](https://github.com/Sais-gallego)
- **Email:** `afranklinromero@gmail.com`
- **Institution:** University of Helsinki - Faculty of Science, Department of Computer Science
- **Course:** DATA20067 Data Science for Monitoring Aquatic Ecosystems (DATA200671 MOOC + DATA200672 Exam)

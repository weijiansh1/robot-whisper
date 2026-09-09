# State/action soft-routing layer and denoise effect audit

## Scope

Effects are fixed-initial-state standardized mean shifts. Positive values mean the failure class is higher than success. No classifier AUC and no hard expert ID is used.

State-token maximum probability difference across denoise steps: `0.00000000`.
Therefore state-token denoise steps are exact duplicates in this cache; state results are reported once per layer. Action-token routing is reported at denoise steps 0-9.
Denoise `d0` is the first, noisiest forward at flow time 1.0; `d9` is the last recorded forward at flow time 0.1 before the Euler update reaches 0.0.

## active_return_vs_success

Episodes `151` = positive `39` + success `112`; mixed initial states `9`.

### State layers

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L4 | entropy_std_query | -1.043 | [-1.823, +0.166] | -0.0217911 | 0.0005 | 0.0090 |
| L14 | query_speed_mean | -1.009 | [-1.404, -0.363] | -0.00349357 | 0.0005 | 0.0180 |
| L12 | entropy_mean | +0.840 | [+0.316, +1.551] | +0.00114475 | 0.0085 | 0.2019 |
| L4 | query_speed_mean | -0.837 | [-1.516, -0.058] | -0.00842905 | 0.0090 | 0.2144 |
| L12 | entropy_std_query | -0.835 | [-1.505, -0.040] | -0.00101354 | 0.0095 | 0.2199 |
| L4 | top1_std_query | -0.811 | [-1.666, +0.222] | -0.0196998 | 0.0140 | 0.2824 |
| L12 | top1_mean | -0.690 | [-1.600, -0.151] | -0.0018894 | 0.0600 | 0.7791 |
| L4 | entropy_mean | +0.639 | [-0.514, +1.490] | +0.0214326 | 0.1164 | 0.9155 |
| L14 | top1_mean | -0.589 | [-1.708, +0.523] | -0.00206274 | 0.2019 | 0.9895 |
| L12 | query_speed_mean | -0.588 | [-1.418, -0.086] | -0.00258982 | 0.2059 | 0.9895 |
| L4 | top1_mean | -0.577 | [-1.451, +0.556] | -0.0229493 | 0.2314 | 0.9925 |
| L14 | entropy_std_query | -0.546 | [-1.154, +0.102] | -0.000455859 | 0.3103 | 0.9985 |

### State front/back and relative-query scan

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15 | entropy_std_query | -0.855 | [-1.527, -0.313] | -0.000393956 | 0.0010 | 0.1704 |
| back_12_15 | entropy_mean | +0.849 | [+0.129, +1.914] | +0.000483738 | 0.0010 | 0.1814 |
| back_12_15 | top1_mean | -0.803 | [-1.830, -0.101] | -0.00115625 | 0.0015 | 0.3058 |
| back_12_15 | query_speed_mean | -0.736 | [-1.510, -0.277] | -0.0022501 | 0.0055 | 0.5807 |
| front_2_5 | entropy_mean | +0.610 | [+0.027, +1.217] | +0.006579 | 0.0385 | 0.9690 |
| front_2_5 | query_speed_mean | -0.563 | [-1.447, +0.124] | -0.00418879 | 0.0695 | 0.9960 |
| front_2_5 | top1_mean | -0.494 | [-1.018, +0.016] | -0.00730737 | 0.1609 | 1.0000 |
| back_12_15 | top1_std_query | -0.448 | [-1.079, +0.166] | -0.000517467 | 0.2559 | 1.0000 |

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15/q+4 | entropy | +0.909 | [+0.345, +1.522] | +0.00150266 | 0.0020 | 0.0810 |
| front_2_5/q+3 | top1_mass | -0.855 | [-1.316, +0.002] | -0.0231316 | 0.0050 | 0.1679 |
| front_2_5/q+3 | entropy | +0.833 | [-0.225, +1.356] | +0.018195 | 0.0080 | 0.2254 |
| front_2_5/q+4 | entropy | +0.709 | [+0.079, +1.343] | +0.0112225 | 0.0645 | 0.6967 |
| back_12_15/q+6 | p4p5_gap | +0.701 | [+0.211, +1.255] | +0.000741857 | 0.0710 | 0.7316 |
| front_2_5/q+5 | top1_mass | -0.700 | [-1.388, +0.213] | -0.0172664 | 0.0720 | 0.7361 |
| front_2_5/q+2 | top1_mass | -0.680 | [-1.313, +0.211] | -0.0283675 | 0.0865 | 0.8161 |
| back_12_15/q+4 | top1_mass | -0.667 | [-1.348, -0.126] | -0.00279252 | 0.1019 | 0.8576 |

### Action layer x denoise

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L15/d0 | entropy_mean | +1.066 | [+0.127, +1.726] | +5.89756e-05 | 0.0025 | 0.0060 |
| L15/d1 | entropy_mean | +0.955 | [+0.019, +1.588] | +5.62006e-05 | 0.0140 | 0.0445 |
| L15/d2 | entropy_mean | +0.923 | [+0.034, +1.536] | +5.90532e-05 | 0.0235 | 0.0685 |
| L15/d3 | entropy_mean | +0.896 | [+0.042, +1.484] | +6.32071e-05 | 0.0360 | 0.1024 |
| L14/d9 | entropy_std_query | -0.882 | [-1.656, -0.223] | -6.67613e-05 | 0.0405 | 0.1219 |
| L13/d9 | top1_std_query | -0.862 | [-1.410, -0.209] | -0.000255366 | 0.0500 | 0.1574 |
| L15/d4 | entropy_mean | +0.848 | [+0.019, +1.425] | +6.68476e-05 | 0.0605 | 0.1849 |
| L14/d8 | entropy_std_query | -0.832 | [-1.629, -0.104] | -4.76159e-05 | 0.0745 | 0.2289 |
| L15/d5 | entropy_mean | +0.817 | [+0.059, +1.420] | +7.08662e-05 | 0.0885 | 0.2659 |
| L14/d7 | entropy_std_query | -0.816 | [-1.589, -0.073] | -3.86203e-05 | 0.0900 | 0.2709 |
| L13/d0 | top1_std_query | -0.815 | [-1.416, -0.283] | -0.00012875 | 0.0905 | 0.2724 |
| L15/d0 | entropy_std_query | -0.807 | [-1.432, -0.247] | -5.73877e-05 | 0.1000 | 0.2949 |
| L15/d1 | entropy_std_query | -0.799 | [-1.411, -0.227] | -6.04349e-05 | 0.1109 | 0.3153 |
| L15/d6 | entropy_mean | +0.790 | [+0.037, +1.452] | +7.65354e-05 | 0.1224 | 0.3478 |
| L15/d2 | entropy_std_query | -0.783 | [-1.277, -0.296] | -6.42487e-05 | 0.1364 | 0.3798 |

### Action front/back x denoise

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15/d9 | top1_std_query | -1.098 | [-1.546, -0.561] | -0.00017615 | 0.0010 | 0.0045 |
| back_12_15/d2 | entropy_std_query | -1.049 | [-1.751, -0.507] | -3.14132e-05 | 0.0025 | 0.0075 |
| back_12_15/d1 | entropy_std_query | -1.047 | [-1.759, -0.486] | -2.92766e-05 | 0.0025 | 0.0075 |
| back_12_15/d0 | entropy_std_query | -1.016 | [-1.679, -0.490] | -2.63371e-05 | 0.0025 | 0.0170 |
| back_12_15/d3 | entropy_std_query | -1.014 | [-1.628, -0.540] | -3.35678e-05 | 0.0025 | 0.0170 |
| back_12_15/d4 | entropy_std_query | -0.973 | [-1.516, -0.543] | -3.5474e-05 | 0.0050 | 0.0325 |
| back_12_15/d5 | entropy_std_query | -0.911 | [-1.404, -0.526] | -3.48079e-05 | 0.0060 | 0.0790 |
| back_12_15/d6 | entropy_std_query | -0.854 | [-1.313, -0.461] | -3.49687e-05 | 0.0155 | 0.1709 |
| back_12_15/d0 | entropy_mean | +0.831 | [+0.098, +1.310] | +2.53522e-05 | 0.0230 | 0.2294 |
| back_12_15/d9 | entropy_std_query | -0.831 | [-1.153, -0.401] | -4.91926e-05 | 0.0230 | 0.2304 |
| back_12_15/d7 | entropy_std_query | -0.816 | [-1.313, -0.371] | -3.5771e-05 | 0.0265 | 0.2704 |
| back_12_15/d8 | entropy_std_query | -0.789 | [-1.281, -0.323] | -3.96168e-05 | 0.0360 | 0.3533 |
| back_12_15/d1 | entropy_mean | +0.764 | [-0.003, +1.317] | +2.52314e-05 | 0.0475 | 0.4508 |
| back_12_15/d2 | entropy_mean | +0.759 | [-0.000, +1.353] | +2.74715e-05 | 0.0490 | 0.4738 |
| back_12_15/d1 | top1_std_query | -0.744 | [-1.213, -0.392] | -0.000117736 | 0.0605 | 0.5472 |

### Soft expert-probability shifts

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L2/e7 | mean_probability | +1.613 | [+0.267, +2.503] | +0.00250044 | 0.0005 | 0.0005 |
| L2/e0 | mean_probability | +1.470 | [+0.346, +2.683] | +0.00248493 | 0.0005 | 0.0005 |
| L4/e14 | mean_probability | +1.412 | [+0.281, +2.421] | +0.0016978 | 0.0005 | 0.0005 |
| L13/e12 | mean_probability | +1.379 | [+0.122, +2.444] | +0.00179138 | 0.0005 | 0.0005 |
| L2/e24 | mean_probability | +1.326 | [+0.556, +2.043] | +0.00262868 | 0.0005 | 0.0005 |
| L3/e15 | mean_probability | -1.315 | [-2.064, -0.342] | -0.00280109 | 0.0005 | 0.0005 |
| L5/e29 | mean_probability | -1.309 | [-2.247, -0.221] | -0.0023292 | 0.0005 | 0.0005 |
| L3/e5 | mean_probability | -1.294 | [-2.276, -0.162] | -0.00209405 | 0.0005 | 0.0005 |

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L12/d6/e14 | mean_probability | +1.481 | [+0.341, +2.547] | +0.000834987 | 0.0005 | 0.0005 |
| L12/d7/e14 | mean_probability | +1.476 | [+0.386, +2.493] | +0.000854451 | 0.0005 | 0.0005 |
| L12/d8/e14 | mean_probability | +1.467 | [+0.448, +2.381] | +0.000877411 | 0.0005 | 0.0005 |
| L12/d5/e14 | mean_probability | +1.413 | [+0.255, +2.493] | +0.000773271 | 0.0005 | 0.0005 |
| L15/d9/e29 | mean_probability | -1.406 | [-2.363, -0.360] | -0.000823941 | 0.0005 | 0.0005 |
| L4/d8/e27 | mean_probability | -1.397 | [-2.054, -0.624] | -0.000295468 | 0.0005 | 0.0005 |
| L13/d2/e18 | mean_probability | -1.397 | [-2.128, -0.483] | -0.000331379 | 0.0005 | 0.0005 |
| L13/d4/e18 | mean_probability | -1.380 | [-2.101, -0.497] | -0.000365618 | 0.0005 | 0.0005 |

## stagnation_vs_success

Episodes `242` = positive `88` + success `154`; mixed initial states `10`.

### State layers

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L14 | top1_mean | -0.887 | [-1.525, -0.331] | -0.00254285 | 0.0005 | 0.0010 |
| L14 | entropy_mean | +0.835 | [+0.300, +1.499] | +0.000778751 | 0.0005 | 0.0020 |
| L4 | entropy_mean | +0.764 | [+0.363, +1.151] | +0.0230115 | 0.0005 | 0.0150 |
| L4 | entropy_std_query | -0.739 | [-1.199, -0.237] | -0.0150379 | 0.0005 | 0.0260 |
| L4 | top1_mean | -0.715 | [-1.125, -0.312] | -0.0247958 | 0.0005 | 0.0430 |
| L14 | top1_std_query | -0.710 | [-1.175, -0.247] | -0.00190217 | 0.0005 | 0.0475 |
| L14 | entropy_std_query | -0.658 | [-1.055, -0.215] | -0.000497133 | 0.0050 | 0.1334 |
| L12 | entropy_std_query | -0.600 | [-0.935, -0.280] | -0.000620672 | 0.0140 | 0.3638 |
| L4 | top1_std_query | -0.535 | [-1.033, +0.020] | -0.0124921 | 0.0510 | 0.7281 |
| L14 | query_speed_mean | -0.471 | [-1.026, +0.179] | -0.00156477 | 0.1369 | 0.9640 |
| L4 | p4p5_gap_mean | +0.444 | [-0.006, +0.965] | +0.000366309 | 0.2229 | 0.9905 |
| L15 | top1_std_query | +0.417 | [-0.009, +0.886] | +0.000459787 | 0.3113 | 0.9980 |

### State front/back and relative-query scan

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15 | top1_mean | -0.720 | [-1.132, -0.284] | -0.000839466 | 0.0010 | 0.0370 |
| back_12_15 | entropy_std_query | -0.653 | [-1.351, -0.127] | -0.000281965 | 0.0020 | 0.1469 |
| back_12_15 | entropy_mean | +0.599 | [+0.081, +1.179] | +0.000305497 | 0.0045 | 0.3663 |
| front_2_5 | entropy_mean | +0.565 | [+0.152, +0.847] | +0.00570109 | 0.0110 | 0.5577 |
| front_2_5 | top1_mean | -0.542 | [-0.827, -0.177] | -0.00702948 | 0.0155 | 0.6872 |
| back_12_15 | top1_std_query | -0.357 | [-0.721, -0.001] | -0.000381957 | 0.2599 | 1.0000 |
| back_12_15 | query_speed_mean | -0.289 | [-1.072, +0.415] | -0.000942211 | 0.5202 | 1.0000 |
| back_12_15 | p4p5_gap_mean | -0.114 | [-0.508, +0.259] | -3.83453e-05 | 0.9965 | 1.0000 |

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| front_2_5/q+4 | entropy | +0.884 | [+0.375, +1.289] | +0.01479 | 0.0005 | 0.0010 |
| front_2_5/q+5 | top1_mass | -0.833 | [-1.273, -0.378] | -0.0197517 | 0.0010 | 0.0025 |
| front_2_5/q+4 | top1_mass | -0.828 | [-1.213, -0.345] | -0.0188751 | 0.0010 | 0.0030 |
| front_2_5/q+5 | entropy | +0.815 | [+0.325, +1.325] | +0.0148673 | 0.0010 | 0.0035 |
| back_12_15/q+6 | p4p5_gap | +0.653 | [+0.276, +0.969] | +0.000634546 | 0.0075 | 0.1469 |
| back_12_15/q+6 | top1_mass | -0.560 | [-0.917, -0.191] | -0.00201669 | 0.0360 | 0.5862 |
| front_2_5/q+3 | top1_mass | -0.537 | [-1.070, -0.077] | -0.0140835 | 0.0565 | 0.7161 |
| front_2_5/q+3 | entropy | +0.485 | [-0.019, +1.023] | +0.0108889 | 0.1564 | 0.9335 |

### Action layer x denoise

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L15/d0 | entropy_mean | +0.738 | [+0.165, +1.222] | +4.10837e-05 | 0.0110 | 0.0275 |
| L15/d1 | entropy_mean | +0.660 | [+0.092, +1.137] | +3.88692e-05 | 0.0420 | 0.1289 |
| L15/d2 | entropy_mean | +0.614 | [+0.072, +1.075] | +3.94754e-05 | 0.1044 | 0.3013 |
| L2/d7 | denoise_delta_mean | -0.586 | [-0.963, -0.244] | -9.24749e-05 | 0.1719 | 0.4343 |
| L15/d3 | entropy_mean | +0.562 | [+0.034, +1.029] | +4.00069e-05 | 0.2549 | 0.5702 |
| L3/d1 | p4p5_gap_mean | +0.539 | [+0.281, +0.781] | +1.80443e-05 | 0.3573 | 0.7091 |
| L3/d7 | top1_mean | +0.517 | [+0.179, +0.927] | +0.000134069 | 0.4523 | 0.8196 |
| L15/d1 | entropy_std_query | -0.512 | [-1.010, -0.128] | -4.17491e-05 | 0.4758 | 0.8421 |
| L12/d8 | top1_mean | -0.511 | [-0.883, -0.068] | -0.000143569 | 0.4788 | 0.8451 |
| L15/d0 | entropy_std_query | -0.510 | [-0.986, -0.146] | -3.8376e-05 | 0.4883 | 0.8521 |
| L15/d4 | entropy_mean | +0.500 | [+0.014, +0.941] | +3.94138e-05 | 0.5502 | 0.8916 |
| L15/d2 | entropy_std_query | -0.495 | [-0.975, -0.117] | -4.44814e-05 | 0.5707 | 0.9045 |
| L15/d3 | entropy_std_query | -0.468 | [-0.922, -0.100] | -4.72178e-05 | 0.7256 | 0.9695 |
| L2/d1 | p4p5_gap_mean | -0.449 | [-0.725, -0.181] | -1.53354e-05 | 0.8291 | 0.9895 |
| L3/d6 | top1_mean | +0.438 | [+0.202, +0.731] | +0.000103124 | 0.8681 | 0.9935 |

### Action front/back x denoise

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15/d1 | entropy_std_query | -0.417 | [-0.991, +0.036] | -1.19989e-05 | 0.5007 | 0.9980 |
| back_12_15/d2 | entropy_std_query | -0.412 | [-0.973, +0.044] | -1.29487e-05 | 0.5237 | 1.0000 |
| back_12_15/d3 | entropy_std_query | -0.388 | [-0.934, +0.069] | -1.37733e-05 | 0.6522 | 1.0000 |
| back_12_15/d0 | entropy_std_query | -0.385 | [-0.912, +0.030] | -1.01602e-05 | 0.6677 | 1.0000 |
| back_12_15/d0 | entropy_mean | +0.348 | [-0.234, +0.892] | +9.05152e-06 | 0.8411 | 1.0000 |
| back_12_15/d4 | entropy_std_query | -0.316 | [-0.858, +0.161] | -1.25064e-05 | 0.9345 | 1.0000 |
| back_12_15/d1 | entropy_mean | +0.307 | [-0.301, +0.878] | +8.56526e-06 | 0.9560 | 1.0000 |
| back_12_15/d8 | p4p5_gap_mean | -0.305 | [-0.720, +0.118] | -8.43998e-06 | 0.9580 | 1.0000 |
| back_12_15/d2 | entropy_mean | +0.305 | [-0.284, +0.879] | +9.23702e-06 | 0.9580 | 1.0000 |
| front_2_5/d5 | p4p5_gap_mean | +0.277 | [+0.073, +0.498] | +6.17512e-06 | 0.9895 | 1.0000 |
| back_12_15/d1 | top1_std_query | -0.275 | [-0.586, +0.006] | -4.47336e-05 | 0.9900 | 1.0000 |
| back_12_15/d6 | p4p5_gap_mean | +0.263 | [-0.195, +0.749] | +6.806e-06 | 0.9965 | 1.0000 |
| back_12_15/d3 | entropy_mean | +0.256 | [-0.325, +0.840] | +8.66632e-06 | 0.9975 | 1.0000 |
| front_2_5/d7 | denoise_delta_mean | -0.251 | [-0.469, -0.065] | -4.66187e-05 | 0.9975 | 1.0000 |
| back_12_15/d0 | top1_std_query | -0.237 | [-0.546, +0.048] | -3.63731e-05 | 0.9995 | 1.0000 |

### Soft expert-probability shifts

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L13/e12 | mean_probability | +1.314 | [+0.521, +2.077] | +0.00164627 | 0.0005 | 0.0005 |
| L13/e21 | mean_probability | -1.231 | [-2.035, -0.539] | -0.00187557 | 0.0005 | 0.0005 |
| L2/e0 | mean_probability | +1.157 | [+0.472, +1.939] | +0.00203115 | 0.0005 | 0.0005 |
| L14/e2 | mean_probability | +1.153 | [+0.647, +1.733] | +0.00152292 | 0.0005 | 0.0005 |
| L15/e4 | mean_probability | -1.076 | [-1.721, -0.567] | -0.00137588 | 0.0005 | 0.0005 |
| L4/e0 | mean_probability | +1.060 | [+0.493, +1.696] | +0.0021164 | 0.0005 | 0.0005 |
| L12/e1 | mean_probability | +1.016 | [+0.401, +1.667] | +0.00129398 | 0.0005 | 0.0005 |
| L13/e2 | mean_probability | -1.002 | [-1.539, -0.444] | -0.001417 | 0.0005 | 0.0005 |

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L12/d8/e14 | mean_probability | +1.266 | [+0.566, +1.966] | +0.00067243 | 0.0005 | 0.0005 |
| L12/d7/e14 | mean_probability | +1.258 | [+0.490, +2.028] | +0.000640262 | 0.0005 | 0.0005 |
| L12/d6/e14 | mean_probability | +1.256 | [+0.474, +2.071] | +0.000620875 | 0.0005 | 0.0005 |
| L12/d9/e14 | mean_probability | +1.229 | [+0.537, +1.880] | +0.000697868 | 0.0005 | 0.0005 |
| L12/d5/e14 | mean_probability | +1.229 | [+0.430, +2.067] | +0.000589754 | 0.0005 | 0.0005 |
| L2/d9/e2 | mean_probability | -1.222 | [-1.654, -0.731] | -0.000246998 | 0.0005 | 0.0005 |
| L12/d3/e14 | mean_probability | +1.203 | [+0.408, +2.025] | +0.000513617 | 0.0005 | 0.0005 |
| L12/d4/e14 | mean_probability | +1.200 | [+0.409, +2.039] | +0.000551456 | 0.0005 | 0.0005 |

## any_failure_vs_success

Episodes `318` = positive `127` + success `191`; mixed initial states `12`.

### State layers

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L4 | entropy_std_query | -0.885 | [-1.331, -0.318] | -0.018293 | 0.0005 | 0.0005 |
| L12 | entropy_std_query | -0.823 | [-1.239, -0.404] | -0.000859098 | 0.0005 | 0.0005 |
| L4 | entropy_mean | +0.705 | [+0.205, +1.146] | +0.022528 | 0.0005 | 0.0045 |
| L14 | query_speed_mean | -0.694 | [-1.160, -0.070] | -0.00232462 | 0.0005 | 0.0070 |
| L14 | top1_mean | -0.693 | [-1.445, +0.052] | -0.00212861 | 0.0005 | 0.0080 |
| L4 | top1_std_query | -0.691 | [-1.164, -0.102] | -0.0162606 | 0.0005 | 0.0080 |
| L4 | top1_mean | -0.637 | [-1.105, -0.134] | -0.0237245 | 0.0015 | 0.0455 |
| L14 | entropy_mean | +0.617 | [-0.026, +1.305] | +0.000625557 | 0.0020 | 0.0690 |
| L14 | top1_std_query | -0.615 | [-1.084, -0.114] | -0.00168226 | 0.0020 | 0.0710 |
| L14 | entropy_std_query | -0.602 | [-0.982, -0.121] | -0.000467384 | 0.0025 | 0.0945 |
| L12 | entropy_mean | +0.472 | [+0.022, +0.943] | +0.000610607 | 0.0545 | 0.7441 |
| L12 | query_speed_mean | -0.417 | [-1.142, +0.138] | -0.00179774 | 0.1624 | 0.9655 |

### State front/back and relative-query scan

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15 | entropy_std_query | -0.856 | [-1.471, -0.357] | -0.000364141 | 0.0005 | 0.0005 |
| back_12_15 | top1_mean | -0.702 | [-1.253, -0.207] | -0.000937229 | 0.0005 | 0.0055 |
| back_12_15 | entropy_mean | +0.655 | [+0.168, +1.281] | +0.000367288 | 0.0005 | 0.0275 |
| front_2_5 | entropy_mean | +0.622 | [+0.260, +0.886] | +0.00646656 | 0.0005 | 0.0600 |
| front_2_5 | top1_mean | -0.570 | [-0.848, -0.240] | -0.00781031 | 0.0010 | 0.1789 |
| back_12_15 | query_speed_mean | -0.470 | [-1.218, +0.158] | -0.00151315 | 0.0120 | 0.7561 |
| back_12_15 | top1_std_query | -0.431 | [-0.793, -0.044] | -0.000458517 | 0.0235 | 0.9325 |
| front_2_5 | query_speed_mean | -0.214 | [-0.750, +0.300] | -0.00164922 | 0.7096 | 1.0000 |

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| front_2_5/q+4 | entropy | +0.864 | [+0.453, +1.175] | +0.0141088 | 0.0005 | 0.0005 |
| front_2_5/q+4 | top1_mass | -0.815 | [-1.135, -0.404] | -0.0182653 | 0.0005 | 0.0005 |
| front_2_5/q+5 | top1_mass | -0.786 | [-1.269, -0.267] | -0.0186545 | 0.0005 | 0.0005 |
| front_2_5/q+5 | entropy | +0.755 | [+0.191, +1.288] | +0.013824 | 0.0005 | 0.0015 |
| back_12_15/q+4 | entropy | +0.702 | [+0.152, +1.205] | +0.00106595 | 0.0005 | 0.0060 |
| back_12_15/q+6 | p4p5_gap | +0.688 | [+0.295, +1.041] | +0.000688412 | 0.0005 | 0.0085 |
| front_2_5/q+3 | top1_mass | -0.668 | [-1.085, -0.225] | -0.0185197 | 0.0010 | 0.0185 |
| front_2_5/q+3 | entropy | +0.621 | [+0.146, +1.039] | +0.0145432 | 0.0020 | 0.0610 |

### Action layer x denoise

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L15/d0 | entropy_mean | +0.888 | [+0.236, +1.423] | +4.82696e-05 | 0.0005 | 0.0005 |
| L15/d1 | entropy_mean | +0.815 | [+0.169, +1.347] | +4.68243e-05 | 0.0005 | 0.0005 |
| L15/d2 | entropy_mean | +0.772 | [+0.150, +1.303] | +4.86993e-05 | 0.0010 | 0.0010 |
| L15/d3 | entropy_mean | +0.737 | [+0.133, +1.260] | +5.14602e-05 | 0.0015 | 0.0035 |
| L15/d4 | entropy_mean | +0.685 | [+0.116, +1.198] | +5.3272e-05 | 0.0020 | 0.0110 |
| L15/d5 | entropy_mean | +0.629 | [+0.096, +1.131] | +5.44525e-05 | 0.0155 | 0.0525 |
| L13/d9 | top1_std_query | -0.619 | [-0.994, -0.179] | -0.000164771 | 0.0205 | 0.0660 |
| L15/d6 | entropy_mean | +0.587 | [+0.094, +1.097] | +5.73028e-05 | 0.0345 | 0.1234 |
| L12/d8 | top1_mean | -0.571 | [-0.973, -0.171] | -0.000172381 | 0.0525 | 0.1769 |
| L2/d7 | denoise_delta_mean | -0.569 | [-0.826, -0.252] | -8.8309e-05 | 0.0530 | 0.1799 |
| L15/d0 | entropy_std_query | -0.565 | [-1.040, -0.168] | -4.13058e-05 | 0.0585 | 0.1939 |
| L15/d1 | entropy_std_query | -0.560 | [-1.040, -0.150] | -4.41715e-05 | 0.0645 | 0.2119 |
| L3/d1 | p4p5_gap_mean | +0.557 | [+0.301, +0.793] | +1.84687e-05 | 0.0715 | 0.2284 |
| L13/d0 | top1_std_query | -0.547 | [-0.820, -0.254] | -8.63671e-05 | 0.0925 | 0.2764 |
| L15/d2 | entropy_std_query | -0.543 | [-0.990, -0.166] | -4.68753e-05 | 0.1034 | 0.2949 |

### Action front/back x denoise

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| back_12_15/d1 | entropy_std_query | -0.598 | [-1.124, -0.116] | -1.69057e-05 | 0.0045 | 0.1000 |
| back_12_15/d0 | entropy_mean | +0.594 | [-0.042, +1.131] | +1.6223e-05 | 0.0045 | 0.1029 |
| back_12_15/d2 | entropy_std_query | -0.586 | [-1.103, -0.111] | -1.79836e-05 | 0.0065 | 0.1274 |
| back_12_15/d0 | entropy_std_query | -0.576 | [-1.063, -0.131] | -1.49823e-05 | 0.0085 | 0.1604 |
| back_12_15/d2 | entropy_mean | +0.562 | [-0.079, +1.120] | +1.79618e-05 | 0.0135 | 0.2079 |
| back_12_15/d1 | entropy_mean | +0.557 | [-0.098, +1.117] | +1.6334e-05 | 0.0160 | 0.2244 |
| back_12_15/d3 | entropy_std_query | -0.547 | [-1.041, -0.091] | -1.88033e-05 | 0.0225 | 0.2714 |
| back_12_15/d9 | top1_std_query | -0.524 | [-0.950, -0.153] | -8.22086e-05 | 0.0360 | 0.4118 |
| back_12_15/d3 | entropy_mean | +0.515 | [-0.127, +1.089] | +1.84266e-05 | 0.0440 | 0.4698 |
| back_12_15/d4 | entropy_mean | +0.491 | [-0.156, +1.072] | +1.97234e-05 | 0.0810 | 0.6242 |
| back_12_15/d4 | entropy_std_query | -0.481 | [-0.952, -0.032] | -1.83634e-05 | 0.0920 | 0.6827 |
| back_12_15/d7 | entropy_mean | +0.467 | [-0.174, +1.041] | +2.90468e-05 | 0.1224 | 0.7761 |
| back_12_15/d6 | entropy_mean | +0.465 | [-0.181, +1.050] | +2.45527e-05 | 0.1234 | 0.7811 |
| back_12_15/d5 | entropy_mean | +0.459 | [-0.179, +1.056] | +2.10734e-05 | 0.1394 | 0.8101 |
| back_12_15/d8 | entropy_mean | +0.446 | [-0.202, +1.020] | +3.39723e-05 | 0.1724 | 0.8811 |

### Soft expert-probability shifts

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L13/e12 | mean_probability | +1.392 | [+0.590, +2.064] | +0.00178202 | 0.0005 | 0.0005 |
| L13/e21 | mean_probability | -1.338 | [-2.061, -0.609] | -0.00205492 | 0.0005 | 0.0005 |
| L2/e0 | mean_probability | +1.337 | [+0.596, +2.081] | +0.00232767 | 0.0005 | 0.0005 |
| L14/e2 | mean_probability | +1.235 | [+0.790, +1.731] | +0.00161947 | 0.0005 | 0.0005 |
| L4/e0 | mean_probability | +1.198 | [+0.568, +1.779] | +0.00237427 | 0.0005 | 0.0005 |
| L12/e11 | mean_probability | -1.098 | [-1.569, -0.606] | -0.00227431 | 0.0005 | 0.0005 |
| L4/e14 | mean_probability | +1.084 | [+0.443, +1.632] | +0.00125092 | 0.0005 | 0.0005 |
| L13/e1 | mean_probability | +1.054 | [+0.303, +1.677] | +0.00186492 | 0.0005 | 0.0005 |

| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |
|---|---|---:|---:|---:|---:|---:|
| L12/d6/e14 | mean_probability | +1.404 | [+0.687, +2.071] | +0.000729451 | 0.0005 | 0.0005 |
| L12/d7/e14 | mean_probability | +1.400 | [+0.701, +2.036] | +0.000746275 | 0.0005 | 0.0005 |
| L12/d8/e14 | mean_probability | +1.392 | [+0.719, +1.957] | +0.000771739 | 0.0005 | 0.0005 |
| L12/d5/e14 | mean_probability | +1.367 | [+0.638, +2.059] | +0.000689075 | 0.0005 | 0.0005 |
| L12/d4/e14 | mean_probability | +1.327 | [+0.601, +2.017] | +0.000641122 | 0.0005 | 0.0005 |
| L12/d9/e14 | mean_probability | +1.321 | [+0.710, +1.837] | +0.000781902 | 0.0005 | 0.0005 |
| L12/d3/e14 | mean_probability | +1.317 | [+0.591, +1.999] | +0.000593367 | 0.0005 | 0.0005 |
| L12/d2/e14 | mean_probability | +1.284 | [+0.563, +1.948] | +0.000540367 | 0.0005 | 0.0005 |

## Statistical interpretation

- `effect sigma` is the class shift divided by within-init residual SD.
- Labels are permuted within initial state. `max-T family p` corrects the full search inside one feature family; `max-T global p` corrects all scanned cells in the contrast.
- Bootstrap intervals are written for the 20 largest absolute effects in each feature family and resample initial states; they do not account for choosing this analysis after inspecting earlier results.
- The active-return contrast is exploratory and has only 39 positive episodes across nine mixed initial states.

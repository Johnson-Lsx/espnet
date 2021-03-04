# RESULTS
## Environments
- date: `Tue Jan 26 18:50:58 CST 2021`
- python version: `3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.6.0`
- Git hash: `7c1b696d80ed4c0648d63d164ee1f238810d9ec9`
  - Commit date: `Fri Jan 22 15:08:19 2021 +0800`


## enh_train_enh_rnn_tf_raw

config: ./conf/tuning/train_enh_rnn_tf.yaml

| dataset                     | PESQ    | STOI     | SAR     | SDR     | SIR     | SI_SNR  |
| --------------------------- | ------- | -------- | ------- | ------- | ------- | ------- |
| enhanced_cv_mix_both_min_8k | 2.02894 | 0.746114 | 6.01086 | 4.96165 | 15.1449 | 3.97731 |
| enhanced_tt_mix_both_min_8k | 2.04627 | 0.768593 | 6.16839 | 5.09177 | 15.1774 | 4.08659 |

## Environments

- date: `Wed Feb  3 04:46:30 CST 2021`
- python version: `3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.6.0`
- Git hash: `7c1b696d80ed4c0648d63d164ee1f238810d9ec9`
  - Commit date: `Fri Jan 22 15:08:19 2021 +0800`


## enh_train_enh_dprnn_tasnet_default_raw

config: ./conf/tuning/train_enh_dprnn_tasnet_default.yaml

| dataset                     | PESQ    | STOI     | SAR     | SDR     | SIR     | SI_SNR  |
| --------------------------- | ------- | -------- | ------- | ------- | ------- | ------- |
| enhanced_cv_mix_both_min_8k | 2.58643 | 0.855801 | 10.5699 | 10.2788 | 25.3606 | 9.62627 |
| enhanced_tt_mix_both_min_8k | 2.57072 | 0.881034 | 10.6937 | 10.356  | 25.0537 | 9.73456 |

## Environments
- date: `Thu Feb  4 12:17:13 CST 2021`
- python version: `3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.6.0`
- Git hash: `7c1b696d80ed4c0648d63d164ee1f238810d9ec9`
  - Commit date: `Fri Jan 22 15:08:19 2021 +0800`


## original_input_raw

config: original_input

| dataset                     | PESQ    | STOI     | SAR     | SDR      | SIR      | SI_SNR   |
| --------------------------- | ------- | -------- | ------- | -------- | -------- | -------- |
| enhanced_cv_mix_both_min_8k | 1.66104 | 0.602989 | 2.00169 | -3.85956 | 0.153048 | -4.11303 |
| enhanced_tt_mix_both_min_8k | 1.65666 | 0.627758 | 1.39529 | -4.23063 | 0.145262 | -4.48842 |
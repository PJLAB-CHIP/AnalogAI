# AnalogAI: A General Accuracy Recovery Framework for Improving the Robustness of DNN
# Overview
Analog in memory Computing (AIMC) is a promising method for accelerating deep neural networks (DNNs) and
reducing energy consumption. However, instabilities caused by
manufacturing and device variations limit its accuracy and
reliability. To cope with complex patterns of realistic noise
behavior and to build a unified framework that leverages the
strengths of various algorithms, we
introduce **AnalogAI**, a new open-source framework designed to
flexibly recover accuracy in analog DNNs. This framework is
centered on a multi-scale noise-aware training approach. The
multi-noise fusion injection (FI) strategy working with multiple
feedback loops (FL) is designed to achieve multi-scale noise
awareness by aggregating distributed weights while improving
training stability. Additionally, the framework enables the design
of custom quantization-aware and supports a range of noise-
insensitive training algorithms for fast accuracy recovery studies.
![The overview of AnalogAI](./figures/overview.png)
# Usage
We provide one example to illustrate the usage of the code. For the IRS instance, we run resent8 with device relative variation 0.3. Training for 200 epochs and then conduct 200 times inference w/ noise Monte Carlo simulation (parameters can be modified in config).

```bash
python res18_main.py \
  --mode tnt \
  --type irs \
```
# Analog Inference
```bash
python res18_main.py \
  --mode tnt \
  --type irs \
```

# Accuracy Improvements
Our proposed multi-scale noise-aware training **(MSNAT)** improves the accuracy compare to vanilla training, single noise injection, MSNAT  without (w/o) fusion injection (FI) strategy and multiple feedback loops (FL).

<div style="text-align: center;"> <img src="./figures/experiments.png" alt="Test results" width="500"/> </div>



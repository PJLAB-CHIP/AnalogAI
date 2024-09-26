# AnalogAI: A General Accuracy Recovery Framework for Improving the Robustness of DNN
# Overview
Analog in memory Computing (AIMC) is a promising method for accelerating deep neural networks (DNNs) and
reducing energy consumption. However, instabilities caused by
manufacturing and device variations limit its accuracy and
reliability. Previous studies typically focus on improving the non-idealities of specific devices and evaluate under ideal conditions.
They struggle to cope with complex patterns of realistic noise
behavior and lack a unified framework that leverages the
strengths of various algorithms. Here, we
introduce AnalogAI, a new open-source framework designed to
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
## Analog Inference
## Accuracy Recovery

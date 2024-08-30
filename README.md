# Examining the Threat Landscape: Foundation Models and Model Stealing

This is the official PyTorch implementation for the paper "Examining the Threat Landscape: Foundation Models and Model Stealing" accepted in BMVC 2024.

## Abstract

Foundation models (FMs) for computer vision learn rich and robust representations, enabling their adaptation to task/domain-specific deployments with little to no fine-tuning. However, we posit that the very same strength can make applications based on FMs vulnerable to model stealing attacks. Through empirical analysis, we reveal that models fine-tuned from FMs harbor heightened susceptibility to model stealing, compared to conventional vision architectures like ResNets. We hypothesize that this behavior is due to the comprehensive encoding of visual patterns and features learned by FMs during pre-training, which are accessible to both the attacker and the victim. We report that an attacker is able to obtain 94.28% agreement (matched predictions with victim) for a Vision Transformer based victim model (ViT-L/16) trained on CIFAR-10 dataset, compared to only 73.20% agreement for a ResNet-18 victim, when using ViT-L/16 as the thief model. We arguably show, for the first time, that utilizing FMs for downstream tasks may not be the best choice for deployment in commercial APIs due to their susceptibility to model theft. We thereby alert model owners towards the associated security risks, and highlight the need for robust security measures to safeguard such models against theft.

## Code

To be released soon ...

<div style="text-align: center;">
  <h1>MLLM-SUL: Multimodal Large Language Model for Semantic Scene Understanding and Localization in Traffic Scenarios</h1>
</div>  

> Multimodal large language models (MLLMs) have shown satisfactory effects in many autonomous driving tasks. In this paper, MLLMs are utilized to solve joint semantic scene understanding and risk localization tasks, while only relying on front-view images. In the proposed MLLM-SUL framework, a dual-branch visual encoder is first designed to extract features from two resolutions, and rich visual information is conducive to the language model describing risk objects of different sizes accurately. Then for the language generation, LLaMA model is fine-tuned to predict scene descriptions, containing the type of driving scenario, actions of risk objects, and driving intentions and suggestions of ego-vehicle. Ultimately, a transformer-based network incorporating a regression token is trained to locate the risk objects. Extensive experiments on the existing DRAMA-ROLISP dataset and the extended DRAMA-SRIS dataset demonstrate that our method is efficient, surpassing many state-of-the-art image-based and video-based methods. Specifically, our method achieves 80.1% BLEU-1 score and 298.5% CIDEr score in the scene understanding task, and 59.6% accuracy in the localization task.

If you have any question, please feel free to email fanjq@tongji.edu.cn.  

## :fire: News
- This paper is submitted to IROS 2025.
- Video demo at: xxxxx
- DRAMA-SRIS dataset at: data/drama/DRAMA-SRIS_dataset_compare/integrated_v7_3_r4_0211.json

## :pill: Installation
1. pytorch
2. torchvision

## :star: Train & Inference
1. 



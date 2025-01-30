# ELMo-reproduce
reproduce the ELMo model from the paper 《Deep contextualized word representations》

## Abstract
We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.





## Project Structure
```Plain Text
elmo_project/  
├── config.py           # 配置文件  
├── data/              # 数据目录  
├── models/            # 模型定义  
│   ├── __init__.py  
│   └── elmo.py  
├── train.py           # 预训练脚本  
├── finetune.py        # 微调脚本  
└── utils.py           # 工具函数
```







## Citation
```bibtex
@misc{peters2018deepcontextualizedwordrepresentations,
      title={Deep contextualized word representations}, 
      author={Matthew E. Peters and Mark Neumann and Mohit Iyyer and Matt Gardner and Christopher Clark and Kenton Lee and Luke Zettlemoyer},
      year={2018},
      eprint={1802.05365},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1802.05365}, 
}

```

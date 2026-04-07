---
license: apache-2.0
library_name: transformers
tags:
  - trl
  - peft
  - lora
---

# {{ model_name | default("Adapter") }}

This repository contains a PEFT/LoRA adapter trained with TRL.

## Base model

{{ base_model | default("unknown") }}

## Training

- **Trainer**: {{ trainer_name | default("TRL") }}
- **Dataset**: {{ dataset_name | default("unknown") }}


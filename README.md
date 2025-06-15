# Exploring Deep Reasoning Paradigms of Large Language Models for Table-based Question Answering

## Step 1: Evaluation of Base and Deep-Reasoning Large Language Models

To assess the performance of both base LLMs and deep reasoning-augmented LLMs, we provide the following evaluation scripts:

* Run `python llm_inference/inference_deepseek_hitab_chat.py` to perform inference on four benchmark datasets: **WikiTQ**, **HiTab**, **TabFact**, and **InfoTabs**.

* Use the following scripts to evaluate model outputs:

  * `python eval_scripts/eval_tabfact_infotabs.py` for evaluating performance on **TabFact** and **InfoTabs**.
  * `python eval_scripts/eval_wikitq_hitab.py` for evaluating **WikiTQ** and **HiTab**.

We also offer an **optional fuzzy matching evaluation** for WikiTQ and HiTab, to address minor lexical or formatting mismatches (e.g., `1,200` vs. `1200`, or `56%` vs. `56 percentage`). This is particularly useful for evaluating semantic equivalence between model predictions and gold answers. For implementation details, refer to `eval_scripts/utils/llm_check_fuzzy.py`.

We observed that using fuzzy matching significantly alters the evaluation outcome. For instance, on WikiTQ, fuzzy evaluation achieves **84.6** accuracy, compared to **82.38** with strict matching.

---

## Step 2: Generation of Reasoning Trajectories

### Step 2.1: Generation of Shallow Reasoning Data

To generate shallow reasoning traces using the **Qwen2.5-14B-Instruct** model, execute:

```bash
python make/distill_data/generate_base_data.py
```

### Step 2.2: Filtering Correctly Answered Instances (using HiTab as an example)

Extract instances where the model's predictions are correct:

```bash
python first_filter_data/filter_data_hitab.py
```

### Step 2.3: Generation of Deep Reasoning Traces via Multi-Sampling

For harder examples, deep reasoning data is synthesized using a multi-sampling strategy:

```bash
python solve_hard_samples/multiple_samlping.py
```

### Step 2.4: Hesitation-Driven Exploration for Hard Cases

To further explore challenging cases, we apply a **hesitation-driven exploration** approach:

```bash
python solve_hard_samples/deep_thinking.py
```

### Step 2.5: Integration of Shallow and Deep Reasoning Data

Unify the generated reasoning trajectories into a single training dataset:

```bash
python convert_chat_messages.py
```

The reasoning format is as follows:

* **Shallow reasoning**:
  `<think>\n\n</think>{solution}`
* **Deep reasoning**:
  `<think>\n{deep_thinking_trace}</think>{solution}`

---

## Step 3: Supervised Fine-Tuning (SFT) of the LLM

### Step 3.1: Configuration of Training Parameters

Edit the configuration file `open-r1-main/recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_peft.yaml` to define:

* Base model
* Learning rate
* Number of training epochs
* Batch size
* LoRA-related parameters (for parameter-efficient tuning)

### Step 3.2: Execute Fine-Tuning

Launch the supervised fine-tuning procedure using `Accelerate`:

```bash
accelerate launch \
  --config_file=open-r1-main/recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_peft.yaml \
  open-r1-main/src/open_r1/sft.py
```

---

### Some essential environment packages include:

numpy=1.25.1  
torch=2.5.1  
transformers=4.52.4  
trl=0.18.1  
openai=1.84.0  
peft=0.15.2  
flash-attn=2.7.3  



# QLoRA Fine-tuning for Dialogue Summarization

This project demonstrates the efficient fine-tuning of a Large Language Model (LLM) for dialogue summarization using **QLoRA (Quantized Low-Rank Adapters)**. QLoRA is a highly efficient technique that allows for fine-tuning large language models on a single GPU by quantizing the pre-trained LLM to 4 bits and adding small "Low-Rank Adapters."

## 🌟 Features

* **Efficient Fine-tuning**: Utilizes QLoRA to fine-tune LLMs with significantly reduced memory footprint.

* **4-bit Quantization**: Loads the base LLM in 4-bit precision, enabling training on consumer-grade GPUs.

* **PEFT Integration**: Leverages the Hugging Face `PEFT` (Parameter-Efficient Fine-tuning) library for seamless implementation of LoRA.

* **Dialogue Summarization**: Fine-tunes the model specifically for the task of summarizing conversational dialogues.

* **Qualitative & Quantitative Evaluation**: Includes steps for both human evaluation and ROUGE metric-based assessment of the fine-tuned model's performance.

## 📚 Dataset

The project uses the **DialogSum** dataset, available on Hugging Face (`knkarthick/dialogsum`). This dataset consists of conversational dialogues paired with human-written summaries, making it ideal for training summarization models.

## 🧠 Base Model

The base Large Language Model used for fine-tuning is:

* **`meta-llama/Llama-3.2-3B-Instruct`**

## 🚀 Setup and Installation

To run this notebook, you'll need to install the required libraries and set up your Hugging Face token for model access.

1. **Install Libraries**:
   All necessary libraries can be installed using `pip`:
   2. **Hugging Face Authentication**:
The `Llama-3.2-3B-Instruct` model requires authentication to access.

* **Kaggle Method (Recommended for Kaggle Notebooks)**: Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`. The notebook automatically attempts to log in using this secret.

* **Manual Method**: If not on Kaggle or if the secret fails, you can manually log in by running `from huggingface_hub import login; login(token="YOUR_HF_TOKEN")` and replacing `"YOUR_HF_TOKEN"` with your actual token.

## 📝 Notebook Structure

The notebook is structured into the following key sections:

1. **Install all the required libraries**: Sets up the environment.

2. **Loading dataset**: Loads and inspects the DialogSum dataset.

3. **Create bitsandbytes configuration**: Defines the 4-bit quantization settings.

4. **Load Base Model**: Loads the `Llama-3.2-3B-Instruct` model with quantization.

5. **Tokenization**: Configures the tokenizer for the model.

6. **Test the Model with Zero Shot Inferencing**: Demonstrates the base model's performance before fine-tuning.

7. **Pre-processing dataset**: Formats and tokenizes the dataset for training.

8. **Setup the PEFT/LoRA model for Fine-Tuning**: Applies LoRA configuration to the base model, significantly reducing trainable parameters.

9. **Train PEFT Adapter**: Trains the LoRA adapter using the `transformers.Trainer`.

10. **Evaluate the Model Qualitatively (Human Evaluation)**: Provides an example of the fine-tuned model's output compared to a human summary.

11. **Evaluate the Model Quantitatively (with ROUGE Metric)**: Calculates ROUGE scores to numerically assess the fine-tuned model's performance against human baselines.

## ✨ Results

The fine-tuning process significantly reduces the number of trainable parameters, making it feasible to train large models on less powerful hardware. Both qualitative and quantitative evaluations are performed to demonstrate the improvements in summarization capabilities after fine-tuning. The ROUGE metric provides a standardized way to compare the generated summaries with human-written references.

## 🙏 Acknowledgements

* **QLoRA Paper**: Efficient Finetuning of Quantized LLMs

* **Hugging Face Transformers & PEFT Libraries**: For providing the tools and frameworks for efficient LLM development.

* **DialogSum Dataset**: For providing a high-quality dataset for dialogue summarization.
